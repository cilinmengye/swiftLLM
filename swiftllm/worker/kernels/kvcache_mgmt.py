import torch
import triton
import triton.language as tl

from swiftllm.worker.mconfigs.llamaconfig import LlamaModelConfig
from swiftllm.engine_config import EngineConfig
from swiftllm.worker.infer_state import InferState
from swiftllm.utils import cdiv

@triton.jit
def _fwd_kvcache_mgmt_prefill_kernel(
    k_cache: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    v_cache: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    k: torch.Tensor,	# [num_prefill_tokens, num_kv_heads, head_dim], contiguous
    v: torch.Tensor,	# [num_prefill_tokens, num_kv_heads, head_dim], contiguous
    block_table: torch.Tensor,  # [*, max_blocks_per_seq], contiguous
    seq_ids: torch.Tensor,  # [num_prefill_seqs], contiguous
    prefill_seq_start_locs: torch.Tensor,  # [num_prefill_seqs], contiguous
    prefill_seq_lens: torch.Tensor,  # [num_prefill_seqs], contiguous
    cur_layer: int,

    num_layers: tl.constexpr,
    num_kv_heads: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
):
    # grid shape: [num_prefill_seqs, cdiv(max_prefill_len, block_size)]
    my_batch_id = tl.program_id(0)
    my_block_id = tl.program_id(1)
    my_seq_len = tl.load(prefill_seq_lens + my_batch_id)
    my_seq_start_loc = tl.load(prefill_seq_start_locs + my_batch_id)
    if my_block_id*block_size >= my_seq_len:
        return
    
    my_token_range = tl.arange(0, block_size).to(tl.int64) + my_block_id*block_size + my_seq_start_loc
    my_seq_id = tl.load(seq_ids + my_batch_id)
    my_block_index = tl.load(block_table + my_seq_id*max_blocks_per_seq + my_block_id).to(tl.int64)

    offs_kv = (my_token_range*num_kv_heads*head_dim).to(tl.int64)[:, None, None] + (tl.arange(0, num_kv_heads)*head_dim)[None, :, None] + tl.arange(0, head_dim)[None, None, :]
    offs_kvcache = (my_block_index*num_layers+cur_layer)*num_kv_heads*block_size*head_dim + \
        (tl.arange(0, num_kv_heads)*block_size*head_dim)[None, :, None] + \
        (tl.arange(0, block_size)*head_dim)[:, None, None] + \
        tl.arange(0, head_dim)[None, None, :]
    
    mask = (my_token_range < my_seq_len + my_seq_start_loc)[:, None, None]
    tl.store(k_cache + offs_kvcache, tl.load(k + offs_kv, mask=mask), mask=mask)
    tl.store(v_cache + offs_kvcache, tl.load(v + offs_kv, mask=mask), mask=mask)

@triton.jit
def _fwd_kvcache_mgmt_decoding_kernel(
    k_cache: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    v_cache: torch.Tensor,	# [num_blocks, num_layers, num_kv_heads, block_size, head_dim], contiguous
    k: torch.Tensor,	# [num_decoding_seqs, num_kv_heads, head_dim], contiguous
    v: torch.Tensor,	# [num_decoding_seqs, num_kv_heads, head_dim], contiguous
    block_table: torch.Tensor,  # [*, max_blocks_per_seq], contiguous
    decoding_seq_ids: torch.Tensor,  # [num_decoding_seqs], contiguous
    decoding_seq_lens: torch.Tensor,  # [num_decoding_seqs], contiguous
    cur_layer: int,

    num_layers: tl.constexpr,
    num_kv_heads: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
):
    # grid shape: [num_decoding_seqs]
    my_batch_id = tl.program_id(0).to(tl.int64)
    my_seq_id = tl.load(decoding_seq_ids + my_batch_id)
    my_seq_len = tl.load(decoding_seq_lens + my_batch_id)
    my_block_id = (my_seq_len-1) // block_size
    my_block_offset = (my_seq_len-1) % block_size
    my_block_index = tl.load(block_table + my_seq_id*max_blocks_per_seq + my_block_id).to(tl.int64)

    offs_kv = my_batch_id*num_kv_heads*head_dim + (tl.arange(0, num_kv_heads)*head_dim)[:, None] + tl.arange(0, head_dim)[None, :]
    offs_kvcache = (my_block_index*num_layers+cur_layer)*num_kv_heads*block_size*head_dim + (tl.arange(0, num_kv_heads)*block_size*head_dim)[:, None] + my_block_offset*head_dim + tl.arange(0, head_dim)[None, :]

    tl.store(k_cache + offs_kvcache, tl.load(k + offs_kv))
    tl.store(v_cache + offs_kvcache, tl.load(v + offs_kv))

def store_kvcache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    model_config: LlamaModelConfig,
    engine_config: EngineConfig,
    infer_state: InferState,
    cur_layer: int
):
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert k_cache.is_contiguous()
    assert v_cache.is_contiguous()
    assert block_table.is_contiguous()
    assert infer_state.seq_ids.is_contiguous()
    assert infer_state.decoding_seq_lens.is_contiguous()

    # 当前 rank 传进来的 k / v 已经只包含 local_num_kv_heads；因此这里不能继续使用
    # model_config.num_kv_heads 这个“全局 head 数”去解释它们，否则 Triton kernel 在
    # 计算 stride 时就会把 local shard 当成 full tensor 来跨行寻址，直接读错输入。
    #
    # 同理，当前 rank 持有的 k_cache / v_cache 也应当只为本 rank 的 local KV heads
    # 分配空间。只有这样，kernel 内部使用 local_num_kv_heads 计算出的线性地址，才能与
    # cache 的实际内存布局保持一致。
    local_num_kv_heads = k.shape[1]
    head_dim = k.shape[2]
    assert v.shape[1] == local_num_kv_heads
    assert v.shape[2] == head_dim
    assert k_cache.shape[2] == local_num_kv_heads
    assert v_cache.shape[2] == local_num_kv_heads
    assert head_dim == model_config.head_dim

    if infer_state.num_prefill_seqs > 0:
        grid = (infer_state.num_prefill_seqs, cdiv(infer_state.max_prefill_len, engine_config.block_size))
        _fwd_kvcache_mgmt_prefill_kernel[grid](
            k_cache, v_cache,
            k, v,
            block_table,
            infer_state.seq_ids, infer_state.prefill_seq_start_locs, infer_state.prefill_seq_lens,
            cur_layer,
            model_config.num_layers, local_num_kv_heads, engine_config.block_size, head_dim, engine_config.max_blocks_per_seq
        )

    if infer_state.num_decoding_seqs > 0:
        grid = (infer_state.num_decoding_seqs,)
        _fwd_kvcache_mgmt_decoding_kernel[grid](
            k_cache, v_cache,
            k[infer_state.num_prefill_tokens:, :, :],
            v[infer_state.num_prefill_tokens:, :, :],
            block_table,
            infer_state.seq_ids[infer_state.num_prefill_seqs:],
            infer_state.decoding_seq_lens,
            cur_layer,
            model_config.num_layers, local_num_kv_heads, engine_config.block_size, head_dim, engine_config.max_blocks_per_seq
        )

        # for my_batch_id in range(infer_state.num_decoding_seqs):
        #     my_k = k[infer_state.num_prefill_tokens+my_batch_id]    # [num_kv_heads, head_dim]
        #     my_v = v[infer_state.num_prefill_tokens+my_batch_id]    # [num_kv_heads, head_dim]
        #     my_new_token_pos = infer_state.decoding_seq_lens[my_batch_id] - 1
        #     my_block_index = block_table[infer_state.seq_ids[infer_state.num_prefill_seqs+my_batch_id]][my_new_token_pos // engine_config.block_size]
        #     my_block_offset = my_new_token_pos % engine_config.block_size

        #     k_cache[my_block_index][cur_layer][:, my_block_offset, :] = my_k
        #     v_cache[my_block_index][cur_layer][:, my_block_offset, :] = my_v
