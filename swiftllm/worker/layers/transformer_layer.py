import torch
import vllm_flash_attn

from swiftllm.model_config import LlamaModelConfig
from swiftllm.engine_config import EngineConfig
from swiftllm.worker.weight import LlamaTransformerLayerWeight
from swiftllm.worker.infer_state import LlamaInferState

from swiftllm.worker.kernels.linear import linear
from swiftllm.worker.kernels.rmsnorm import fused_add_rmsnorm_inplace
from swiftllm.worker.kernels.rotary_emb import rotary_embedding_inplace
from swiftllm.worker.kernels.paged_attn import paged_attention
from swiftllm.worker.kernels.kvcache_mgmt import store_kvcache
from swiftllm.worker.kernels.silu_and_mul import silu_and_mul_inplace

class LlamaTransformerLayer:
    def __init__(
        self,
        model_config: LlamaModelConfig,
        engine_config: EngineConfig,
        weight: LlamaTransformerLayerWeight,
        decoding_piggyback_stream: torch.cuda.Stream,
        layer_id: int
    ):
        self.model_config = model_config
        self.engine_config = engine_config
        self.weight = weight
        self.decoding_piggyback_stream = decoding_piggyback_stream
        self.layer_id = layer_id
    
    def forward(
        self,
        input_embds: torch.Tensor,  # [num_tokens, hidden_size]
        residual_buf: torch.Tensor, # [num_tokens, hidden_size]
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        infer_state: LlamaInferState,
    ) -> torch.Tensor:
        # (fused) Add last layer's residual, and perform RMSNorm
        # Before: input_embds is the output of the last FFN block, and residual_buf
        #         is the residual to be added to input_embds
        # After: input_embds will be RMSNorm(input_embds + residual_buf), and
        #        residual_buf will be input_embds + residual_buf (which will be
        #        used as the residual after the attention block)
        # 即fused_add_rmsnorm_inplace 融合了 residual add 和 rmsnorm 操作
        # residual_buf 为提前开辟的显存空间，可以避免 residual tensor 频繁地分配、释放
        # 其核心操作为： input_embds =  RMSNorm(input_embds + residual_buf); residual_buf = input_embds + residual_buf
        fused_add_rmsnorm_inplace(
            input_embds,
            residual_buf,
            self.weight.attn_norm,
            self.model_config.rms_norm_eps
        )

        # Calculate QKV
        q = linear(input_embds, self.weight.q_proj)		# [num_total_tokens, hidden_size]
        k = linear(input_embds, self.weight.k_proj)		# [num_total_tokens, num_kv_heads*head_dim]
        v = linear(input_embds, self.weight.v_proj)		# [num_total_tokens, num_kv_heads*head_dim]
        q = q.view(-1, self.model_config.num_q_heads,  self.model_config.head_dim)	# [num_total_tokens, num_q_heads, head_dim]
        k = k.view(-1, self.model_config.num_kv_heads, self.model_config.head_dim)	# [num_total_tokens, num_kv_heads, head_dim]
        v = v.view(-1, self.model_config.num_kv_heads, self.model_config.head_dim)	# [num_total_tokens, num_kv_heads, head_dim]

        # Rotary emb
        rotary_embedding_inplace(
            q,
            k,
            infer_state
        )

        # 注意我们开创了两个CUDA Stream: 1. torch 默认的CUDA Stream, 
        # 2. decoding_piggyback_stream CUDA Stream
        # 当我们没有指定使用何种Stream时，会默认使用 default cuda stream
        # 所以接下来的 store_kcache 和 针对prefill request的flash attention都是在default cuda stream
        # 针对decode request的page attention在我们手动创建的decoding_piggyback_stream CUDA Stream
        # 不同Stream上的CUDA kernel是并发地，即GPU在执行其中一个cuda kernel时会利用剩余的资源启动另一个CUDA kernel
        # 我们将default cuda stream 类比为主干道: store_kvcache , flash attention 在主干道上串行
        # decoding_piggyback_stream 类比为次干道: page attention 在次干道上执行，但是其必须等待store_kvcache执行完毕
        if not infer_state.ignore_kvcache:
            store_kvcache(
                k, v,
                k_cache, v_cache,
                block_table,
                self.model_config,
                self.engine_config,
                infer_state,
                self.layer_id
            )
        store_kvcache_event = torch.cuda.Event()
        store_kvcache_event.record()

        # Attention
        o = input_embds    # [num_total_tokens, hidden_size]
        if infer_state.num_prefill_seqs > 0:
            # Here the performance of vLLM's flash attention is better than us,
            # so use vllm_flash_attn
            o[:infer_state.num_prefill_tokens, :] = vllm_flash_attn.flash_attn_varlen_func(
                q[:infer_state.num_prefill_tokens, :, :],
                k[:infer_state.num_prefill_tokens, :, :],
                v[:infer_state.num_prefill_tokens, :, :],
                infer_state.prefill_seq_start_locs_with_end,
                infer_state.prefill_seq_start_locs_with_end,
                infer_state.max_prefill_len,
                infer_state.max_prefill_len,
                softmax_scale=infer_state.softmax_scale,
                causal=True
            ).reshape(-1, self.model_config.hidden_size)
            # prefill_attention(
            #     q, k, v, o[:infer_state.num_prefill_tokens, :],
            #     self.model_config, self.engine_config, infer_state
            # )
        if infer_state.num_decoding_seqs > 0:
            assert not infer_state.ignore_kvcache
            with torch.cuda.stream(self.decoding_piggyback_stream):
                torch.cuda.current_stream().wait_event(store_kvcache_event)
                #  前 num_prefill_tokens 行是 prefill 序列的 query，切片后只留解码 query
                paged_attention(
                    q[infer_state.num_prefill_tokens:, :, :],
                    k_cache, v_cache, block_table,
                    self.model_config, self.engine_config, infer_state,
                    self.layer_id,
                    o[infer_state.num_prefill_tokens:, :],
                )
                event = torch.cuda.Event()
                event.record()
            torch.cuda.default_stream().wait_event(event)
        
        # Output GEMM
        o = linear(o, self.weight.o_proj)	# [num_total_tokens, hidden_size]

        # residual & FFN norm
        fused_add_rmsnorm_inplace(o, residual_buf, self.weight.ffn_norm, self.model_config.rms_norm_eps)
        q = None
        k = None
        v = None

        # FFN
        up_gate_proj = linear(o, self.weight.up_gate_proj)
        silu_and_mul_inplace(up_gate_proj)
        ffn_out = linear(up_gate_proj[:, :self.model_config.ffn_inter_dim], self.weight.down_proj)

        return ffn_out
    