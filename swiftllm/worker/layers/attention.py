import torch
from torch import nn
import vllm_flash_attn

from swiftllm.worker.infer_state import InferState
from swiftllm.worker.kernels.paged_attn import paged_attention
from swiftllm.worker.kernels.kvcache_mgmt import store_kvcache
from swiftllm.worker.mconfigs.llamaconfig import LlamaModelConfig
from swiftllm.engine_config import EngineConfig

class Attention(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.decoding_piggyback_stream = torch.cuda.Stream()
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor
    ) -> torch.Tensor:
        """
        可处理 batch 中混有prefill request 和 decode request

        注意我们开创了两个CUDA Stream: 
            1. torch 默认的CUDA Stream
            2. decoding_piggyback_stream CUDA Stream
        
        当我们没有指定使用何种 Stream 时, 会默认使用 default cuda stream. 所以接下来
        的store_kvcache 和针对 prefill request 的 flash attention 都是在 default 
        cuda stream
        
        针对 decode request 的 page attention 在 我们手动创建的 decoding_piggyback_stream 
        CUDA Stream; 
        
        不同 Stream 上的 CUDA kernel 是并发地, 即 GPU 在执行其中一个 cuda kernel 时
        会利用剩余的资源启动另一个 CUDA kernel。
        
        我们将 default cuda stream 类比为主干道: store_kvcache, flash attention
        在主干道上串行
        
        decoding_piggyback_stream 类比为次干道: page attention 在次干道上执行, 
        但是其必须等待store_kvcache执行完毕
        """
        infer_state = InferState.get_inferstate()
        k_cache = infer_state.k_cache
        v_cache = infer_state.v_cache
        block_table = infer_state.gpu_block_table
        model_config = LlamaModelConfig.get_model_config()
        engine_config = EngineConfig.get_engine_config()

        if not infer_state.ignore_kvcache:
            store_kvcache(
                k, v,
                k_cache, v_cache,
                block_table,
                model_config,
                engine_config,
                infer_state,
                self.layer_id
            )
        store_kvcache_event = torch.cuda.Event()
        store_kvcache_event.record()

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
            ).reshape(-1, o.shape[-1])

        if infer_state.num_decoding_seqs > 0:
            assert not infer_state.ignore_kvcache
            with torch.cuda.stream(self.decoding_piggyback_stream):
                torch.cuda.current_stream().wait_event(store_kvcache_event)
                #  前 num_prefill_tokens 行是 prefill 序列的 query，切片后只留解码 query
                paged_attention(
                    q[infer_state.num_prefill_tokens:, :, :],
                    k_cache, v_cache, block_table,
                    model_config, engine_config, infer_state,
                    self.layer_id,
                    o[infer_state.num_prefill_tokens:, :],
                )
                event = torch.cuda.Event()
                event.record()
            torch.cuda.default_stream().wait_event(event)
        
        return o