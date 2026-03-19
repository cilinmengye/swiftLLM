import torch

from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import LlamaWeight
from swiftllm.worker.kernels.rmsnorm import rmsnorm_inplace
from swiftllm.worker.infer_state import LlamaInferState
from swiftllm.worker.kernels.linear import linear

class LlamaPostLayer:
    def __init__(
        self,
        model_config: LlamaModelConfig,
        weights: LlamaWeight,
    ):
        self.model_config = model_config
        self.weights = weights
    
    def forward(
        self,
        input_embds: torch.Tensor,	# [num_total_tokens, hidden_size]
        infer_state: LlamaInferState
    ) -> torch.Tensor:
        # Slice to get the last token embedding for each request
        # 如下代码主要目的为取出 flatten request 中每个request中最后一个token进行RMS NORM and lm_head, sampler
        last_token_indices = torch.cat(
            (
                # infer_state.prefill_seq_start_locs 为 [b*s, *] 中 prefill request 在这种 flatten request 的下标
                # infer_state.prefill_seq_start_locs + infer_state.prefill_seq_lens - 1 求得  prefill request 最后一个token 在这种 flatten request 的下标
                # torch.arange(infer_state.num_prefill_tokens, infer_state.num_tokens) 求得 decode request 最后一个token 在这种 flatten request 的下标
                infer_state.prefill_seq_start_locs + infer_state.prefill_seq_lens - 1,
                torch.arange(infer_state.num_prefill_tokens, infer_state.num_tokens, device=input_embds.device, dtype=torch.int32)
            ), dim=0
        )
        last_input = torch.empty((infer_state.batch_size, self.model_config.hidden_size), device=input_embds.device, dtype=input_embds.dtype)
        # 注意此处 last_token_indices shape[0] == infer_state.batch_size
        last_input[:, :] = input_embds[last_token_indices, :]
        # Apply RMS-norm
        rmsnorm_inplace(
            last_input,
            self.weights.final_norm,
            self.model_config.rms_norm_eps
        )
        logits = linear(last_input, self.weights.lm_head)    # [batch_size, vocab_size]
        output_tokens = torch.argmax(logits, dim=1)
        return output_tokens
    