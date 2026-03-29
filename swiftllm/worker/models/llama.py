import torch
from torch import nn
import torch.distributed as dist

from swiftllm.worker.mconfigs.llamaconfig import LlamaModelConfig
from swiftllm.worker.layers.embed_head import RowVocabParallelEmbedding
from swiftllm.worker.layers.embed_head import ParallelLMHead
from swiftllm.worker.layers.linear import RowParallelLinear
from swiftllm.worker.layers.linear import GateUPParallelLinear
from swiftllm.worker.layers.linear import QKVParallelLinear
from swiftllm.worker.layers.layernorm import RMSNorm
from swiftllm.worker.layers.attention import Attention
from swiftllm.worker.kernels.silu_and_mul import silu_and_mul_inplace
from swiftllm.worker.kernels.rotary_emb import rotary_embedding_inplace
from swiftllm.worker.infer_state import InferState

class LlamaMLP(nn.Module):
    def __init__(
        self,
        config: LlamaModelConfig,
        layer_id: int
    ):
        super().__init__()
        self.layer_id = layer_id

        self.gate_up_proj = GateUPParallelLinear(
            config.hidden_size,
            [config.ffn_inter_dim, config.ffn_inter_dim]
        )   # 矩阵合并优化
        self.down_proj = RowParallelLinear(
            config.ffn_inter_dim,
            config.hidden_size
        )

        # 只保存 "权重名 -> 目标加载模块" 的映射
        self.name_to_module = {
            f"model.layers.{self.layer_id}.mlp.gate_proj.weight": self.gate_up_proj,
            f"model.layers.{self.layer_id}.mlp.up_proj.weight": self.gate_up_proj,
            f"model.layers.{self.layer_id}.mlp.down_proj.weight": self.down_proj,
        }
    
    def forward(
        self,
        hidden_state: torch.Tensor 
    ) -> torch.Tensor:
        gate_up = self.gate_up_proj(hidden_state)
        silu_and_mul_inplace(gate_up)
        ffn_inter_dim = gate_up.shape[1] // 2
        hidden_state = self.down_proj(gate_up[:, :ffn_inter_dim])
        return hidden_state


class LlamaAttention(nn.Module):
    def __init__(
        self,
        config: LlamaModelConfig,
        layer_id: int
    ):
        super().__init__()
        self.layer_id = layer_id

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            config.head_dim,
            config.num_q_heads,
            config.num_kv_heads
        )   # 此处是一个优化, 我们将q_proj, k_proj, v_proj
            # 合并成一个大矩阵一起参与计算，可以减少kernel启
            # 动次数, 减少延迟, 提高计算强度充分利用计算资源
        self.attn = Attention(layer_id)
        self.o_proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size
        )

        tp_size = dist.get_world_size()
        assert config.num_q_heads % tp_size == 0
        assert config.num_kv_heads % tp_size == 0
        self.num_heads = config.num_q_heads // tp_size
        self.num_kv_heads = config.num_kv_heads // tp_size
        self.head_dim = config.head_dim
        
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.name_to_module = {
            f"model.layers.{self.layer_id}.self_attn.q_proj.weight": self.qkv_proj,
            f"model.layers.{self.layer_id}.self_attn.k_proj.weight": self.qkv_proj,
            f"model.layers.{self.layer_id}.self_attn.v_proj.weight": self.qkv_proj,
            f"model.layers.{self.layer_id}.self_attn.o_proj.weight": self.o_proj,
        }
    
    def forward(
        self,
        hidden_state: torch.Tensor
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_state)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Problem 2 的根因在这里：
        # 当前新版为了减少 GEMM / kernel launch，把 q_proj、k_proj、v_proj 合并成了
        # 一次 fused QKV projection；但 split() 返回的是原始 qkv 大张量上的 view。
        # 这意味着后面的 q/k/v 虽然 shape 已经被解释成 [tokens, heads, head_dim]，
        # 底层 storage / stride 仍然可能延续 packed QKV 的布局，而不是旧 kernel
        # 所假定的“独立且连续”的三块张量。
        #
        # 下游两个路径都仍然依赖这个旧契约：
        # 1. store_kvcache() 要求 k / v.is_contiguous()；
        # 2. paged_attention() 要求 q.is_contiguous()。
        #
        # 因此这里不能只修 k / v，否则 decode 阶段仍可能在 q 上再次失败。
        # 这次采取的策略是：保留 fused QKV 架构不动，只在 attention 的边界处把
        # q / k / v 一次性收口成连续张量，让现有 Triton kernel 契约继续成立。
        # 这不是长期唯一方案，但它是当前最小、最稳妥的 correctness 修复。
        q = q.view(-1, self.num_heads, self.head_dim).contiguous()
        k = k.view(-1, self.num_kv_heads, self.head_dim).contiguous()
        v = v.view(-1, self.num_kv_heads, self.head_dim).contiguous()
        rotary_embedding_inplace(
            q,
            k,
            InferState.get_inferstate()
        )
        o = self.attn(q, k, v, hidden_state)    # 因为后续 hidden_state 不会再被使用了, 所以为避免
                                                # 在开辟空间的开销, 我们直接复用 hidden_state 的空间
        # 释放资源
        q = None
        k = None
        v = None

        output = self.o_proj(o)
        return output

class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaModelConfig,
        layer_id: int
    ) -> None:
        super().__init__()
        self.layer_id = layer_id

        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = LlamaAttention(config, layer_id)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = LlamaMLP(config, layer_id)

        self.name_to_module = {
            f"model.layers.{self.layer_id}.input_layernorm.weight": self.input_layernorm,
            f"model.layers.{self.layer_id}.post_attention_layernorm.weight": self.post_attention_layernorm,
        }
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual_buf: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.input_layernorm(hidden_states, residual_buf)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states, residual_buf)
        hidden_states = self.mlp(hidden_states)
        return hidden_states


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaModelConfig
    ) -> None:
        super().__init__()
        self.embed_tokens = RowVocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size
        )
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, i) for i in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.name_to_module = {
            "model.embed_tokens.weight": self.embed_tokens,
            "model.norm.weight": self.norm,
        }
    
    def forward(
        self,
        input_ids: torch.Tensor, 
    ):
        hidden_states = self.embed_tokens(input_ids)
        residual_buf = torch.zeros_like(hidden_states)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                residual_buf
            )
        hidden_states += residual_buf
        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    def __init__(
        self,
        config: LlamaModelConfig
    ) -> None:
        super().__init__()
        self.model = LlamaModel(config)
        # lm_head 不能继续复用普通 ColParallelLinear。
        # 原因是 TP 下每个 rank 只持有 vocab 的一个 shard，我们最终必须把各 rank 的
        # shard logits 聚合到 rank0 后，才能在完整 vocab 上做采样。
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

        self.name_to_module = {
            "lm_head.weight": self.lm_head,
        }

    def forward(
        self,
        input_ids: torch.Tensor # [total_token_num]
    ) -> torch.Tensor:
        # 这里 forward 只负责 transformer 主体，把 server-facing 的“取最后一个 token
        # 做采样”语义延后到 compute_logits()。
        # 这样做有两个直接好处：
        # 1. profile / prefill 路径不会再对所有 token 都做一次超大的 vocab projection；
        # 2. 上层 sampler 看到的 logits 行数会重新对齐到 batch_size，而不是 num_tokens。
        return self.model(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor):
        # server-facing 语义是一条请求只生成一个 next token，因此这里只挑出每条请求
        # 最后一个 token 的 hidden states，再做 lm_head。
        last_token_hidden_states = self.lm_head.get_last_token_hidden_states(hidden_states)
        return self.lm_head(last_token_hidden_states)
