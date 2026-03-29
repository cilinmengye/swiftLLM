import torch
from torch import nn
import torch.distributed as dist

from swiftllm.worker.mconfigs.llamaconfig import LlamaModelConfig
from swiftllm.worker.layers.embed_head import RowVocabParallelEmbedding
from swiftllm.worker.layers.linear import ColParallelLinear
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
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
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
        self.lm_head = ColParallelLinear(config.hidden_size, config.vocab_size)

        self.name_to_module = {
            "lm_head.weight": self.lm_head,
        }
    
    def forward(
        self, 
        input_ids: torch.Tensor # [total_token_num]
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids)
        return self.lm_head(hidden_states)