import os
import json
import torch

_LLAMAMODELCONFIG = None


def _parse_torch_dtype(torch_dtype: str | torch.dtype | None) -> torch.dtype:
    """
    Convert the HuggingFace config field `torch_dtype` into an actual `torch.dtype`.

    config.json usually stores this field as a string such as "float16" or "bfloat16".
    We parse it once in the model config so that the worker can set the correct default
    dtype *before* constructing CUDA parameters.
    """
    if torch_dtype is None:
        return torch.float16

    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype

    if isinstance(torch_dtype, str):
        attr_name = torch_dtype.removeprefix("torch.")
        parsed_dtype = getattr(torch, attr_name, None)
        if isinstance(parsed_dtype, torch.dtype):
            return parsed_dtype

    raise ValueError(f"Unsupported torch dtype in model config: {torch_dtype}")


class LlamaModelConfig:
    """
    The configuration of a LLaMA model (including LLaMA 1/2/3).
    """
    
    def __init__(
        self,
        model_config: dict
    ):
        """
        Initialize a LLaMA model configuration from a dict, which should be generated
        from a huggingface transformers config.json file.
        """
        
        assert model_config["model_type"] == "llama"
        self.num_layers = model_config["num_hidden_layers"]
        self.num_q_heads = model_config["num_attention_heads"]
        self.num_kv_heads = model_config.get("num_key_value_heads", self.num_q_heads)
        self.hidden_size = model_config["hidden_size"]
        self.head_dim = self.hidden_size // self.num_q_heads
        self.vocab_size = model_config["vocab_size"]
        self.max_position_embeddings = model_config["max_position_embeddings"]
        self.ffn_inter_dim = model_config["intermediate_size"]
        self.rotary_base = model_config.get("rope_theta", model_config.get("rotary_base", 10000))
        self.rms_norm_eps = model_config["rms_norm_eps"]
        self.torch_dtype = _parse_torch_dtype(model_config.get("torch_dtype"))
        self.rope_scaling = model_config.get("rope_scaling", 1.0)
        self.rope_theta = model_config.get("rope_theta", 10000)
        if self.rope_scaling is None:
            self.rope_scaling = 1.0
        assert model_config["hidden_act"] == "silu"

        # 需要通过`profile_num_blocks`来修正此值
        self.num_gpu_blocks = -1

    def get_kvslot_size(self, dtype: torch.dtype = torch.float16) -> int:
        """
        Get the size of one kv slot (the kv cache of one token) (in bytes)
        """
        return (2 * self.num_layers * self.num_kv_heads * self.head_dim) * dtype.itemsize

    def get_rotary(self) -> tuple[torch.Tensor, torch.Tensor]:
        rope_scaling = self.rope_scaling
        base = self.rope_theta
        max_position_embeddings = self.max_position_embeddings

        # Handle the case where rope_scaling is a dictionary (Llama 3.2)
        if isinstance(rope_scaling, dict):
            rope_type = rope_scaling.get('rope_type')
            assert rope_type == "llama3"

            scaling_factor = rope_scaling.get('factor', 4.0)
            low_freq_factor = rope_scaling.get('low_freq_factor', 1.0)
            high_freq_factor = rope_scaling.get('high_freq_factor', 1.0)
            
            original_max_position_embeddings = rope_scaling.get('original_max_position_embeddings', max_position_embeddings)

            # Calculate maximum sequence length based on scaling factor
            max_seq_len = int(original_max_position_embeddings * scaling_factor)

            # Generate position indices
            dim = self.head_dim
            t = torch.arange(max_seq_len + 128, device="cuda", dtype=torch.float32)

            # Create frequency array with dimensions split between low and high frequency parts
            dim_half = dim // 2
            split_point = int(dim_half * low_freq_factor / (low_freq_factor + high_freq_factor))

            # Apply different scaling factors to different parts of the frequency spectrum
            inv_freq_low = 1.0 / (base ** (torch.arange(0, split_point * 2, 2, device="cuda", dtype=torch.float32) / dim))
            inv_freq_high = 1.0 / (base ** (torch.arange(split_point * 2, dim, 2, device="cuda", dtype=torch.float32) / dim))

            # Apply scaling factors
            low_positions = t / low_freq_factor
            high_positions = t / high_freq_factor

            # Calculate frequencies for both parts
            freqs_low = torch.outer(low_positions, inv_freq_low)
            freqs_high = torch.outer(high_positions, inv_freq_high)

            # Combine frequencies
            freqs = torch.cat([freqs_low, freqs_high], dim=-1)
        else:
            # Original implementation for scalar rope_scaling
            rope_scaling_factor = rope_scaling      # 标量，比如 1.0 或 2.0，表示支持扩展的序列长度的大小
            max_seq_len = max_position_embeddings * rope_scaling_factor     # 支持的最大序列长度， max_position_embeddings 为原始预训练时支持的序列长度

            inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2, device="cuda", dtype=torch.float32) / self.head_dim))
            t = torch.arange(max_seq_len + 128, device="cuda", dtype=torch.float32) / rope_scaling_factor       # t 决定缓存表实际有多少行（能覆盖多长的序列）， / rope_scaling_factor 目的是将长度缩放回原来的预训练序列长度
            freqs = torch.outer(t, inv_freq)

        _cos_cached = torch.cos(freqs).to(torch.float16)
        _sin_cached = torch.sin(freqs).to(torch.float16)

        return _cos_cached, _sin_cached
    
    @staticmethod
    def set_model_config(model_path: str):
        with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
            model_config_dict = json.loads(f.read())
        # model_config_dict is dict read from model config.json
        # print(model_config_dict)
        global _LLAMAMODELCONFIG

        if _LLAMAMODELCONFIG is not None:
            _LLAMAMODELCONFIG = None

        _LLAMAMODELCONFIG = LlamaModelConfig(model_config_dict)
    
    @staticmethod
    def get_model_config():
        assert _LLAMAMODELCONFIG is not None
        return _LLAMAMODELCONFIG
