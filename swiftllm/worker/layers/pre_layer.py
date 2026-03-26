import torch
import torch.distributed as dist

from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.weight import LlamaWeight

class LlamaPreLayer:
    def __init__(
        self,
        model_config: LlamaModelConfig,
        weights: LlamaWeight,
    ):
        self.model_config = model_config
        self.weights = weights
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()
    
    def forward(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        # 因为已经有tensor的weight，即self.weights.wte，所以不使用nn.Embedding
        # 因为nn.Embedding会内部自带 self.weight，如果使用nn.Embedding，那么我还需要将weight copy一下
        input_embdings = torch.embedding(self.weights.wte, input_ids, padding_idx=-1)
        return input_embdings
    