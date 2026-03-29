import torch
from torch import nn
from swiftllm.worker.kernels.rmsnorm import fused_add_rmsnorm_inplace
from swiftllm.worker.kernels.rmsnorm import rmsnorm_inplace

class RMSNorm(nn.Module):
    def __init__(
        self, 
        hidden_size: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(
        self, 
        x: torch.Tensor,
        residual_buf: torch.Tensor = None
    ) -> torch.Tensor:
        if residual_buf is not None:
            fused_add_rmsnorm_inplace(
                x,
                residual_buf,
                self.weight,
                self.eps,
            )
        else:
            rmsnorm_inplace(
                x,
                self.weight,
                self.eps
            )
        return x

    def load_weight(self, weight: torch.Tensor, weight_name: str = None):
        self.weight.data.copy_(weight)