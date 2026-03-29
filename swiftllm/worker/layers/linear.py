"""
Tensor Parallel Linear Layer

我们按照习惯的 shape == (input_size, output_size) 来定义是 ColParallelLinear 
还是 RowParallelLinear; 其中 input_size 表示 tensor 经过 Linear Layer 之前的
维度, output_size 表示 tensor 经过 Linear Layer 之后的维度; 

其中若沿 input_size 维度切分, 即切分后 shape == (input_size / n, output_size), 
我们则定义为其 RowParallelLinear, 反之则为 ColParallelLinear
"""
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator

class LinearBase(nn.Module):
    def __init__(
        self, 
        input_size: int,
        output_size: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        # 一般情况下，我们习惯写成 shape==(input_size, output_size)
        # 但是在这里我们需要如下写法，因为load weight时，weight是按照
        # 如下shape进行存放的
        self.weight = nn.Parameter(torch.empty(output_size, input_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def load_weight(self, weight: torch.Tensor, weight_name: str = None):
        raise NotImplementedError


class ColParallelLinear(LinearBase):
    """
    典型用到 ColParallelLinear Layer 为: 
    1. attention q_proj, k_proj, v_proj
    2. mlp gate_proj, up_proj 
    3. lm_head
    """
    def __init__(
        self, 
        input_size,
        output_size,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)

    def load_weight(self, weight: torch.Tensor, weight_name: str = None):
        shard_size = self.weight.data.size(0)   # 注意在我们真正实现Linear维度时,是output_size为第一维
        # 分片起点必须由当前 rank 决定，而不是 world size。
        # 如果这里误用 tp_size，那么 rank0 也不会从 0 开始取 shard，TP 权重一定整体错位。
        start_idx = self.tp_rank * shard_size
        weight = weight.narrow(0, start_idx, shard_size)
        self.weight.data.copy_(weight)


class RowParallelLinear(LinearBase):
    """
    典型用到 RowParallelLinear Layer 为:
    1. attention o_proj
    2. mlp down_proj
    
    RowParallelLinear Layer 产生出来的 Tensor 是不完整的, 
    需要与其他GPU进行通信--AllReudce Sum, 让每个GPU上都有
    完整的, 可用于下一层进行计算的 Tensor
    """
    def __init__(
        self, 
        input_size,
        output_size,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight)
        if self.tp_size > 1:
            dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return y

    def load_weight(self, weight: torch.Tensor, weight_name: str = None):
        shard_size = self.weight.data.size(1)   # 注意在我们真正实现Linear维度时,是output_size为第一维
        # RowParallelLinear 沿输入维切分，因此这里同样必须按 tp_rank 选择本 rank 负责的列分片。
        start_idx = self.tp_rank * shard_size
        weight = weight.narrow(1, start_idx, shard_size)
        self.weight.data.copy_(weight)


class QKVParallelLinear(ColParallelLinear):
    """
    合并 q_proj, k_proj, v_proj 进行 GEMM
    """
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
    ):
        tp_size = dist.get_world_size()
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)

        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
        super().__init__(hidden_size, output_size)

    def load_weight(self, weight: torch.Tensor, weight_name: str = None):
        assert weight_name is not None

        if "q_proj" in weight_name:
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif "k_proj" in weight_name:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        elif "v_proj" in weight_name:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
        else:
            raise KeyError(f"Unexpected weight name: {weight_name}")

        local_weight = weight.chunk(self.tp_size, dim=0)[self.tp_rank]

        assert local_weight.shape[0] == shard_size, (
            f"Loaded shard rows mismatch for {weight_name}: "
            f"expected {shard_size}, got {local_weight.shape[0]}"
        )
        assert local_weight.shape[1] == self.weight.shape[1], (
            f"Loaded shard cols mismatch for {weight_name}: "
            f"expected {self.weight.shape[1]}, got {local_weight.shape[1]}"
        )

        target_weight = self.weight.data.narrow(0, shard_offset, shard_size)
        target_weight.copy_(local_weight)



class GateUPParallelLinear(ColParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
    ):
        """
        output_sizes:
            - output_sizes[0]: gate_proj output size
            - output_sizes[1]: up_proj output size
        """
        assert len(output_sizes) == 2, "GateUPParallelLinear expects two output sizes"
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes))
    
    def load_weight(self, weight: torch.Tensor, weight_name: str = None):
        assert weight_name is not None

        gate_out, up_out = self.output_sizes

        if "gate_proj" in weight_name:
            global_size = gate_out
            global_offset = 0
        elif "up_proj" in weight_name:
            global_size = up_out
            global_offset = gate_out
        else:
            raise KeyError(f"Unexpected weight name: {weight_name}")

        assert global_size % self.tp_size == 0, (
            f"Output size {global_size} must be divisible by tp_size {self.tp_size}"
        )
        assert global_offset % self.tp_size == 0, (
            f"Offset {global_offset} must be divisible by tp_size {self.tp_size}"
        )

        local_size = global_size // self.tp_size
        local_offset = global_offset // self.tp_size

        local_weight = weight.chunk(self.tp_size, dim=0)[self.tp_rank]
        target_weight = self.weight.data.narrow(0, local_offset, local_size)
        target_weight.copy_(local_weight)

        