"""
词嵌入层我们习惯其 shape == (vocab_size, embedding_size), 若
沿 vocab_size 方向进行分割, 则我们称其为 RowVocabParallelEmbedding,
反之, 我们称其为 CowVocabParallelEmbedding
"""
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

class VocabEmbeddingBase(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        embedding_size: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(vocab_size, embedding_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def load_weight(self, weight: torch.Tensor, weight_name: str = None):
        raise NotImplementedError


class RowVocabParallelEmbedding(VocabEmbeddingBase):
    """
    RowVocabParallelEmbedding 得到的是不完整的 Tensor
    我们需要通过 All Reduec Sum 得到完整的 Tensor
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
    ):
        tp_size = dist.get_world_size()
        super().__init__(vocab_size // tp_size, embedding_size)
        
        assert vocab_size % self.tp_size == 0
        self.vocab_size = vocab_size
        self.num_embeddings_per_partition = self.vocab_size // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            # 首先设置掩码, 让不属于本GPU映射范围的下标为0
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 然后让 x 不要超过当前GPU能映射的下标范围
            x = mask * (x - self.vocab_start_idx)

        y = F.embedding(x, self.weight)
        
        if self.tp_size > 1:
            # 由于0也属于某个映射下标，所以最终映射出来还是会有值
            # 我们这里就是要清除这个值，让其值为0
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y

    def load_weight(self, weight: torch.Tensor, weight_name: str = None):
        shard_size = self.weight.data.size(0)               # 注意我们此时的weight是已经做好TP分片后的大小
        start_idx = self.tp_rank * shard_size               # 计算起始位置
        weight = weight.narrow(0, start_idx, shard_size)    # narrow(维度, 起始位置, 长度)
        self.weight.data.copy_(weight)                      # 同步式拷贝，此时cuda并不是异步的


    