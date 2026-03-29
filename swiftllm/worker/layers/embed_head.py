"""
词嵌入层我们习惯其 shape == (vocab_size, embedding_size), 若
沿 vocab_size 方向进行分割, 则我们称其为 RowVocabParallelEmbedding,
反之, 我们称其为 CowVocabParallelEmbedding
"""
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from swiftllm.worker.infer_state import InferState


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
    RowVocabParallelEmbedding 得到的是不完整的 Tensor。
    我们需要通过 AllReduce Sum 得到完整的 Tensor。
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
            # 首先设置掩码, 让不属于本 GPU 映射范围的 token id 变成 0。
            # 后面 embedding 之后我们会再把这些位置清零，然后通过 all_reduce
            # 把各 rank 负责的词表分片重新合成为完整 embedding。
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)

        y = F.embedding(x, self.weight)

        if self.tp_size > 1:
            # 由于 0 本身也可能是合法 token id，所以这里必须显式把“不归本 rank 负责”
            # 的位置清成 0；否则 all_reduce 之后会把错误 embedding 混进去。
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y

    def load_weight(self, weight: torch.Tensor, weight_name: str = None):
        shard_size = self.weight.data.size(0)               # 注意我们此时的 weight 是已经做好 TP 分片后的大小
        start_idx = self.tp_rank * shard_size               # 分片起点取决于当前 rank，而不是 world size
        weight = weight.narrow(0, start_idx, shard_size)    # narrow(维度, 起始位置, 长度)
        self.weight.data.copy_(weight)                      # 同步式拷贝，此时 cuda 并不是异步的


class ParallelLMHead(RowVocabParallelEmbedding):
    """
    TP 版 lm_head。

    这里和 embedding 共享同一套“沿 vocab 维切分权重”的装载方式，但 forward 语义不同：
    1. 输入不再是 token ids，而是已经选好的 last-token hidden states；
    2. 每个 rank 只计算自己 vocab shard 的 logits；
    3. TP>1 时只在 rank0 聚合完整 vocab logits，并由 rank0 继续采样。

    之所以必须只对 last-token hidden states 做 lm_head，而不是对所有 token 做，
    是因为 server-facing 语义是一条请求只生成一个 next token。对全部 token 做 vocab
    projection 不但会让 profile / prefill 路径额外占显存，还会把返回值错误地扩成
    `num_tokens` 个 token，而不是 `batch_size` 个 token。
    """

    def _get_last_token_indices(self, hidden_states: torch.Tensor) -> torch.Tensor:
        infer_state = InferState.get_inferstate()

        # Prefill 部分需要取“每条请求最后一个 prompt token”；
        # Decode 部分本来就是一条请求一个 token，所以它们在打平 Tensor 中正好位于
        # [num_prefill_tokens, num_tokens) 这段区间。
        last_token_indices = torch.cat(
            (
                infer_state.prefill_seq_start_locs + infer_state.prefill_seq_lens - 1,
                torch.arange(
                    infer_state.num_prefill_tokens,
                    infer_state.num_tokens,
                    device=hidden_states.device,
                    dtype=torch.int32,
                ),
            ),
            dim=0,
        )
        return last_token_indices.to(dtype=torch.int64)

    def get_last_token_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        last_token_indices = self._get_last_token_indices(hidden_states)
        return hidden_states.index_select(0, last_token_indices).contiguous()

    def forward(self, hidden_states: torch.Tensor):
        logits = F.linear(hidden_states, self.weight)
        if self.tp_size > 1:
            # 每个 rank 只持有 vocab 的一个 shard，因此本地 logits 只是“局部词表 logits”。
            # 只有 rank0 拿到所有 shard 并拼接之后，argmax 才是全 vocab 上正确的 next token。
            gathered_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, gather_list=gathered_logits, dst=0)
            logits = torch.cat(gathered_logits, dim=-1) if self.tp_rank == 0 else None
        return logits
