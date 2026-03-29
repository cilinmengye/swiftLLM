import dataclasses
import torch

_INFERSTATE = None

@dataclasses.dataclass
class InferState:
    batch_size: int
    num_tokens: int

    seq_ids: torch.Tensor   # [batch_size]
    softmax_scale: float    # Equal to 1/sqrt(head_dim)

    num_prefill_seqs: int
    num_prefill_tokens: int
    prefill_seq_start_locs: torch.Tensor # [batch_size]
    prefill_seq_start_locs_with_end: torch.Tensor # [batch_size+1], = prefill_seq_start_locs + [num_prefill_tokens]
    prefill_seq_lens: torch.Tensor # [batch_size]
    max_prefill_len: int

    num_decoding_seqs: int
    decoding_seq_lens: torch.Tensor # [batch_size]
    max_decoding_len: int

    seq_block_size: int
    num_seq_blocks: int

    position_cos: torch.Tensor	# [num_tokens, hidden_size]
    position_sin: torch.Tensor	# [num_tokens, hidden_size]

    ignore_kvcache: bool    # Skip storing the key/value cache, useful when profiling the number of kv blocks
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    gpu_block_table: torch.Tensor

    @staticmethod
    def get_inferstate():
        assert _INFERSTATE is not None
        return _INFERSTATE

    @staticmethod
    def set_inferstate(
        batch_size: int,
        num_tokens: int,
        seq_ids: torch.Tensor,
        softmax_scale: float,
        num_prefill_seqs: int,
        num_prefill_tokens: int,
        prefill_seq_start_locs: torch.Tensor,
        prefill_seq_start_locs_with_end: torch.Tensor,
        prefill_seq_lens: torch.Tensor,
        max_prefill_len: int,
        num_decoding_seqs: int,
        decoding_seq_lens: torch.Tensor,
        max_decoding_len: int,
        seq_block_size: int,
        num_seq_blocks: int,
        position_cos: torch.Tensor,
        position_sin: torch.Tensor,
        ignore_kvcache: bool,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        gpu_block_table: torch.Tensor,
    ):
        global _INFERSTATE

        if _INFERSTATE is not None:
            _INFERSTATE = None

        _INFERSTATE = InferState(
            batch_size=batch_size,
            num_tokens=num_tokens,
            seq_ids=seq_ids,
            softmax_scale=softmax_scale,
            num_prefill_seqs=num_prefill_seqs,
            num_prefill_tokens=num_prefill_tokens,
            prefill_seq_start_locs=prefill_seq_start_locs,
            prefill_seq_start_locs_with_end=prefill_seq_start_locs_with_end,
            prefill_seq_lens=prefill_seq_lens,
            max_prefill_len=max_prefill_len,
            num_decoding_seqs=num_decoding_seqs,
            decoding_seq_lens=decoding_seq_lens,
            max_decoding_len=max_decoding_len,
            seq_block_size=seq_block_size,
            num_seq_blocks=num_seq_blocks,
            position_cos=position_cos,
            position_sin=position_sin,
            ignore_kvcache=ignore_kvcache,
            k_cache=k_cache,
            v_cache=v_cache,
            gpu_block_table=gpu_block_table
        )