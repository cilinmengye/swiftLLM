import dataclasses
import argparse

_ENGINECONFIG = None

@dataclasses.dataclass
class EngineConfig:
    """
    Configuration for the SwiftLLM engine.
    """
    
    # Model loading parameters
    model_path: str
    use_dummy: bool

    # PagedAttention-related parameters
    block_size: int
    gpu_mem_utilization: float
    num_cpu_blocks: int
    max_seqs_in_block_table: int
    max_blocks_per_seq: int

    # Scheduling-related parameters
    # 就获取剩余可用KV Cache Size而言，刚需知道如下两个参数
    # 因为需要模拟运行一遍以最大负荷执行的Prefill，获得模型推理时整个显存最大剩余显存
    max_batch_size: int
    max_tokens_in_batch: int

    # Model tensor parallel
    tensor_parallel_size: int

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        """
        Add CLI arguments for the engine configuration
        """
        parser.add_argument(
            "--model-path",
            type=str,
            required=True,
            help="Path to the model directory (currently SwiftLLM does not support downloading from HuggingFace, so please download in advance)",
        )
        parser.add_argument(
            "--use-dummy",
            action="store_true",
            help="Use dummy weights (mainly for profiling)",
        )

        parser.add_argument(
            "--block-size",
            type=int,
            default=16,
            help="Block size for PagedAttention",
        )
        parser.add_argument(
            "--gpu-mem-utilization",
            type=float,
            default=0.97,
            help="Fraction of GPU memory to be used",
        )
        parser.add_argument(
            "--num-cpu-blocks",
            type=int,
            default=2048,
            help="Number of CPU blocks",
        )
        parser.add_argument(
            "--max-seqs-in-block-table",
            type=int,
            default=4096,
            help="Maximum number of sequences in the block table",
        )
        parser.add_argument(
            "--max-blocks-per-seq",
            type=int,
            default=32768,
            help="Maximum number of blocks per sequence",
        )

        parser.add_argument(
            "--max-batch-size",
            type=int,
            default=512,
            help="Maximum batch size",
        )
        parser.add_argument(
            "--max-tokens-in-batch",
            type=int,
            default=32768,
            help="Maximum number of tokens in a batch",
        )
        
        parser.add_argument(
            "--tensor-parallel-size",
            type=int,
            default=1,
            help="Model tensor parallel size",
        )
    
    @staticmethod
    def set_engine_config(
        model_path: str,
        use_dummy: bool = False,
        block_size: int = 16,
        gpu_mem_utilization: float = 0.97,
        num_cpu_blocks: int = 2048,
        max_seqs_in_block_table: int = 4096,
        max_blocks_per_seq: int = 32768,
        max_batch_size: int = 512,
        max_tokens_in_batch: int = 32768,
        tensor_parallel_size: int = 1,
    ):
        """
        Initialize the global engine config from explicit arguments.

        Returns:
            EngineConfig: the initialized global config
        """
        global _ENGINECONFIG
        
        if _ENGINECONFIG is not None:
            _ENGINECONFIG = None

        _ENGINECONFIG = EngineConfig(
            model_path=model_path,
            use_dummy=use_dummy,
            block_size=block_size,
            gpu_mem_utilization=gpu_mem_utilization,
            num_cpu_blocks=num_cpu_blocks,
            max_seqs_in_block_table=max_seqs_in_block_table,
            max_blocks_per_seq=max_blocks_per_seq,
            max_batch_size=max_batch_size,
            max_tokens_in_batch=max_tokens_in_batch,
            tensor_parallel_size=tensor_parallel_size,
        )
        return _ENGINECONFIG

    @staticmethod
    def get_engine_config():
        assert _ENGINECONFIG is not None
        return _ENGINECONFIG