"""
ModelInstance 负责如下任务:
* model 初始化逻辑
* model forward 逻辑, 包括 forward 前的预处理逻辑, forward 后善后逻辑
* model 物理 GPU and CPU KV Cache
* model swap KV Cache from GPU to CPU
"""
import math
import itertools
import torch
import torch.distributed as dist

from swiftllm.engine_config import EngineConfig
from swiftllm.worker.mconfigs.llamaconfig import LlamaModelConfig
from swiftllm.worker.models.llama import LlamaForCausalLM
from swiftllm.worker.block_manager import BlockManager
from swiftllm.worker.loader import load_weight
from swiftllm.worker.infer_state import InferState
from swiftllm.utils import GB
import swiftllm_c

class ModelInstance:
    @torch.inference_mode()
    def __init__(
        self,
        engine_config: EngineConfig,
        rank: int,
    ):  
        # Config
        self.rank = rank
        self.engine_config = engine_config
        LlamaModelConfig.set_model_config(engine_config.model_path)
        self.model_config = LlamaModelConfig.get_model_config()

        # Initialize model
        self.model = LlamaForCausalLM(self.model_config)
        load_weight(self.model, self.engine_config.model_path)
        
        # Initialize rotary embeddings
        self._cos_cached, self._sin_cached = self.model_config.get_rotary()

        # KV Cache
        self.num_blocks = None
        self.k_cache = self.v_cache = None
        self.k_swap = self.v_swap = None

        # Block manager
        self.cpu_block_manager = self.gpu_block_manager = None

        # Initialize KV Cache
        num_gpu_blocks = self.profile_num_blocks()
        self.init_kvcache_and_swap(num_gpu_blocks)
        

    @torch.inference_mode()
    def profile_num_blocks(self) -> int:
        """
        Profiler the number of GPU blocks

        We run a forged prefill batch with the maximum number of tokens and
        sequences, record the peak memory usage, and infer the number of blocks
        that can be allocated.
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Synthesis a prefill batch
        num_tokens = self.engine_config.max_tokens_in_batch
        batch_size = self.engine_config.max_batch_size
        input_lens = [num_tokens // batch_size] * batch_size
        input_lens[-1] += num_tokens % batch_size
        input_ids = [
            [0 for _ in range(input_len)]
            for input_len in input_lens
        ]
        seq_ids = list(range(batch_size))

        if self.rank == 0:
            print(f"[Model.profile] profile num block in batch size {batch_size} and input len {input_lens[0]}")
        
        self.k_cache = self.v_cache = None # pylint: disable=attribute-defined-outside-init

        _ = self.forward(input_ids, seq_ids, [], ignore_kvcache=True)
        torch.cuda.synchronize()

        # peak_memory = torch.cuda.max_memory_allocated()
        # total_memory = torch.cuda.get_device_properties(0).total_memory
        free_memory, total_memory = torch.cuda.mem_get_info()
        peak_memory = total_memory - free_memory
        useable_memory = total_memory*self.engine_config.gpu_mem_utilization

        if self.rank == 0:
            print(f"[Model.profile] GPU total memory: {total_memory/GB:.2f} GB, runtime peak memory: {peak_memory/GB:.2f} GB")
        
        if useable_memory < peak_memory:
            raise RuntimeError(f"[Model.profile rank {self.rank}] Peak memory {peak_memory/GB:.2f} GB exceeds usable memory " +
                               f"{useable_memory/GB:.2f} GB ({total_memory/GB:.2f} GB * {self.engine_config.gpu_mem_utilization})")
        
        block_size_bytes = self.engine_config.block_size * self.model_config.get_kvslot_size()
        num_gpu_blocks = math.floor((useable_memory - peak_memory) / block_size_bytes)

        target_num_blocks = [(input_len + self.engine_config.block_size - 1) // self.engine_config.block_size for input_len in input_lens]
        
        if self.rank == 0:
            print(f"[Model.profile] Available GPU block {num_gpu_blocks} " \
                  f"and least needed GPU block for batch size {batch_size}, input len {input_lens[0]} " \
                  f"is {sum(target_num_blocks)}")

        torch.cuda.empty_cache()

        # 注意我们在这里修改了model_config
        self.model_config.num_gpu_blocks = num_gpu_blocks
        return num_gpu_blocks


    @torch.inference_mode()
    def init_kvcache_and_swap(self, num_blocks: int):
        self.num_blocks = num_blocks

        # Initialize KV cache
        kvcache_shape = (
            self.num_blocks,
            self.model_config.num_layers,
            self.model_config.num_kv_heads,
            self.engine_config.block_size,
            self.model_config.head_dim
        )
        # Here we use torch.zeros instead of torch.empty, since that torch.empty
        # has the possibility to contain NaNs, which will cause the model to output NaNs.
        self.k_cache = torch.zeros(kvcache_shape, dtype=torch.float16, device="cuda")
        self.v_cache = torch.zeros(kvcache_shape, dtype=torch.float16, device="cuda")

        # Initialize KV swap space
        kvswap_shape = (
            self.engine_config.num_cpu_blocks,
            self.model_config.num_layers,
            self.model_config.num_kv_heads,
            self.engine_config.block_size,
            self.model_config.head_dim
        )
        self.k_swap = torch.zeros(kvswap_shape, dtype=torch.float16, device="cpu")
        self.v_swap = torch.zeros(kvswap_shape, dtype=torch.float16, device="cpu")

        # Initialize block manager
        self.gpu_block_manager = BlockManager(
            "GPU",
            self.num_blocks,
            self.engine_config.max_seqs_in_block_table,
            self.engine_config.max_blocks_per_seq,
            self.engine_config.block_size
        )
        self.cpu_block_manager = BlockManager(
            "CPU",
            self.engine_config.num_cpu_blocks,
            self.engine_config.max_seqs_in_block_table,
            self.engine_config.max_blocks_per_seq,
            self.engine_config.block_size
        )
    

    @torch.inference_mode()
    def forward(
        self,
        input_ids_list: list[list[int]], # [batch_size, *]
        seq_ids_list: list[int],     # [batch_size]
        decoding_seq_lens_list: list[int], # [num_decoding_seqs]
        ignore_kvcache: bool = False,   # Skip actions related to kv cache, useful when profiling the number of kv blocks
    ) -> list[int]:
        """
        Run a forward pass of the Model.

        It prepares the infer_state and calls the model `forward` function.

        This function is intended to be called by the server.
        """
        # input_ids_list 混合着 Prefill 和 Decode 的 Input ids, 且前部分是Prefill, 后部分是Decode
        # seq_ids_list 含义为 request sequence 在 request lists 中的下标
        # decoding_seq_lens_list 为当前decode requests已经生成序列长度

        # 计算 Prefill（新请求）的数量
        num_prefill_seqs = len(input_ids_list) - len(decoding_seq_lens_list)
        
        # 将每个请求在打平
        flattened_input_ids = list(itertools.chain(*input_ids_list))
        
        # 汇总每个序列的长度：Prefill 算实际长度，Decoding 算当前已生成的长度
        seq_lengths_list = [len(seq) for seq in input_ids_list[:num_prefill_seqs]] + decoding_seq_lens_list

        seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32, device="cuda")
        seq_lengths = torch.tensor(seq_lengths_list, dtype=torch.int32, device="cuda")

        batch_size = len(input_ids_list)
        num_tokens = len(flattened_input_ids)

        # 取出 Prefill request 长度
        prefill_seq_lens_list = seq_lengths_list[:num_prefill_seqs]
        prefill_seq_lens = torch.tensor(prefill_seq_lens_list, dtype=torch.int32, device="cuda")
        
        # cumsum 是累加和，用来计算每个请求在打平后的 Tensor 中的起始位置
        prefill_start_locs = torch.cumsum(prefill_seq_lens, dim=0, dtype=torch.int32) - prefill_seq_lens
        max_prefill_len = max(prefill_seq_lens_list) if prefill_seq_lens_list else 0

        decoding_seq_lens = torch.tensor(decoding_seq_lens_list, dtype=torch.int32, device="cuda")
        max_decoding_len = max(decoding_seq_lens_list) if decoding_seq_lens_list else 0

        # 让模型知道每个 token 的顺序
        # 对于 prefill request 生成的是 [0,.., prefill_len - 1]
        # 对于 decode request 生成的是 [decode_len - 1]
        # 最后将其拼成一个一维tensor
        # 例如 [0, 1, 2, 0, 1, 5]，我就可知道其有三个request，2个prefill request, 1个decode request
        # [0, 1, 2], [0, 1], [5] 其作用主要在于RoPE
        position_indices = torch.cat((
            torch.concat([
                torch.arange(
                    0,
                    prefill_seq_len,
                    device="cuda",
                    dtype=torch.int32
                )
                for prefill_seq_len in prefill_seq_lens_list
            ]) if prefill_seq_lens_list else torch.empty(0, device="cuda", dtype=torch.int32),
            decoding_seq_lens - 1
        ), dim=0)

        if not ignore_kvcache:
            self.gpu_block_manager.allocate_blocks_for_seqs(
                seq_ids,
                seq_lengths
            )

        # Select the seq_block_size
        #
        # Here we use a simple heuristic:
        #
        # In paged attention phase 1, the grid shape is (num_decoding_seqs, num_kv_heads, cdiv(max_decoding_len, seq_block_size))
        # and among these blocks, num_kv_heads * sum(cdiv(decoding_seq_lens, seq_block_size)) blocks are useful.
        # Thus we set seq_block_size to be the largest integer that satisfies
        #      num_kv_heads * sum(cdiv(decoding_seq_lens, seq_block_size)) >= 1024
        # to fully utilize the GPU. Here 1024 is a magic number (since most high-end
        # GPUs have ~128 SMs, so ~512 SMSPs. Since the decoding-stage attention
        # is mostly a memory-bound operation, I think 1024 is a reasonable number.)
        #
        # In practice, we use `decoding_seq_lens_sum/seq_block_size` to approximate
        # sum(cdiv(decoding_seq_lens, seq_block_size))

        # 从如下代码中可以窥见作者实现 Page attention 的思路
        # 作者启动 CUDA kernel :  (num_decoding_seqs, num_kv_heads, cdiv(max_decoding_len, seq_block_size))
        # 先来说明下为什么是 num_kv_heads, 而不是 num_attention_heads: 作者的观点：内存受限（Memory-Bound）下的工程优化
        # 在GQA的推理 Decoding 阶段，模型是极其吃显存带宽的。加载 KV Cache 的开销远大于计算本身的开销
        # 如果设置为 num_attention_heads 那么依赖相同KV Head的 thread block 会各自去显存里读取同一份 KV Cache 块
        # 但是如果设置为 num_kv_heads，thread block 只需要加载一份KV Cache, 处理多个Q HEAD
        # 所以上述CUDA kernel, 每个 thread block 处理 num_attention_head // num_kv_head 个 Q HEAD 中一份 kv cache block 的 attention
        
        # 作者如下代码的目的是为了 选择出最佳的seq_block_size, 因为其决定了CUDA kernel启动时有多少thread block, 我们thread block需要 >= GPU SM number 以最大利用计算资源
        # 同时 我们要保证 seq_block_size 不能太小，需要保证 thread block 中的计算量充足，而且要保证在 Page attention phase 2 压力不会过大
        # (Phase 1：每个 Block 计算自己负责的那一小段 KV Cache 的 Attention 分数，得到一个局部结果。
        # (Phase 2 (Reduction)：将同一个decode request局部结果合并（LogSumExp 累加）成最终的输出。)
        seq_block_size = 2048
        decoding_seq_lens_sum = sum(decoding_seq_lens_list)
        while self.model_config.num_kv_heads*(decoding_seq_lens_sum/seq_block_size) < 1024 and seq_block_size//2 >= 64 and \
            max_decoding_len / (seq_block_size//2) <= 128:
            seq_block_size //= 2

        InferState.set_inferstate(
            batch_size = batch_size,
            num_tokens = num_tokens,

            seq_ids = seq_ids,
            softmax_scale = self.model_config.head_dim ** -0.5,

            num_prefill_seqs = num_prefill_seqs,
            num_prefill_tokens = num_tokens - (batch_size - num_prefill_seqs),
            prefill_seq_start_locs = prefill_start_locs,
            prefill_seq_start_locs_with_end = torch.cat([
                prefill_start_locs,
                torch.tensor([num_tokens], dtype=torch.int32, device="cuda")
            ]),
            prefill_seq_lens = prefill_seq_lens,
            max_prefill_len = max_prefill_len,

            num_decoding_seqs = batch_size - num_prefill_seqs,
            decoding_seq_lens = decoding_seq_lens,
            max_decoding_len = max_decoding_len,

            seq_block_size = seq_block_size,
            num_seq_blocks = (max_decoding_len + seq_block_size-1) // seq_block_size,

            position_cos = self._cos_cached[position_indices],
            position_sin = self._sin_cached[position_indices],

            ignore_kvcache = ignore_kvcache,
            k_cache = self.k_cache,
            v_cache = self.v_cache,
            gpu_block_table = self.gpu_block_manager.block_table if self.gpu_block_manager else None
        )

        # 调用 model forward
        input_ids = torch.tensor(flattened_input_ids, dtype=torch.int32, device="cuda")
        logits = self.model(input_ids)

        # 执行采样算法
        output_tokens = self.sampler(logits)
        
        return output_tokens

    def sampler(self, logits: torch.Tensor) -> list[int]:
        """
        目前直接进行贪婪采样
        """
        output_tokens = torch.argmax(logits, dim=1)
        return output_tokens.tolist()

    def _swap(
        self,
        seq_ids_list: list[int],
        is_swap_in: bool
    ):
        src_block_manager = self.cpu_block_manager if is_swap_in else self.gpu_block_manager
        dst_block_manager = self.gpu_block_manager if is_swap_in else self.cpu_block_manager
        seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32, device="cuda")
        seq_lengths = src_block_manager.get_num_allocated_blocks(seq_ids) * self.engine_config.block_size
        src_block_ids = src_block_manager.gather_allocated_blocks_and_free(seq_ids)
        dst_block_ids = dst_block_manager.allocate_blocks_for_seqs(seq_ids, seq_lengths)
        swiftllm_c.swap_blocks(
            src_block_ids.tolist(),
            dst_block_ids.tolist(),
            is_swap_in,

            self.k_cache, self.v_cache,
            self.k_swap, self.v_swap
        )
        
    @torch.inference_mode()
    def swap_in_seqs(
        self,
        seq_ids_list: list[int]
    ):
        """
        Swap in (move blocks from CPU to GPU) the specified sequences.
        """
        self._swap(seq_ids_list, True)
    
    @torch.inference_mode()
    def swap_out_seqs(
        self,
        seq_ids_list: list[int]
    ):
        """
        Swap out (move blocks from GPU to CPU) the specified sequences.
        """
        self._swap(seq_ids_list, False)

    @torch.inference_mode()
    def free_seqs_resources(self, seq_ids_list: list[int]):
        """
        Free the resources of the specified sequences.
        """
        seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32, device="cuda")
        self.gpu_block_manager.free_blocks_for_seqs(seq_ids)
        self.cpu_block_manager.free_blocks_for_seqs(seq_ids)
