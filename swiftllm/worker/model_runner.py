'''
model_runner.py主要实现单个进程如何进行model forward, 如何与其他进程完成通信, 同步
'''
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
import torch.distributed as dist
import torch
import pickle

from swiftllm.engine_config import EngineConfig
from swiftllm.worker.mconfigs.llamaconfig import LlamaModelConfig
from swiftllm.worker.model_instance import ModelInstance


class ModelRunner:
    """
    ModelRunner 负责一个 TP rank 对应进程的生命周期管理。

    这里有两个和当前问题直接相关的关键职责：
    1. 在真正构造模型参数之前，先把默认 dtype / device 调整到与 HF 配置一致；
    2. 在 TP>1 时维护 rank0 -> 其他 rank 的共享内存 RPC 协议。

    之所以必须在这里集中处理 dtype / device，而不是分散到每个 layer 里显式传参，
    是因为当前代码会在 nn.Module 的 __init__ 阶段直接创建大量 CUDA Parameter。
    如果只设置 default device 为 cuda，而没有同步把 default dtype 改成模型配置里的
    float16 / bfloat16，那么这些参数就会按 PyTorch 默认 float32 直接落到 GPU 上，
    建模期显存会接近翻倍，这正是当前离线启动 OOM 的根因之一。
    """
    def __init__(
        self,
        engine_config: EngineConfig,
        rank: int,
        event: Event | list[Event]
    ):
        """
        parameters
        : engine_config
            config about initizate model
        : rank
            cpu process id, also gpu id; rank == 0 is main process
        : event
            flag, use to multiprocess communication; main process
            have all sub-process events, use to control sub-process
        """
        # 我们需要在此做得工作有: 1. 初始化GPU通信组
        # 2. 初始化模型  3. 初始化多进程之间通信
        # 显示设置让全部多进程都能看见全局的engine_config
        EngineConfig.set_engine_config(
            model_path=engine_config.model_path,
            use_dummy=engine_config.use_dummy,
            block_size=engine_config.block_size,
            gpu_mem_utilization=engine_config.gpu_mem_utilization,
            num_cpu_blocks=engine_config.num_cpu_blocks,
            max_seqs_in_block_table=engine_config.max_seqs_in_block_table,
            max_blocks_per_seq=engine_config.max_blocks_per_seq,
            max_batch_size=engine_config.max_batch_size,
            max_tokens_in_batch=engine_config.max_tokens_in_batch,
            tensor_parallel_size=engine_config.tensor_parallel_size,
        )
        self.engine_config = EngineConfig.get_engine_config()
        self.rank = rank
        self.event = event
        self.world_size = engine_config.tensor_parallel_size

        # 先在所有进程里读取一次模型配置。后面构造模型时会依赖其中的 torch_dtype
        # 去临时修改 default dtype，从而避免 CUDA 上错误地以 float32 建参。
        LlamaModelConfig.set_model_config(engine_config.model_path)
        self.model_config = LlamaModelConfig.get_model_config()

        # 1. 初始化GPU通信组
        dist.init_process_group(
            "nccl",
            "tcp://localhost:2333",
            world_size=self.world_size,
            rank=rank,
        )

        # 创建映射,这意味着 process rank i
        # -> GPU cuda: i, 即此进程绑定了指定cuda rank的GPU
        torch.cuda.set_device(rank)

        # 这里必须同时管理 default dtype 和 default device。
        # 原因是当前模型参数是在 __init__ 阶段通过 torch.empty / torch.ones 直接创建的。
        # 如果只把 default device 设成 cuda，而不把 default dtype 设成 HF 配置中的
        # torch_dtype，那么这些参数就会以默认 float32 落到 GPU，导致建模阶段显存暴涨。
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.model_config.torch_dtype)
        torch.set_default_device("cuda")

        # 2. 初始化模型
        # 为防止多进程打印内容出现混乱，我们规定只允许rank == 0主进程打印信息
        self.modelinstance = ModelInstance(self.engine_config, rank)

        # 无论模型初始化是否成功，都把进程级默认状态恢复掉，避免污染后续 CPU tensor
        # 创建逻辑。这里恢复顺序也保持和 nano-vllm 一致：先切回 CPU，再恢复 dtype。
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 3. 初始化多进程之间通信
        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(
                    name="swiftllm",
                    create=True,
                    size=32 * 2**20
                )   # 开 32MB 共享内存, 占用共享内存主要空间的是传递的prompt
                    # 开 32MB 共享内存, 那么batch size == 1024时，all prompt
                    # token number 需要 <= 8192
                dist.barrier()
            else:
                dist.barrier()  # 子进程等待主进程创建完成共享内存
                self.shm = SharedMemory(name="swiftllm")
                self.loop()

    def run(
        self,
        input_ids_list: list[list[int]],
        seq_ids_list: list[int],
        decoding_seq_lens_list: list[int],
        ignore_kvcache: bool = False,
    ) -> list[int] | None:
        return self.modelinstance.forward(
            input_ids_list=input_ids_list,
            seq_ids_list=seq_ids_list,
            decoding_seq_lens_list=decoding_seq_lens_list,
            ignore_kvcache=ignore_kvcache,
        )

    def swap_in_seqs(self, seq_ids_list: list[int]):
        self.modelinstance.swap_in_seqs(seq_ids_list)

    def swap_out_seqs(self, seq_ids_list: list[int]):
        self.modelinstance.swap_out_seqs(seq_ids_list)

    def free_seqs_resources(self, seq_ids_list: list[int]):
        self.modelinstance.free_seqs_resources(seq_ids_list)

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            # 如果是主进程，那么其需要命令其他子进程也执行相应的命令
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def exit(self):
        """
        Process releases resources
        """
        # 释放共享内存(只在多进程时)
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        # 解除nccl通信组
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """
        Subprocess main behave: Use shared memory to listen
        for messages from the main process and execute commands
        from the main process when needed.
        """
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()   # 相当于检查自己的flag, 可以想象event为红绿灯，初始event是红灯
                            # event.wait()相当于检查自己的flag, 如果是红灯，那么将进程挂起
                            # 如果是绿灯则继续执行, event.set()会设置flag为绿灯，event.clear()会设置flag为红灯
        n = int.from_bytes(self.shm.buf[0: 4], "little")    # 前4bit为数据大小, 小端模式
        method_name, *args = pickle.loads(self.shm.buf[4: n + 4])   # pickle为序列化工具
        self.event.clear()  # 消费者消费完毕再次设置flag为红灯

        # 这里必须固定返回 (method_name, args) 这个二元组契约，原因是 loop() 会按
        # `method_name, args = self.read_shm()` 解包；如果这里返回的是展开后的多元组，
        # 那么一旦 RPC 参数数量大于 1，主从进程的协议就会立刻错位。
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0

        # 这里必须使用 pickle.dumps 而不是 pickle.dump。
        # dump 是“写到文件句柄”，不会返回要写入共享内存的 bytes；
        # dumps 才会返回可直接塞进 shm buffer 的序列化结果。
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4: n + 4] = data
        for event in self.event:    # 生产者生产完毕, 让消费者开始消费
            event.set()
