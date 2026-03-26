'''
model_runner.py主要实现单个进程如何进行model forward, 如何与其他进程完成通信, 同步
'''
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
import torch.distributed as dist
import torch
import pickle

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.model import LlamaModel
from swiftllm.utils import GB

class ModelRunner:
    """
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
        self.engine_config = engine_config
        self.rank = rank
        self.event = event
        self.world_size = engine_config.tensor_parallel_size

        # 1. 初始化GPU通信组 
        dist.init_process_group(
            "nccl", 
            "tcp://localhost:2333", 
            world_size=self.world_size, 
            rank=rank,
        )
        # 创建映射,这意味着 process rank i
        # -> GPU cuda: i, 即此进程绑定了
        # 指定cuda rank的GPU
        torch.cuda.set_device(rank)

        # 2. 初始化模型
        # 为防止多进程打印内容出现混乱，我们
        # 规定只允许rank == 0主进程打印信息
        self.model_config = LlamaModelConfig.load_from_model_path(engine_config.model_path)

        if rank == 0:
            print("[Engine] Initializing model...")
        self.model = LlamaModel(self.engine_config)

        if rank == 0:
            print("[Engine] Loading weights...")
        self.model.load_weights()

        if rank == 0:
            print("[Engine] Profiling kv blocks...")
        num_gpu_blocks = self.model.profile_num_blocks()
        self.num_gpu_blocks = num_gpu_blocks

        num_cpu_blocks = self.engine_config.num_cpu_blocks
        block_size_bytes = self.engine_config.block_size*self.model_config.get_kvslot_size()
        if rank == 0:
            print(f"[Engine] Number of GPU blocks Per GPU: {num_gpu_blocks} ({num_gpu_blocks*block_size_bytes/GB:.2f} GB)")
            print(f"[Engine] Number of CPU blocks Per Process: {num_cpu_blocks} ({num_cpu_blocks*block_size_bytes/GB:.2f} GB)")

        if rank == 0:
            print("[Engine] Allocating kv cache and swap...")
        self.model.init_kvcache_and_swap(num_gpu_blocks)

        # 3. 初始化多进程之间通信
        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(
                    name="swiftllm", 
                    create=True, 
                    size=32*2**20
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
    ) -> list[int]:
        return self.model.forward(
            input_ids_list=input_ids_list,
            seq_ids_list=seq_ids_list,
            decoding_seq_lens_list=decoding_seq_lens_list,
            ignore_kvcache=ignore_kvcache,
        )
    
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
                            # 如果是绿灯则继续执行, event.set()会设置flag为绿灯，event.cl
                            # ear()会设置flag为红灯
        n = int.from_bytes(self.shm.buf[0: 4], "little")    # 前4bit为数据大小, 小端模式    
        method_name, *args = pickle.loads(self.shm.buf[4: n + 4])   # pickle为序列化工具
        self.event.clear()  # 消费者消费完毕再次设置flag为红灯
        return method_name, *args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dump([method_name, *args])    # 先序列化
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4: n + 4] = data
        for event in self.event:    # 生产者生产完毕, 让消费者开始消费
            event.set()

    def get_num_gpu_blocks(self):
        return self.num_gpu_blocks





    