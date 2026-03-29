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
        torch.set_default_device("cuda")    # 接下来的tensor默认全部在GPU上

        # 2. 初始化模型
        # 为防止多进程打印内容出现混乱，我们
        # 规定只允许rank == 0主进程打印信息
        self.modelinstance = ModelInstance(self.engine_config, rank)
        
        torch.set_default_device("cpu")     # 模型初始化结束后，tensor默认在CPU上

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
        return self.modelinstance.forward(
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





    