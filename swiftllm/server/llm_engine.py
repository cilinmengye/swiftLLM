'''
llm_engine.py主要实现如何组织多进程, 如何实现Tensor Parallel启动
model forward, 由于需要添加Tensor Parallel功能, 需要完成如下工作:

* 实现多进程之间初始化
* 实现多进程之间通信
* 实现多进程之间合作完成model forward

如上工作并不适合添加在engine.py上, 因为engine.py主要工作在于如何实现
online server上

llm_engine.py需要能够充当offline运行的主体, 以后我们想要运行offline 
benmark, 只需要实体化llm_engine, 然后就能够调用相关API完成model运行

同时llm_engine.py需要能够被engine.py调用API, 以完成online model的运行
'''
import atexit
import torch.multiprocessing as mp

from swiftllm.engine_config import EngineConfig
from swiftLLM.swiftllm.worker.model_runner import ModelRunner

class LLMEngine:
    def __init__(
        self,
        engine_config: EngineConfig
    ):
        """
        LLMEngine - A Engine manage llm model runner.

        This class provides a unified API for manipulating 
        the model, We will implement Tensor Parallel models 
        distributed across different GPUs.
        """
        # 我第一件事情是要创建多进程, 每个进程负责一个GPU
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")

        # 创建子进程的ModelRunner
        for i in range(1, engine_config.tensor_parallel_size):
            # 每个进程拥有一个event, 方便主进程控制
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(engine_config, i, event))
            process.start()

            self.ps.append(process)
            self.events.append(event)
        
        # 创建主进程的ModelRunner
        self.model_runner = ModelRunner(engine_config=engine_config, rank=0, event=self.events)

        # 注册主进程结束程序
        #   atexit 是 Python 标准库里的一个模块，用来做：在 
        #   Python 解释器正常退出时，自动调用你注册的清理函数。
        atexit.register(self.exit)
    
    def exit(self):
        """
        clear main-process and sub-process resource
        """
        # 利用主进程控制子进程结束执行, 释放资源
        # 同时自己也要释放资源
        self.model_runner.call("exit")

        del self.model_runner
        for p in self.ps:
            p.join()
    
    def step(
        self,
        input_ids_list: list[list[int]],
        seq_ids_list: list[int],
        decoding_seq_lens_list: list[int],
        ignore_kvcache: bool = False,
    ) -> list[int]:
        """
        tensor parallel model forward

        对标 model.forward: 参数和返回值都是一样的
        """
        # 利用主进程控制子进程执行forward
        # 同时自己也要执行forward
        return self.model_runner.call(
            "run",
            input_ids_list,
            seq_ids_list,
            decoding_seq_lens_list,
            ignore_kvcache,
        )


