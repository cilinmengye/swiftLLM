import asyncio
import functools
from typing import AsyncGenerator

import torch

from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.model import LlamaModel
from swiftllm.utils import GB

from .tokenization_engine import TokenizationEngine
from .structs import Request, RawRequest, StepOutput
from .scheduler import Scheduler

class Engine:
    def __init__(self, engine_config: EngineConfig):
        self.engine_config = engine_config
        # self.model_config = LlamaModelConfig.load_from_model_path(engine_config.model_path)
        self.initialized = False

        # The following fields will be created on `init_model()`
        self.model = None
        self.event_loop = None
        self.scheduler = None
        self.tokenization_engine = None
        self.eos_token_id = None

        self.untokenized_raw_requests: list[tuple[Request, str]] = []

    async def _run_on_model_async(self, func, *args, **kwargs):
        """
        Run a function on the model asynchronously, and return the result
        """
        func_partial = functools.partial(func, *args, **kwargs)
        # TODO: 此处性能可以优化，因为其参数为None, 默认开启的是线程池，但是在
        # Python下线程池是伪多线程，遇到计算强度大的任务需要开启进程池
        # Update:
        # 考虑了下执行_run_on_model_async函数的主要有3种情况：
        # 1. self.model.swap_out_seqs 2. self.model.swap_in_seqs, 3. self.model.forward
        # 前两种情况是I/O密集型非常适合python多线程; 后一种情况主要在GPU上运行，并不会受到太多GIL影响
        # 进程池不是一定更快，因为它有额外成本：进程创建开销；参数序列化 / 反序列化开销；大对象跨进程传输很贵
        return await self.event_loop.run_in_executor(None, func_partial)

    async def initialize(self):
        self.event_loop = asyncio.get_event_loop()

        # print("[Engine] Initializing model...")
        # self.model = LlamaModel(self.engine_config)

        # print("[Engine] Loading weights...")
        # self.model.load_weights()

        # print("[Engine] Profiling kv blocks...")
        # num_gpu_blocks = self.model.profile_num_blocks()
        # num_cpu_blocks = self.engine_config.num_cpu_blocks
        # block_size_bytes = self.engine_config.block_size*self.model_config.get_kvslot_size()
        # print(f"[Engine] Number of GPU blocks: {num_gpu_blocks} ({num_gpu_blocks*block_size_bytes/GB:.2f} GB)")
        # print(f"[Engine] Number of CPU blocks: {num_cpu_blocks} ({num_cpu_blocks*block_size_bytes/GB:.2f} GB)")

        # print("[Engine] Allocating kv cache and swap...")
        # self.model.init_kvcache_and_swap(num_gpu_blocks)

        print("[Engine] Initializing scheduler...")
        self.scheduler = Scheduler(self.model, self.engine_config, num_gpu_blocks)

        print("[Engine] Initializing tokenization engine...")
        self.tokenization_engine = TokenizationEngine.remote(self.engine_config)
        self.eos_token_id = await self.tokenization_engine.get_eos_token_id.remote()

        print("[Engine] Model initialized")
        self.initialized = True

    async def add_request_and_stream(self, raw_request: RawRequest) -> AsyncGenerator[StepOutput, None]:
        """
        Add a raw request to the engine and stream the output of the request (streaming mode)
        """
        request = Request(raw_request)
        self.untokenized_raw_requests.append((request, raw_request.prompt))
        while True:
            step_output = await request.output_q.get()
            # 把当前拿到的这一步输出，立刻返回给调用方，
            # 但函数本身不结束；当调用方处理结束后,
            # 下次继续从这里往后执行。
            yield step_output
            request.output_q.task_done()
            if step_output.request.is_finished():
                break

    async def add_request_and_wait(self, raw_request: RawRequest) -> tuple[Request, list[int]]:
        """
        Add a raw request to the engine and wait for the completion (non-streaming mode)

        Return the output token ids
        """
        request = Request(raw_request)
        self.untokenized_raw_requests.append((request, raw_request.prompt))
        await request.finished_event.wait()
        return (request, request.output_token_ids)

    async def _tokenize_raw_request_event_loop(self):
        """
        Event loop for tokenizing raw requests
        """
        while True:
            if not self.untokenized_raw_requests:
                # No new raw requests, sleep for a bit
                await asyncio.sleep(0.002)
                continue

            # Tokenize the raw request in batch
            cur_untokenized_raw_requests = self.untokenized_raw_requests
            self.untokenized_raw_requests = []

            prompts = [prompt for _, prompt in cur_untokenized_raw_requests]
            prompt_token_ids = await self.tokenization_engine.batched_tokenize.remote(prompts)

            new_requests = []
            for (request, _), prompt_token_id in zip(cur_untokenized_raw_requests, prompt_token_ids):
                request.prompt_token_ids = prompt_token_id
                request.prompt_len = len(prompt_token_id)
                new_requests.append(request)

            self.scheduler.on_requests_arrival(new_requests)
            await asyncio.sleep(0.001)  # yield the event loop

    async def _main_event_loop(self):
        """
        Event loop for forwarding the model
        """
        while True:
            # Get the next batch from the scheduler
            cur_batch, cur_swap_in, cur_swap_out = self.scheduler.get_next_batch()
            if not cur_batch and not cur_swap_in and not cur_swap_out:
                # No new batch, sleep for a bit
                await asyncio.sleep(0.005)
                continue

            # Perform swap in/out
            if cur_swap_out:
                await self._run_on_model_async(
                    self.model.swap_out_seqs,
                    [req.request_id for req in cur_swap_out]
                )
            if cur_swap_in:
                await self._run_on_model_async(
                    self.model.swap_in_seqs,
                    [req.request_id for req in cur_swap_in]
                )

            # Forward the model
            input_ids = [
                req.prompt_token_ids if req.is_prefill_stage() else [req.output_token_ids[-1]]
                for req in cur_batch
            ]
            seq_ids = [req.request_id for req in cur_batch]
            decoding_seq_lens_list = [
                req.prompt_len + req.get_cur_output_len()
                for req in cur_batch
                if not req.is_prefill_stage()
            ]
            output_tokens = await self._run_on_model_async(
                self.model.forward,
                input_ids,
                seq_ids,
                decoding_seq_lens_list
            )

            # Deal with output tokens
            finished_req_ids = []
            for req, output_token in zip(cur_batch, output_tokens):
                req.output_token_ids.append(output_token)
                req.maybe_mark_finished(output_token, self.eos_token_id)
                req.output_q.put_nowait(StepOutput(output_token, req))
                if req.is_finished():
                    finished_req_ids.append(req.request_id)
                    req.finished_event.set()
            if finished_req_ids:
                await self._run_on_model_async(
                    self.model.free_seqs_resources,
                    finished_req_ids
                )

            # Inform the scheduler
            self.scheduler.on_batch_finish(cur_batch)

    async def start_all_event_loops(self):
        """
        Start all event loops
        """
        assert self.initialized, "Engine not initialized. Please call `initialize()` before starting the event loop."
        await asyncio.gather(
            self._tokenize_raw_request_event_loop(),
            self._main_event_loop()
        )
