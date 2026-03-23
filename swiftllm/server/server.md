# SwiftLLM Server 如何支撑整个 Online Inference

本文聚焦 `swiftllm/server/` 这一层，目标不是逐行翻译源码，而是回答一个更核心的问题：

> SwiftLLM 的 server 到底是如何把一个在线请求，变成一次次真正的模型推理，并最终把结果持续返回给客户端的？

如果用一句话概括：

**`swiftllm/server` 是 SwiftLLM 的 control plane（控制面），负责接请求、做分词、排队、调度 batch、管理请求状态，并驱动底层 `worker` 层执行模型前向与 KV cache 管理。**

---

## 1. 先建立整体视角：server 在整个架构中的位置

在项目的总说明里，SwiftLLM 明确把架构分成两部分：

- **control plane**：决定“算什么、什么时候算、谁先算”
- **data plane**：决定“具体怎么算”，并真正执行 GPU 上的计算

对应到代码上：

- **control plane**：`swiftllm/server/`
- **data plane**：`swiftllm/worker/`

也就是说：

- `server/` 不负责实现 Transformer 的数学计算细节；
- `server/` 负责把在线服务需要的各种“控制逻辑”串起来；
- `worker/` 里的 `LlamaModel`、`BlockManager`、各层实现、kernel 才是真正把 token 跑出来的执行层。

所以如果你想理解 **online inference 是怎么“服务起来”的**，重点就要看清楚：

1. 请求是怎么进入 server 的；
2. server 是怎么把请求组织成可执行的 batch 的；
3. server 是怎么驱动 model 一轮一轮 forward 的；
4. 每轮 forward 之后，结果又是怎么回流给客户端的。

---

## 2. `swiftllm/server/` 目录里的核心文件分别做什么

`swiftllm/server/` 目录不大，但分工非常清晰：

### 2.1 `api_server.py`：HTTP 入口

文件：[api_server.py](api_server.py)

职责：

- 用 FastAPI 暴露 `POST /generate`
- 解析用户请求 JSON
- 把请求转成 `RawRequest`
- 根据 `stream` 参数选择：
  - 流式返回
  - 非流式等待完整结果返回
- 启动整个服务：Uvicorn + Engine 的后台事件循环

它是“门面层”，负责接住外部请求，但**它自己不做推理调度**。

---

### 2.2 `engine.py`：在线推理的总控中枢

文件：[engine.py](engine.py)

这是整个 online serving 的核心。

职责：

- 初始化模型与运行环境
- 创建 tokenizer 服务
- 创建 scheduler
- 维护未分词请求队列
- 开两个后台事件循环：
  - 一个负责分词
  - 一个负责调度并驱动模型 forward
- 把每轮生成出来的 token 分发给对应请求

可以把 `Engine` 理解成：

> 一个在线推理控制器（orchestrator）

它本身不实现 Transformer 计算，但它控制着整个生命周期。

---

### 2.3 `scheduler.py`：请求调度器

文件：[scheduler.py](scheduler.py)

职责：

- 管理请求状态队列：
  - `waiting_q`
  - `running_q`
  - `swapped_q`
- 严格按 **FCFS（先来先服务）** 组织调度
- 决定下一轮是：
  - 发起 prefill batch
  - 还是继续 decoding batch
- 在 GPU block 不够时，决定哪些请求 swap out 到 CPU
- 在资源有余量时，决定哪些请求 swap in 回 GPU

这层本质上在回答：

> 下一轮 GPU 应该算谁？

---

### 2.4 `structs.py`：请求与输出的状态载体

文件：[structs.py](structs.py)

定义了三个最核心的数据结构：

- `RawRequest`
  - 用户刚提交的原始请求
  - 只包含最基本输入：`prompt`、`output_len`
- `Request`
  - 进入引擎后的内部请求对象
  - 持有分词结果、输出 token、请求状态、同步原语等
- `StepOutput`
  - 单步 decoding 的输出
  - 表示“这个请求这一轮生成了一个 token”

这三个结构把“外部请求”和“内部执行状态”区分得很清楚。

---

### 2.5 `tokenization_engine.py`：分词/解码服务

文件：[tokenization_engine.py](tokenization_engine.py)

职责：

- 用 `@ray.remote` 定义一个 Ray Actor
- 封装 HuggingFace `AutoTokenizer`
- 提供：
  - `batched_tokenize(prompts)`
  - `decode(token_ids)`

为什么它单独拎出来？

因为在线服务里，“接请求”和“分词”最好不要和主控制循环完全耦死。这里作者把 tokenizer 做成一个远程 actor，本质是在把：

- 文本 -> token ids
- token ids -> 文本

这件事变成一个独立服务。

---

## 3. 整个 server 的核心逻辑，其实就是“两条后台循环 + 一个调度器”

如果把细节全部折叠起来，`server/` 的核心框架可以总结成下面这张图：

```text
Client
  │
  ▼
FastAPI /generate
  │
  ▼
Engine.add_request_*()
  │
  ├─> 未分词请求队列 untokenized_raw_requests
  │
  ├─ 后台循环1：tokenization loop
  │      └─ 批量分词 -> Scheduler.waiting_q
  │
  └─ 后台循环2：main loop
         ├─ Scheduler.get_next_batch()
         ├─ 必要时 swap in / swap out
         ├─ 调用 LlamaModel.forward()
         ├─ 产出本轮 token
         ├─ 更新请求状态
         └─ 把结果流式/非流式返回给调用方
```

这个结构非常关键。

SwiftLLM 的 online inference **不是“一次请求触发一次完整推理”**，而是：

- 请求先进入系统；
- 进入统一调度；
- 引擎不断地“滴答、滴答”地推进每一轮 forward；
- 每轮 forward 为若干请求各生成一个 token（或为 prefill 请求做首轮处理）；
- 直到请求完成。

所以它本质上是一个 **事件驱动 + batch 调度 + 增量生成** 的系统。

---

## 4. 服务启动时，server 是怎么被“拉起来”的

入口在 [api_server.py](api_server.py)。

### 4.1 创建 FastAPI app 和全局 engine

`api_server.py` 一开始定义了：

- `app = fastapi.FastAPI()`
- `engine = None`

说明 API 层会持有一个全局 `Engine` 实例。

---

### 4.2 解析命令行参数，构造 `EngineConfig`

在 `__main__` 中：

- 解析 `--host`、`--port`
- 调用 `swiftllm.EngineConfig.add_cli_args(parser)` 增加引擎相关参数

配置定义在 [../engine_config.py](../engine_config.py)，主要包括：

- `model_path`
- `use_dummy`
- `block_size`
- `gpu_mem_utilization`
- `num_cpu_blocks`
- `max_seqs_in_block_table`
- `max_blocks_per_seq`
- `max_batch_size`
- `max_tokens_in_batch`

这些参数直接决定：

- 模型从哪里加载
- GPU KV cache 可以吃多少显存
- CPU swap 空间多大
- 单轮 batch 能塞多少请求/多少 token
- block table 可以支持多少序列

也就是说，**`EngineConfig` 本质上决定了 online serving 的容量边界。**

---

### 4.3 创建 `Engine`

在 [api_server.py](api_server.py) 中：

```python
engine = swiftllm.Engine(swiftllm.EngineConfig(**args))
```

从这里开始，控制中枢建立起来了。

---

### 4.4 `Engine.initialize()`：把在线服务真正初始化完成

初始化逻辑在 [engine.py](engine.py)。

`Engine.initialize()` 做了几件非常关键的事情。

#### 第一步：加载模型配置

`Engine.__init__` 中先通过 `LlamaModelConfig.load_from_model_path(...)` 读取模型配置。

这一步的意义是：

- 后面要知道 head 数、layer 数、KV slot 大小等
- 这些信息会直接影响 KV cache 和 block 数量计算

---

#### 第二步：创建 `LlamaModel`

`initialize()` 里：

- `self.model = LlamaModel(self.engine_config)`
- `self.model.load_weights()`

也就是把 data plane 的执行器建起来，并把权重真正加载到 GPU 相关结构中。

---

#### 第三步：profile GPU 能容纳多少 KV blocks

`initialize()` 调用：

- `self.model.profile_num_blocks()`

这一步很关键。

它不是拍脑袋决定 GPU 能容纳多少 KV cache，而是：

1. 先伪造一个“最大负荷”的 prefill batch；
2. 真正跑一次 forward；
3. 观测这时运行期峰值显存；
4. 根据剩余可用显存推断还能分配多少 KV blocks。

所以这里算出来的 `num_gpu_blocks`，本质上是：

> 在当前模型、当前 batch 上限、当前显存利用率约束下，GPU 能拿出多少块空间给 KV cache。

这也是后续 scheduler 做在线调度的资源基础。

---

#### 第四步：初始化 GPU KV cache 和 CPU swap

`initialize()` 接着调用：

- `self.model.init_kvcache_and_swap(num_gpu_blocks)`

这一步在 [../worker/model.py](../worker/model.py) 里完成：

- 在 GPU 上分配 `k_cache` / `v_cache`
- 在 CPU 上分配 `k_swap` / `v_swap`
- 初始化 `gpu_block_manager`
- 初始化 `cpu_block_manager`

到这里，online inference 的“状态存储层”就齐了：

- GPU 上存正在活跃推理的序列状态
- CPU 上存被换出的序列状态

---

#### 第五步：创建 `Scheduler`

`initialize()` 中：

- `self.scheduler = Scheduler(...)`

从这一步开始，Engine 有了调度器。

---

#### 第六步：创建 `TokenizationEngine`

`initialize()` 中：

- `self.tokenization_engine = TokenizationEngine.remote(self.engine_config)`

也就是启动了一个 Ray Actor，专门负责：

- 批量分词
- token decode

---

### 4.5 启动两个并行世界：HTTP 服务 + Engine 后台循环

在 [api_server.py](api_server.py) 里，`main_coroutine()` 中会并发启动两个任务：

- `uvicorn_server.serve()`
- `engine.start_all_event_loops()`

这表示：

1. 一个世界负责收外部请求；
2. 一个世界负责内部不断推进推理。

这两个世界并不是“一请求一执行”的同步关系，而是并发运行、松耦合衔接的。

---

## 5. 一条请求进入后，到底经历了什么

下面按一条真实请求的生命周期展开。

---

## 5.1 第 0 步：客户端发请求到 `/generate`

HTTP 入口在 [api_server.py](api_server.py)。

接口是：

- `POST /generate`

请求体至少包含：

- `prompt`
- `output_len`

还支持：

- `stream`：是否流式输出
- `decode`：是否把 token ids 解码成人类可读文本

API 层读到 JSON 后，会构造：

```python
raw_request = swiftllm.RawRequest(
    prompt=req_dict["prompt"],
    output_len=req_dict["output_len"]
)
```

也就是说，用户输入在进入引擎前还是一个很“原始”的结构。

---

## 5.2 第 1 步：API 层根据返回模式调用 Engine

这里分两种模式。

### 流式模式

调用：

- `engine.add_request_and_stream(raw_request)`

它返回一个异步生成器，API 层可以边拿 token 边往客户端写。

### 非流式模式

调用：

- `engine.add_request_and_wait(raw_request)`

它会一直等到这个请求全部生成完成，然后一次性返回全部 token ids。

所以：

- **streaming 和 non-streaming 的核心差别不在调度层**；
- 它们的差别主要在“结果是如何被消费”的方式上。

底层请求进入 Engine 后，走的都是同一套调度与执行系统。

---

## 5.3 第 2 步：`RawRequest` 变成内部 `Request`

在 [engine.py](engine.py) 中，不管是 `add_request_and_stream()` 还是 `add_request_and_wait()`，第一件事都是：

```python
request = Request(raw_request)
```

`Request` 定义在 [structs.py](structs.py)。

它比 `RawRequest` 多了很多内部状态：

- `prompt_token_ids`
- `prompt_len`
- `output_q`
- `finished_event`
- `request_id`
- `output_token_ids`

这说明 SwiftLLM 把：

- **用户视角的请求**（prompt + 目标输出长度）
- **引擎内部的可调度请求**

做了明确分层。

其中两个同步原语尤其关键：

### `output_q`

- 一个 `asyncio.Queue`
- 每生成一个 token，就往里面塞一个 `StepOutput`
- 给流式返回使用

### `finished_event`

- 一个 `asyncio.Event`
- 请求全部完成后置位
- 给非流式等待使用

所以 streaming / non-streaming 实际上只是：

- 一个在消费 `output_q`
- 一个在等待 `finished_event`

---

## 5.4 第 3 步：请求先进入“未分词队列”

构造完 `Request` 之后，Engine 不会立刻把它送去调度，而是先追加到：

- `self.untokenized_raw_requests`

这个队列里存的是：

- `(request, prompt_text)`

注意这个时刻：

- 请求已经进入系统；
- 但还没有 `prompt_token_ids`；
- 也还没进入 scheduler。

这一步的设计很重要，因为它意味着：

> “请求接入”和“分词执行”被解耦了。

请求先快速入队，不阻塞 API 路径，然后由后台分词循环批量处理。

---

## 5.5 第 4 步：后台 tokenization loop 批量分词

这部分逻辑在 `Engine._tokenize_raw_request_event_loop()`，位于 [engine.py](engine.py)。

它会一直循环：

1. 检查 `untokenized_raw_requests` 是否为空
2. 如果为空，就 sleep 一小会儿
3. 如果不为空，就把当前积累的一批请求全部取出来
4. 调用：
   - `self.tokenization_engine.batched_tokenize.remote(prompts)`
5. 得到一批 `prompt_token_ids`
6. 回填到各个 `Request` 上：
   - `request.prompt_token_ids`
   - `request.prompt_len`
7. 最后调用：
   - `self.scheduler.on_requests_arrival(new_requests)`

也就是说，一条请求只有在**完成分词之后**，才真正进入调度器的等待队列。

这一步有两个核心价值：

### 价值 1：批处理分词

多个请求一起 tokenize，减少开销。

### 价值 2：把文本世界转换成 token 世界

Scheduler 不关心字符串 prompt，它只关心：

- prompt 长度是多少
- 会占多少 blocks
- 能不能塞进当前 batch

所以 tokenization loop 是整个系统的一个“边界转换层”。

---

## 5.6 第 5 步：请求进入 Scheduler 的 `waiting_q`

`Scheduler.on_requests_arrival()` 很简单：

- 把请求 append 到 `waiting_q`

但从系统语义上，这一步非常重要。

它意味着请求已经从：

- “刚进入系统、还没准备好”

变成：

- “已准备好被调度执行”

此后，这个请求就归 scheduler 管了。

---

## 6. Scheduler 是如何决定“下一轮算谁”的

调度核心在 `Scheduler.get_next_batch()`，位于 [scheduler.py](scheduler.py)。

这是 online inference 最关键的控制决策点之一。

---

## 6.1 Scheduler 管理三类请求队列

### `waiting_q`

- 已分词
- 尚未开始执行
- 等待进入 prefill

### `running_q`

- 已经在执行流程中
- 其 KV 状态当前驻留在 GPU
- 后续还能继续 decode

### `swapped_q`

- 逻辑上还没结束
- 但因 GPU block 不够，被换到了 CPU swap
- 等以后有资源再换回 GPU

所以请求状态流转通常是：

```text
waiting_q -> running_q -> 完成
                │
                └-> swapped_q -> running_q
```

---

## 6.2 为什么要区分 prefill 和 decode

在线推理里，一条请求通常分两个阶段：

### prefill

- 把整段 prompt 喂给模型
- 建立这条请求的初始 KV cache
- 通常是一次输入很多 token

### decode

- 在已有 KV cache 基础上，一轮轮只喂最后一个 token
- 每轮生成 1 个新 token
- 是持续推进式的

这两种阶段的资源特征很不一样：

- prefill 吃的是“本轮输入 token 数”
- decode 更依赖“已有序列状态”和 KV cache 占用

所以 scheduler 必须把两者区分开。

---

## 6.3 `get_next_batch()` 的第一优先级：尝试发起 prefill batch

当 `swapped_q` 为空时，scheduler 会优先尝试从 `waiting_q` 中取请求，组一个新的 prefill batch。

判断能不能继续往 batch 里塞新请求时，会同时受这些条件约束：

- `len(cur_batch)+1 <= max_batch_size`
- `len(running_q)+len(cur_batch)+1 <= max_batch_size`
- `cur_batch_block_needed + cur_seq_block_needed + num_decoding_gpu_blocks <= num_gpu_blocks`
- `cur_num_tokens_sum + cur_seq.prompt_len <= max_tokens_in_batch`

这四类约束分别控制：

1. 新 prefill batch 自身不能太大
2. 系统整体活跃序列数不能太多
3. GPU blocks 不能超容量
4. 本轮 prefill token 总量不能超上限

这里采用的是**严格 FCFS**：

- 只看队头请求
- 如果队头请求塞不进去，就直接停止
- 不会跳过队头去拿后面的短请求

所以它优先保证“先来先服务”，而不是“吞吐最优”。

---

## 6.4 prefill 请求何时拿到 `request_id`

一旦一个 prefill batch 被正式选中，这些请求会被分配 `request_id`：

- `req.request_id = self.request_id_manager.get_id()`

这个 `request_id` 很重要。

它不是简单的业务 ID，而是：

> 这条序列在 block table 中的索引

后续：

- KV block 分配
- swap in/out
- 资源释放

都要靠这个 ID 定位。

---

## 6.5 如果这轮不做 prefill，就进入 decoding 调度

如果没有新的 prefill batch 可发，scheduler 就会继续处理正在运行的 decoding 请求。

这里先重新统计：

- 当前 `running_q` 一共占了多少 GPU blocks

然后判断是否超限：

- 活跃解码请求数是否超过 `max_batch_size`
- 总 block 占用是否超过 `num_gpu_blocks`

如果超限，就从 `running_q` 的尾部开始把请求弹出，加入 `newly_swapped_out`。

这表示：

- 这些请求不会被丢弃；
- 它们只是暂时从 GPU 上挪走，等待以后恢复。

---

## 6.6 swap out / swap in 的含义

如果某轮触发了 `swap_out`：

- 请求的 KV blocks 会从 GPU 移到 CPU swap 空间
- 这些请求进入 `swapped_q`

如果某轮没有触发新的 swap out，且 GPU 有余量，则 scheduler 会尝试从 `swapped_q` 头部开始把请求换回 GPU，即 `swap in`。

所以 `swapped_q` 可以理解为：

> 一个“暂停区”——请求没结束，只是暂时不占 GPU。

这正是 online serving 能承受更多并发请求的关键机制之一。

---

## 6.7 当前实现中的一个重要事实：调度层基本按“纯 prefill”或“纯 decode”批次推进

底层 `LlamaModel.forward()` 其实支持混合 batch：

- 前半部分是 prefill 请求
- 后半部分是 decode 请求

但从当前 `Scheduler.get_next_batch()` 的实现来看：

- 要么返回一个新 prefill batch
- 要么返回 `running_q` 做 decoding batch

源码里还有一句注释：

- “If you want decoding requests to be piggybacked, you can do it here”

说明：

- **执行层支持混合 prefill+decode**
- **但当前调度策略没有开启这种 piggyback**

这点很重要，因为它说明：

> 目前 online serving 的核心复杂度主要在控制调度与内存状态，而不是复杂的混合批策略。

---

## 7. Engine 是如何驱动每一轮模型执行的

核心逻辑在 `Engine._main_event_loop()`，位于 [engine.py](engine.py)。

这条循环就是整个 online inference 的“心跳”。

---

## 7.1 每一轮先向 scheduler 要执行计划

每次循环先调用：

- `cur_batch, cur_swap_in, cur_swap_out = self.scheduler.get_next_batch()`

可以把这三个返回值理解成：

- `cur_batch`：这轮真正要 forward 的请求列表
- `cur_swap_in`：在 forward 前，要先从 CPU 搬回 GPU 的请求
- `cur_swap_out`：在 forward 前，要先从 GPU 搬去 CPU 的请求

如果三个都空，说明当前系统暂时没活可干，Engine 就 sleep 一小会儿继续轮询。

---

## 7.2 真正 forward 前，先处理 swap

如果有 `cur_swap_out`：

- 调 `self.model.swap_out_seqs(...)`

如果有 `cur_swap_in`：

- 调 `self.model.swap_in_seqs(...)`

注意这里 Engine 没有自己操作 KV cache，而是把命令发给 `LlamaModel`。

也就是说：

- **swap 的决策由 scheduler 做**
- **swap 的执行由 model / block manager 做**

这是 control plane / data plane 分离的一个非常典型的体现。

---

## 7.3 构造这一轮 forward 的输入

Engine 接下来会根据请求当前所处阶段，构造 `input_ids`：

- 如果请求还没产出任何 token，说明处于 prefill 阶段：
  - 输入整段 `prompt_token_ids`
- 否则说明处于 decode 阶段：
  - 只输入最后一个生成 token，即 `[req.output_token_ids[-1]]`

同时会构造：

- `seq_ids`
- `decoding_seq_lens_list`

其中 `decoding_seq_lens_list` 表示：

- 对于正在 decode 的请求，它当前总序列长度是多少
- 即 `prompt_len + 已生成 token 数`

这是底层 paged attention / RoPE / block 分配都需要知道的信息。

---

## 7.4 为什么 Engine 要用 `run_in_executor()` 调模型

`Engine._run_on_model_async()` 内部会调用：

- `event_loop.run_in_executor(None, func_partial)`

这是个非常关键的工程点。

原因是：

- `LlamaModel.forward()` 是同步的、重量级的 GPU 计算入口
- 如果直接在 asyncio event loop 里同步执行，会阻塞整个异步系统
- 阻塞之后：
  - API 响应处理会受影响
  - tokenization loop 会受影响
  - 整个在线系统的并发性会变差

所以这里的设计是：

- **Engine 继续保持 async 控制逻辑**
- **模型执行丢到 executor 中做**

这让“控制”和“重计算”保持相对解耦。

---

## 7.5 模型这一轮 forward 完之后，Engine 怎么处理结果

`model.forward(...)` 返回后，Engine 会得到每个请求这一轮对应的输出 token。

随后它会逐个请求做几件事：

1. `req.output_token_ids.append(output_token)`
2. `req.output_q.put_nowait(StepOutput(output_token, req))`
3. 如果请求完成：
   - 记录到 `finished_req_ids`
   - `req.finished_event.set()`

这三步分别服务于：

- 保持请求内部状态
- 支持流式输出
- 支持非流式等待结束

最后 Engine 还会调用：

- `self.model.free_seqs_resources(finished_req_ids)`

把已经完成的请求所占用的 GPU/CPU blocks 释放掉。

再调用：

- `self.scheduler.on_batch_finish(cur_batch)`

通知调度器更新状态。

也就是说，一轮 forward 结束后，Engine 做的是：

- 更新请求状态
- 唤醒等待者
- 回收资源
- 通知调度器进入下一轮

---

## 8. 底层 `worker` 是怎么接住这些调度命令的

虽然本文重点是 `server/`，但如果不看一点 `worker/`，就很难真正理解 online inference 为什么能成立。

核心执行器在 [../worker/model.py](../worker/model.py)。

---

## 8.1 `LlamaModel.forward()` 是在线执行层的主入口

`Engine` 最终调用的是：

- `self.model.forward(input_ids, seq_ids, decoding_seq_lens_list)`

`LlamaModel.forward()` 做的事情可以概括成：

1. 识别 batch 中哪些是 prefill，哪些是 decode
2. 把所有输入 token flatten 成一维 tensor
3. 构造每条序列的长度信息
4. 计算每个 token 的 position index
5. 必要时为序列分配/扩展 KV blocks
6. 选择 paged attention 的 `seq_block_size`
7. 构造 `LlamaInferState`
8. 调 `_forward()` 真正执行模型前向

所以 `forward()` 不只是“跑一遍网络”，它其实还承担了：

- batch 元数据整理
- KV cache 对应关系建立
- 推理状态对象封装

---

## 8.2 `BlockManager`：在线推理的内存状态管理员

文件：[../worker/block_manager.py](../worker/block_manager.py)

它管理的核心东西有三个：

- `num_seq_allocated_blocks`
- `block_table`
- `is_block_free`

可以理解为：

- 某条序列已经分到多少块
- 这条序列的第 k 块映射到哪个物理 block id
- 整个设备上哪些 blocks 还空闲

它提供的关键方法包括：

- `allocate_blocks_for_seqs()`
- `free_blocks_for_seqs()`
- `gather_allocated_blocks_and_free()`
- `get_num_allocated_blocks()`

这意味着 scheduler 决定“要 swap”之后，底层真的能完成“把序列状态迁移/释放/恢复”，靠的就是这些 block 管理结构。

---

## 8.3 `swap_in_seqs()` / `swap_out_seqs()` 让在线服务具备“超出 GPU 容量仍能继续排队推进”的能力

在 [../worker/model.py](../worker/model.py) 中：

- `swap_in_seqs()`
- `swap_out_seqs()`

最终都会走到 `_swap()`：

- 从源 block manager 收集 block ids
- 在目标 block manager 上重新分配 block ids
- 调用底层 C 扩展 `swiftllm_c.swap_blocks(...)`

这就完成了：

- GPU KV cache <-> CPU swap 空间

之间的数据迁移。

没有这套机制，online inference 只能同时容纳非常有限的活跃序列；
有了它，系统就可以在 GPU 紧张时临时“挂起”一部分请求。

---

## 9. streaming 和 non-streaming 本质差异在哪里

这一点源码里体现得很清楚：

### streaming

- API 层消费 `engine.add_request_and_stream()` 返回的异步生成器
- 每拿到一个 `StepOutput`，就向客户端写一次
- 如果 `decode=True`，还会调用 tokenizer actor 做 token -> 文本解码

### non-streaming

- API 层调用 `engine.add_request_and_wait()`
- 等 `finished_event` 被置位
- 一次性返回全部 `output_token_ids`
- 如果 `decode=True`，最后再整体 decode 一次

所以本质上：

- **生成过程相同**
- **调度过程相同**
- **模型执行过程相同**
- **差别只在输出消费路径**

你可以把它理解成：

- streaming：边生产边消费
- non-streaming：全部生产完再消费

---

## 10. 一个具体案例：3 个请求并发时，server 是如何推进 online inference 的

下面用一个稍微贴近真实服务的案例，把整个流程串起来。

> 为了便于理解，下面的 token 数和 block 占用是示意性的，不是精确 profiling 结果。

### 请求设定

有三个请求：

- **请求 A**
  - prompt 长度约 40 tokens
  - 需要输出 4 tokens
  - `stream=true`
- **请求 B**
  - prompt 长度约 120 tokens
  - 需要输出 3 tokens
  - `stream=false`
- **请求 C**
  - 稍晚一点到达
  - prompt 长度约 20 tokens
  - 需要输出 2 tokens
  - `stream=true`

同时假设：

- 当前配置允许一定数量的 GPU blocks
- `max_batch_size` 足以先容纳 A、B
- C 到来时，系统已有正在 decode 的请求

---

### 阶段 1：A、B 到达 API 层

客户端分别调用 `/generate`。

在 [api_server.py](api_server.py) 中：

- A、B 都先被转成 `RawRequest`
- A 因为是流式，调用 `engine.add_request_and_stream()`
- B 因为是非流式，调用 `engine.add_request_and_wait()`

这时：

- A、B 都已被包装成 `Request`
- 但都还没有 `prompt_token_ids`
- 都先进入 `Engine.untokenized_raw_requests`

系统状态大概是：

```text
untokenized_raw_requests = [A, B]
waiting_q = []
running_q = []
swapped_q = []
```

---

### 阶段 2：tokenization loop 把 A、B 一起分词

后台 `Engine._tokenize_raw_request_event_loop()` 发现未分词队列非空，于是：

- 取出 A、B
- 调 `TokenizationEngine.batched_tokenize.remote([promptA, promptB])`
- 得到各自 `prompt_token_ids`
- 回填到请求对象中
- 调 `Scheduler.on_requests_arrival([A, B])`

现在状态变成：

```text
untokenized_raw_requests = []
waiting_q = [A, B]
running_q = []
swapped_q = []
```

这一步完成后，A、B 才算真正进入“可调度状态”。

---

### 阶段 3：scheduler 组织 A、B 的 prefill batch

主循环 `Engine._main_event_loop()` 调 `scheduler.get_next_batch()`。

因为当前 `swapped_q` 为空，scheduler 先尝试发起 prefill。

假设 A、B 都满足：

- batch size 限制
- max tokens in batch 限制
- GPU block 限制

那么本轮会返回：

- `cur_batch = [A, B]`
- `cur_swap_in = []`
- `cur_swap_out = []`

并且：

- A、B 获得各自 `request_id`
- A、B 进入 `running_q`

状态变成：

```text
waiting_q = []
running_q = [A, B]
swapped_q = []
```

---

### 阶段 4：Engine 驱动第一轮 forward（prefill）

Engine 根据 A、B 仍处于 prefill 阶段，构造输入：

- A 输入整段 `prompt_token_ids(A)`
- B 输入整段 `prompt_token_ids(B)`

然后调用：

- `LlamaModel.forward(...)`

底层会：

- 给 A、B 分配对应 blocks
- 建立 block table 映射
- 构造推理状态
- 完成 prefill 前向
- 返回每个请求本轮采样出的 token

注意：

虽然这轮输入是整段 prompt，但从 Engine 的逻辑看，它仍把返回值视作“每个请求一个输出 token”。

于是：

- A 的第 1 个输出 token 加入 `A.output_token_ids`
- B 的第 1 个输出 token 加入 `B.output_token_ids`
- A 的 `output_q` 被塞进一个 `StepOutput`

对 A 来说，因为它是 streaming，请求处理协程会立刻从 `output_q` 取到结果，并开始给客户端回传第一个增量输出。

对 B 来说，因为是 non-streaming，调用方仍在等 `finished_event`，不会立即返回。

---

### 阶段 5：系统进入 decode 推进阶段

下一轮再调 `scheduler.get_next_batch()` 时，已经没有 waiting 请求了，于是 scheduler 开始组织 decoding batch：

- `cur_batch = running_q = [A, B]`

此时 Engine 构造输入时就不再传整段 prompt，而是：

- A 只传上轮刚生成的最后一个 token
- B 也只传上轮刚生成的最后一个 token

同时还会告诉 model：

- A 当前总序列长度 = `prompt_len(A) + 已生成数`
- B 当前总序列长度 = `prompt_len(B) + 已生成数`

底层利用现有 KV cache，只增量算下一 token。

每完成一轮：

- A streaming 继续收到新 token
- B 还继续等待

这就是 online decoding 的本质：

> 每轮只推进一步，但可同时推进多条活跃请求。

---

### 阶段 6：C 在 A、B decode 过程中到达

这时请求 C 到来。

它会先走和前面一样的 API 路径：

- 变成 `RawRequest`
- 再变成 `Request`
- 先进入 `untokenized_raw_requests`

随后 tokenization loop 会把它分词好，送入 `waiting_q`。

此时系统状态可能变成：

```text
waiting_q = [C]
running_q = [A, B]
swapped_q = []
```

接下来 scheduler 会根据当时资源判断：

#### 情况 1：资源够

那就可以在合适时机发起 C 的 prefill。

#### 情况 2：资源不够

那 C 会继续留在 `waiting_q`。

#### 情况 3：decode 活跃序列太多 / GPU blocks 紧张

那 scheduler 可能把 `running_q` 末尾的某个请求 swap out 到 CPU，释放 GPU 空间。

比如它可能暂时把 B 换出：

```text
waiting_q = [C]
running_q = [A]
swapped_q = [B]
```

然后再为 C 创造进入计算的机会。

这就体现出 online serving 和 offline batch inference 的根本区别：

- 请求不是“一批输入，一次跑完”
- 而是在系统里被持续调度、推进、换入换出

---

### 阶段 7：请求完成与资源回收

假设 A 先达到 `output_len=4`。

某一轮 forward 后，Engine 发现：

- `A.is_finished() == True`

于是会：

- `A.finished_event.set()`
- 把 A 的 `request_id` 收集到 `finished_req_ids`
- 调 `model.free_seqs_resources([A.request_id])`
- `scheduler.on_batch_finish(...)` 把 A 从 `running_q` 中移除，并回收 request id

这时：

- A 的流式响应结束
- A 占用的 KV blocks 被释放
- 系统有更多空间继续推进 B / C

如果 B 随后完成，则等待它的 non-streaming 调用者会被唤醒，一次性拿到全部 `output_token_ids`。

---

## 11. 从这个案例里，能提炼出什么“核心逻辑”

到这里，可以把 SwiftLLM server 支撑 online inference 的核心逻辑总结成 5 句话。

### 1）API 层只是入口，不是调度中心

`api_server.py` 负责接请求和返回结果，真正驱动在线推理的是 `Engine`。

### 2）Engine 是一个双循环控制器

- 一个循环负责把文本请求转成 token 请求
- 一个循环负责持续向 scheduler 取 batch 并驱动 model 执行

这是整个 online serving 的主框架。

### 3）Scheduler 决定“谁先算、谁继续算、谁先下 GPU”

它通过：

- FCFS
- prefill / decode 区分
- swap in / swap out
- batch 与 block 约束

来决定每轮的执行计划。

### 4）Worker 层负责真正执行和维护 KV 状态

`LlamaModel`、`BlockManager`、KV cache、CPU swap 这些东西，才是 online inference 能持续增量推进的底座。

### 5）Streaming / non-streaming 只是输出消费方式不同

底层生成逻辑一致，区别只在：

- 是每轮立即消费一个 token
- 还是等全部生成完再统一返回

---

## 12. 你可以如何理解 SwiftLLM 的 online serving 核心框架

如果再往抽象层提一层，我会把这个 server 的设计概括为：

> **一个轻量但完整的在线推理控制面。**

它具备 online inference 的几个关键要素：

- 有统一请求入口
- 有请求对象状态管理
- 有分词与解码服务
- 有异步后台循环
- 有调度器
- 有 KV cache / swap 支持
- 有流式与非流式两种输出路径

虽然它不像成熟工业系统那样拥有更复杂的：

- admission control
- priority scheduling
- cancellation
- retry / timeout / backpressure
- 更激进的 mixed batching 策略

但它已经把 online serving 的骨架搭得非常清楚了。

对于理解 LLM serving 来说，这套实现非常有代表性，因为它把几个最关键的问题都直接暴露在代码里：

- 请求如何建模
- 调度如何建模
- KV cache 如何建模
- 在线 decoding 如何一轮轮推进
- 资源不足时如何 swap

这也是为什么 `swiftllm/server/` 很值得仔细读：

**它不是把 online inference 封装得看不见，而是把整个控制逻辑几乎裸露出来了。**

---

## 13. 推荐的源码阅读顺序

如果你接下来想继续深入，我建议按这个顺序读：

1. [api_server.py](api_server.py)
   - 看外部请求如何进入系统
2. [structs.py](structs.py)
   - 看请求对象长什么样
3. [engine.py](engine.py)
   - 看双循环和总控逻辑
4. [scheduler.py](scheduler.py)
   - 看调度决策
5. [tokenization_engine.py](tokenization_engine.py)
   - 看 tokenizer actor 如何解耦
6. [../worker/model.py](../worker/model.py)
   - 看 engine 最终如何驱动执行层
7. [../worker/block_manager.py](../worker/block_manager.py)
   - 看 KV blocks 如何分配/释放/迁移

如果你只想抓住主线，优先看：

- [engine.py](engine.py)
- [scheduler.py](scheduler.py)
- [../worker/model.py](../worker/model.py)

因为这三者合起来，基本就是 SwiftLLM online inference 的心脏。

# api server use

## Context
用户想根据 `swiftLLM/swiftllm/server/api_server.py` 理解如何启动 SwiftLLM 的 API 服务，而不是修改代码。目标是给出最直接的启动方式、必需参数、依赖前提，以及一个最小可验证请求。

## Recommended approach
1. 先按仓库说明安装运行时依赖：`fastapi`、`uvicorn`、`ray`、`transformers`、PyTorch，以及 SwiftLLM 本体和 `csrc` 扩展。
2. 准备一个本地模型目录，并将其作为 `--model-path` 传入；这里必须是本地路径，代码不会自动从 Hugging Face 下载模型。
3. 进入 `swiftLLM/` 目录后，直接运行 `swiftllm/server/api_server.py`：
   - 推荐命令：`python swiftllm/server/api_server.py --model-path /path/to/model --host 0.0.0.0 --port 8000`
   - `--model-path` 是必填。
   - `--host` 默认 `localhost`，`--port` 默认 `8000`。
4. 首次启动时，服务会先执行初始化：加载权重、profile KV blocks、分配 KV cache / swap、创建 scheduler、启动 tokenization Ray actor，然后再并发跑 Uvicorn HTTP 服务和 engine 事件循环。
5. 用 `POST /generate` 验证：
   - 最小请求体：`{"prompt":"Hello","output_len":16}`
   - 如果希望直接拿到可读文本，务必显式传 `"decode": true`。虽然注释写的是默认 true，但代码实际默认是 false。
   - 流式输出时加 `"stream": true`。

## Critical files
- `swiftLLM/swiftllm/server/api_server.py`
- `swiftLLM/swiftllm/engine_config.py`
- `swiftLLM/swiftllm/server/engine.py`
- `swiftLLM/swiftllm/server/tokenization_engine.py`
- `swiftLLM/README.md`
- `swiftLLM/requirements.txt`

## Reuse / code paths found
- `swiftllm.EngineConfig.add_cli_args(...)`：定义 API 服务可用 CLI 参数。
- `swiftllm.Engine(...)`：构造在线 serving 引擎。
- `Engine.initialize()`：完成模型、KV cache、scheduler、tokenizer actor 初始化。
- `Engine.start_all_event_loops()`：启动 tokenization loop + main scheduling loop。
- `@app.post("/generate")`：唯一 HTTP 入口。

## Verification
1. 启动服务。
2. 发送非流式请求：`POST http://127.0.0.1:8000/generate`，JSON 里带 `prompt`、`output_len`、`decode=true`。
3. 发送流式请求：同一路径，JSON 里带 `stream=true`，确认返回按 token/文本增量输出。
4. 如需远程访问，确认 `--host 0.0.0.0` 而不是默认 `localhost`。

```
可以，按这个文件的实现，启用 SwiftLLM API 服务的方式就是直接运行它：


cd /home/yxlin/github/swift/swiftLLM
python swiftllm/server/api_server.py --model-path /你的本地模型目录 --host 0.0.0.0 --port 8000
关键点如下：

--model-path 是必填，定义在 swiftLLM/swiftllm/engine_config.py:32-37
--host 默认是 localhost，--port 默认是 8000，定义在 swiftLLM/swiftllm/server/api_server.py:87-97
启动流程是：
创建 Engine
await engine.initialize()
并发启动 Uvicorn HTTP 服务和 engine 后台事件循环
见 swiftLLM/swiftllm/server/api_server.py:97-121
你需要先准备好环境，至少包括这些依赖：

torch
fastapi
uvicorn
ray[default]
transformers
依赖列表在 swiftLLM/requirements.txt:1-7，README 的安装步骤在 swiftLLM/README.md:66-87。

一个比较完整的初始化流程通常是：


cd /home/yxlin/github/swift/swiftLLM

pip install -r requirements.txt
pip install -e .
pip install -e csrc

python swiftllm/server/api_server.py \
  --model-path /path/to/model \
  --host 0.0.0.0 \
  --port 8000
服务接口是：

POST /generate
定义在 swiftLLM/swiftllm/server/api_server.py:16-84
最小请求示例：


curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello",
    "output_len": 16,
    "decode": true
  }'
如果你要流式输出：


curl -N -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello",
    "output_len": 16,
    "stream": true,
    "decode": true
  }'
再补充两个容易踩坑的点：

模型路径必须是本地目录

代码不会自动从 Hugging Face 拉模型，说明见 swiftLLM/swiftllm/engine_config.py:32-37 和 swiftLLM/README.md:84-87。

decode 默认实际是 False，不是注释里写的 True

实际代码是 swiftLLM/swiftllm/server/api_server.py:31-33。

所以如果你想直接返回文本，最好显式传 "decode": true。

另外，这个服务里 TokenizationEngine 用了 Ray actor，swiftLLM/swiftllm/server/tokenization_engine.py:18-28，但我在仓库里没看到显式 ray.init()；如果你启动时报 Ray 相关错误，通常要自己先初始化 Ray，或者补一层启动逻辑。

如果你愿意，我下一步可以直接帮你整理成一份可复制执行的启动脚本，或者根据你的模型路径给你写出一条完整命令。
```