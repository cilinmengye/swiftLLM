# `swiftllm_benchmark.py` 阅读导图

这份文档不是重复代码注释，而是给你一个**按阅读顺序展开的地图**，帮助你快速建立对脚本整体结构的心智模型。

对应脚本：[`swiftllm_benchmark.py`](./swiftllm_benchmark.py)

---

## 1. 先用一句话理解这个脚本在做什么

这个脚本的本质是：

> 用和 vLLM 在线基准尽量一致的方法，向 SwiftLLM 的真实 `/generate` 服务持续发请求，测量首 token、相邻 token、整体请求延迟，并把结果保存成 JSON 和 CSV。

它不是训练脚本，不是离线推理脚本，也不是直接调用 `Engine` 的内部测试脚本。
它是一个**在线服务 benchmark client**。

---

## 2. 你应该先带着这三个问题去读

读这个脚本时，最容易迷路的原因是“函数很多，但不知道每层在回答什么问题”。

你可以先把它拆成三个核心问题：

### 问题 A：发什么请求？
也就是：
- 请求数据从哪里来？
- `prompt` / `output_len` 是怎么定的？
- 为什么这些请求能和 vLLM 基准尽量对齐？

这一部分主要看：
- `SampleRequest`：`swiftllm_benchmark.py:87`
- `extract_prompt_completion()`：`swiftllm_benchmark.py:387`
- `is_valid_sequence()`：`swiftllm_benchmark.py:421`
- `maybe_oversample_requests()`：`swiftllm_benchmark.py:446`
- `load_sharegpt_requests()`：`swiftllm_benchmark.py:487`

### 问题 B：按什么节奏发？
也就是：
- 请求速率 `request_rate` 是怎样变成“每个请求什么时候发”的？
- 为什么这里会出现 Poisson / Gamma / burstiness？
- 最大并发是怎么限制的？

这一部分主要看：
- `build_delay_schedule()`：`swiftllm_benchmark.py:551`
- `benchmark_request_rate()`：`swiftllm_benchmark.py:1059`

### 问题 C：发出去之后测什么？
也就是：
- TTFT / ITL / TPOT 是怎么从流式返回里测出来的？
- 最后 JSON 和 CSV 是怎么汇总出来的？

这一部分主要看：
- `RequestMetrics`：`swiftllm_benchmark.py:107`
- `issue_request()`：`swiftllm_benchmark.py:614`
- `compute_peak_metrics()`：`swiftllm_benchmark.py:836`
- `build_result_payload()`：`swiftllm_benchmark.py:890`
- `benchmark_summary_row()`：`swiftllm_benchmark.py:990`

---

## 3. 先看最外层主流程：`main()`

入口在：`swiftllm_benchmark.py:1126`

如果你只想先抓住大框架，就只看 `main()`，因为它直接定义了脚本的执行顺序。

它做的事情可以概括成 7 步：

1. `parse_args()`：读取命令行参数
2. 创建结果输出目录
3. 加载 tokenizer
4. `load_sharegpt_requests()`：构造固定请求集合
5. 对每个 `request_rate` 调 `benchmark_request_rate()`
6. 每个速率写一份 `rate_x.json`
7. 最后汇总写 `result.csv`

你可以把 `main()` 理解为一个总控器：

```text
准备参数
  -> 准备请求样本
  -> 按不同速率重复压测
  -> 保存结果
```

所以读代码时，最推荐的顺序其实是：

```text
main()
  -> load_sharegpt_requests()
  -> benchmark_request_rate()
       -> build_delay_schedule()
       -> issue_request()
       -> build_result_payload()
```

---

## 4. 模块分层阅读法

这个脚本虽然是一个文件，但其实已经自然分成了 5 层。

---

### 第 1 层：配置与参数层

对应位置：
- 常量区：`swiftllm_benchmark.py:56`
- 参数 dataclass：`swiftllm_benchmark.py:136`
- 参数解析函数：`swiftllm_benchmark.py:180` 到 `swiftllm_benchmark.py:345`

这一层只回答一个问题：

> 这次 benchmark 要怎么跑？

包括：
- 连哪个服务：`host` / `port` / `endpoint`
- 用哪个模型 tokenizer：`model_path`
- 用哪个数据集：`dataset_path`
- 跑多少请求：`num_prompts`
- 每秒发多少请求：`request_rates`
- 到达过程有多“抖动”：`burstiness`
- 最大并发多少：`max_concurrency`
- 是否保存详细结果：`save_result` / `save_detailed`

#### 这一层最关键的认识
这里的参数不负责真正执行 benchmark，它们只是为后面的流程提供“运行规则”。

你可以把 `BenchmarkArgs` 看成这份脚本的**全局配置快照**。

---

### 第 2 层：请求样本构造层

对应位置：
- `SampleRequest`：`swiftllm_benchmark.py:87`
- `tokenize_text()`：`swiftllm_benchmark.py:366`
- `extract_prompt_completion()`：`swiftllm_benchmark.py:387`
- `is_valid_sequence()`：`swiftllm_benchmark.py:421`
- `maybe_oversample_requests()`：`swiftllm_benchmark.py:446`
- `load_sharegpt_requests()`：`swiftllm_benchmark.py:487`

这一层回答的问题是：

> benchmark 到底会发哪些请求？

### 这一层的数据流

```text
原始 ShareGPT JSON
  -> 提取前两轮对话
  -> 得到 prompt / completion
  -> tokenizer 计算 prompt_len / output_len
  -> 按规则过滤
  -> 形成 SampleRequest 列表
```

### 为什么这里这么重要
因为 benchmark 是否“公平”，首先取决于**你拿什么请求去打服务端**。

这个脚本刻意复用了 vLLM 的 ShareGPT 采样思路：
- 只取前两轮对话
- 要求 prompt/output 都有最小长度
- 总长度不能超限
- 可以保持原始顺序
- 样本不够时做确定性 oversample

### 你读这层时可以重点看这几个点

#### 1. `extract_prompt_completion()` 干了什么
它不是通用对话解析器，而是一个非常“benchmark-oriented”的提取器：
- 只接受 `dict`
- 只看 `conversations`
- 要求至少有两轮
- 直接取：
  - `conversations[0]["value"]` 作为 `prompt`
  - `conversations[1]["value"]` 作为 `completion`

所以它回答的是：

> “这条 ShareGPT 记录能不能变成 benchmark 请求？”

#### 2. `is_valid_sequence()` 在守什么边界
它把样本过滤规则集中到一个地方：
- `prompt_len >= 4`
- `output_len >= 4`
- `prompt_len <= 1024`
- `prompt_len + output_len <= 2048`

这一步很关键，因为它保证最后送去压测的请求，和 vLLM 那套 benchmark 的样本空间尽量一致。

#### 3. `maybe_oversample_requests()` 为什么存在
如果过滤后样本不够 `num_prompts`，这个函数会从已有有效样本里做**确定性补样**。

你可以把它理解成：

> “先保证方法一致；如果数量不够，再可复现地补足数量。”

#### 4. `load_sharegpt_requests()` 是这层真正的总控函数
前面的函数都像零件，`load_sharegpt_requests()` 才是把这些零件串起来的人。

它真正完成了：
- 读 JSON
- 可选 shuffle
- 提取 prompt/completion
- tokenizer 算长度
- 过滤
- 编号成 `req-0`, `req-1`, ...
- 必要时 oversample

所以你如果只想抓住“样本是怎么来的”，就重点看这个函数。

---

### 第 3 层：请求调度层

对应位置：
- `build_delay_schedule()`：`swiftllm_benchmark.py:551`
- `benchmark_request_rate()` 中 schedule 的使用：`swiftllm_benchmark.py:1111` 到 `swiftllm_benchmark.py:1118`

这一层回答的问题是：

> 这些请求不是一起发出去的，那它们分别在什么时候发？

### 这一层的数据流

```text
request_rate + burstiness + seed
  -> 采样 inter-arrival delay
  -> 转成 cumulative delay
  -> 得到每个请求的目标发射时刻
```

### `build_delay_schedule()` 应该怎么理解
这个函数返回的是一个列表，例如：

```python
[0.18, 0.73, 1.02, 1.61, ...]
```

它的含义不是“sleep 多久”，而是：

- 第 1 个请求在 benchmark 开始后 0.18 秒发出
- 第 2 个请求在 benchmark 开始后 0.73 秒发出
- 第 3 个请求在 benchmark 开始后 1.02 秒发出

也就是说，它是一个**绝对发射时间表（相对 benchmark 起点）**。

### 为什么这里会有 Gamma / Poisson
这里复用了 vLLM 那边常见的建模方式：
- `burstiness = 1.0` 时，对应标准 Poisson 风格到达
- `burstiness > 1.0` 时，更均匀
- `burstiness < 1.0` 时，更容易扎堆

所以 `request_rate` 只决定平均强度，`burstiness` 决定“抖动形状”。

### 为什么后面还要做 normalize
脚本不是只采样 delay 就结束了，而是还会做一次归一化：

```text
target_total_delay = total_requests / request_rate
```

这样做的目的不是改变随机性，而是把整轮请求的总发射跨度拉回理论目标，减少随机采样造成的整体漂移。

这是一个很容易忽略但非常重要的细节。

---

### 第 4 层：单请求执行与测量层

对应位置：
- `issue_request()`：`swiftllm_benchmark.py:614`
- `run_readiness_check()`：`swiftllm_benchmark.py:694`

这一层回答的问题是：

> 一个请求真正发出去之后，脚本怎么知道 TTFT、ITL、latency 分别是多少？

### 先看请求长什么样
`issue_request()` 里发给 SwiftLLM 的请求体是：

```json
{
  "prompt": "...",
  "output_len": 128,
  "stream": true,
  "decode": false
}
```

其中最关键的是：
- `stream = true`：因为要测 token 到达时间
- `decode = false`：因为 benchmark 关心的是服务路径里的生成节奏，而不是逐 token 文本解码开销

### `RequestMetrics` 是这一层的结果容器
每发一个请求，最后都会得到一个 `RequestMetrics`。

它记录：
- 是否成功
- 总 latency
- 收到了多少输出 token
- TTFT
- ITL 列表
- 错误信息
- 开始时间

所以它本质上是：

> “单个请求被测量之后留下的原始观测结果”

### `issue_request()` 的核心测量逻辑
可以把它简化理解成：

```text
记录 started_at
  -> 发 HTTP POST
  -> 流式读每一行 token id
  -> 第一个 token 到达：记 TTFT
  -> 后续 token 到达：记 ITL
  -> 最后一个 token 到达：记 latency
```

### 为什么读取流时要 `int(line)`
这一步不是多余的。

它其实是在做一个很强的假设校验：

> “服务端流式返回的每一行都应该是 token id。”

如果某一行不是整数，就直接进异常分支，整个请求记为失败，而不是把脏数据误记成正常 token。

### `run_readiness_check()` 的角色
它不是 benchmark 主流程的一部分指标计算器，而是一个**开跑前探活器**。

作用是：
- 先拿一个真实样本试一下
- 如果服务端根本打不通，就立刻报错
- 避免 benchmark 跑了几十分钟才发现服务没起来

这一步把“服务没准备好”和“性能差”分开了。

---

### 第 5 层：统计汇总与结果输出层

对应位置：
- 统计辅助函数：`swiftllm_benchmark.py:720` 到 `swiftllm_benchmark.py:833`
- `compute_peak_metrics()`：`swiftllm_benchmark.py:836`
- `build_result_payload()`：`swiftllm_benchmark.py:890`
- `benchmark_summary_row()`：`swiftllm_benchmark.py:990`
- `write_result_csv()`：`swiftllm_benchmark.py:1038`

这一层回答的问题是：

> 所有请求都跑完以后，怎么把大量逐请求数据变成你最终看到的 JSON 和 CSV？

### 先分清“原始结果”和“汇总结果”
- `RequestMetrics`：单请求原始结果
- `result` 字典：一整个请求速率下的汇总结果
- `rows`：多个请求速率汇总成的 CSV 行

### 这层内部又可以分成 3 小块

#### 小块 A：基础统计函数
包括：
- `safe_mean()`
- `safe_median()`
- `safe_pstdev()`
- `compute_percentile()`
- `format_percentile()`
- `add_metric_summary()`

这些函数本身不懂 benchmark 业务，它们只负责一件事：

> “给一组数，稳定地算均值/中位数/标准差/分位数，并写成统一字段名。”

#### 小块 B：峰值统计
`compute_peak_metrics()` 负责两个近似峰值：
- `max_output_tokens_per_s`
- `max_concurrent_requests`

注意它不是精确的连续时间峰值，而是**按秒分桶的近似峰值**。

所以你应该把它理解为：

> “一个足够直观、适合横向对比的峰值近似量。”

#### 小块 C：真正的结果拼装
`build_result_payload()` 是这一层的中心函数。

它会把 `outputs` 拆成多个数组：
- `ttfts`
- `itls`
- `tpots`
- `e2els`
- `output_lens`
- `per_input_token_latencies`

然后再写成最终 JSON。

### `TPOT` 在代码里不是直接存的，而是现算的
这是阅读时一个很值得注意的点。

脚本里没有在 `RequestMetrics` 里直接保存 `tpot`，而是在聚合阶段现算：

```text
(latency - ttft) / (output_tokens - 1)
```

这说明作者在概念上把它看成：

> “基于原始观测值派生出来的聚合指标”，而不是单独的第一类原始测量值。

### 为什么 JSON 里会多出 `mean_per_input_token_latency_ms`
这是为了补 README 与当前 CSV 定义不完全一致的问题。

当前 CSV 中间列仍然是：
- `mean_first_token_latency`，也就是 `mean_ttft_ms`

不是严格意义上的 per-input-token latency。

所以脚本额外在 JSON 里保存：

```text
ttft / prompt_len
```

这样你既能保持和现有 vLLM CSV 对齐，又不会丢掉 README 风格的派生指标。

### `benchmark_summary_row()` 为什么单独存在
因为 JSON 很详细，但 CSV 很克制。

CSV 只保留三列核心延迟：
- `mean_per_token_latency` <- `mean_itl_ms`
- `mean_first_token_latency` <- `mean_ttft_ms`
- `mean_per_output_token_latency` <- `mean_tpot_ms`

`benchmark_summary_row()` 的作用就是把“详细 JSON 的大字典”压缩成“兼容现有 vLLM 基线的一行 CSV”。

### `write_result_csv()` 为什么也值得看
虽然它很短，但它承担了一个很重要的职责：

> 固定输出列顺序，保证和现有基线文件兼容。

很多 benchmark 脚本最后的问题，不是算错，而是输出格式漂了。这里专门把列顺序钉死了。

---

## 5. `benchmark_request_rate()` 是整份脚本最像“主引擎”的函数

位置：`swiftllm_benchmark.py:1059`

如果 `main()` 是最外层总控，那么 `benchmark_request_rate()` 就是：

> “在某一个 request rate 下，真正把一轮 benchmark 跑起来的主引擎。”

它内部主要做 6 件事：

1. 拼 API URL
2. 建 `aiohttp` 连接池和 timeout
3. 建并发信号量 `Semaphore`
4. 先做 `run_readiness_check()`
5. 根据 `build_delay_schedule()` 按时发请求
6. `gather()` 全部结果并交给 `build_result_payload()`

### 这里有两个容易忽略的细节

#### 细节 1：并发不是只靠一个机制限制的
它同时用了：
- `aiohttp.TCPConnector(limit=...)`
- `asyncio.Semaphore(...)`

所以它不是只限制“连接数”，也不是只限制“协程进入数量”，而是两层一起卡。

#### 细节 2：schedule 是发射时刻，不是执行顺序
代码虽然是 `for request, target_delay in zip(...)` 顺序遍历，但真正决定请求何时起跑的是 `target_delay`。

也就是说，这个循环的重点不是“依次执行”，而是“依次安排发射时间”。

---

## 6. 推荐阅读顺序

如果你准备真正把这份脚本吃透，推荐按这个顺序读：

### 第一遍：只抓骨架
1. `main()`：`swiftllm_benchmark.py:1126`
2. `benchmark_request_rate()`：`swiftllm_benchmark.py:1059`
3. `build_result_payload()`：`swiftllm_benchmark.py:890`

目标：先知道这脚本总流程怎么走。

### 第二遍：抓 benchmark 方法论
1. `load_sharegpt_requests()`：`swiftllm_benchmark.py:487`
2. `build_delay_schedule()`：`swiftllm_benchmark.py:551`
3. `issue_request()`：`swiftllm_benchmark.py:614`

目标：搞清楚请求样本、到达过程、测量逻辑三件事。

### 第三遍：抓结果定义
1. `RequestMetrics`：`swiftllm_benchmark.py:107`
2. `compute_peak_metrics()`：`swiftllm_benchmark.py:836`
3. `benchmark_summary_row()`：`swiftllm_benchmark.py:990`

目标：搞清楚最终 JSON / CSV 每一项到底是什么意思。

---

## 7. 你可以把整份脚本压缩成这一张心智图

```text
BenchmarkArgs
  -> 决定怎么跑

ShareGPT dataset
  -> extract_prompt_completion()
  -> tokenize_text()
  -> is_valid_sequence()
  -> load_sharegpt_requests()
  -> 得到 SampleRequest[]

SampleRequest[] + request_rate
  -> build_delay_schedule()
  -> benchmark_request_rate()
       -> issue_request()
       -> 得到 RequestMetrics[]

RequestMetrics[]
  -> build_result_payload()
  -> benchmark_summary_row()
  -> write_result_csv()
```

如果你脑子里已经能稳定放下这张图，那你再回去看具体函数，就不会容易迷路了。

---

## 8. 最后给你一个“读代码时不要误解”的提醒

### 不要把这个脚本误解成“模型 benchmark 本体”
它不是模型内部 kernel benchmark。
它测的是：

- 真实 HTTP 请求路径
- 服务端流式返回路径
- 客户端并发与调度行为
- 最终端到端在线服务延迟

所以它更接近：

> “在线 serving benchmark”，而不是“离线推理速度测试”。

### 不要把中间那列 CSV 误解成 per-input-token latency
CSV 中间列当前仍然是：
- `mean_first_token_latency`

也就是 TTFT，不是严格意义上的 per-input-token latency。

真正的派生 per-input-token 指标在 JSON 里。

### 不要把 `output_len` 理解成“真实生成长度预测”
这里的 `output_len` 本质上是：

> benchmark 希望服务端最多生成多少 token。

它来自 ShareGPT completion 的 token 长度，是为了让 SwiftLLM benchmark 在请求规模上尽量对齐 vLLM。

---

## 9. 一句话收尾

如果你只记住一句话，那就是：

> 这个脚本做的是“按 vLLM 的方法造请求、按在线服务的方式发请求、按流式 token 到达时间测指标、最后输出可比较结果”。

抓住这条主线，整份代码就会清晰很多。