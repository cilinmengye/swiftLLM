# SwiftLLM online benchmark 排查与修正计划

## 背景

你观察到 `dummys/benchmark/online/swiftllm_benchmark.py` 跑出的结果，与 vLLM 官方 online benchmark 的结果差距非常大，尤其高 `request_rate` 下的 TTFT 明显异常放大。核心问题是：这到底是 SwiftLLM 服务端真的更慢，还是 benchmark 客户端本身的计时/解析逻辑有问题。

## 结论摘要

### 1. 计时起点大概率不是主因

我对照了 vLLM 官方 benchmark 的实现：

- `formal_vllm_env/lib64/python3.12/site-packages/vllm/benchmarks/lib/endpoint_request_func.py`
- `formal_vllm_env/lib64/python3.12/site-packages/vllm/benchmarks/serve.py`

结论是：

- vLLM 的 per-request 计时起点在 `session.post(...)` 之前。
- SwiftLLM benchmark 原来的 `started_at = time.perf_counter()` 也在 `session.post(...)` 之前。

所以从语义上看，两边的 TTFT / latency 起点是基本对齐的。也就是说，**“计时起点放错了”不是当前最主要的嫌疑。**

### 2. 更可疑的问题是流式解析方式

原来的 `swiftllm_benchmark.py` 使用的是：

- `async for raw_line in response.content`

然后把每个 `raw_line` 直接当成一条完整 token 行来解析。

这在 `aiohttp` 里是不安全的，因为：

- `response.content` 迭代到的是底层 transport chunk
- transport chunk 不保证和逻辑上的 `\n` 分隔 token 行一一对应

这会导致几个严重问题：

- 一个 chunk 里如果包含多个 token 行，可能被误合并
- 一个 token 行如果被拆到两个 chunk，可能被延后计时或直接解析失败
- TTFT / ITL / output_tokens / failure classification 都可能被污染

这和 vLLM 官方 benchmark 的做法不同。vLLM 会先把 chunk 重组为完整消息，再做计时。

### 3. 两边可能还存在停止语义/实际输出长度不一致

即使请求数相同，两边实际生成的 output token 总量可能不一致。

这会直接影响：

- batch 占用
- 排队压力
- ITL / TPOT
- 高负载下的 TTFT

所以即便计时点没问题，如果两边实际工作量不同，也不能直接把结果差距都归因到服务端实现差异。

### 4. 仍然保留“SwiftLLM 服务端本身更慢”的可能

在 benchmark 客户端解析修正之后，如果高压下 TTFT 仍然显著高于 vLLM，那就更像是 SwiftLLM 服务端真实路径上的代价：

- tokenization / enqueue
- scheduling / polling
- stream flush
- batch 资源竞争

但在修复客户端解析之前，不能先下这个结论。

---

## 这次已经完成的修正

### A. 修复流式解析

已修改：

- `dummys/benchmark/online/swiftllm_benchmark.py`

修正思路：

- 不再把 `aiohttp` transport chunk 直接当成 token 行
- 改成先把 chunk 累积到 buffer
- 只有当读到完整换行后，才提取一条完整 token line
- 对完整 line 再做 `int(...)` 校验和 TTFT / ITL 计时
- 对流结束时残留的尾部内容做收尾处理

这样可以保证测量点更接近真实的“完整 token 行到达客户端”的时间。

### B. 增加诊断字段

已在 benchmark 结果中增加用于归因的字段，包括：

- `expected_output_len`
- `response_headers_time`
- `first_token_time`
- `last_token_time`
- `invalid_line_count`
- `hit_max_output_len`
- `mean_expected_output_tokens`
- `mean_actual_output_tokens`
- `median_actual_output_tokens`
- `hit_max_output_len_count`
- `hit_max_output_len_ratio`
- `total_invalid_stream_lines`
- `mean_response_headers_latency_ms`
- `mean_headers_to_first_token_latency_ms`

这些字段不改变现有 CSV 兼容性，但能帮助判断延迟主要花在：

- 请求发出到响应头
- 响应头到首 token
- 生成过程本身
- 或者流解析异常

---

## 下一步建议

### P0. 重新跑一轮小规模 benchmark

先不要直接跑完整 500 prompts 全 sweep，建议先做小规模验证，例如：

- 较小 `num_prompts`
- 只测 1~2 个 `request_rate`

重点看新增 JSON 字段：

- `total_invalid_stream_lines` 是否为 0
- `output_lens` 是否明显更合理
- `mean_response_headers_latency_ms`
- `mean_headers_to_first_token_latency_ms`
- `hit_max_output_len_ratio`

### P1. 再和 vLLM 做对齐比较

重新生成 SwiftLLM 结果后，重点比较：

- `mean_first_token_latency`
- `mean_per_token_latency`
- `mean_per_output_token_latency`

同时不要只看 CSV，还要看详细 JSON 里的：

- 总输出 token 数
- 平均实际输出长度
- 是否大量请求打满 `output_len`

### P2. 如果差距仍然大，再进入服务端排查

如果修复解析后：

- `invalid line` 问题消失
- 输出长度统计也更合理
- 但高压下 TTFT 仍然显著高

那下一步就该重点看：

- `swiftllm/server/api_server.py`
- `swiftllm/server/engine.py`
- `swiftllm/server/scheduler.py`

也就是进一步确认：

- 请求进入服务端后是否排队过久
- scheduler 是否在高负载下放大等待
- token 首次吐出前是否存在轮询/flush 延迟

---

## 验证情况

这次修改后我已经做了这些检查：

1. `python -m py_compile dummys/benchmark/online/swiftllm_benchmark.py`
2. `python dummys/benchmark/online/swiftllm_benchmark.py --help`
3. 一个最小 streaming smoke test：故意把 token 行拆到多个 chunk 中返回，确认现在仍能正确得到：
   - 3 个 output tokens
   - 正确的 ITL 数量
   - 0 个 invalid lines
   - 正确的 `hit_max_output_len`

---

## 当前判断

当前最稳妥的判断是：

1. **已基本排除“计时起点放错位置”是主因。**
2. **已确认原 benchmark 的流式解析方式存在较大风险，并已修复。**
3. **接下来需要重新跑数据，判断剩余差距里有多少来自实际输出长度/停止语义不一致。**
4. **只有在 benchmark 客户端确认可靠后，才能继续把剩余差距归因到 SwiftLLM 服务端实现。**

---

## 追加结论：output tokens 差异的根因与本轮修正

### 1. `total_input_tokens` 一致但 `total_output_tokens` 差异巨大的根因

在 `num_prompts=500` 时，SwiftLLM 与 vLLM 的 `total_input_tokens` 都是 `109944`，这说明两边拿到的 ShareGPT 输入样本基本一致，问题不在输入集。

真正的差异在于**输出长度语义不一致**：

- vLLM benchmark 把 ShareGPT completion token 长度当成 `max_tokens` 上限；
- vLLM 服务端允许请求在遇到 EOS 时提前停止，因此统计的是**实际生成 token 数**；
- SwiftLLM 原先把同一个长度字段当成“必须生成这么多 token”，也就是固定长度生成；
- 因此 SwiftLLM 的 `total_output_tokens` 会接近所有样本 `output_len` 之和，而不是实际生成长度之和。

这正是为什么：

- SwiftLLM 的 `mean_expected_output_tokens == mean_actual_output_tokens`；
- `hit_max_output_len_ratio == 1.0`；
- `total_output_tokens` 长期固定在 `101045` 左右；
- 而 vLLM 的 `total_output_tokens` 只在约 `5.8w ~ 6.2w` 间波动。

所以，**当前两边 benchmark 比较的不是相同工作量**。在这种前提下比较 TTFT / TPOT / throughput，没有解释价值。

### 2. 线程池不是这次 output token 失真的主因

`swiftllm/server/engine.py` 中 `_run_on_model_async()` 目前使用：

- `event_loop.run_in_executor(None, ...)`

这确实可能影响：

- TTFT
- ITL
- 吞吐
- 高并发下的调度抖动

但它**不能解释**下面这些现象：

- 所有请求几乎都打满输出上限；
- `total_output_tokens` 基本固定；
- `mean_actual_output_tokens` 与 `mean_expected_output_tokens` 基本相等。

这些现象只能由“停止条件本身写成固定长度停止”解释，而不是线程池类型解释。

因此本轮结论是：

- **线程池问题是第二阶段性能优化议题；**
- **不是本轮 SwiftLLM output token 明显偏大的主因。**

### 3. 本轮已完成的代码修正

本轮已经把 SwiftLLM 的生成语义从“固定长度生成”改成“`max_tokens` 上限 + EOS 提前停止”语义，核心改动如下。

#### 服务端

- `swiftllm/server/structs.py`
  - `RawRequest` 从 `output_len` 改为 `max_tokens`
  - 新增 `ignore_eos`
  - `Request` 新增 `finish_reason`
  - 新增 `maybe_mark_finished()`，按以下顺序停止：
    - 达到 `max_tokens`
    - 命中 EOS 且 `ignore_eos=False`

- `swiftllm/server/tokenization_engine.py`
  - 新增 `get_eos_token_id()`，供 engine 初始化时读取 tokenizer 的 EOS id

- `swiftllm/server/engine.py`
  - 初始化时缓存 `eos_token_id`
  - 在每步生成后调用 `req.maybe_mark_finished(output_token, self.eos_token_id)`
  - 保持现有 scheduler / resource free 路径不变，只修正停止语义

- `swiftllm/server/api_server.py`
  - `/generate` 同时兼容读取：
    - `max_tokens`
    - 旧字段 `output_len`（向后兼容）
  - 非流式返回中新增 `finish_reason`

#### benchmark 客户端

- `dummys/benchmark/online/swiftllm_benchmark.py`
  - 请求 payload 从 `output_len` 改成 `max_tokens`
  - 新增 CLI 参数：`--ignore-eos`
  - 单请求指标新增 `finish_reason`
  - 聚合结果新增：
    - `finish_reason_counts`
    - `eos_stop_count` / `eos_stop_ratio`
    - `max_tokens_stop_count` / `max_tokens_stop_ratio`
  - 修复了 `issue_request()` / `run_readiness_check()` / `benchmark_request_rate()` 之间的参数传递不一致问题
  - 恢复了 `total_invalid_stream_lines` 所需的聚合变量定义

### 4. 当前 stop reason 统计口径

本轮 streaming 接口没有额外引入复杂的结束事件协议；当前 benchmark 采用的是最小可用口径：

- 若请求输出 token 数打满 `expected_output_len`，记为 `max_tokens`；
- 若未打满且请求成功完成，当前口径记为 `eos`。

在目前 SwiftLLM 仅支持 `max_tokens` / `eos` 两种停止条件的前提下，这个口径已经足够用于本轮 benchmark 对齐。

如果未来再引入：

- `stop_token_ids`
- stop string
- 其他 finish reason

则 streaming 协议需要再扩展，以避免把所有“未打满上限”的完成都归为 `eos`。

### 5. 已完成的验证

本轮已完成的验证是：

1. `python -m py_compile` 检查以下文件，均已通过：
   - `swiftllm/server/api_server.py`
   - `swiftllm/server/structs.py`
   - `swiftllm/server/engine.py`
   - `swiftllm/server/tokenization_engine.py`
   - `dummys/benchmark/online/swiftllm_benchmark.py`
   - `examples/online.py`
2. `python dummys/benchmark/online/swiftllm_benchmark.py --help` 已通过，CLI 参数正常，包含新的 `--ignore-eos`
3. 编辑器诊断结果显示本轮改动文件没有语法错误；仅 `engine.py` 中已有一个未使用 `torch` 的提示，不影响运行

### 6. 下一步验证重点

下一步不该继续猜测线程池，而应先重新跑 benchmark，验证这次语义修正是否真正把工作量口径对齐。建议顺序如下：

1. 小规模验证（少量 prompts、少量 request rates）
   - 看 `mean_actual_output_tokens` 是否明显下降；
   - 看 `hit_max_output_len_ratio` 是否明显低于 `1.0`；
   - 看 `finish_reason_counts` 是否出现大量 `eos`。

2. 再跑 `num_prompts=500` 对照 vLLM
   - 比较 `total_input_tokens` 是否继续一致；
   - 比较 `total_output_tokens` 是否不再锁死在 `101045`；
   - 比较 SwiftLLM 与 vLLM 的 `total_output_tokens` 是否进入同一量级。

3. 只有在 output token 工作量口径基本对齐后，再讨论：
   - `_run_on_model_async()` 的线程池是否该优化；
   - scheduler / polling / flush 是否仍然拖慢高压 TTFT。
