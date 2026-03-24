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
