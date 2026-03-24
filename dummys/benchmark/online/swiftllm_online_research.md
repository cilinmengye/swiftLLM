# SwiftLLM online benchmark research

## Goal

Implement a SwiftLLM online benchmark that matches the methodology of the existing vLLM ShareGPT benchmark in [vllm_benchmark.py](./vllm_benchmark.py) while using SwiftLLM's real serving path in [api_server.py](../../swiftllm/server/api_server.py).

The target comparison conditions are:

- ShareGPT dataset
- `REQUEST_RATE = [1, 2, 3, 4, 5, 6]`
- `num_prompts = 500`
- `burstiness = 1.0`
- `max_concurrency = 256`
- `seed = 0`
- `disable_shuffle = True`

The implemented benchmark runner is [swiftllm_benchmark.py](./swiftllm_benchmark.py).

## Why the vLLM client cannot be reused directly

Your current vLLM benchmark wrapper in [vllm_benchmark.py](./vllm_benchmark.py) shells out to:

```bash
vllm bench serve
```

That client expects an OpenAI-compatible serving backend. It relies on vLLM's benchmark stack under:

- `formal_vllm_env/lib64/python3.12/site-packages/vllm/benchmarks/serve.py`
- `formal_vllm_env/lib64/python3.12/site-packages/vllm/benchmarks/datasets.py`
- `formal_vllm_env/lib64/python3.12/site-packages/vllm/benchmarks/lib/endpoint_request_func.py`

SwiftLLM does not expose `/v1/completions` or `/v1/chat/completions`. Its online server exposes only:

- `POST /generate` in [api_server.py](../../swiftllm/server/api_server.py)

with request body:

```json
{
  "prompt": "...",
  "output_len": 128,
  "stream": true,
  "decode": false
}
```

So exact client reuse is impossible. The correct approach is to reuse **vLLM's benchmark methodology** while implementing a SwiftLLM-specific HTTP client.

## SwiftLLM serving path used for the benchmark

The benchmark measures the real online serving stack, not an in-process toy path.

Relevant files:

- [api_server.py](../../swiftllm/server/api_server.py): FastAPI server entrypoint and `/generate` API
- [engine.py](../../swiftllm/server/engine.py): request enqueueing, tokenization, scheduling loop, and token streaming
- [structs.py](../../swiftllm/server/structs.py): `RawRequest`, `Request`, and `StepOutput`
- [scheduler.py](../../swiftllm/server/scheduler.py): FCFS scheduling with batch and KV-cache constraints
- [engine_config.py](../../swiftllm/engine_config.py): server CLI and engine capacity knobs

The benchmark uses streaming mode with `decode=false`.

This choice matters because `decode=true` in [api_server.py](../../swiftllm/server/api_server.py) performs extra token decoding work during streaming. That would pollute TTFT and ITL with tokenization overhead. Using `decode=false` lets the benchmark measure token arrival timing from the serving path itself.

## How ShareGPT sampling is matched to vLLM

The dataset logic mirrors vLLM's `ShareGPTDataset` in:

- `formal_vllm_env/lib64/python3.12/site-packages/vllm/benchmarks/datasets.py`

The implemented runner in [swiftllm_benchmark.py](./swiftllm_benchmark.py) does the following:

1. loads the JSON ShareGPT file from `--dataset-path`
2. keeps only records with at least two conversation turns
3. uses:
   - `prompt = conversations[0]["value"]`
   - `completion = conversations[1]["value"]`
4. tokenizes both prompt and completion with `AutoTokenizer.from_pretrained(model_path)`
5. applies the same vLLM filtering constraints:
   - `prompt_len >= 4`
   - `output_len >= 4`
   - `prompt_len <= 1024`
   - `prompt_len + output_len <= 2048`
6. preserves original order when `disable_shuffle=True`
7. takes the first `num_prompts` valid requests
8. oversamples deterministically if the filtered set is smaller than `num_prompts`

This makes the SwiftLLM benchmark request set align with the vLLM benchmark as closely as possible.

## How Poisson arrivals and concurrency are matched to vLLM

The arrival process mirrors vLLM's online benchmark logic in:

- `formal_vllm_env/lib64/python3.12/site-packages/vllm/benchmarks/serve.py`

In vLLM:

- `request_rate` controls launch rate
- `burstiness=1.0` means Poisson arrivals
- inter-arrival delays are sampled from a gamma distribution
- cumulative delay is normalized to `num_prompts / request_rate`
- concurrency is capped by a semaphore and client connection limits

The implemented runner reproduces the same structure:

- one async benchmark task per request
- one client-side semaphore capped by `--max-concurrency`
- one gamma-based delay schedule per request rate
- one normalization pass to align total launch span with `num_prompts / request_rate`

Implementation reference:

- [swiftllm_benchmark.py](./swiftllm_benchmark.py)

The implementation uses NumPy's gamma sampler so the traffic synthesis follows the same family of distribution as vLLM.

## How metrics are measured

The metric definitions are aligned with vLLM's request-function and result aggregation logic from:

- `formal_vllm_env/lib64/python3.12/site-packages/vllm/benchmarks/lib/endpoint_request_func.py`
- `formal_vllm_env/lib64/python3.12/site-packages/vllm/benchmarks/serve.py`

Per request, the benchmark records:

- `start_time`
- `prompt_len`
- `output_tokens`
- `ttft`: elapsed time from request send to first token line
- `itls`: elapsed gaps between subsequent token lines
- `latency`: elapsed time from request send to last token line
- `error`

Aggregate metrics are computed as:

- `mean_ttft_ms`: mean of per-request TTFT
- `mean_itl_ms`: mean over all flattened ITL samples
- `mean_tpot_ms`: mean of per-request TPOT where
  - `tpot = (latency - ttft) / (output_tokens - 1)`
- `mean_e2el_ms`: mean request end-to-end latency

The detailed JSON also stores:

- `input_lens`
- `output_lens`
- `ttfts`
- `itls`
- `start_times`
- `errors`
- `request_ids`
- `per_input_token_latencies`

## CSV schema and mapping

The summary CSV is intentionally kept compatible with your existing vLLM output in [vllm_result/result.csv](./vllm_result/result.csv).

Output columns:

- `request_rate`
- `mean_per_token_latency`
- `mean_first_token_latency`
- `mean_per_output_token_latency`

Mapping:

- `mean_per_token_latency = mean_itl_ms`
- `mean_first_token_latency = mean_ttft_ms`
- `mean_per_output_token_latency = mean_tpot_ms`

So SwiftLLM results can be compared directly against the vLLM CSV row-by-row.

## Important metric mismatch in README vs current vLLM CSV

The README figure names the three curves as:

- Per-Token Latency
- Per Input Token Latency
- Per Output Token Latency

But your current vLLM benchmark wrapper writes:

- ITL as per-token latency
- TTFT as first-token latency
- TPOT as per-output-token latency

That means the middle curve in your current CSV is **not** per-input-token latency.

To avoid silently changing your existing baseline, the implemented SwiftLLM benchmark keeps the same CSV contract as your vLLM wrapper.

At the same time, it also records an extra derived metric in JSON:

- `mean_per_input_token_latency_ms`

computed from per-request:

```text
ttft / prompt_len
```

This lets you later reconstruct the README-style middle metric if needed, without breaking direct CSV comparison with vLLM.

## Generated files

The new implementation writes results under:

- `swiftLLM/dummys/benchmark/online/swiftllm_result/result.csv`
- `swiftLLM/dummys/benchmark/online/swiftllm_result/rate_1.json`
- `swiftLLM/dummys/benchmark/online/swiftllm_result/rate_2.json`
- `swiftLLM/dummys/benchmark/online/swiftllm_result/rate_3.json`
- `swiftLLM/dummys/benchmark/online/swiftllm_result/rate_4.json`
- `swiftLLM/dummys/benchmark/online/swiftllm_result/rate_5.json`
- `swiftLLM/dummys/benchmark/online/swiftllm_result/rate_6.json`

## How to launch the SwiftLLM server

Run the SwiftLLM API server first.

From the `swiftLLM/` directory:

```bash
python swiftllm/server/api_server.py \
  --host 127.0.0.1 \
  --port 8000 \
  --model-path /mnt/hdd/data/yxlin/huggingface/Meta-Llama-3.1-8B
```

The server also accepts all engine capacity knobs from [engine_config.py](../../swiftllm/engine_config.py), including:

- `--block-size`
- `--gpu-mem-utilization`
- `--num-cpu-blocks`
- `--max-seqs-in-block-table`
- `--max-blocks-per-seq`
- `--max-batch-size`
- `--max-tokens-in-batch`

If you want the fairest 4090 comparison, keep these settings fixed across repeated runs.

## How to run the SwiftLLM benchmark

From the `swiftLLM/` directory:

```bash
python dummys/benchmark/online/swiftllm_benchmark.py \
  --host 127.0.0.1 \
  --port 8000 \
  --endpoint /generate \
  --model-path /mnt/hdd/data/yxlin/huggingface/Meta-Llama-3.1-8B \
  --dataset-path /mnt/hdd/data/yxlin/huggingface_data/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 500 \
  --request-rates 1,2,3,4,5,6 \
  --burstiness 1.0 \
  --max-concurrency 256 \
  --seed 0 \
  --disable-shuffle \
  --save-result 
```

## How to use swiftllm api server

start: `python api_server.py --model-path /mnt/hdd/data/yxlin/huggingface/Meta-Llama-3.1-8B --host 0.0.0.0 --port 8000`

test: 
```
curl -N -X POST http://127.0.0.1:8000/generate   -H "Content-Type: application/json"   -d '{
    "prompt": "Summarize the main ideas of Jeff Walker'\''s Product Launch Formula into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients...",
    "max_tokens": 1000,
    "stream": true,
    "decode": true
  }'
```

## How to compare with vLLM

Your existing vLLM baseline is in:

- [vllm_result/result.csv](./vllm_result/result.csv)

After running SwiftLLM, compare it against:

- `swiftLLM/dummys/benchmark/online/swiftllm_result/result.csv`

The rows will align by `request_rate`, and the three latency columns use the same meaning as the current vLLM wrapper.

## Verification performed during implementation

The following checks were run while implementing the benchmark:

1. Python syntax check for [swiftllm_benchmark.py](./swiftllm_benchmark.py)
2. CLI help check for [swiftllm_benchmark.py](./swiftllm_benchmark.py)
3. Mock streaming-server smoke test that validated:
   - ShareGPT loading and filtering
   - readiness check behavior
   - streaming token timing collection
   - TTFT / ITL / TPOT aggregation

## Key caveats

1. SwiftLLM is not OpenAI-compatible, so exact client reuse is impossible.
2. SwiftLLM currently stops generation when `output_len` is reached, as defined by [structs.py](../../swiftllm/server/structs.py), not necessarily at EOS.
3. SwiftLLM only supports greedy decoding, so output text behavior is not guaranteed to match other backends even when request lengths match.
4. `decode=true` should not be used for latency benchmarking because it adds extra decoding overhead inside the server.
5. The current CSV remains aligned with your existing vLLM wrapper, which uses TTFT rather than true per-input-token latency for the middle column.

## Files changed

- [swiftllm_benchmark.py](./swiftllm_benchmark.py)
- [swiftllm_online_research.md](./swiftllm_online_research.md)
- [requirements.txt](../../requirements.txt)

## Summary

The implemented solution benchmarks SwiftLLM through its real `/generate` server path, reproduces vLLM's ShareGPT sampling and Poisson arrival methodology, and writes results in a CSV format that is directly comparable with your existing vLLM online benchmark outputs.

# 为什么经过上述这么多工作，依旧还是无法和vllm输出token相同？

根据我手动测试 -- 以相同prompt输入到vllm 和 swiftllm 上，在使用相同模型下，发现确实vllm更说人话，同时会有更好的eos结束；但是swiftllm仿佛在打乱话，我怀疑是采样相关实现的问题。

最后我打算测试swiftllm online benchmark手动设置 max_token长度为120，大致和vllm输出token对齐

以下为AI 推测：

这轮工作的目标已经从“先做广义协议改造”收敛为一个更具体的问题：先解释 **vLLM 为什么在 ShareGPT 500 条请求上平均只有约 121.5 个 output tokens**，以及 **为什么 SwiftLLM 即使 `ignore_eos=False` 仍然 500/500 都因为 `max_tokens` 结束**。

当前已确认的事实：

- `dummys/benchmark/online/vllm_result/rate_1.json` 中：
  - `num_prompts = 500`
  - `total_input_tokens = 109944`
  - `total_output_tokens = 60750`
  - 平均每条实际输出约 `121.5` tokens。
- `dummys/benchmark/online/test_swiftllm_result/rate_1.json` 中：
  - `total_input_tokens = 109944`
  - `total_output_tokens = 101045`
  - `mean_expected_output_tokens = mean_actual_output_tokens = 202.09`
  - `hit_max_output_len_ratio = 1.0`
  - `finish_reason_counts = {"max_tokens": 500}`。
- vLLM benchmark 的 ShareGPT 逻辑并不是“强制生成满 completion 长度”，而是：
  - 只取每条样本的前两轮对话；
  - 对 prompt/output 长度做过滤；
  - 默认会 shuffle；
  - 把 completion token 长度作为请求里的 `max_tokens` 上限；
  - 默认允许 EOS/stop 提前结束。
- SwiftLLM 当前代码虽然已经支持 `max_tokens + EOS` 的停止语义，但实际结果仍然全部打满上限，说明“协议字段改名”并没有解释当前运行结果。
- `swiftllm/server/engine.py` 里的 `run_in_executor(None, ...)` 线程池路径可能影响 TTFT/ITL/吞吐，但不能解释“500/500 全部打满上限”，因此不是本轮主因。

因此，这轮最重要的不是继续泛化重构，而是先做**证明型排查**：把“vLLM 为什么更短”与“SwiftLLM 为什么仍然不提前停”分开验证，再决定是否需要继续改代码。

# Recommended Approach

## 1. 先证明 vLLM 的较短输出来自数据子集与 early-stop 语义，而不是隐藏截断

继续沿用 vLLM benchmark 现有代码路径做解释，不假设它有额外“强制短截断”：

- 在 `vllm/benchmarks/datasets.py` 里确认 ShareGPT 只取前两轮，并带有长度过滤与 shuffle；
- 在 `vllm/benchmarks/lib/endpoint_request_func.py` 里确认请求只把 ShareGPT completion 长度下发为 `max_tokens` / `max_completion_tokens`；
- 在 `vllm/benchmarks/serve.py` 与 `vllm/sampling_params.py` 里确认 ShareGPT 场景默认并不会设置 `ignore_eos=True`，且会接纳 generation config 里的多个 EOS/stop ids。

推荐把结论收敛为：

- vLLM benchmark 比较的是“同一筛选后子集上的实际生成长度”；
- `total_output_tokens ≈ 60750` 更可能来自 **筛选后的 500 条样本 + 模型提前停止**；
- 不是 benchmark 客户端又额外做了一次隐藏截断。

## 2. 先验证 SwiftLLM 当前运行结果里的 `finish_reason` 是否真实来自服务端

当前 `swiftllm_benchmark.py` 的 streaming 结果里，`finish_reason` 仍然主要来自客户端 fallback 推断，而不是服务端 authoritative streaming 元数据。因此下一步验证不应继续依赖现有 streaming JSON 直接下结论。

推荐先做两类小规模验证：

- **non-streaming 验证**：直接调用 `/generate` 的非流式路径，因为该路径会返回 `finish_reason`；
- **进程一致性验证**：确认当前实际 benchmark 命中的 SwiftLLM 服务进程，确实是已经包含 `max_tokens + EOS` 改动的那份代码，而不是旧进程/旧环境。

如果 non-streaming 仍然全部返回 `max_tokens`，那说明问题在服务端真实停止逻辑，而不是 benchmark streaming 聚合口径。

## 3. 把 SwiftLLM 的真实根因排查优先聚焦在“EOS/stop 识别不足”

如果确认跑的是最新服务端，下一优先级应是检查 SwiftLLM 的 stop 判断是否过窄：

- 当前 `tokenization_engine.py` 只暴露单个 `tokenizer.eos_token_id`；
- 当前 `Request.maybe_mark_finished()` 也只检查单个 `eos_token_id`；
- 而 vLLM 会把 generation config 里的多个 EOS/stop ids 一并纳入 stopping。

因此，最值得优先验证的假设是：

- **SwiftLLM 当前只识别了一个 EOS id，但模型真实停止条件依赖多个 terminator/stop ids**；
- 所以即使 `ignore_eos=False`，也几乎没有请求命中当前被监控的那个 token，最终全部走到 `max_tokens`。

## 4. greedy 解码是次优先级假设，不应先于 stop-id 诊断

`swiftllm/worker/layers/post_layer.py` 当前是：

- `torch.argmax(logits, dim=1)`

这意味着 SwiftLLM 现在是纯 greedy。它确实可能让模型比 vLLM 的默认采样路径更少自然命中 EOS，但在当前证据下，它应排在 stop-id 诊断之后：

- 因为“单一 EOS 检查”已经足以解释大量请求不提前停；
- 只有在 stop-id 口径补齐后仍然长期全部 `max_tokens`，才值得进一步做 greedy vs sampled 的对照设计。

## 5. 线程池问题后置，只在 output-length 口径对齐后再讨论

`engine.py:_run_on_model_async()` 的线程池逻辑应明确后置。当前计划不把它作为本轮核心修改方向：

- 它更可能影响 TTFT / ITL / throughput；
- 不足以解释 `total_output_tokens = 101045` 且 500/500 全部打满；
- 应等到“真实 stop 行为”查清楚之后，再决定是否做 executor / scheduler / polling 的性能优化。

# Critical Files To Inspect / Potentially Modify

## SwiftLLM

- `/home/yxlin/github/swift/swiftLLM/swiftllm/server/api_server.py`
  - 用于 non-streaming 验证真实 `finish_reason`，以及确认 streaming 路径是否缺少 authoritative finish metadata。

- `/home/yxlin/github/swift/swiftLLM/swiftllm/server/structs.py`
  - 当前 `Request.maybe_mark_finished()` 的 stop 判断核心入口；若要补齐多 EOS/stop ids，这里是关键修改点。

- `/home/yxlin/github/swift/swiftLLM/swiftllm/server/tokenization_engine.py`
  - 当前只返回单个 `eos_token_id`；若模型需要多个 stop ids，这里需要扩展获取接口。

- `/home/yxlin/github/swift/swiftLLM/swiftllm/server/engine.py`
  - 承接 stop 决策的调用链；线程池问题也在这里，但本轮只做记录，不优先改它。

- `/home/yxlin/github/swift/swiftLLM/swiftllm/worker/layers/post_layer.py`
  - 用于确认当前是 greedy argmax；仅在 stop-id 诊断后仍有必要时，才考虑把它纳入第二阶段方案。

- `/home/yxlin/github/swift/swiftLLM/dummys/benchmark/online/swiftllm_benchmark.py`
  - 现有 streaming 聚合使用 fallback `finish_reason`；若需要更可靠诊断，可能要补充 non-streaming 小规模验证路径或更明确的 streaming 结束元数据。

## vLLM

- `/home/yxlin/github/swift/formal_vllm_env/lib/python3.12/site-packages/vllm/benchmarks/datasets.py`
- `/home/yxlin/github/swift/formal_vllm_env/lib/python3.12/site-packages/vllm/benchmarks/serve.py`
- `/home/yxlin/github/swift/formal_vllm_env/lib/python3.12/site-packages/vllm/benchmarks/lib/endpoint_request_func.py`
- `/home/yxlin/github/swift/formal_vllm_env/lib/python3.12/site-packages/vllm/sampling_params.py`
- `/home/yxlin/github/swift/formal_vllm_env/lib64/python3.12/site-packages/vllm/...`
  - 需要确认 `lib` 与 `lib64` 两套安装中的关键代码路径是否一致，避免把错误版本当成运行版本。

# Existing Functions / Utilities To Reuse

- `generate()` in `/home/yxlin/github/swift/swiftLLM/swiftllm/server/api_server.py`
  - 直接复用现有 non-streaming 返回 `finish_reason` 的路径做验证，不必先设计新协议。

- `Request.maybe_mark_finished()` in `/home/yxlin/github/swift/swiftLLM/swiftllm/server/structs.py`
  - 继续作为 SwiftLLM stop 逻辑的唯一核心入口；如果要补多个 stop ids，应在这里扩展，而不是新增平行分支。

- `TokenizationEngine.get_eos_token_id()` in `/home/yxlin/github/swift/swiftLLM/swiftllm/server/tokenization_engine.py`
  - 作为现有 EOS 信息入口；若验证后确需支持多 stop ids，优先在这一层扩展而不是散落到 benchmark 里硬编码。

- ShareGPT loading/filtering path in `/home/yxlin/github/swift/formal_vllm_env/lib/python3.12/site-packages/vllm/benchmarks/datasets.py`
  - 复用它来解释 vLLM 为什么不是在“原始直觉观察的整个数据集”上运行，而是在过滤后的前两轮子集上运行。

# Implementation Sequence

## 阶段 A：先完成证明型验证，不急着继续改代码

1. 用现有 vLLM 源码路径整理出“为什么平均 output 只有约 121.5”的证据链：
   - 前两轮采样；
   - 长度过滤；
   - shuffle；
   - `max_tokens` 只是上限；
   - EOS/stop 可提前结束。
2. 用 SwiftLLM 的 non-streaming `/generate` 路径验证真实 `finish_reason`；
3. 确认 benchmark 实际命中的 SwiftLLM 进程与当前 inspected code 一致。

只有这三步完成后，才进入是否修改代码的决策。

## 阶段 B：若 non-streaming 仍然全部 `max_tokens`，优先补齐 stop-id 语义

推荐的最小修改方向是：

1. 查明模型/tokenizer/generation config 当前实际 stop ids；
2. 扩展 SwiftLLM 获取与保存 stop ids 的方式；
3. 让 `Request.maybe_mark_finished()` 按“多个 stop ids”而不是“单个 eos_token_id”判断停止；
4. 再重新做小规模验证，看 `finish_reason` 是否出现明显的 `eos` / stop 提前结束。

## 阶段 C：若 stop-id 补齐后仍然极少提前结束，再评估 greedy 解码差异

此时再进入第二层判断：

- SwiftLLM 的纯 greedy 是否与 vLLM 当前默认采样语义差异过大；
- 是否需要最小采样参数链路，或者先让两边都显式 greedy 做 apples-to-apples 对照。

## 阶段 D：最后才讨论线程池与调度性能

只有在以下条件成立后，才值得讨论 `_run_on_model_async()`：

- SwiftLLM 与 vLLM 的 output-length 口径已经基本对齐；
- `finish_reason` 分布已经合理，不再是 500/500 全 `max_tokens`；
- 这时再比较 TTFT / TPOT / throughput，线程池问题才有分析价值。

# Verification

## 1. vLLM 侧解释验证

需要能用源码证据解释以下现象：

- 为什么用户主观上看到很多对话很长，但 benchmark 平均 output 仍只有约 121.5；
- 为什么这更像“筛选后子集 + early stop”，而不是 benchmark 隐藏截断。

## 2. SwiftLLM 真实 stop 行为验证

优先看 non-streaming 返回的 `finish_reason`：

- 如果 non-streaming 也几乎全是 `max_tokens`，说明问题在服务端真实 stop 逻辑；
- 如果 non-streaming 与 streaming 不一致，先修 benchmark/streaming 统计口径，而不是先改模型逻辑。

## 3. stop-id 修正后的验证

如果进入代码修改阶段，最关键的成功标准是：

- 小规模样本下不再 500/500 全部 `max_tokens`；
- `mean_actual_output_tokens` 明显低于 `mean_expected_output_tokens`；
- `finish_reason_counts` 中出现可观比例的 `eos` / stop；
- `total_output_tokens` 不再固定锁死在 `101045` 附近。

## 4. 第二阶段性能验证

只有在 stop 行为合理后，再验证：

- TTFT / ITL / TPOT 是否仍显著落后 vLLM；
- 这时再单独评估线程池、scheduler、polling、flush 的贡献。

# Notes / Non-Goals

- 这轮不建议先做大规模重构，也不建议先改线程池。
- 这轮首要目标不是“让 benchmark 先跑完”，而是先解释当前结果为什么失真、以及最小改动应该打在哪里。
- 若最终证明根因只是 stale server / 运行版本不一致，则应优先修复运行环境，而不是继续改代码。
