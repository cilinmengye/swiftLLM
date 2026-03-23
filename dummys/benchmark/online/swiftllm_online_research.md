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
  --save-result \
  --save-detailed
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