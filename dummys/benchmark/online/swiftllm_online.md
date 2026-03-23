# Context

You want a SwiftLLM online benchmark that is directly comparable to your existing vLLM ShareGPT run under the same traffic settings: ShareGPT dataset, `REQUEST_RATE = [1, 2, 3, 4, 5, 6]`, `num_prompts=500`, `burstiness=1.0`, `max_concurrency=256`, `seed=0`, and `disable_shuffle=True`. The current vLLM wrapper in `swiftLLM/dummys/benchmark/online/vllm_benchmark.py` writes a compact CSV with three latency metrics derived from vLLM’s benchmark JSON:

- `mean_itl_ms` -> `mean_per_token_latency`
- `mean_ttft_ms` -> `mean_first_token_latency`
- `mean_tpot_ms` -> `mean_per_output_token_latency`

The main constraint is that SwiftLLM is **not** OpenAI-compatible today. Its online API is `POST /generate` in `swiftLLM/swiftllm/server/api_server.py`, with request fields `prompt`, `output_len`, `stream`, and `decode`. So the vLLM client cannot be reused as-is; the benchmark logic must be adapted to SwiftLLM’s HTTP contract while preserving the same dataset sampling, arrival process, concurrency control, and metric definitions.

## Recommended approach

Implement a dedicated SwiftLLM online benchmark client under `swiftLLM/dummys/benchmark/online/` that keeps the **benchmark methodology** aligned with vLLM, while talking to SwiftLLM’s existing `/generate` endpoint.

### 1. Add a new benchmark driver for SwiftLLM

Create `swiftLLM/dummys/benchmark/online/swiftllm_benchmark.py`.

This script should:

- sweep `REQUEST_RATE = [1, 2, 3, 4, 5, 6]`
- reuse the same ShareGPT dataset file path and same tokenizer/model path inputs as the vLLM run
- select the same 500 prompts under the same seed and `disable_shuffle=True` behavior
- launch requests against `POST /generate`
- measure TTFT / ITL / TPOT-equivalent metrics from SwiftLLM’s streaming response
- write one detailed JSON per request rate plus one summary CSV aligned with `swiftLLM/dummys/benchmark/online/vllm_result/result.csv`

### 2. Benchmark through HTTP, not through `Engine` directly

Use the running SwiftLLM API server (`swiftLLM/swiftllm/server/api_server.py`) as the benchmark target instead of calling `swiftllm.Engine` in-process.

Why this is the right comparison:

- vLLM’s `bench serve` measures the online serving path, not just raw model stepping
- benchmarking through HTTP includes FastAPI/uvicorn/server-path overhead, which is part of the online-serving scenario in `README.md`
- reusing the existing server avoids invasive changes to SwiftLLM itself

Do **not** benchmark via `examples/online.py` except as a debugging aid.

### 3. Use SwiftLLM streaming mode with `decode=False`

For each request, send:

```json
{
  "prompt": "...",
  "output_len": <expected_output_len>,
  "stream": true,
  "decode": false
}
```

This is important.

Reason:

- `decode=True` in `swiftLLM/swiftllm/server/api_server.py` decodes tokens during streaming by making per-token tokenizer calls through the tokenization engine, which would add extra Python/Ray overhead and distort TTFT/ITL
- `decode=False` returns one token id per streamed line, which is enough to count output tokens and timestamp token arrivals accurately
- this keeps the measured latency closer to model/server behavior rather than text-decoding overhead

### 4. Reproduce vLLM’s ShareGPT sampling logic locally

Mirror the logic from:

- `formal_vllm_env/lib64/python3.12/site-packages/vllm/benchmarks/datasets.py`
- class `ShareGPTDataset`

The SwiftLLM benchmark script should:

1. load the JSON dataset from `--dataset-path`
2. keep only entries with at least two conversation turns
3. with `disable_shuffle=True`, preserve original order
4. take:
   - `prompt = conversations[0]["value"]`
   - `completion = conversations[1]["value"]`
5. tokenize both using `AutoTokenizer.from_pretrained(model_path)`
6. compute:
   - `prompt_len = len(tokenizer(prompt).input_ids)`
   - `output_len = len(tokenizer(completion).input_ids)`
7. keep only valid sequences using the same vLLM pruning rules:
   - `prompt_len >= 4`
   - `output_len >= 4`
   - `prompt_len <= 1024`
   - `prompt_len + output_len <= 2048`
8. take the first 500 valid samples

This gives the closest possible alignment to the request set used by `vllm bench serve`.

### 5. Reproduce vLLM’s arrival process and concurrency behavior

Mirror the logic from:

- `formal_vllm_env/lib64/python3.12/site-packages/vllm/benchmarks/serve.py`
- especially `get_request(...)`

Implementation details to preserve:

- request rate controls **launch rate**, not completion rate
- `burstiness=1.0` means Poisson arrivals
- arrival intervals should follow the same gamma/exponential schedule semantics as vLLM
- cumulative delays should be normalized to `num_prompts / request_rate` to reduce seed drift, same as vLLM does
- enforce `max_concurrency=256` with a client-side semaphore

Recommended implementation shape:

- one async benchmark driver
- one semaphore limiting in-flight requests
- one request scheduler that yields prompts at the configured cumulative launch times
- one per-request coroutine that performs the HTTP POST and parses streamed token-id lines

### 6. Measure metrics using the same aggregation semantics as vLLM

Use the vLLM metric definitions from:

- `formal_vllm_env/lib64/python3.12/site-packages/vllm/benchmarks/lib/endpoint_request_func.py`
- `formal_vllm_env/lib64/python3.12/site-packages/vllm/benchmarks/serve.py`

Per request, collect:

- `start_time`
- `prompt_len`
- `output_tokens`
- `ttft` = time from request send to first streamed token line
- `itl[]` = time gaps between subsequent token lines
- `latency` = time from request send to last token line
- `success/error`

Aggregate exactly like vLLM:

- `mean_ttft_ms` = mean of per-request TTFT
- `mean_itl_ms` = mean over the flattened list of all inter-token latencies
- `tpot` per request = `(latency - ttft) / (output_tokens - 1)` when `output_tokens > 1`
- `mean_tpot_ms` = mean of per-request TPOT values

Then write the same compact CSV schema used by your current vLLM wrapper:

- `request_rate`
- `mean_per_token_latency`
- `mean_first_token_latency`
- `mean_per_output_token_latency`

with mapping:

- `mean_per_token_latency = mean_itl_ms`
- `mean_first_token_latency = mean_ttft_ms`
- `mean_per_output_token_latency = mean_tpot_ms`

### 7. Also document the README metric mismatch explicitly

Your screenshot and `README.md` describe three curves as:

- Per-Token Latency
- Per Input Token Latency
- Per Output Token Latency

But your current vLLM CSV stores:

- per-token latency
- **first-token latency (TTFT)**
- per-output-token latency

These are not the same as “per input token latency”.

Recommended handling:

- keep the CSV compatible with the existing vLLM result file, because that is your current baseline
- in the research report, explicitly explain the mismatch
- if you still want the README-style middle curve, derive and report an additional metric:
  - per-request input-token latency = `ttft / prompt_len`
  - aggregate mean across requests as a separate derived statistic
- do **not** silently rename TTFT to “per input token latency”

### 8. Write the research report after the benchmark logic is in place

Create `swiftLLM/dummys/benchmark/online/swiftllm_online_research.md`.

The report should include:

- why `vllm_benchmark.py` cannot be reused directly
- the SwiftLLM server contract and why `/generate` + streaming is the correct path
- exact ShareGPT sampling rules reused from vLLM
- exact request scheduling logic reused from vLLM
- metric definitions and CSV field mapping
- caveats that affect strict apples-to-apples comparison
- exact commands to launch the SwiftLLM server and run the benchmark sweep
- expected output files and how to compare them against `vllm_result/result.csv`

## Critical files to modify

Primary changes:

- `swiftLLM/dummys/benchmark/online/swiftllm_benchmark.py` (new)
- `swiftLLM/dummys/benchmark/online/swiftllm_online_research.md` (new)

Generated outputs:

- `swiftLLM/dummys/benchmark/online/swiftllm_result/result.csv`
- `swiftLLM/dummys/benchmark/online/swiftllm_result/rate_1.json`
- `swiftLLM/dummys/benchmark/online/swiftllm_result/rate_2.json`
- ...
- `swiftLLM/dummys/benchmark/online/swiftllm_result/rate_6.json`

Existing code to reuse without modification if possible:

- `swiftLLM/swiftllm/server/api_server.py`
- `swiftLLM/swiftllm/server/engine.py`
- `swiftLLM/swiftllm/server/structs.py`
- `swiftLLM/swiftllm/engine_config.py`
- `swiftLLM/examples/online.py`
- `swiftLLM/dummys/benchmark/online/vllm_benchmark.py`
- `swiftLLM/dummys/benchmark/offline/benchmark.py`

Possible dependency touchpoint if needed:

- `swiftLLM/requirements.txt` only if the chosen HTTP client library is not already available in the environment

## Important implementation details to preserve

### Server launch path

Use the existing API server entrypoint:

- `swiftLLM/swiftllm/server/api_server.py`

This server exposes CLI args from `swiftLLM/swiftllm/engine_config.py`, so later execution should launch it with the target model path and explicit engine settings for the 4090 benchmark environment.

### Initial readiness check

Mirror vLLM’s benchmark behavior by doing one untimed initial request before the timed sweep for each run/rate. This confirms the endpoint is alive and avoids counting obvious cold-start errors inside the benchmark.

### Detailed JSON format

The per-rate JSON should preserve enough fields to recompute metrics later, similar to vLLM’s `--save-detailed` output. At minimum include:

- `request_rate`
- `duration`
- `completed`
- `failed`
- `total_input_tokens`
- `total_output_tokens`
- `request_throughput`
- `output_throughput`
- `total_token_throughput`
- `mean_ttft_ms`
- `mean_tpot_ms`
- `mean_itl_ms`
- `mean_e2el_ms`
- `input_lens`
- `output_lens`
- `ttfts`
- `itls`
- `start_times`
- `errors`

That will let you compute the compact CSV and any derived “per input token latency” later without rerunning the experiment.

## Known comparability caveats to call out in the report

1. **SwiftLLM does not expose an OpenAI-compatible API today.**
   So comparison must happen by reproducing vLLM’s client-side methodology, not by reusing the exact same client.

2. **SwiftLLM only supports greedy decoding.**
   vLLM’s OpenAI completion path may not match that behavior exactly unless its sampling options are pinned. For latency benchmarking, the main thing to match is the prompt/output length distribution, not the generated text.

3. **SwiftLLM currently generates until `output_len` is reached.**
   Its request lifecycle in `swiftLLM/swiftllm/server/structs.py` finishes when `len(output_token_ids) == output_len`, not when EOS appears. That means strict parity with a server that may stop early on EOS should be described as “same benchmark methodology and target lengths”, not “identical token-by-token generation behavior”.

4. **Do not enable response decoding while timing.**
   `decode=True` would benchmark tokenizer/decode overhead rather than just serving latency.

## Verification plan

After implementation, verify in this order:

1. **Server sanity check**
   - launch `swiftLLM/swiftllm/server/api_server.py` with the target model path
   - send one manual `/generate` request with `stream=true, decode=false`
   - confirm streamed lines arrive and each line corresponds to one token id

2. **Dataset alignment check**
   - run only the sample-selection portion
   - confirm it produces 500 valid ShareGPT requests
   - confirm prompt order is deterministic with `disable_shuffle=True`
   - confirm prompt/output length filtering matches vLLM rules

3. **Single-rate smoke test**
   - run only `request_rate=1`
   - confirm JSON and CSV are produced
   - confirm all three summary columns are populated
   - confirm detailed JSON includes per-request `ttfts`, `itls`, and token counts

4. **Metric sanity check**
   - verify `mean_per_token_latency` equals `mean_itl_ms`
   - verify `mean_first_token_latency` equals `mean_ttft_ms`
   - verify `mean_per_output_token_latency` equals `mean_tpot_ms`
   - if the report includes per-input-token latency, verify it is computed from request-level `ttft / prompt_len`

5. **Full sweep**
   - run rates `[1, 2, 3, 4, 5, 6]`
   - confirm one JSON per rate and one final `result.csv`
   - compare SwiftLLM CSV against `swiftLLM/dummys/benchmark/online/vllm_result/result.csv`

6. **Report validation**
   - ensure the report cites the exact reference files used from both SwiftLLM and vLLM
   - ensure the report distinguishes TTFT from per-input-token latency

## Minimal scope / non-goals

To stay focused on the benchmark request, do **not** expand scope to:

- making SwiftLLM fully OpenAI-compatible
- adding a new `/v1/completions` server endpoint
- adding plotting unless explicitly requested later
- refactoring the server internals unless benchmarking reveals a blocker
