# Offline benchmark implementation plan

## Phase 1 [completed] Review requirements and design

- [x] Confirm the benchmark follows SwiftLLM README's offline single-forward path.
- [x] Confirm the benchmark cases use paired `batch_size` / `input_len` values.
- [x] Confirm the implementation reuses `examples/offline.py` and `swiftllm/worker/model.py`.

## Phase 2 [completed] Implement benchmark workflow

- [x] Implement `benchmark.py` to generate deterministic input data.
- [x] Implement `benchmark.py` to measure prefill and single-step decode.
- [x] Implement `benchmark.py` to save JSON and CSV results.
- [x] Implement `plot.py` to render `prefill.png` and `decode.png`.
- [x] Keep generated data and outputs under `swiftLLM/dummys/benchmark/offline/`.

## Phase 3 [completed] Run checks and fix issues

- [x] Run available type or syntax checks after each major edit.
- [x] Fix all issues found by those checks.

## Phase 4 [completed] Finish documentation and mark completion

- [x] Update this plan document as each phase completes.
- [x] Ensure all phases and tasks are marked completed.

## Benchmark scope

This benchmark reproduces SwiftLLM README’s **"A Single Forward Operation"** scenario through the same direct model path used by [offline.py](../../../examples/offline.py).

The benchmark uses the paired cases:

- `(512, 32)`
- `(256, 64)`
- `(128, 128)`
- `(64, 256)`
- `(32, 512)`
- `(16, 1024)`
- `(8, 2048)`
- `(4, 4096)`
- `(2, 8192)`
- `(1, 16384)`

## Files

- [benchmark.py](./benchmark.py): generate deterministic input data, run prefill/decode benchmark, write JSON/CSV
- [plot.py](./plot.py): read CSV results and render `prefill.png` / `decode.png`
- `data/*.json`: saved synthetic token batches
- `results.json`, `results.csv`: raw measurements
- `prefill.png`, `decode.png`: plots

## Benchmark method

1. Generate deterministic token IDs from `model_path/config.json`’s `vocab_size`.
2. Initialize one `swiftllm.LlamaModel` with:
   - `max_batch_size = 512`
   - `max_tokens_in_batch = 16384`
   - `max_seqs_in_block_table = 512`
   - `max_blocks_per_seq = ceil((16384 + 1) / 16)`
3. Measure **prefill** with `model.forward(input_ids, seq_ids, [])`.
4. Measure **single-step decode** by first doing an untimed prefill, then timing:
   - `model.forward([[tok] for tok in last_tokens], seq_ids, [input_len + 1] * batch_size)`
5. Use `torch.cuda.synchronize()` before and after each timed region.
6. Call `model.free_seqs_resources(seq_ids)` after every run.

## Commands

Generate only:

```bash
python dummys/benchmark/offline/benchmark.py --model-path /path/to/model --generate-only
```

Run benchmark:

```bash
python dummys/benchmark/offline/benchmark.py --model-path /path/to/model
```

Plot results:

```bash
python dummys/benchmark/offline/plot.py
```

## Verification

- confirm `data/*.json` exists for all 10 cases
- confirm `results.json` and `results.csv` contain 10 rows
- confirm `prefill.png` and `decode.png` are created
- smoke test first with a smaller subset if needed:

```bash
python dummys/benchmark/offline/benchmark.py \
  --model-path /path/to/model \
  --batch-sizes 512,1 \
  --input-lens 32,16384 \
  --warmup-runs 1 \
  --measure-runs 1
```


改动内容
把原来的 benchmark.py 改成 orchestrator

仍负责：
解析 case
生成输入数据
汇总最终 results.json / results.csv
不再在同一进程里直接跑所有 case。
现在会对每个 case 启动一个独立 Python 子进程。
新增单 case runner

新文件：benchmark_case.py
每次只跑一个 (batch_size, input_len) case。
跑完后写一个单独的 JSON 结果文件，然后进程退出，显存自然释放。
新增 per-case 结果目录

benchmark.py 增加了参数：
--case-results-dir
默认输出到：
offline/case_results/
关键实现位置
orchestrator: benchmark.py
单 case 入口: benchmark_case.py
新增/保留的核心函数
在 benchmark.py 中新增了这些能力：

case_result_path(...)
benchmark_single_case(...)
benchmark_result_from_dict(...)
write_single_result_json(...)
load_single_result_json(...)
run_case_subprocess(...)
验证结果
我已经做了最小验证：

py_compile 语法检查通过
两个脚本的 --help 参数检查通过
现在的使用方式
1. 只生成数据

python swiftLLM/dummys/benchmark/offline/benchmark.py \
  --model-path /path/to/model \
  --generate-only
2. 跑完整 benchmark

python swiftLLM/dummys/benchmark/offline/benchmark.py \
  --model-path /path/to/model
它会：

先生成数据
再逐个 case 启动 benchmark_case.py
每个 case 写入一个独立 JSON
最后汇总成总的 results.json 和 results.csv
3. 只跑单个 case

python swiftLLM/dummys/benchmark/offline/benchmark_case.py \
  --model-path /path/to/model \
  --data-dir swiftLLM/dummys/benchmark/offline/data \
  --batch-size 32 \
  --input-len 512 \
  --warmup-runs 2 \
  --measure-runs 5 \
  --output-path swiftLLM/dummys/benchmark/offline/case_results/bs32_len512.json