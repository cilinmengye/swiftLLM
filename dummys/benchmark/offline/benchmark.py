from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import subprocess
import sys
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import swiftllm

# 注意model max_position_embeddings大小
DEFAULT_BATCH_SIZES = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
DEFAULT_INPUT_LENS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
# DEFAULT_BATCH_SIZES = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
# DEFAULT_INPUT_LENS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
DEFAULT_SEED = 20260319
BLOCK_SIZE = 16
GPU_MEM_UTILIZATION = 0.99
NUM_CPU_BLOCKS = 0


@dataclass(frozen=True)
class BenchmarkCase:
    batch_size: int
    input_len: int


@dataclass
class GeneratedCaseData:
    batch_size: int
    input_len: int
    seed: int
    input_ids: list[list[int]]


@dataclass
class BenchmarkResult:
    batch_size: int
    input_len: int
    total_tokens: int
    prefill_ms_mean: float
    prefill_ms_std: float
    decode_ms_mean: float
    decode_ms_std: float
    prefill_tokens_per_s: float
    decode_tokens_per_s: float
    warmup_runs: int
    measure_runs: int
    prefill_samples_ms: list[float]
    decode_samples_ms: list[float]


RUNNER_PATH = Path(__file__).resolve().with_name("benchmark_case.py")


def parse_int_list(raw: str | None, default: list[int]) -> list[int]:
    if raw is None:
        return list(default)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected a non-empty comma-separated integer list")
    return [int(item) for item in values]


def build_cases(batch_sizes: list[int], input_lens: list[int]) -> list[BenchmarkCase]:
    if len(batch_sizes) != len(input_lens):
        raise ValueError("batch_sizes and input_lens must have the same length")
    return [BenchmarkCase(batch_size=batch_size, input_len=input_len) for batch_size, input_len in zip(batch_sizes, input_lens)]


def read_vocab_size(model_path: Path) -> int:
    config_path = model_path / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return int(config["vocab_size"])


def build_model(model_path: str, case: BenchmarkCase) -> swiftllm.LlamaModel:
    max_batch_size = case.batch_size
    max_tokens_in_batch = case.batch_size * case.input_len
    max_input_len = case.input_len
    max_blocks_per_seq = (max_input_len + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE

    engine_config = swiftllm.EngineConfig(
        model_path=model_path,
        use_dummy=False,
        block_size=BLOCK_SIZE,
        gpu_mem_utilization=GPU_MEM_UTILIZATION,
        num_cpu_blocks=NUM_CPU_BLOCKS,
        max_seqs_in_block_table=max_batch_size,
        max_blocks_per_seq=max_blocks_per_seq,
        max_batch_size=max_batch_size,
        max_tokens_in_batch=max_tokens_in_batch,
    )
    model = swiftllm.LlamaModel(engine_config)
    model.load_weights()
    num_blocks = model.profile_num_blocks()
    model.init_kvcache_and_swap(num_blocks)
    return model


def make_input_ids(vocab_size: int, batch_size: int, input_len: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(0, vocab_size) for _ in range(input_len)] for _ in range(batch_size)]


def case_seed(case: BenchmarkCase, base_seed: int) -> int:
    return base_seed + case.batch_size * 100_000 + case.input_len


def case_data_path(data_dir: Path, case: BenchmarkCase) -> Path:
    return data_dir / f"bs{case.batch_size}_len{case.input_len}.json"


def case_result_path(results_dir: Path, case: BenchmarkCase) -> Path:
    return results_dir / f"bs{case.batch_size}_len{case.input_len}.json"


def save_case_data(data_dir: Path, data: GeneratedCaseData) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = case_data_path(data_dir, BenchmarkCase(batch_size=data.batch_size, input_len=data.input_len))
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(data), f)
    return output_path


def load_case_data(data_dir: Path, case: BenchmarkCase) -> GeneratedCaseData:
    with case_data_path(data_dir, case).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return GeneratedCaseData(
        batch_size=int(payload["batch_size"]),
        input_len=int(payload["input_len"]),
        seed=int(payload["seed"]),
        input_ids=[[int(token) for token in row] for row in payload["input_ids"]],
    )


def generate_case_data(model_path: Path, cases: list[BenchmarkCase], seed: int, data_dir: Path) -> list[Path]:
    vocab_size = read_vocab_size(model_path)
    saved_paths: list[Path] = []
    for case in cases:
        generated = GeneratedCaseData(
            batch_size=case.batch_size,
            input_len=case.input_len,
            seed=case_seed(case, seed),
            input_ids=make_input_ids(vocab_size, case.batch_size, case.input_len, case_seed(case, seed)),
        )
        saved_paths.append(save_case_data(data_dir, generated))
    return saved_paths


def summarize_samples(samples: list[float]) -> tuple[float, float]:
    mean_value = statistics.fmean(samples)
    std_value = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return mean_value, std_value


def time_prefill(model: swiftllm.LlamaModel, input_ids: list[list[int]], seq_ids: list[int]) -> float:
    torch.cuda.synchronize()
    started_at = time.perf_counter()
    try:
        model.forward(input_ids, seq_ids, [])
        torch.cuda.synchronize()
        return (time.perf_counter() - started_at) * 1000
    finally:
        model.free_seqs_resources(seq_ids)


def time_decode(
    model: swiftllm.LlamaModel,
    input_ids: list[list[int]],
    seq_ids: list[int],
    input_len: int,
) -> float:
    try:
        last_tokens = model.forward(input_ids, seq_ids, [])
        torch.cuda.synchronize()
        started_at = time.perf_counter()
        model.forward([[token] for token in last_tokens], seq_ids, [input_len + 1] * len(seq_ids))
        torch.cuda.synchronize()
        return (time.perf_counter() - started_at) * 1000
    finally:
        model.free_seqs_resources(seq_ids)


def benchmark_case(
    model: swiftllm.LlamaModel,
    case: BenchmarkCase,
    input_ids: list[list[int]],
    warmup_runs: int,
    measure_runs: int,
) -> BenchmarkResult:
    seq_ids = list(range(case.batch_size))

    print(f"[Info] Start Profile Prefill Stage")
    for _ in range(warmup_runs):
        time_prefill(model, input_ids, seq_ids)
    prefill_samples = [time_prefill(model, input_ids, seq_ids) for _ in range(measure_runs)]

    print(f"[Info] Start Profile Decode Stage")
    for _ in range(warmup_runs):
        time_decode(model, input_ids, seq_ids, case.input_len)
    decode_samples = [time_decode(model, input_ids, seq_ids, case.input_len) for _ in range(measure_runs)]

    prefill_ms_mean, prefill_ms_std = summarize_samples(prefill_samples)
    decode_ms_mean, decode_ms_std = summarize_samples(decode_samples)
    total_tokens = case.batch_size * case.input_len

    return BenchmarkResult(
        batch_size=case.batch_size,
        input_len=case.input_len,
        total_tokens=total_tokens,
        prefill_ms_mean=prefill_ms_mean,
        prefill_ms_std=prefill_ms_std,
        decode_ms_mean=decode_ms_mean,
        decode_ms_std=decode_ms_std,
        prefill_tokens_per_s=total_tokens / (prefill_ms_mean / 1000.0),
        decode_tokens_per_s=case.batch_size / (decode_ms_mean / 1000.0),
        warmup_runs=warmup_runs,
        measure_runs=measure_runs,
        prefill_samples_ms=prefill_samples,
        decode_samples_ms=decode_samples,
    )


def benchmark_single_case(
    model_path: Path,
    data_dir: Path,
    case: BenchmarkCase,
    warmup_runs: int,
    measure_runs: int,
) -> BenchmarkResult:
    generated = load_case_data(data_dir, case)
    model = build_model(str(model_path), case)
    result = benchmark_case(
        model=model,
        case=case,
        input_ids=generated.input_ids,
        warmup_runs=warmup_runs,
        measure_runs=measure_runs,
    )
    print(
        f"batch_size={result.batch_size} input_len={result.input_len} "
        f"prefill_ms={result.prefill_ms_mean:.3f} decode_ms={result.decode_ms_mean:.3f}"
    )
    return result


def benchmark_result_from_dict(payload: dict) -> BenchmarkResult:
    return BenchmarkResult(
        batch_size=int(payload["batch_size"]),
        input_len=int(payload["input_len"]),
        total_tokens=int(payload["total_tokens"]),
        prefill_ms_mean=float(payload["prefill_ms_mean"]),
        prefill_ms_std=float(payload["prefill_ms_std"]),
        decode_ms_mean=float(payload["decode_ms_mean"]),
        decode_ms_std=float(payload["decode_ms_std"]),
        prefill_tokens_per_s=float(payload["prefill_tokens_per_s"]),
        decode_tokens_per_s=float(payload["decode_tokens_per_s"]),
        warmup_runs=int(payload["warmup_runs"]),
        measure_runs=int(payload["measure_runs"]),
        prefill_samples_ms=[float(value) for value in payload["prefill_samples_ms"]],
        decode_samples_ms=[float(value) for value in payload["decode_samples_ms"]],
    )


def write_single_result_json(output_path: Path, result: BenchmarkResult) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)


def load_single_result_json(result_path: Path) -> BenchmarkResult:
    with result_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return benchmark_result_from_dict(payload)


def write_results_json(results: list[BenchmarkResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(result) for result in results], f, indent=2)


def write_results_csv(results: list[BenchmarkResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "batch_size",
        "input_len",
        "total_tokens",
        "prefill_ms_mean",
        "prefill_ms_std",
        "decode_ms_mean",
        "decode_ms_std",
        "prefill_tokens_per_s",
        "decode_tokens_per_s",
        "warmup_runs",
        "measure_runs",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "batch_size": result.batch_size,
                    "input_len": result.input_len,
                    "total_tokens": result.total_tokens,
                    "prefill_ms_mean": result.prefill_ms_mean,
                    "prefill_ms_std": result.prefill_ms_std,
                    "decode_ms_mean": result.decode_ms_mean,
                    "decode_ms_std": result.decode_ms_std,
                    "prefill_tokens_per_s": result.prefill_tokens_per_s,
                    "decode_tokens_per_s": result.decode_tokens_per_s,
                    "warmup_runs": result.warmup_runs,
                    "measure_runs": result.measure_runs,
                }
            )


def run_case_subprocess(
    model_path: Path,
    data_dir: Path,
    case: BenchmarkCase,
    warmup_runs: int,
    measure_runs: int,
    case_results_dir: Path,
) -> Path:
    output_path = case_result_path(case_results_dir, case)
    command = [
        sys.executable,
        str(RUNNER_PATH),
        "--model-path",
        str(model_path),
        "--data-dir",
        str(data_dir),
        "--batch-size",
        str(case.batch_size),
        "--input-len",
        str(case.input_len),
        "--warmup-runs",
        str(warmup_runs),
        "--measure-runs",
        str(measure_runs),
        "--output-path",
        str(output_path),
    ]
    print(f"[Info] Launch subprocess for batch_size {case.batch_size} input_len {case.input_len}")
    subprocess.run(command, check=True)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark SwiftLLM offline prefill and single-step decode performance.")
    parser.add_argument("--model-path", type=Path, required=True, help="Local Hugging Face model directory.")
    parser.add_argument("--batch-sizes", type=str, default=None, help="Comma-separated batch sizes. Must align with --input-lens.")
    parser.add_argument("--input-lens", type=str, default=None, help="Comma-separated input lengths. Must align with --batch-sizes.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base seed for deterministic synthetic token generation.")
    parser.add_argument("--warmup-runs", type=int, default=2, help="Number of warmup runs before timing.")
    parser.add_argument("--measure-runs", type=int, default=5, help="Number of timed runs per case.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data", help="Directory for generated input data.")
    parser.add_argument("--case-results-dir", type=Path, default=Path(__file__).resolve().parent / "case_results", help="Directory for per-case result files.")
    parser.add_argument("--results-json", type=Path, default=Path(__file__).resolve().parent / "results.json", help="Output path for JSON results.")
    parser.add_argument("--results-csv", type=Path, default=Path(__file__).resolve().parent / "results.csv", help="Output path for CSV results.")
    parser.add_argument("--generate-only", action="store_true", help="Generate synthetic input data and exit without running the benchmark.")
    return parser.parse_args()

def count_files(folder_path):
    # 列出所有文件和文件夹，并过滤出文件
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return len(files)

def main() -> None:
    args = parse_args()
    batch_sizes = parse_int_list(args.batch_sizes, DEFAULT_BATCH_SIZES)
    input_lens = parse_int_list(args.input_lens, DEFAULT_INPUT_LENS)
    cases = build_cases(batch_sizes, input_lens)

    if count_files(args.data_dir) != len(batch_sizes) or args.generate_only:
        # 如果有文件就不再生成了
        saved_paths = generate_case_data(args.model_path, cases, args.seed, args.data_dir)
        print(f"Generated {len(saved_paths)} input files under {args.data_dir}")
    if args.generate_only:
        return 

    result_paths: list[Path] = []
    for case in cases:
        result_paths.append(
            run_case_subprocess(
                model_path=args.model_path,
                data_dir=args.data_dir,
                case=case,
                warmup_runs=args.warmup_runs,
                measure_runs=args.measure_runs,
                case_results_dir=args.case_results_dir,
            )
        )

    results = [load_single_result_json(result_path) for result_path in result_paths]
    write_results_json(results, args.results_json)
    write_results_csv(results, args.results_csv)
    print(f"Wrote results to {args.results_json} and {args.results_csv}")


if __name__ == "__main__":
    main()
