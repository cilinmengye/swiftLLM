import csv
import json
import math
import os
import torch
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from dataclasses import asdict, dataclass
import statistics

@dataclass
class BenchmarkResult:
    batch_size: int
    input_len: int
    total_tokens: int
    prefill_s_mean: float
    prefill_s_std: float
    decode_s_mean: float
    decode_s_std: float
    prefill_tokens_per_s: float
    decode_tokens_per_s: float
    warmup_runs: int
    measure_runs: int
    prefill_samples_s: list[float]
    decode_samples_s: list[float]

MODEL = "/mnt/hdd/data/yxlin/huggingface/Meta-Llama-3.1-8B"
OUT_DIR = Path("./vllm_result")
RAW_DIR = OUT_DIR / "raw_json"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZES = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
INPUT_LENS  = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

WARMUP_RUNS = 2
MEASURE_RUNS = 5

# 你可以按机器情况加这些公共参数
COMMON_ARGS = [
    "--disable-detokenize",
    "--gpu-memory-utilization", "0.90",
]

RESULT_CSV = OUT_DIR / "results.csv"

def summarize_samples(samples: list[float]) -> tuple[float, float]:
    mean_value = statistics.fmean(samples)
    std_value = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return mean_value, std_value

def run_cmd(cmd):
    print("RUN:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    torch.cuda.synchronize()
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return proc.stdout

def bench_prefill(batch_size: int, input_len: int) -> list[float]:
    out_json = RAW_DIR / f"prefill_bs{batch_size}_in{input_len}.json"
    max_model_len = input_len + 8
    cmd = [
        "vllm", "bench", "latency",
        "--model", MODEL,
        "--batch-size", str(batch_size),
        "--input-len", str(input_len),
        "--output-len", "1",
        "--max-model-len", str(max_model_len),
        "--num-iters-warmup", str(WARMUP_RUNS),
        "--num-iters", str(MEASURE_RUNS),
        "--output-json", str(out_json),
        *COMMON_ARGS,
    ]
    run_cmd(cmd)
    with open(out_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload['latencies']


def bench_decode(batch_size: int, input_len: int) -> list[float]:
    out_json = RAW_DIR / f"decode_bs{batch_size}_in{input_len}.json"
    max_model_len = input_len + 8
    cmd = [
        "vllm", "bench", "latency",
        "--model", MODEL,
        "--batch-size", str(batch_size),
        "--input-len", str(input_len),
        "--output-len", "2",
        "--max-model-len", str(max_model_len),
        "--num-iters-warmup", str(WARMUP_RUNS),
        "--num-iters", str(MEASURE_RUNS),
        "--output-json", str(out_json),
        *COMMON_ARGS,
    ]
    run_cmd(cmd)
    with open(out_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload['latencies']

def main():
    fieldnames=[
        "batch_size",
        "input_len",
        "total_tokens",
        "prefill_s_mean",
        "prefill_s_std",
        "decode_s_mean",
        "decode_s_std",
        "prefill_tokens_per_s",
        "decode_tokens_per_s",
        "warmup_runs",
        "measure_runs",
    ]
    with open(RESULT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for batch_size, input_len in zip(BATCH_SIZES, INPUT_LENS):
        total_tokens = batch_size * input_len

        prefill_s = bench_prefill(batch_size, input_len)
        prefill_s_mean, prefill_s_std = summarize_samples(prefill_s)

        decode_s = bench_decode(batch_size, input_len)
        decode_s = [x - prefill_s_mean for x in decode_s]
        decode_s_mean, decode_s_std = summarize_samples(decode_s)

        result = BenchmarkResult(
            batch_size=batch_size,
            input_len=input_len,
            total_tokens=total_tokens,
            prefill_s_mean=prefill_s_mean,
            prefill_s_std=prefill_s_std,
            decode_s_mean=decode_s_mean,
            decode_s_std=decode_s_std,
            prefill_tokens_per_s=total_tokens / (prefill_s_mean),
            decode_tokens_per_s=batch_size / (decode_s_mean),
            warmup_runs=WARMUP_RUNS,
            measure_runs=MEASURE_RUNS,
            prefill_samples_s=None,
            decode_samples_s=None,
        )

        with open(RESULT_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(
                {
                    "batch_size": result.batch_size,
                    "input_len": result.input_len,
                    "total_tokens": result.total_tokens,
                    "prefill_s_mean": result.prefill_s_mean,
                    "prefill_s_std": result.prefill_s_std,
                    "decode_s_mean": result.decode_s_mean,
                    "decode_s_std": result.decode_s_std,
                    "prefill_tokens_per_s": result.prefill_tokens_per_s,
                    "decode_tokens_per_s": result.decode_tokens_per_s,
                    "warmup_runs": result.warmup_runs,
                    "measure_runs": result.measure_runs,
                }
            )
        
        print(f"[INFO] result write over")

    print(f"\nSaved: {RESULT_CSV}")


if __name__ == "__main__":
    main()