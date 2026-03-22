import argparse
import csv
import json
import subprocess
import argparse
import torch
import sys

RESULT_CSV = "result.csv"
REQUEST_RATE = [1, 2, 3, 4, 5, 6]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark vllm online performance."
    )
    parser.add_argument("--backend", type=str, default="openai")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--endpoint", type=str, default="/v1/completions")
    parser.add_argument(
        "--model",
        type=str,
        default="/mnt/hdd/data/yxlin/huggingface/Meta-Llama-3.1-8B",
    )
    parser.add_argument("--dataset-name", type=str, default="sharegpt")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/mnt/hdd/data/yxlin/huggingface_data/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json",
    )
    parser.add_argument("--num-prompts", type=int, default=500)
    parser.add_argument("--burstiness", type=float, default=1.0)
    parser.add_argument("--max-concurrency", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--disable-shuffle",
        default=True,
        help="Do not disturb or remove the prompt.")
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl,e2el",
        help="Comma-separated percentile metrics.",
    )
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="50,95,99",
        help="Comma-separated percentile values.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        default=True,
        help="Whether to save benchmark results.",
    )
    parser.add_argument(
        "--result-filepath",
        type=str,
        default="/home/yxlin/github/swift/swiftLLM/dummys/benchmark/online/vllm_result",
    )

    return parser.parse_args()

def run_cmd(cmd):
    print("RUN:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    torch.cuda.synchronize()
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return proc.stdout

def benchmark_vllm(args, request_rate):
    out_json = args.result_filepath + "/" + f"rate_{request_rate}.json"
    cmd = [
        "vllm", "bench", "serve",
        "--backend", str(args.backend),
        "--host", str(args.host),
        "--port", str(args.port),
        "--endpoint", str(args.endpoint),
        "--model", str(args.model),
        "--dataset-name", str(args.dataset_name),
        "--dataset-path", str(args.dataset_path),
        "--num-prompts", str(args.num_prompts),
        "--request-rate", str(request_rate),
        "--burstiness", str(args.burstiness),
        "--max-concurrency", str(args.max_concurrency),
        "--seed", str(args.seed),
        "--percentile-metrics", str(args.percentile_metrics),
        "--metric-percentiles", str(args.metric_percentiles),
        "--result-filename", str(out_json),
    ]
    if args.disable_shuffle:
        cmd.append("--disable-shuffle")
    if args.save_result:
        cmd.append("--save-result")
    
    run_cmd(cmd)

    with open(out_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    
    return payload["mean_itl_ms"], payload["mean_ttft_ms"], payload["mean_tpot_ms"]


def main():
    # 获取下参数
    # 其中--disable-shuffle保证vllm 和 swiftllm 使用的是同一些prompt
    args = parse_args()

    # 写设置下最终的输出结果
    fieldnames = [
        "request_rate",
        "mean_per_token_latency",
        "mean_first_token_latency",
        "mean_per_output_token_latency",
    ]
    result_csv =  args.result_filepath + "/" + RESULT_CSV
    with open(result_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
    # 进行测试
    for request_rate in REQUEST_RATE:
        # 最终会将不同request_rate下的性能数据以json格式保存到result-filepath目录下
        # per_token_latency 和 per_output_token_latency 的区别是
        # 是否把首个输出 token 算进去。
        per_token_latency, first_token_latency, per_output_token_latency = benchmark_vllm(args, request_rate)

        with open(result_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(
                {
                    "request_rate": request_rate,
                    "mean_per_token_latency": per_token_latency,
                    "mean_first_token_latency": first_token_latency,
                    "mean_per_output_token_latency":  per_output_token_latency,
                }
            )
        print(f"[INFO] result write over")
    
    print(f"\nSaved: {result_csv}")


if __name__ == '__main__':
    main()