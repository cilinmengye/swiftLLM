from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

RESULT_CSV = "result.csv"
REQUEST_RATES = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
SUPPORTED_PERCENTILE_METRICS = {"ttft", "tpot", "itl", "e2el"}
REQUEST_TIMEOUT_SECONDS = 6 * 60 * 60
DEFAULT_MODEL_PATH = "/mnt/hdd/data/yxlin/huggingface/Meta-Llama-3.1-8B"
DEFAULT_DATASET_PATH = "/mnt/hdd/data/yxlin/huggingface_data/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json"
DEFAULT_RESULT_PATH = "/home/yxlin/github/swift/swiftLLM/dummys/benchmark/online/swiftllm_result"
CSV_FIELDNAMES = [
    "request_rate",
    "mean_per_token_latency",
    "mean_first_token_latency",
    "mean_per_output_token_latency",
]
JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class SampleRequest:
    request_id: str
    prompt: str
    prompt_len: int
    output_len: int


@dataclass
class RequestMetrics:
    request_id: str
    prompt_len: int
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0
    itls: list[float] = field(default_factory=list)
    error: str = ""
    start_time: float = 0.0


@dataclass(frozen=True)
class BenchmarkArgs:
    host: str
    port: int
    endpoint: str
    model_path: Path
    dataset_name: str
    dataset_path: Path
    num_prompts: int
    burstiness: float
    max_concurrency: int
    seed: int
    disable_shuffle: bool
    percentile_metrics: tuple[str, ...]
    metric_percentiles: tuple[float, ...]
    save_result: bool
    save_detailed: bool
    result_filepath: Path
    request_rates: tuple[float, ...]


def parse_request_rates(raw_value: str) -> tuple[float, ...]:
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated request rate list.")
    parsed: list[float] = []
    for value in values:
        parsed_value = float(value)
        if parsed_value <= 0.0:
            raise argparse.ArgumentTypeError("Request rates must be positive.")
        parsed.append(parsed_value)
    return tuple(parsed)


def parse_metric_percentiles(raw_value: str) -> tuple[float, ...]:
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated percentile list.")
    parsed: list[float] = []
    for value in values:
        parsed_value = float(value)
        if parsed_value < 0.0 or parsed_value > 100.0:
            raise argparse.ArgumentTypeError("Percentiles must be in [0, 100].")
        parsed.append(parsed_value)
    return tuple(parsed)


def parse_percentile_metrics(raw_value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw_value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one percentile metric.")
    invalid = [value for value in values if value not in SUPPORTED_PERCENTILE_METRICS]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unsupported percentile metrics: {', '.join(invalid)}"
        )
    return values


def parse_args() -> BenchmarkArgs:
    parser = argparse.ArgumentParser(description="Benchmark SwiftLLM online performance.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--endpoint", type=str, default="/generate")
    parser.add_argument("--model-path", "--model", dest="model_path", type=Path, default=Path(DEFAULT_MODEL_PATH))
    parser.add_argument("--dataset-name", type=str, default="sharegpt")
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_DATASET_PATH))
    parser.add_argument("--num-prompts", type=int, default=500)
    parser.add_argument("--burstiness", type=float, default=1.0)
    parser.add_argument("--max-concurrency", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--disable-shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--percentile-metrics",
        type=parse_percentile_metrics,
        default=parse_percentile_metrics("ttft,tpot,itl,e2el"),
    )
    parser.add_argument(
        "--metric-percentiles",
        type=parse_metric_percentiles,
        default=parse_metric_percentiles("50,95,99"),
    )
    parser.add_argument(
        "--save-result",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-detailed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--result-filepath",
        type=Path,
        default=Path(DEFAULT_RESULT_PATH),
    )
    parser.add_argument(
        "--request-rates",
        type=parse_request_rates,
        default=REQUEST_RATES,
    )
    namespace = parser.parse_args()
    if namespace.dataset_name != "sharegpt":
        raise ValueError("Only the ShareGPT dataset is supported by this benchmark.")
    if namespace.num_prompts <= 0:
        raise ValueError("num_prompts must be positive.")
    if namespace.burstiness <= 0.0:
        raise ValueError("burstiness must be positive.")
    if namespace.max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive.")
    return BenchmarkArgs(
        host=namespace.host,
        port=namespace.port,
        endpoint=namespace.endpoint,
        model_path=namespace.model_path,
        dataset_name=namespace.dataset_name,
        dataset_path=namespace.dataset_path,
        num_prompts=namespace.num_prompts,
        burstiness=namespace.burstiness,
        max_concurrency=namespace.max_concurrency,
        seed=namespace.seed,
        disable_shuffle=namespace.disable_shuffle,
        percentile_metrics=namespace.percentile_metrics,
        metric_percentiles=namespace.metric_percentiles,
        save_result=namespace.save_result,
        save_detailed=namespace.save_detailed,
        result_filepath=namespace.result_filepath,
        request_rates=namespace.request_rates,
    )


def build_endpoint_url(host: str, port: int, endpoint: str) -> str:
    normalized_endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    return f"http://{host}:{port}{normalized_endpoint}"


def tokenize_text(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    input_ids = tokenizer(text).input_ids
    if not isinstance(input_ids, list) or not all(isinstance(token_id, int) for token_id in input_ids):
        raise TypeError("Unexpected tokenizer output shape.")
    return [int(token_id) for token_id in input_ids]


def extract_prompt_completion(entry: object) -> tuple[str, str] | None:
    if not isinstance(entry, dict):
        return None
    conversations = entry.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        return None
    first_turn = conversations[0]
    second_turn = conversations[1]
    if not isinstance(first_turn, dict) or not isinstance(second_turn, dict):
        return None
    prompt = first_turn.get("value")
    completion = second_turn.get("value")
    if not isinstance(prompt, str) or not isinstance(completion, str):
        return None
    return prompt, completion


def is_valid_sequence(prompt_len: int, output_len: int) -> bool:
    return (
        prompt_len >= 4
        and output_len >= 4
        and prompt_len <= 1024
        and (prompt_len + output_len) <= 2048
    )


def maybe_oversample_requests(
    requests: list[SampleRequest],
    num_requests: int,
    seed: int,
) -> list[SampleRequest]:
    if len(requests) >= num_requests:
        return requests[:num_requests]
    if not requests:
        raise ValueError("No valid ShareGPT requests were found.")
    rng = random.Random(seed)
    oversampled = list(requests)
    while len(oversampled) < num_requests:
        chosen = requests[rng.randrange(len(requests))]
        oversampled.append(
            SampleRequest(
                request_id=f"req-{len(oversampled)}",
                prompt=chosen.prompt,
                prompt_len=chosen.prompt_len,
                output_len=chosen.output_len,
            )
        )
    return oversampled


def load_sharegpt_requests(
    dataset_path: Path,
    tokenizer: PreTrainedTokenizerBase,
    num_requests: int,
    seed: int,
    disable_shuffle: bool,
) -> list[SampleRequest]:
    with dataset_path.open("r", encoding="utf-8") as file:
        raw_entries = json.load(file)
    if not isinstance(raw_entries, list):
        raise TypeError("ShareGPT dataset must be a JSON array.")
    entries = list(raw_entries)
    if not disable_shuffle:
        random.Random(seed).shuffle(entries)
    requests: list[SampleRequest] = []
    for entry in entries:
        extracted = extract_prompt_completion(entry)
        if extracted is None:
            continue
        prompt, completion = extracted
        prompt_len = len(tokenize_text(tokenizer, prompt))
        output_len = len(tokenize_text(tokenizer, completion))
        if not is_valid_sequence(prompt_len, output_len):
            continue
        requests.append(
            SampleRequest(
                request_id=f"req-{len(requests)}",
                prompt=prompt,
                prompt_len=prompt_len,
                output_len=output_len,
            )
        )
        if len(requests) >= num_requests:
            break
    return maybe_oversample_requests(requests, num_requests, seed)


def build_delay_schedule(
    total_requests: int,
    request_rate: float,
    burstiness: float,
    seed: int,
) -> list[float]:
    if total_requests <= 0:
        raise ValueError("total_requests must be positive.")
    if math.isinf(request_rate):
        return [0.0] * total_requests
    generator = np.random.default_rng(seed)
    delay_schedule: list[float] = []
    cumulative_delay = 0.0
    for _ in range(total_requests):
        if math.isinf(burstiness):
            sampled_delay = 1.0 / request_rate
        else:
            scale = 1.0 / (request_rate * burstiness)
            sampled_delay = float(generator.gamma(shape=burstiness, scale=scale))
        cumulative_delay += sampled_delay
        delay_schedule.append(cumulative_delay)
    if delay_schedule[-1] > 0.0:
        target_total_delay = total_requests / request_rate
        normalize_factor = target_total_delay / delay_schedule[-1]
        delay_schedule = [delay * normalize_factor for delay in delay_schedule]
    return delay_schedule


async def issue_request(
    session: aiohttp.ClientSession,
    api_url: str,
    request: SampleRequest,
) -> RequestMetrics:
    result = RequestMetrics(request_id=request.request_id, prompt_len=request.prompt_len)
    payload = {
        "prompt": request.prompt,
        "output_len": request.output_len,
        "stream": True,
        "decode": False,
    }
    started_at = time.perf_counter()
    result.start_time = started_at
    most_recent_timestamp = started_at
    try:
        async with session.post(api_url, json=payload) as response:
            if response.status != 200:
                response_text = await response.text()
                result.error = f"HTTP {response.status}: {response_text}"
                return result
            async for raw_line in response.content:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                int(line)
                timestamp = time.perf_counter()
                if result.output_tokens == 0:
                    result.ttft = timestamp - started_at
                else:
                    result.itls.append(timestamp - most_recent_timestamp)
                most_recent_timestamp = timestamp
                result.output_tokens += 1
            if result.output_tokens == 0:
                result.error = "Never received a valid token from the stream."
                return result
            result.success = True
            result.latency = most_recent_timestamp - started_at
            return result
    except Exception as error:
        result.error = str(error)
        return result


async def run_readiness_check(
    session: aiohttp.ClientSession,
    api_url: str,
    request: SampleRequest,
) -> None:
    result = await issue_request(session, api_url, request)
    if not result.success:
        raise RuntimeError(f"SwiftLLM readiness check failed: {result.error}")


def safe_mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def safe_median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def safe_pstdev(values: list[float]) -> float:
    return statistics.pstdev(values) if values else 0.0


def compute_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = rank - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + (upper_value - lower_value) * fraction


def format_percentile(percentile: float) -> str:
    return str(int(percentile)) if percentile.is_integer() else str(percentile).replace(".", "_")


def add_metric_summary(
    result: dict[str, JsonValue],
    metric_name: str,
    values: list[float],
    selected_metrics: tuple[str, ...],
    selected_percentiles: tuple[float, ...],
) -> None:
    result[f"mean_{metric_name}_ms"] = safe_mean(values) * 1000.0
    result[f"median_{metric_name}_ms"] = safe_median(values) * 1000.0
    result[f"std_{metric_name}_ms"] = safe_pstdev(values) * 1000.0
    if metric_name not in selected_metrics:
        return
    for percentile in selected_percentiles:
        percentile_key = format_percentile(percentile)
        result[f"p{percentile_key}_{metric_name}_ms"] = compute_percentile(values, percentile) * 1000.0


def compute_peak_metrics(outputs: list[RequestMetrics]) -> tuple[float, int]:
    successful_outputs = [output for output in outputs if output.success]
    if not successful_outputs:
        return 0.0, 0
    min_start_time = min(output.start_time for output in successful_outputs)
    max_end_time = max(output.start_time + output.latency for output in successful_outputs)
    duration_seconds = int(math.ceil(max_end_time - min_start_time)) + 1
    tokens_per_second = [0.0 for _ in range(duration_seconds)]
    concurrent_requests_per_second = [0 for _ in range(duration_seconds)]
    for output in successful_outputs:
        if output.output_tokens > 0:
            token_time = output.start_time + output.ttft
            token_bucket = int(token_time - min_start_time)
            if 0 <= token_bucket < duration_seconds:
                tokens_per_second[token_bucket] += 1.0
            for itl_value in output.itls:
                token_time += itl_value
                token_bucket = int(token_time - min_start_time)
                if 0 <= token_bucket < duration_seconds:
                    tokens_per_second[token_bucket] += 1.0
        request_start_second = int(output.start_time - min_start_time)
        request_end_second = int((output.start_time + output.latency) - min_start_time)
        for second in range(request_start_second, request_end_second + 1):
            if 0 <= second < duration_seconds:
                concurrent_requests_per_second[second] += 1
    return max(tokens_per_second), max(concurrent_requests_per_second)


def build_result_payload(
    args: BenchmarkArgs,
    request_rate: float,
    requests: list[SampleRequest],
    outputs: list[RequestMetrics],
    benchmark_duration: float,
) -> dict[str, JsonValue]:
    ttfts = [output.ttft for output in outputs if output.success]
    tpots = [
        (output.latency - output.ttft) / (output.output_tokens - 1)
        for output in outputs
        if output.success and output.output_tokens > 1
    ]
    itls = [itl for output in outputs if output.success for itl in output.itls]
    e2els = [output.latency for output in outputs if output.success]
    total_input_tokens = sum(request.prompt_len for request, output in zip(requests, outputs) if output.success)
    output_lens = [output.output_tokens if output.success else 0 for output in outputs]
    total_output_tokens = sum(output_lens)
    completed = sum(1 for output in outputs if output.success)
    failed = len(outputs) - completed
    per_input_token_latencies = [
        (output.ttft / output.prompt_len) if output.success and output.prompt_len > 0 else 0.0
        for output in outputs
    ]
    successful_per_input_token_latencies = [
        latency
        for latency, output in zip(per_input_token_latencies, outputs)
        if output.success
    ]
    max_output_tokens_per_s, max_concurrent_requests = compute_peak_metrics(outputs)
    result: dict[str, JsonValue] = {
        "date": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "endpoint_type": "swiftllm",
        "backend": "swiftllm",
        "label": None,
        "model_id": str(args.model_path),
        "tokenizer_id": str(args.model_path),
        "num_prompts": len(requests),
        "request_rate": request_rate,
        "burstiness": args.burstiness,
        "max_concurrency": args.max_concurrency,
        "duration": benchmark_duration,
        "completed": completed,
        "failed": failed,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "request_throughput": completed / benchmark_duration if benchmark_duration > 0.0 else 0.0,
        "request_goodput": None,
        "output_throughput": total_output_tokens / benchmark_duration if benchmark_duration > 0.0 else 0.0,
        "total_token_throughput": (total_input_tokens + total_output_tokens) / benchmark_duration if benchmark_duration > 0.0 else 0.0,
        "max_output_tokens_per_s": max_output_tokens_per_s,
        "max_concurrent_requests": max_concurrent_requests,
        "rtfx": 0.0,
        "mean_per_input_token_latency_ms": safe_mean(successful_per_input_token_latencies) * 1000.0,
    }
    add_metric_summary(result, "ttft", ttfts, args.percentile_metrics, args.metric_percentiles)
    add_metric_summary(result, "tpot", tpots, args.percentile_metrics, args.metric_percentiles)
    add_metric_summary(result, "itl", itls, args.percentile_metrics, args.metric_percentiles)
    add_metric_summary(result, "e2el", e2els, args.percentile_metrics, args.metric_percentiles)
    if args.save_detailed:
        result["input_lens"] = [output.prompt_len for output in outputs]
        result["output_lens"] = output_lens
        result["ttfts"] = [output.ttft for output in outputs]
        result["itls"] = [output.itls for output in outputs]
        result["start_times"] = [output.start_time for output in outputs]
        result["errors"] = [output.error for output in outputs]
        result["request_ids"] = [output.request_id for output in outputs]
        result["per_input_token_latencies"] = per_input_token_latencies
    return result


def benchmark_summary_row(result: dict[str, JsonValue], request_rate: float) -> dict[str, float]:
    mean_itl_ms = result.get("mean_itl_ms", 0.0)
    mean_ttft_ms = result.get("mean_ttft_ms", 0.0)
    mean_tpot_ms = result.get("mean_tpot_ms", 0.0)
    return {
        "request_rate": request_rate,
        "mean_per_token_latency": float(mean_itl_ms) if isinstance(mean_itl_ms, (int, float)) else 0.0,
        "mean_first_token_latency": float(mean_ttft_ms) if isinstance(mean_ttft_ms, (int, float)) else 0.0,
        "mean_per_output_token_latency": float(mean_tpot_ms) if isinstance(mean_tpot_ms, (int, float)) else 0.0,
    }


def rate_to_filename(request_rate: float) -> str:
    if request_rate.is_integer():
        return str(int(request_rate))
    return str(request_rate).replace(".", "_")


def write_result_csv(result_csv: Path, rows: list[dict[str, float]]) -> None:
    with result_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


async def benchmark_request_rate(
    args: BenchmarkArgs,
    requests: list[SampleRequest],
    request_rate: float,
) -> dict[str, JsonValue]:
    api_url = build_endpoint_url(args.host, args.port, args.endpoint)
    connector = aiohttp.TCPConnector(
        limit=args.max_concurrency,
        limit_per_host=args.max_concurrency,
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=60,
        enable_cleanup_closed=True,
        force_close=False,
    )
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
    semaphore = asyncio.Semaphore(args.max_concurrency)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=False) as session:
        await run_readiness_check(session, api_url, requests[0])

        async def limited_issue(request: SampleRequest) -> RequestMetrics:
            async with semaphore:
                return await issue_request(session, api_url, request)

        benchmark_start_time = time.perf_counter()
        tasks: list[asyncio.Task[RequestMetrics]] = []
        schedule = build_delay_schedule(len(requests), request_rate, args.burstiness, args.seed)
        schedule_start_time = time.perf_counter()
        for request, target_delay in zip(requests, schedule):
            sleep_duration = schedule_start_time + target_delay - time.perf_counter()
            if sleep_duration > 0.0:
                await asyncio.sleep(sleep_duration)
            tasks.append(asyncio.create_task(limited_issue(request)))
        outputs = await asyncio.gather(*tasks)
        benchmark_duration = time.perf_counter() - benchmark_start_time
    return build_result_payload(args, request_rate, requests, outputs, benchmark_duration)


async def main() -> None:
    args = parse_args()
    args.result_filepath.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    requests = load_sharegpt_requests(
        dataset_path=args.dataset_path,
        tokenizer=tokenizer,
        num_requests=args.num_prompts,
        seed=args.seed,
        disable_shuffle=args.disable_shuffle,
    )
    csv_rows: list[dict[str, float]] = []
    for request_rate in args.request_rates:
        result = await benchmark_request_rate(args, requests, request_rate)
        if args.save_result:
            output_json = args.result_filepath / f"rate_{rate_to_filename(request_rate)}.json"
            with output_json.open("w", encoding="utf-8") as file:
                json.dump(result, file, indent=2)
        csv_rows.append(benchmark_summary_row(result, request_rate))
    write_result_csv(args.result_filepath / RESULT_CSV, csv_rows)


if __name__ == "__main__":
    asyncio.run(main())
