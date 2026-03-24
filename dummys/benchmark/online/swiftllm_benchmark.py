"""SwiftLLM 在线基准测试脚本。

模块整体逻辑架构
================
这个脚本的目标是：
1. 从 ShareGPT 数据集中挑选一批和 vLLM 基准方法尽量一致的请求样本。
2. 按给定请求速率（request rate）生成请求到达时间表，模拟在线服务压力。
3. 通过 HTTP 调用 SwiftLLM 的 `/generate` 接口，并在流式返回过程中记录时延。
4. 将单请求指标聚合成整体验证结果。
5. 为每个请求速率输出一份详细 JSON，并汇总为一份 CSV，方便和 vLLM 结果对比。

从执行流程上看，可以分成 5 层：
- 参数层：`parse_args()` 负责读取命令行参数并做基本校验。
- 数据层：`load_sharegpt_requests()` 负责加载 ShareGPT、过滤无效样本、计算 prompt/output 长度。
- 调度层：`build_delay_schedule()` 负责生成请求发射时间表，用来模拟 Poisson / Gamma 到达过程。
- 请求层：`issue_request()` 负责真正发 HTTP 请求并从流式 token 返回中记录 TTFT / ITL / E2EL。
- 聚合层：`build_result_payload()` 负责把所有请求结果汇总成统计指标，再由 `write_result_csv()` 落盘。

核心运行链路：
`main()`
  -> `parse_args()`
  -> `load_sharegpt_requests()`
  -> 对每个 request rate 调用 `benchmark_request_rate()`
       -> `build_delay_schedule()`
       -> `issue_request()`
       -> `build_result_payload()`
  -> `write_result_csv()`

术语说明
========
- TTFT (time to first token): 从请求发出到第一个输出 token 到达的时间。
- ITL (inter-token latency): 相邻两个输出 token 之间的时间间隔。
- TPOT (time per output token): 除第一个 token 以外，平均每个输出 token 的生成时间。
- E2EL (end-to-end latency): 单个请求从发出到最后一个 token 到达的总耗时。
"""

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

# 汇总 CSV 文件名。
RESULT_CSV = "result.csv"

# 默认扫过的请求速率列表，单位是 requests / second。
REQUEST_RATES = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

# 允许做 percentile 统计的指标名称。
SUPPORTED_PERCENTILE_METRICS = {"ttft", "tpot", "itl", "e2el"}

# 单个 HTTP 请求的最大超时时间。
# 这里给得很长，是为了避免长序列或高压力场景下被客户端过早打断。
REQUEST_TIMEOUT_SECONDS = 6 * 60 * 60

# 默认模型和数据集路径，方便直接在当前环境运行。
DEFAULT_MODEL_PATH = "/mnt/hdd/data/yxlin/huggingface/Meta-Llama-3.1-8B"
DEFAULT_DATASET_PATH = "/mnt/hdd/data/yxlin/huggingface_data/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json"
DEFAULT_RESULT_PATH = "/home/yxlin/github/swift/swiftLLM/dummys/benchmark/online/swiftllm_result"

# 输出 CSV 的列顺序。
# 这里故意保持和现有 vLLM 结果兼容，方便直接横向比较。
CSV_FIELDNAMES = [
    "request_rate",
    "mean_per_token_latency",
    "mean_first_token_latency",
    "mean_per_output_token_latency",
]

# 便于给结果 JSON 标注一个递归类型别名。
JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class SampleRequest:
    """表示一个待发送的基准请求样本。

    这个结构体只保存“发请求时必须知道的信息”，也就是：
    - `request_id`：请求唯一标识，便于在结果里追踪。
    - `prompt`：真正发送到服务端的输入文本。
    - `prompt_len`：输入 token 数，用于统计吞吐和派生 per-input-token latency。
    - `output_len`：期望生成的输出 token 上限，会作为 `/generate` 的 `max_tokens` 参数传给服务端。

    之所以把它单独定义成 dataclass，是为了把“输入样本”和“运行结果”分开，
    避免后面在 benchmark 执行时混淆。
    """

    request_id: str
    prompt: str
    prompt_len: int
    output_len: int


@dataclass
class RequestMetrics:
    """表示单个请求执行后的测量结果。

    字段说明：
    - `request_id`：对应哪个请求样本。
    - `prompt_len`：输入 token 数，后面聚合指标时会用到。
    - `expected_output_len`：请求期望生成的输出 token 上限。
    - `success`：这个请求是否成功拿到了至少一个有效 token。
    - `latency`：端到端总时延，即最后一个 token 到达减去请求发起时间。
    - `output_tokens`：实际收到的输出 token 数。
    - `ttft`：首 token 时延。
    - `itls`：每两个相邻 token 之间的时间差列表。
    - `error`：如果失败，这里记录错误信息。
    - `start_time`：请求开始发送时的绝对时间戳（perf_counter 基准）。
    - `response_headers_time`：收到响应头时的绝对时间戳。
    - `first_token_time`：首个合法 token 行到达时的绝对时间戳。
    - `last_token_time`：最后一个合法 token 行到达时的绝对时间戳。
    - `invalid_line_count`：流里出现的非法非空行数量。
    - `hit_max_output_len`：实际输出 token 数是否打满期望上限。
    - `finish_reason`：服务端返回的停止原因，例如 `eos` / `max_tokens`。

    这个结构体是后续所有统计汇总的原始数据来源。
    """

    request_id: str
    prompt_len: int
    expected_output_len: int = 0
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0
    itls: list[float] = field(default_factory=list)
    error: str = ""
    start_time: float = 0.0
    response_headers_time: float = 0.0
    first_token_time: float = 0.0
    last_token_time: float = 0.0
    invalid_line_count: int = 0
    hit_max_output_len: bool = False
    finish_reason: str = ""


@dataclass(frozen=True)
class BenchmarkArgs:
    """命令行参数在程序内部的只读表示。

    把 argparse 的 Namespace 转成 dataclass 有两个好处：
    1. 字段更明确，调用时能看出参数的语义；
    2. 设置为 `frozen=True` 后，运行过程中不会被意外修改。

    字段说明：
    - `host` / `port` / `endpoint`：SwiftLLM 服务地址。
    - `model_path`：tokenizer 和模型路径。这里主要用于 tokenizer。
    - `dataset_name` / `dataset_path`：数据集名称与路径。当前只支持 ShareGPT。
    - `num_prompts`：本轮 benchmark 要使用的请求数。
    - `burstiness`：控制到达间隔分布形状；1.0 时接近标准 Poisson 到达。
    - `max_concurrency`：客户端允许的最大并发请求数。
    - `seed`：随机种子，用于可复现的 shuffle / oversample / arrival schedule。
    - `disable_shuffle`：是否保留数据集原始顺序。
    - `ignore_eos`：是否忽略 EOS 并继续生成到上限。
    - `percentile_metrics`：哪些指标需要额外计算分位数。
    - `metric_percentiles`：需要计算的百分位列表，例如 50/95/99。
    - `save_result`：是否保存每个请求速率的详细 JSON。
    - `save_detailed`：是否在 JSON 中保留逐请求明细。
    - `result_filepath`：输出目录。
    - `request_rates`：本次 sweep 要测试的请求速率集合。
    """

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
    ignore_eos: bool
    percentile_metrics: tuple[str, ...]
    metric_percentiles: tuple[float, ...]
    save_result: bool
    save_detailed: bool
    result_filepath: Path
    request_rates: tuple[float, ...]
    max_tokens: int


def parse_request_rates(raw_value: str) -> tuple[float, ...]:
    """把命令行里的请求速率字符串解析成浮点元组。

    参数：
    - `raw_value`：形如 `"1,2,3,4"` 的逗号分隔字符串。

    返回：
    - 一个 `tuple[float, ...]`，表示所有请求速率。

    这里会做两类校验：
    1. 不能为空；
    2. 每个请求速率都必须是正数。
    """

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
    """解析需要统计的百分位列表。

    参数：
    - `raw_value`：形如 `"50,95,99"` 的字符串。

    返回：
    - 一个百分位元组，例如 `(50.0, 95.0, 99.0)`。

    限制：
    - 每个百分位都必须在 `[0, 100]` 区间内。
    """

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
    """解析需要计算 percentile 的指标名称列表。

    参数：
    - `raw_value`：形如 `"ttft,tpot,itl,e2el"` 的字符串。

    返回：
    - 一个指标名元组。

    这个函数的作用不是算分位数，而是先约束“哪些指标允许算分位数”，
    避免后续出现拼写错误或不支持的指标名。
    """

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
    """读取命令行参数并转换成 `BenchmarkArgs`。

    返回：
    - 一个已经完成基础校验的 `BenchmarkArgs` 对象。

    这个函数负责两件事：
    1. 定义 CLI 接口；
    2. 做最基本的参数合法性校验，尽早在程序入口处失败。

    这里不负责检查服务端是否可用，也不负责检查数据集内容是否有效；
    那些属于运行期校验，会在后面的流程里处理。
    """

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
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=False,
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
        default=False,
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
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=-1,
    )
    namespace = parser.parse_args()

    # 当前脚本只实现了 ShareGPT 的采样逻辑，所以这里直接限制数据集类型。
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
        ignore_eos=namespace.ignore_eos,
        percentile_metrics=namespace.percentile_metrics,
        metric_percentiles=namespace.metric_percentiles,
        save_result=namespace.save_result,
        save_detailed=namespace.save_detailed,
        result_filepath=namespace.result_filepath,
        request_rates=namespace.request_rates,
        max_tokens=namespace.max_tokens,
    )


def build_endpoint_url(host: str, port: int, endpoint: str) -> str:
    """拼出完整的 SwiftLLM HTTP 接口地址。

    参数：
    - `host`：服务端主机地址，例如 `127.0.0.1`。
    - `port`：服务端端口，例如 `8000`。
    - `endpoint`：接口路径，例如 `/generate`。

    返回：
    - 完整 URL，例如 `http://127.0.0.1:8000/generate`。

    这里会自动补齐 endpoint 前导 `/`，避免命令行传入 `generate` 时拼接错误。
    """

    normalized_endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    return f"http://{host}:{port}{normalized_endpoint}"


def tokenize_text(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    """用 tokenizer 把文本转成 token id 列表。

    参数：
    - `tokenizer`：Hugging Face tokenizer。
    - `text`：待编码文本。

    返回：
    - `list[int]`，表示编码后的 token id 序列。

    这个函数的主要目的不是“封装 tokenizer 调用”，而是：
    - 强制检查返回形状是否符合预期；
    - 把 token id 统一成 Python `int`，避免后面混入其他数值类型。
    """

    input_ids = tokenizer(text).input_ids
    if not isinstance(input_ids, list) or not all(isinstance(token_id, int) for token_id in input_ids):
        raise TypeError("Unexpected tokenizer output shape.")
    return [int(token_id) for token_id in input_ids]


def extract_prompt_completion(entry: object) -> tuple[str, str] | None:
    """从一条 ShareGPT 原始记录中提取 prompt 和 completion。

    参数：
    - `entry`：数据集中的一条原始 JSON 记录。

    返回：
    - 如果结构合法，返回 `(prompt, completion)`；
    - 如果结构不合法，返回 `None`。

    提取规则：
    - 只看前两个 conversation turn；
    - 第一个 turn 当作 prompt；
    - 第二个 turn 当作 completion。

    这样做是为了尽量贴近 vLLM ShareGPT 基准里的取样方法。
    """

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
    """判断一个样本是否满足 ShareGPT 基准过滤规则。

    参数：
    - `prompt_len`：输入 token 数。
    - `output_len`：输出 token 数。

    返回：
    - `True` 表示保留；`False` 表示过滤掉。

    过滤条件与文档中对齐：
    - prompt 至少 4 token；
    - output 至少 4 token；
    - prompt 不超过 1024 token；
    - prompt + output 总长度不超过 2048 token。
    """

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
    """当有效样本不足时，对现有样本做确定性过采样。

    参数：
    - `requests`：已经过滤后的有效请求列表。
    - `num_requests`：目标请求总数。
    - `seed`：随机种子，保证过采样结果可复现。

    返回：
    - 长度恰好等于 `num_requests` 的请求列表。

    行为说明：
    - 如果有效样本已经足够，直接截断到目标长度；
    - 如果有效样本不够，就从已有样本里随机重复抽取，直到补满；
    - 新补出来的请求会分配新的 `request_id`，避免结果中 ID 冲突。
    """

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
    max_tokens: int,
) -> list[SampleRequest]:
    """从 ShareGPT 数据集中构造基准测试请求。

    参数：
    - `dataset_path`：ShareGPT JSON 文件路径。
    - `tokenizer`：用于计算 prompt/output token 长度的 tokenizer。
    - `num_requests`：最终需要多少条请求。
    - `seed`：随机种子，用于 shuffle 和 oversample。
    - `disable_shuffle`：若为 `True`，保留原始顺序；否则先打乱再取样。
    - `max_tokens`: 每个request能够生成的最大output token number。

    返回：
    - `list[SampleRequest]`，长度会尽量达到 `num_requests`。

    主要步骤：
    1. 读取整个 ShareGPT JSON 数组；
    2. 如果允许 shuffle，就先按 seed 打乱；
    3. 对每条记录提取前两轮对话；
    4. 对 prompt/completion 做分词并计算长度；
    5. 用 `is_valid_sequence()` 做过滤；
    6. 收集到足够样本后停止；
    7. 如果不足 `num_requests`，交给 `maybe_oversample_requests()` 补齐。
    """

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
        
        if max_tokens != -1:    # 即max_tokens不是默认值
                                # 用户确实传了这个参数
            output_len = min(max_tokens, output_len)

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
    """生成每个请求的目标发射时间表。

    参数：
    - `total_requests`：这轮要发送多少个请求。
    - `request_rate`：目标平均请求速率，单位 req/s。
    - `burstiness`：到达分布的 burstiness 参数。
      - `1.0` 时，对应标准 Poisson 风格到达；
      - 越大越接近规律到达；
      - 越小越容易出现更抖动、更聚集的到达。
    - `seed`：随机种子，保证调度时间表可复现。

    返回：
    - 一个长度为 `total_requests` 的列表；
    - 列表中的每个元素都是“相对 benchmark 开始时”的累计延迟秒数。

    例子：
    - 返回 `[0.2, 0.8, 1.1]` 表示：
      - 第 1 个请求在开始后 0.2 秒发出；
      - 第 2 个请求在开始后 0.8 秒发出；
      - 第 3 个请求在开始后 1.1 秒发出。

    实现细节：
    - 先采样 inter-arrival delay；
    - 再转成 cumulative delay；
    - 最后做一次归一化，使总时间跨度更接近 `total_requests / request_rate`，
      从而减少随机采样引入的总体偏移。
    """

    if total_requests <= 0:
        raise ValueError("total_requests must be positive.")
    if math.isinf(request_rate):
        return [0.0] * total_requests

    generator = np.random.default_rng(seed)
    delay_schedule: list[float] = []
    cumulative_delay = 0.0

    for _ in range(total_requests):
        # 如果 burstiness 是无穷大，就退化成固定间隔发射。
        if math.isinf(burstiness):
            sampled_delay = 1.0 / request_rate
        else:
            # Gamma 分布参数的设置方式和目标平均速率相匹配：
            # mean = shape * scale = burstiness * (1 / (request_rate * burstiness)) = 1 / request_rate
            scale = 1.0 / (request_rate * burstiness)
            sampled_delay = float(generator.gamma(shape=burstiness, scale=scale))
        cumulative_delay += sampled_delay
        delay_schedule.append(cumulative_delay)

    # 归一化累计时间，使本轮请求总发射跨度更接近理论目标总时长。
    if delay_schedule[-1] > 0.0:
        target_total_delay = total_requests / request_rate
        normalize_factor = target_total_delay / delay_schedule[-1]
        delay_schedule = [delay * normalize_factor for delay in delay_schedule]
    return delay_schedule


def parse_stream_lines(chunk_buffer: bytearray) -> list[str]:
    """从累计的字节缓冲区里提取完整的 UTF-8 文本行。

    参数：
    - `chunk_buffer`：不断追加响应原始字节后的可变缓冲区。

    返回：
    - 当前已经完整接收到的文本行列表；
    - 缓冲区中最后一个不完整的尾巴会被保留，等待下一个 chunk 补齐。

    这样做是为了把 aiohttp 的底层 transport chunk 重组为真正按换行切分的 token 行，
    避免把 chunk 边界误当成 token 边界。
    """

    complete_lines: list[str] = []
    while True:
        newline_index = chunk_buffer.find(b"\n")
        if newline_index < 0:
            break
        raw_line = bytes(chunk_buffer[:newline_index])
        del chunk_buffer[: newline_index + 1]
        complete_lines.append(raw_line.decode("utf-8").strip())
    return complete_lines


def consume_stream_line(
    result: RequestMetrics,
    line: str,
    started_at: float,
    most_recent_timestamp: float,
) -> float:
    """消费一条已经按换行切好的 token 行，并更新单请求指标。"""

    if not line:
        return most_recent_timestamp

    try:
        int(line)
    except ValueError as error:
        result.invalid_line_count += 1
        raise ValueError(f"Invalid token line: {line!r}") from error

    timestamp = time.perf_counter()
    if result.output_tokens == 0:
        result.ttft = timestamp - started_at
        result.first_token_time = timestamp
    else:
        result.itls.append(timestamp - most_recent_timestamp)

    result.last_token_time = timestamp
    result.output_tokens += 1
    return timestamp


async def issue_request(
    session: aiohttp.ClientSession,
    api_url: str,
    request: SampleRequest,
    args: BenchmarkArgs,
) -> RequestMetrics:
    """向 SwiftLLM 发送一个流式请求，并记录延迟指标。

    参数：
    - `session`：复用的 aiohttp ClientSession。
    - `api_url`：完整接口地址，通常是 `/generate`。
    - `request`：要发送的请求样本。

    返回：
    - 一个 `RequestMetrics`，里面包含这个请求的成功/失败状态和时延数据。

    这个函数是整份脚本里最关键的“测量点”。
    它做的事情是：
    1. 构造 SwiftLLM `/generate` 请求体；
    2. 记录请求开始时间；
    3. 流式读取服务端返回的 token id；
    4. 用 token 到达时间计算 TTFT 和 ITL；
    5. 在请求结束后计算总 latency。

    为什么 `decode=False`：
    - 因为 benchmark 关心的是服务路径中的 token 生成时延；
    - 如果服务端在流式输出时还做逐 token 文本解码，会引入额外 Python 开销，污染 TTFT / ITL。
    """

    result = RequestMetrics(
        request_id=request.request_id,
        prompt_len=request.prompt_len,
        expected_output_len=request.output_len,
    )
    payload = {
        "prompt": request.prompt,
        "max_tokens": request.output_len,
        "ignore_eos": args.ignore_eos,
        "stream": True,
        "decode": False,
    }

    started_at = time.perf_counter()
    result.start_time = started_at
    most_recent_timestamp = started_at
    chunk_buffer = bytearray()

    try:
        async with session.post(api_url, json=payload) as response:
            result.response_headers_time = time.perf_counter()
            finish_reason_header = response.headers.get("X-SwiftLLM-Finish-Reason", "")
            if finish_reason_header:
                result.finish_reason = finish_reason_header
            if response.status != 200:
                response_text = await response.text()
                result.error = f"HTTP {response.status}: {response_text}"
                return result

            async for chunk in response.content.iter_any():
                if not chunk:
                    continue
                chunk_buffer.extend(chunk)
                for line in parse_stream_lines(chunk_buffer):
                    most_recent_timestamp = consume_stream_line(
                        result=result,
                        line=line,
                        started_at=started_at,
                        most_recent_timestamp=most_recent_timestamp,
                    )

            if chunk_buffer:
                trailing_line = chunk_buffer.decode("utf-8").strip()
                chunk_buffer.clear()
                if trailing_line:
                    most_recent_timestamp = consume_stream_line(
                        result=result,
                        line=trailing_line,
                        started_at=started_at,
                        most_recent_timestamp=most_recent_timestamp,
                    )

            # 如果 HTTP 成功但一个合法 token 都没收到，也视为失败。
            if result.output_tokens == 0:
                result.error = "Never received a valid token from the stream."
                return result

            result.success = True
            result.latency = most_recent_timestamp - started_at
            result.hit_max_output_len = result.output_tokens >= result.expected_output_len
            if not result.finish_reason:
                result.finish_reason = "max_tokens" if result.hit_max_output_len else "eos"
            return result
    except Exception as error:
        result.error = str(error)
        return result


async def run_readiness_check(
    session: aiohttp.ClientSession,
    api_url: str,
    request: SampleRequest,
    args: BenchmarkArgs,
) -> None:
    """在正式压测前，先用一个样本请求检查服务是否可用。

    参数：
    - `session`：aiohttp 会话。
    - `api_url`：目标接口地址。
    - `request`：用于探活的样本请求。

    返回：
    - 无返回值；
    - 若探活失败，直接抛出异常终止 benchmark。

    作用：
    - 避免 benchmark 跑了一整轮之后才发现服务端根本不可用；
    - 尽量把“服务不可用”和“高压下性能退化”分开。
    """

    result = await issue_request(session, api_url, request, args)
    if not result.success:
        raise RuntimeError(f"SwiftLLM readiness check failed: {result.error}")


def safe_mean(values: list[float]) -> float:
    """安全地计算均值；空列表时返回 0。

    参数：
    - `values`：待求均值的数值列表。

    返回：
    - 列表非空时返回算术平均值；为空时返回 `0.0`。

    这样做是为了避免聚合阶段因为某个指标没有样本而抛异常。
    """

    return statistics.fmean(values) if values else 0.0


def safe_median(values: list[float]) -> float:
    """安全地计算中位数；空列表时返回 0。"""

    return statistics.median(values) if values else 0.0


def safe_pstdev(values: list[float]) -> float:
    """安全地计算总体标准差；空列表时返回 0。"""

    return statistics.pstdev(values) if values else 0.0


def compute_percentile(values: list[float], percentile: float) -> float:
    """手动计算一个百分位值。

    参数：
    - `values`：待统计的样本列表。
    - `percentile`：目标百分位，例如 `95.0`。

    返回：
    - 对应百分位上的值；若列表为空则返回 `0.0`。

    实现方式：
    - 先排序；
    - 再按线性插值方式在相邻 rank 之间取值。

    这里自己实现而不是直接依赖外部函数，是为了让输出行为更可控、依赖更少。
    """

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
    """把百分位数格式化成结果 JSON 键名里可用的字符串。

    参数：
    - `percentile`：百分位数，例如 `95.0` 或 `99.9`。

    返回：
    - 若是整数百分位，返回 `"95"`；
    - 若带小数，返回把点替换为下划线的形式，例如 `"99_9"`。

    这样可以生成稳定的键名，如 `p95_ttft_ms` 或 `p99_9_itl_ms`。
    """

    return str(int(percentile)) if percentile.is_integer() else str(percentile).replace(".", "_")


def add_metric_summary(
    result: dict[str, JsonValue],
    metric_name: str,
    values: list[float],
    selected_metrics: tuple[str, ...],
    selected_percentiles: tuple[float, ...],
) -> None:
    """把某一类指标的统计摘要写入结果字典。

    参数：
    - `result`：最终要输出的结果字典，会被原地修改。
    - `metric_name`：指标名，例如 `ttft` / `itl` / `tpot` / `e2el`。
    - `values`：该指标对应的样本列表，单位是秒。
    - `selected_metrics`：用户要求做 percentile 的指标集合。
    - `selected_percentiles`：用户要求输出的百分位列表。

    返回：
    - 无返回值；结果直接写进 `result`。

    写入内容包括：
    - mean / median / std；
    - 如果该指标在 `selected_metrics` 中，再补充 p50/p95/p99 等分位数。

    注意：
    - 这里统一把秒转换成毫秒后再写入结果 JSON；
    - 所以输出键名都以 `_ms` 结尾。
    """

    result[f"mean_{metric_name}_ms"] = safe_mean(values) * 1000.0
    result[f"median_{metric_name}_ms"] = safe_median(values) * 1000.0
    result[f"std_{metric_name}_ms"] = safe_pstdev(values) * 1000.0
    if metric_name not in selected_metrics:
        return
    for percentile in selected_percentiles:
        percentile_key = format_percentile(percentile)
        result[f"p{percentile_key}_{metric_name}_ms"] = compute_percentile(values, percentile) * 1000.0


def compute_peak_metrics(outputs: list[RequestMetrics]) -> tuple[float, int]:
    """估算峰值 token 吞吐和峰值并发请求数。

    参数：
    - `outputs`：所有请求的测量结果。

    返回：
    - 一个二元组 `(max_output_tokens_per_s, max_concurrent_requests)`。
      - `max_output_tokens_per_s`：任意 1 秒桶里的最大输出 token 数。
      - `max_concurrent_requests`：任意 1 秒桶里的最大在途请求数。

    计算思路：
    - 先找出所有成功请求的最早开始时间和最晚结束时间；
    - 用按秒分桶的方式近似统计：
      - 每个 token 到达时，落入对应秒桶；
      - 每个请求覆盖的时间区间内，对应秒桶的并发数加一。

    这不是“精确到亚秒级”的瞬时峰值，而是一个足够直观、便于横向比较的近似峰值指标。
    """

    successful_outputs = [output for output in outputs if output.success]
    if not successful_outputs:
        return 0.0, 0

    min_start_time = min(output.start_time for output in successful_outputs)
    max_end_time = max(output.start_time + output.latency for output in successful_outputs)
    duration_seconds = int(math.ceil(max_end_time - min_start_time)) + 1

    tokens_per_second = [0.0 for _ in range(duration_seconds)]
    concurrent_requests_per_second = [0 for _ in range(duration_seconds)]

    for output in successful_outputs:
        # 先把 token 到达轨迹折算到按秒桶里。
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

        # 再把请求存活区间折算到按秒桶里，统计每秒在途请求数。
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
    """把一轮请求速率测试的原始结果聚合成最终 JSON 结构。

    参数：
    - `args`：全局 benchmark 参数。
    - `request_rate`：当前这轮测试的请求速率。
    - `requests`：本轮使用的输入请求列表。
    - `outputs`：每个请求对应的测量结果列表。
    - `benchmark_duration`：整轮测试的总耗时（秒）。

    返回：
    - 一个可直接写入 JSON 的字典。

    主要工作：
    1. 从 `outputs` 中拆出各类原始指标样本；
    2. 计算吞吐、成功数、失败数、峰值指标；
    3. 计算 TTFT / ITL / TPOT / E2EL 的均值、中位数、标准差、分位数；
    4. 如果 `save_detailed=True`，把逐请求的明细数组也带上。

    这里构造出来的 JSON 既服务于详细分析，也兼容后面生成摘要 CSV 的需要。
    """

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
    expected_output_lens = [output.expected_output_len for output in outputs]
    total_output_tokens = sum(output_lens)
    completed = sum(1 for output in outputs if output.success)
    failed = len(outputs) - completed
    response_header_latencies = [
        (output.response_headers_time - output.start_time)
        for output in outputs
        if output.success and output.response_headers_time > 0.0
    ]
    first_token_after_headers = [
        (output.first_token_time - output.response_headers_time)
        for output in outputs
        if output.success and output.first_token_time > 0.0 and output.response_headers_time > 0.0
    ]
    actual_output_tokens = [output.output_tokens for output in outputs if output.success]
    total_invalid_lines = sum(output.invalid_line_count for output in outputs)
    max_output_len_hits = sum(1 for output in outputs if output.success and output.hit_max_output_len)
    finish_reasons = [output.finish_reason for output in outputs if output.success and output.finish_reason]
    finish_reason_counts: dict[str, int] = {}
    for finish_reason in finish_reasons:
        finish_reason_counts[finish_reason] = finish_reason_counts.get(finish_reason, 0) + 1
    eos_stop_count = finish_reason_counts.get("eos", 0)
    max_tokens_stop_count = finish_reason_counts.get("max_tokens", 0)

    # 这个指标不是现有 CSV 的一部分，但有助于补足 README 里“per input token latency”的讨论。
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
        "mean_expected_output_tokens": safe_mean(expected_output_lens),
        "mean_actual_output_tokens": safe_mean(actual_output_tokens),
        "median_actual_output_tokens": safe_median(actual_output_tokens),
        "hit_max_output_len_count": max_output_len_hits,
        "hit_max_output_len_ratio": (max_output_len_hits / completed) if completed > 0 else 0.0,
        "total_invalid_stream_lines": total_invalid_lines,
        "finish_reason_counts": finish_reason_counts,
        "eos_stop_count": eos_stop_count,
        "eos_stop_ratio": (eos_stop_count / completed) if completed > 0 else 0.0,
        "max_tokens_stop_count": max_tokens_stop_count,
        "max_tokens_stop_ratio": (max_tokens_stop_count / completed) if completed > 0 else 0.0,
        "request_throughput": completed / benchmark_duration if benchmark_duration > 0.0 else 0.0,
        "request_goodput": None,
        "output_throughput": total_output_tokens / benchmark_duration if benchmark_duration > 0.0 else 0.0,
        "total_token_throughput": (total_input_tokens + total_output_tokens) / benchmark_duration if benchmark_duration > 0.0 else 0.0,
        "max_output_tokens_per_s": max_output_tokens_per_s,
        "max_concurrent_requests": max_concurrent_requests,
        "rtfx": 0.0,
        "mean_per_input_token_latency_ms": safe_mean(successful_per_input_token_latencies) * 1000.0,
        "mean_response_headers_latency_ms": safe_mean(response_header_latencies) * 1000.0,
        "mean_headers_to_first_token_latency_ms": safe_mean(first_token_after_headers) * 1000.0,
    }


    add_metric_summary(result, "ttft", ttfts, args.percentile_metrics, args.metric_percentiles)
    add_metric_summary(result, "tpot", tpots, args.percentile_metrics, args.metric_percentiles)
    add_metric_summary(result, "itl", itls, args.percentile_metrics, args.metric_percentiles)
    add_metric_summary(result, "e2el", e2els, args.percentile_metrics, args.metric_percentiles)

    # save_detailed=True 时，把逐请求原始数据也保存在 JSON 里，便于后续排查异常样本。
    if args.save_detailed:
        result["input_lens"] = [output.prompt_len for output in outputs]
        result["expected_output_lens"] = expected_output_lens
        result["output_lens"] = output_lens
        result["ttfts"] = [output.ttft for output in outputs]
        result["itls"] = [output.itls for output in outputs]
        result["start_times"] = [output.start_time for output in outputs]
        result["response_headers_times"] = [output.response_headers_time for output in outputs]
        result["first_token_times"] = [output.first_token_time for output in outputs]
        result["last_token_times"] = [output.last_token_time for output in outputs]
        result["response_headers_latencies"] = [
            (output.response_headers_time - output.start_time) if output.response_headers_time > 0.0 else 0.0
            for output in outputs
        ]
        result["headers_to_first_token_latencies"] = [
            (output.first_token_time - output.response_headers_time)
            if output.first_token_time > 0.0 and output.response_headers_time > 0.0
            else 0.0
            for output in outputs
        ]
        result["invalid_line_counts"] = [output.invalid_line_count for output in outputs]
        result["hit_max_output_len"] = [output.hit_max_output_len for output in outputs]
        result["finish_reasons"] = [output.finish_reason for output in outputs]
        result["errors"] = [output.error for output in outputs]
        result["request_ids"] = [output.request_id for output in outputs]
        result["per_input_token_latencies"] = per_input_token_latencies
    return result


def benchmark_summary_row(result: dict[str, JsonValue], request_rate: float) -> dict[str, float]:
    """从详细结果 JSON 中抽取一行 CSV 摘要。

    参数：
    - `result`：`build_result_payload()` 生成的详细结果字典。
    - `request_rate`：当前这一行对应的请求速率。

    返回：
    - 一个只包含 CSV 所需字段的字典。

    字段映射关系：
    - `mean_per_token_latency` <- `mean_itl_ms`
    - `mean_first_token_latency` <- `mean_ttft_ms`
    - `mean_per_output_token_latency` <- `mean_tpot_ms`

    这样做是为了让 SwiftLLM 输出和现有 vLLM 结果文件格式保持一致。
    """

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
    """把请求速率转成适合文件名的字符串。

    参数：
    - `request_rate`：请求速率，例如 `1.0` 或 `2.5`。

    返回：
    - 若是整数，返回 `"1"`；
    - 若有小数，返回把点替换成下划线后的字符串，例如 `"2_5"`。

    用途：
    - 生成 `rate_1.json`、`rate_2_5.json` 这类文件名。
    """

    if request_rate.is_integer():
        return str(int(request_rate))
    return str(request_rate).replace(".", "_")


def write_result_csv(result_csv: Path, rows: list[dict[str, float]]) -> None:
    """把所有请求速率的摘要结果写入 CSV。

    参数：
    - `result_csv`：目标 CSV 路径。
    - `rows`：每个请求速率对应的一行摘要数据。

    返回：
    - 无返回值。

    这个函数很简单，但它承担了“统一输出格式”的职责：
    保证 CSV 的列顺序和已有基线一致，不会因为字典顺序变化而漂移。
    """

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
    """在一个给定请求速率下跑完整轮 benchmark。

    参数：
    - `args`：全局 benchmark 配置。
    - `requests`：这一轮要发送的请求样本列表。
    - `request_rate`：当前测试目标速率，单位 req/s。

    返回：
    - 当前请求速率对应的一份详细结果 JSON 字典。

    这个函数相当于“一轮压测的主控制器”，主要做四件事：
    1. 创建 HTTP 连接池和超时配置；
    2. 先做 readiness check，确保服务端可用；
    3. 根据 `build_delay_schedule()` 逐个按时间发射请求；
    4. 等待全部请求完成后，调用 `build_result_payload()` 汇总结果。

    并发控制方式：
    - `aiohttp.TCPConnector` 限制连接数；
    - `asyncio.Semaphore` 再限制同时在途的请求数；
    两者共同保证客户端不会超出设定并发。
    """

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
        await run_readiness_check(session, api_url, requests[0], args)

        async def limited_issue(request: SampleRequest) -> RequestMetrics:
            """在并发信号量保护下发出单个请求。"""

            async with semaphore:
                return await issue_request(session, api_url, request, args)

        benchmark_start_time = time.perf_counter()
        tasks: list[asyncio.Task[RequestMetrics]] = []

        # 这里的 schedule 是“每个请求应该在 benchmark 开始后多久发出”。
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
    """脚本入口：按所有请求速率顺序执行 benchmark，并写出结果文件。

    执行步骤：
    1. 解析命令行参数；
    2. 创建输出目录；
    3. 加载 tokenizer；
    4. 从 ShareGPT 构造固定请求集合；
    5. 按每个 request rate 依次运行 benchmark；
    6. 对每个 request rate 保存详细 JSON；
    7. 最后汇总写出一份 CSV。

    参数来源：
    - 不直接接收函数参数，统一从命令行解析。

    返回：
    - 无返回值。
    """

    args = parse_args()
    args.result_filepath.mkdir(parents=True, exist_ok=True)

    # tokenizer 只用于把 prompt/completion 转成 token 长度，不参与真正的在线推理。
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    requests = load_sharegpt_requests(
        dataset_path=args.dataset_path,
        tokenizer=tokenizer,
        num_requests=args.num_prompts,
        seed=args.seed,
        disable_shuffle=args.disable_shuffle,
        max_tokens=args.max_tokens,
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
