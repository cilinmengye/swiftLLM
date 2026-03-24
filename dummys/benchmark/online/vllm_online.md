可以，下面我直接给你一套**按 ShareGPT + Poisson 请求到达 + 多个 Request Rate 扫描**来做 vLLM online serving 测试的方法，并把你要的 3 个指标都落到可执行代码里。

先说一个关键点：
现在 vLLM 的旧脚本 `benchmarks/benchmark_serving.py` 已经废弃，官方推荐直接用 `vllm bench serve`。它原生支持 `sharegpt` 数据集、`--request-rate` 控制请求率、`--burstiness 1.0` 表示 Poisson 到达过程，还会直接输出 TTFT / TPOT / ITL 等统计。ShareGPT 也是官方列出的在线 benchmark 数据集之一。([GitHub][1])

---

# 一、你要测的 3 个指标怎么定义

vLLM 官方直接给出的在线 latency 里，最接近你需求的是：

* **ITL (Inter-Token Latency)**：相邻输出 token 之间的延迟
* **TPOT (Time Per Output Token)**：每个输出 token 的延迟，**不含第一个输出 token**
* **TTFT (Time To First Token)**：首 token 延迟 ([vLLM][2])

你要求的 3 个指标我建议这样对应：

1. **Per-Token Latency**
   直接用 **ITL**
   原因：这是 vLLM 官方已经统计好的 token 级 latency 指标。([vLLM][2])

2. **Per Input Token Latency**
   这个不是 vLLM 官方现成指标，建议你定义为：

   [
   \text{Per Input Token Latency} = \frac{\text{TTFT}}{\text{Input Tokens}}
   ]

   含义：把 prefill 阶段的首 token 等待时间均摊到输入 token 上。
   这是做 online serving 时最常见、也最合理的“每输入 token 成本”近似。

3. **Per Output Token Latency**
   直接用 **TPOT**
   因为 TPOT 本来就是“每个输出 token 的时间（不含首 token）”。([vLLM][2])

---

# 二、整体实验流程

你要做的是下面 5 步：

## 步骤 1：安装环境

```bash
pip install -U "vllm" pandas matplotlib requests
```

如果你打算从 Hugging Face 拉模型和数据，通常还需要：

```bash
pip install -U datasets huggingface_hub
```

---

## 步骤 2：下载 ShareGPT 数据集

vLLM benchmark 文档里给出的 ShareGPT 文件就是这个：([vLLM][2])

```bash
mkdir -p data
wget -O data/ShareGPT_V3_unfiltered_cleaned_split.json \
  https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

---

## 步骤 3：启动 vLLM OpenAI 兼容服务

例如你测试一个 chat/instruct 模型：

```bash
vllm serve /mnt/hdd/data/yxlin/huggingface/Meta-Llama-3.1-8B \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9
```

vLLM 的 `vllm serve` 会启动 OpenAI-compatible server，可用于 `/v1/chat/completions` 或 `/v1/completions`。([vLLM][3])


简单的测试命令:
```bash
curl -N http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/mnt/hdd/data/yxlin/huggingface/Meta-Llama-3.1-8B",
    "prompt": "Summarize the main ideas of Jeff Walker'\''s Product Launch Formula into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients...",
    "max_tokens": 1000,
    "stream": true
  }'
```

### 建议

* 如果模型是 **Instruct/Chat 模型**，benchmark 用：

  * `--backend openai-chat`
  * `--endpoint /v1/chat/completions`
* 如果是基础 completion 模型，可以改成：

  * `--backend openai`
  * `--endpoint /v1/completions`

---

## 步骤 4：对多个 Request Rate 做 sweep

官方 `vllm bench serve` 支持：

* `--dataset-name sharegpt`
* `--dataset-path`
* `--request-rate`
* `--burstiness 1.0`（Poisson）
* `--save-result`
* `--save-detailed` ([vLLM][4])

你可以扫一组请求率，比如：

```text
0.5, 1, 2, 4, 8, 12, 16
```

---

## 步骤 5：从详细结果里提取指标并画图

因为你有一个**自定义指标** `Per Input Token Latency`，所以最稳妥的做法是：

* benchmark 时保存 detailed json
* 用 Python 脚本解析每个请求的 token 数与 latency
* 再汇总成每个 request rate 下的一组均值/分位数
* 最后画图

---

# 三、推荐目录结构

```text
bench_online/
├── data/
│   └── ShareGPT_V3_unfiltered_cleaned_split.json
├── results_raw/
├── results_agg/
├── run_bench.sh
└── parse_and_plot.py
```

---

# 四、压测脚本：run_bench.sh

下面脚本会对多组 request rate 依次跑 online benchmark。

## 版本 A：适合 chat/instruct 模型

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL="/mnt/hdd/data/yxlin/huggingface/Meta-Llama-3.1-8B"
HOST="127.0.0.1"
PORT="8000"
DATASET_PATH="data/ShareGPT_V3_unfiltered_cleaned_split.json"
OUTDIR="results_raw"

mkdir -p "${OUTDIR}"

# 你可以按自己机器性能调整
REQUEST_RATES=(0.5 1 2 4 8 12 16)
NUM_PROMPTS=500
MAX_CONCURRENCY=256
SEED=0

for RATE in "${REQUEST_RATES[@]}"; do
  echo "==== Running request_rate=${RATE} req/s ===="

  vllm bench serve \
    --backend openai-chat \
    --host "${HOST}" \
    --port "${PORT}" \
    --endpoint /v1/chat/completions \
    --model "${MODEL}" \
    --dataset-name sharegpt \
    --dataset-path "${DATASET_PATH}" \
    --num-prompts "${NUM_PROMPTS}" \
    --request-rate "${RATE}" \
    --burstiness 1.0 \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --seed "${SEED}" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99 \
    --save-result \
    --save-detailed \
    --result-filename "${OUTDIR}/sharegpt_rate_${RATE}.json"
done
```

给脚本加执行权限：

```bash
chmod +x run_bench.sh
./run_bench.sh
```

<hr>

后续我实测可以启动的命令为:

vllm bench serve \
  --backend openai \
  --host 127.0.0.1 \
  --port 8000 \
  --endpoint /v1/completions \
  --model /mnt/hdd/data/yxlin/huggingface/Meta-Llama-3.1-8B \
  --dataset-name sharegpt \
  --dataset-path /mnt/hdd/data/yxlin/huggingface_data/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 500 \
  --request-rate 4 \
  --burstiness 1.0 \
  --max-concurrency 256 \
  --seed 0 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,95,99 \
  --save-result \
  --result-filename /home/yxlin/github/swift/swiftLLM/dummys/benchmark/online/vllm_result/sharegpt_rate_4.json

其中参数各个含义为:

可以。你可以把这条命令理解成两部分：

* 前半部分：**怎么生成请求**
* 后半部分：**怎么记录结果**

你特别问到的 4 个参数，正好都属于“怎么生成请求”这一层。

---

# 先整体看这条命令在做什么

你的命令是在做一件事：

> 从 ShareGPT 数据集里取出一批 prompt，然后按某种到达规律把请求发到 `vllm serve` 的 `/v1/completions` 接口，再统计吞吐和延迟指标。`vllm bench serve` 官方就把它定义为 online serving benchmark。([vLLM][1])

所以它不是“离线批处理”，而是在模拟“在线系统里请求不断到来”的场景。([vLLM][1])

---

# 你的每个参数分别是什么意思

我按命令顺序解释。

## 连接和接口相关

### `--backend openai`

表示 benchmark 客户端按 **OpenAI 兼容接口** 的方式发请求。
`vllm bench serve` 目前支持多种 backend，例如 `openai`、`openai-chat` 等；`openai` 是默认值。([vLLM][1])

### `--host 127.0.0.1`

目标服务地址。你这里表示 benchmark 去连本机的 vLLM 服务。
文档里这个参数默认就是 `127.0.0.1`。([vLLM][1])

### `--port 8000`

目标服务端口。默认也是 `8000`。([vLLM][1])

### `--endpoint /v1/completions`

请求发往哪个 API 路径。
你这里测的是 completion 接口，不是 chat completion。文档默认值也是 `/v1/completions`。([vLLM][1])

### `--model /mnt/.../Meta-Llama-3.1-8B`

请求里带上的模型名。
如果不写，benchmark 会去服务端 `/v1/models` 拿第一个模型。([vLLM][1])

---

## 数据集相关

### `--dataset-name sharegpt`

表示 benchmark 使用 ShareGPT 数据集来生成请求。
`sharegpt` 是 `vllm bench serve` 支持的内置数据集之一。([vLLM][1])

### `--dataset-path /mnt/.../ShareGPT_V3_unfiltered_cleaned_split.json`

指定 ShareGPT 数据文件位置。
对 `sharegpt` 这类数据集，需要提供文件路径。([vLLM][1])

### `--seed 0`

随机种子。
它会影响数据抽样和请求顺序等带随机性的环节，方便复现实验。文档中这个参数默认就是 `0`。([vLLM][1])

---

## 你最关心的 4 个参数

---

## 1）`--num-prompts 500`

这个参数的意思是：

> **总共要处理多少个请求样本。**

官方定义就是 “Number of prompts to process”，默认值是 `1000`。([vLLM][1])

在你的命令里，意思就是：

* 从 ShareGPT 里取出 **500 条样本**
* 把这 500 条样本依次作为 benchmark 请求发出去
* 最后所有统计指标，都是基于这 500 个请求算出来的

### 它影响什么

它主要影响两件事：

**第一，统计稳定性。**
500 条通常比 50 条更稳定，因为平均值、P95、P99 不那么容易被偶然样本影响。

**第二，测试时长。**
请求越多，跑得越久。

### 怎么选

* 你想快速试跑：可以先用 `20`、`50`
* 你想正式作图：建议至少 `300` 到 `1000`

---

## 2）`--request-rate 4`

这个参数的意思是：

> **目标请求到达速率是每秒 4 个请求。**

官方对它的语义是“控制请求被启动的速率”；也就是说 benchmark 会尽量按这个速率把请求送出去。([vLLM][1])

所以：

* `--request-rate 1`：平均每秒来 1 个请求
* `--request-rate 4`：平均每秒来 4 个请求
* `--request-rate 16`：平均每秒来 16 个请求

### 直观理解

你可以把它看成系统“外部用户流量强度”的控制旋钮。

如果每秒只来 1 个请求，系统压力小。
如果每秒来 16 个请求，系统压力大，排队和 latency 往往会上升。

### 为什么你画图时横轴要用它

因为你的图是：

* 横轴：Request Rate
* 纵轴：Latency(ms)

那就意味着你会固定其它条件，只改 `--request-rate`，比如：

```text
1, 2, 4, 8, 12, 16
```

然后分别跑几次，比较不同流量强度下的 latency。

---

## 3）`--burstiness 1.0`

这个参数是：

> **控制请求到达间隔的“随机性/突发性”形状。**

你前面引用的那段话里说的是“用 Poisson process with different lambdas 模拟请求到达”。
在 vLLM 这里，`--request-rate` 决定平均速率，而 `--burstiness` 决定这个到达过程到底有多“平滑”或多“扎堆”。文档说明里，当你把它和 request rate 结合起来时，可以模拟不同的到达模式；`1.0` 对应的就是最常见的 Poisson 风格到达。([vLLM][1])

### 你这里为什么写 `1.0`

因为你就是想模拟：

* 平均每秒 4 个请求
* 但不是“严格每 0.25 秒整齐到一个”
* 而是符合现实系统中更自然的随机到达

### 直观理解

假设平均速率是 4 req/s：

* **严格均匀**：每 0.25 秒来 1 个
* **Poisson / burstiness=1.0**：有时 0.05 秒连续来两个，有时 0.4 秒一个都不来，但长期平均还是 4 req/s

所以 `burstiness` 是在控制“流量是否成团涌入”。

### 实验里怎么用

如果你的目标是复现“Poisson arrival”，那就固定：

```bash
--burstiness 1.0
```

不要乱改。

---

## 4）`--max-concurrency 256`

这个参数的意思是：

> **同一时刻最多允许多少个请求处于执行中。**

官方定义很明确：
`--request-rate` 控制“请求发起速率”，而 `--max-concurrency` 控制“同时允许多少请求实际在跑”。如果服务端处理不够快，那么实际请求速率可能低于你设定的 `--request-rate`。([vLLM][1])

这是你最容易混淆的点，我重点解释一下。

---

# `request-rate` 和 `max-concurrency` 的区别

这两个不是一回事。

## `request-rate`

控制的是：

> **请求“到来”的速度**

就像水龙头流量。

## `max-concurrency`

控制的是：

> **桶里同时最多能装多少个活跃请求**

就像闸门上限。

---

## 一个具体例子

假设：

```bash
--request-rate 4
--max-concurrency 256
```

含义是：

* benchmark 试图按平均每秒 4 个请求的速度发请求
* 只要当前正在执行的请求数没超过 256，就继续发
* 对大多数单机实验来说，256 基本相当于“几乎不限制”

如果改成：

```bash
--request-rate 4
--max-concurrency 2
```

那就表示：

* benchmark 仍然希望每秒发 4 个
* 但同一时刻只允许 2 个请求在飞
* 如果这 2 个请求还没结束，新请求就得等着
* 所以**实际到达到服务端的速率可能低于 4 req/s**。官方文档也明确说了这一点。([vLLM][1])

---

## 你这组参数 `256` 是什么意思

通常表示：

> 我不希望 benchmark 客户端的人为卡住并发，我希望尽量把瓶颈留给服务端自己。

所以 `256` 更像一个“放得很宽”的上限。

### 什么时候应该把它调小

如果你想模拟上层网关、前端服务、API gateway 的限流行为，就可以故意把它调小，比如：

* `8`
* `16`
* `32`

这样会更像真实系统里“前面还有一道并发门”。

---

# 这 4 个参数放在一起该怎么理解

你的这组：

```bash
--num-prompts 500
--request-rate 4
--burstiness 1.0
--max-concurrency 256
```

组合起来，完整意思就是：

> 从 ShareGPT 中取 500 个请求样本；
> benchmark 以平均每秒 4 个请求的速度发起它们；
> 请求到达模式按 Poisson 风格随机波动；
> 同时最多允许 256 个请求并发执行，因此几乎不会由客户端并发上限限制吞吐。([vLLM][1])

---

# 再讲 `--save-result` 和 `--save-detailed`

---

## `--save-result`

这个参数表示：

> **把 benchmark 的结果保存到文件里。**

通常保存的是一份**聚合后的结果摘要**，例如：

* 吞吐
* 平均 TTFT
* 平均 TPOT
* 平均 ITL
* E2EL
* P50 / P95 / P99 等

也就是你最后做表格、做总体比较最常用的那份结果。文档里对应的就是“save benchmark results”。([vLLM][1])

---

## `--save-detailed`

这个参数表示：

> **除了总体摘要，再额外保存每个请求级别的详细信息。**

官方文档说明它会保存 per-request information，例如 response、error、ttfts、tpots 等。([vLLM][1])

也就是说，开了它以后，你拿到的不只是“整体平均值”，还会有更细的内容，比如某个请求：

* 输入 token 数
* 输出 token 数
* 它自己的 TTFT
* 它自己的 TPOT
* 它自己的错误信息
* 有时还包括响应文本或其它请求级字段

---

# 如果去掉 `--save-detailed` 会怎样

直接说结论：

> **benchmark 仍然能跑，整体结果仍然能保存；但你会失去请求级明细。**

也就是说：

## 去掉后你仍然能得到

* 总体平均 latency
* 各种 percentile
* 总吞吐
* 你的结果文件

## 去掉后你得不到

* 每条请求的 TTFT / TPOT / ITL 明细
* 每条请求的输入输出 token 统计
* 针对失败请求的逐条排查信息
* 你自定义衍生指标所需的底层数据

---

# 为什么你这个实验其实建议保留 `--save-detailed`

因为你前面要测的 3 个指标里，有一个不是官方直接现成给的：

* Per-Token Latency → 可以直接用 ITL
* Per Output Token Latency → 可以直接用 TPOT
* **Per Input Token Latency** → 你通常要自己算，比如 `TTFT / input_tokens`

而这个“自己算”的过程，就依赖每个请求级别的详细记录。
所以如果你去掉 `--save-detailed`，你大概率就只能拿到 overall mean_ttft_ms，却拿不到逐请求的 input_tokens 对齐结果，自定义 `Per Input Token Latency` 会变困难甚至做不了。这个推断基于文档对 `save-detailed` 的定义，以及 `save-result` 主要保存 benchmark summary 的用途。([vLLM][1])

---

# 一句话区分这两个参数

你可以这样记：

* `--save-result`：**保存“总成绩单”**
* `--save-detailed`：**保存“每道题的答题记录”**

---

# 什么时候可以不加 `--save-detailed`

下面这种情况可以不加：

* 你只关心总体吞吐和总体 latency
* 你只想看 p50 / p95 / p99
* 你不打算后处理每条请求
* 你不打算排查单个异常请求

---

# 什么时候强烈建议加 `--save-detailed`

下面这种情况建议保留：

* 你要自己定义指标
* 你要按请求长度分桶分析
* 你要画更细粒度的图
* 你要排查为什么某些请求特别慢
* 你要检查失败请求长什么样

你的实验属于这一类，所以我建议你继续保留。

---

# 你这几个参数在实验里的推荐理解

最后我给你一个最实用的实验视角：

## 固定这些不变

```bash
--num-prompts 500
--burstiness 1.0
--max-concurrency 256
--save-result
--save-detailed
```

## 只扫这个

```bash
--request-rate 1
--request-rate 2
--request-rate 4
--request-rate 8
--request-rate 16
```

这样你得到的图就能干净地表达：

> 在 Poisson 到达、500 条样本、客户端并发几乎不设限的条件下，系统 latency 随 request rate 上升如何变化。([vLLM][1])

---

如果你愿意，我下一条可以继续直接给你画一张“`num-prompts / request-rate / burstiness / max-concurrency` 之间关系的时序示意图”，你会一下子彻底看懂。

[1]: https://docs.vllm.ai/en/latest/cli/bench/serve/?utm_source=chatgpt.com "vllm bench serve - vLLM"


---

## 版本 B：如果你测的是 completion 模型

把核心几项改成：

```bash
--backend openai \
--endpoint /v1/completions
```

其余不变。

---

# 五、如何理解 benchmark 输出

vLLM benchmark 文档明确展示了 TTFT、TPOT、ITL 的输出格式，也支持用 `--percentile-metrics` 和 `--metric-percentiles` 指定统计项。([vLLM][2])

你最终会拿到一批 `results_raw/sharegpt_rate_*.json` 文件。
其中通常有两层信息：

1. **整体汇总指标**

   * `mean_ttft_ms`
   * `mean_tpot_ms`
   * `mean_itl_ms`
   * 以及 p50/p95/p99 等

2. **详细请求级信息**（因为你开了 `--save-detailed`）

   * 每个请求的输入 token 数
   * 输出 token 数
   * ttft / tpot / itl 等明细

`--save-detailed` 官方说明就是会保存 “per request information such as response, error, ttfts, tpots, etc.”。([vLLM][4])

---

# 六、解析与画图脚本：parse_and_plot.py

这个脚本做三件事：

1. 读取每个 request rate 的结果 json
2. 计算：

   * `Per-Token Latency = mean ITL`
   * `Per Input Token Latency = mean(TTFT / input_tokens)`
   * `Per Output Token Latency = mean TPOT`
3. 画图：

   * 横轴：Request Rate
   * 纵轴：Latency (ms)

> 这个脚本写得比较“容错”，因为不同 vLLM 版本保存 detailed json 的字段名可能略有差异。

```python
import json
import math
import glob
import os
import re
from statistics import mean
import pandas as pd
import matplotlib.pyplot as plt


RAW_DIR = "results_raw"
AGG_DIR = "results_agg"
os.makedirs(AGG_DIR, exist_ok=True)


def safe_get(d, keys, default=None):
    """Try multiple candidate keys."""
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def flatten_requests(obj):
    """
    Try to locate per-request detailed records from possible json layouts.
    """
    candidates = [
        "requests",
        "detailed_results",
        "per_request_results",
        "request_results",
        "detailed",
        "results",
    ]
    for k in candidates:
        v = obj.get(k)
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            return v

    # Some versions may nest
    for outer_k, outer_v in obj.items():
        if isinstance(outer_v, dict):
            for k in candidates:
                v = outer_v.get(k)
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                    return v
    return []


def extract_input_tokens(req):
    return safe_get(req, [
        "prompt_len",
        "input_len",
        "num_input_tokens",
        "prompt_tokens",
        "input_tokens",
    ], default=None)


def extract_output_tokens(req):
    return safe_get(req, [
        "output_len",
        "completion_tokens",
        "num_output_tokens",
        "output_tokens",
        "generated_tokens",
    ], default=None)


def extract_ttft_ms(req):
    return safe_get(req, [
        "ttft_ms",
        "ttft",
        "first_token_latency_ms",
    ], default=None)


def extract_tpot_ms(req):
    return safe_get(req, [
        "tpot_ms",
        "mean_tpot_ms",
        "tpot",
    ], default=None)


def extract_itl_ms(req):
    # itl may be a scalar mean or a list of inter-token latencies
    scalar = safe_get(req, [
        "itl_ms",
        "mean_itl_ms",
        "itl",
    ], default=None)
    if scalar is not None:
        if isinstance(scalar, (int, float)):
            return float(scalar)
        if isinstance(scalar, list) and len(scalar) > 0:
            vals = [float(x) for x in scalar if isinstance(x, (int, float))]
            return mean(vals) if vals else None

    for k in ["itls", "itl_list", "inter_token_latencies_ms"]:
        v = req.get(k)
        if isinstance(v, list) and len(v) > 0:
            vals = [float(x) for x in v if isinstance(x, (int, float))]
            return mean(vals) if vals else None
    return None


def extract_summary_metric(obj, keys):
    for k in keys:
        if k in obj:
            return obj[k]
    # nested summary block
    for outer_k, outer_v in obj.items():
        if isinstance(outer_v, dict):
            for k in keys:
                if k in outer_v:
                    return outer_v[k]
    return None


rows = []

for path in sorted(glob.glob(os.path.join(RAW_DIR, "sharegpt_rate_*.json"))):
    filename = os.path.basename(path)
    m = re.search(r"sharegpt_rate_([0-9.]+)\.json", filename)
    if not m:
        continue
    request_rate = float(m.group(1))

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # First, try official summary fields
    mean_itl = extract_summary_metric(obj, ["mean_itl_ms"])
    mean_tpot = extract_summary_metric(obj, ["mean_tpot_ms"])

    # Detailed per-request calculation for custom metric
    reqs = flatten_requests(obj)

    pitl_values = []   # Per Input Token Latency = TTFT / input_tokens
    itl_values = []
    tpot_values = []

    for req in reqs:
        input_tokens = extract_input_tokens(req)
        output_tokens = extract_output_tokens(req)
        ttft_ms = extract_ttft_ms(req)
        tpot_ms = extract_tpot_ms(req)
        itl_ms = extract_itl_ms(req)

        if input_tokens is not None and ttft_ms is not None and input_tokens > 0:
            pitl_values.append(float(ttft_ms) / float(input_tokens))

        if tpot_ms is not None:
            tpot_values.append(float(tpot_ms))

        if itl_ms is not None:
            itl_values.append(float(itl_ms))

    # Prefer detailed averages if available; otherwise fall back to summary
    per_token_latency_ms = mean(itl_values) if itl_values else mean_itl
    per_output_token_latency_ms = mean(tpot_values) if tpot_values else mean_tpot
    per_input_token_latency_ms = mean(pitl_values) if pitl_values else None

    rows.append({
        "request_rate": request_rate,
        "per_token_latency_ms": per_token_latency_ms,
        "per_input_token_latency_ms": per_input_token_latency_ms,
        "per_output_token_latency_ms": per_output_token_latency_ms,
        "num_requests_with_pitl": len(pitl_values),
        "num_requests_with_itl": len(itl_values),
        "num_requests_with_tpot": len(tpot_values),
    })

df = pd.DataFrame(rows).sort_values("request_rate")
csv_path = os.path.join(AGG_DIR, "latency_vs_request_rate.csv")
df.to_csv(csv_path, index=False)

print("Saved aggregated csv:", csv_path)
print(df)

# Plot 1: three metrics in one figure
plt.figure(figsize=(8, 5))
plt.plot(df["request_rate"], df["per_token_latency_ms"], marker="o", label="Per-Token Latency (ITL)")
plt.plot(df["request_rate"], df["per_input_token_latency_ms"], marker="s", label="Per Input Token Latency (TTFT/InputTokens)")
plt.plot(df["request_rate"], df["per_output_token_latency_ms"], marker="^", label="Per Output Token Latency (TPOT)")
plt.xlabel("Request Rate (req/s)")
plt.ylabel("Latency (ms)")
plt.title("Latency vs Request Rate")
plt.grid(True, alpha=0.3)
plt.legend()
fig1 = os.path.join(AGG_DIR, "latency_vs_request_rate.png")
plt.tight_layout()
plt.savefig(fig1, dpi=200)
plt.close()

# Plot 2: separate figures if you want cleaner paper-style plots
metric_map = {
    "per_token_latency_ms": "Per-Token Latency (ITL)",
    "per_input_token_latency_ms": "Per Input Token Latency (TTFT/InputTokens)",
    "per_output_token_latency_ms": "Per Output Token Latency (TPOT)",
}

for col, title in metric_map.items():
    plt.figure(figsize=(7, 4.5))
    plt.plot(df["request_rate"], df[col], marker="o")
    plt.xlabel("Request Rate (req/s)")
    plt.ylabel("Latency (ms)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(AGG_DIR, f"{col}.png")
    plt.savefig(out, dpi=200)
    plt.close()

print("Plots saved to:", AGG_DIR)
```

运行：

```bash
python parse_and_plot.py
```

---

# 七、你最终会得到什么

运行完成后，你会得到：

## 1）聚合表格

`results_agg/latency_vs_request_rate.csv`

类似这样：

```csv
request_rate,per_token_latency_ms,per_input_token_latency_ms,per_output_token_latency_ms
0.5,6.8,0.42,7.0
1,7.1,0.44,7.3
2,7.9,0.49,8.2
4,9.6,0.58,10.0
8,14.2,0.87,14.9
12,22.8,1.40,24.1
16,38.5,2.31,40.3
```

## 2）图

* `results_agg/latency_vs_request_rate.png`
* `results_agg/per_token_latency_ms.png`
* `results_agg/per_input_token_latency_ms.png`
* `results_agg/per_output_token_latency_ms.png`

---

# 八、实验时的注意事项

## 1）Poisson 到达过程要怎么设

你给出的描述里是“用不同 lambda 的 Poisson process 模拟请求到达”。

在 vLLM 里：

* `--request-rate λ`
* `--burstiness 1.0`

这就是 Poisson 到达；如果 `burstiness != 1.0`，就变成 gamma-distributed 间隔。([vLLM][4])

---

## 2）为什么建议保存 detailed json

因为：

* 官方直接给你 `ITL` 和 `TPOT`
* 但 **Per Input Token Latency** 不是内置指标
* 所以必须基于每个请求的 `TTFT / input_tokens` 自己算

而 `--save-detailed` 正是为保存请求级明细准备的。([vLLM][4])

---

## 3）为什么我用 ITL 作为 Per-Token Latency

因为你写的是 **Per-Token Latency**，这是最容易和官方指标对齐的定义。
如果你想把它定义成“每个生成 token 的平均耗时”，那就更接近 TPOT。
所以你最好在报告里明确写一句：

* **Per-Token Latency = ITL**
* **Per Output Token Latency = TPOT**
* **Per Input Token Latency = TTFT / input_tokens**

这样不会歧义。

---

## 4）不同请求率下，别让样本数太少

建议：

* `NUM_PROMPTS >= 300`
* 更稳一点用 `500` 或 `1000`

官方默认 `--num-prompts` 是 1000。([vLLM][4])

---

## 5）建议固定随机性

为了可复现：

```bash
--seed 0
```

并且：

```bash
--disable-shuffle
```

也可以加上，这样不同轮次更稳定。文档中支持 `--seed` 和 `--disable-shuffle`。([vLLM][4])

---

# 九、一套更稳的最终命令模板

如果你要直接拷贝一套“比较论文式”的实验配置，我建议用这个：

```bash
vllm bench serve \
  --backend openai-chat \
  --host 127.0.0.1 \
  --port 8000 \
  --endpoint /v1/chat/completions \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset-name sharegpt \
  --dataset-path data/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 500 \
  --request-rate 4 \
  --burstiness 1.0 \
  --max-concurrency 256 \
  --seed 0 \
  --disable-shuffle \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,95,99 \
  --save-result \
  --save-detailed \
  --result-filename results_raw/sharegpt_rate_4.json
```

---

# 十、如果你想要更严格的“Per Input Token Latency”

上面我给的是：

[
\text{PITL} = \frac{TTFT}{input_tokens}
]

这是最常用、最好解释的定义。

但你也可以用另一种定义：

[
\text{PITL}_{alt} = \frac{E2EL}{input_tokens}
]

不过这个会把 decode 阶段也混进去，不再只反映 prefill，所以我**不建议**。

---

# 十一、你写实验报告时可以直接这样写

你可以直接用下面这段表述：

> We evaluate vLLM under the online serving scenario by launching the OpenAI-compatible API server and sampling prompts from the ShareGPT dataset. Request arrivals are generated using a Poisson process with rate (\lambda), implemented via `vllm bench serve --request-rate λ --burstiness 1.0`. For each request rate, we measure:
> (1) Per-Token Latency, defined as inter-token latency (ITL);
> (2) Per Input Token Latency, defined as TTFT divided by the number of input tokens;
> (3) Per Output Token Latency, defined as time per output token (TPOT).
> We then plot latency (ms) against request rate (req/s). ([vLLM][4])

---

如果你愿意，我下一条可以继续直接给你一份：
**“适配单机单卡 / 双卡 / 8 卡”的完整实验脚本模板**，把模型名、GPU 数、结果目录都参数化。

[1]: https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py "vllm/benchmarks/benchmark_serving.py at main · vllm-project/vllm · GitHub"
[2]: https://docs.vllm.ai/en/latest/benchmarking/cli/ "Benchmark CLI - vLLM"
[3]: https://docs.vllm.ai/en/latest/serving/openai_compatible_server/?utm_source=chatgpt.com "OpenAI-Compatible Server - vLLM"
[4]: https://docs.vllm.ai/en/latest/cli/bench/serve/ "vllm bench serve - vLLM"
