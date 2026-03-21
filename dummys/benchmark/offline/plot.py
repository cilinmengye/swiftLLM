import pandas as pd
import matplotlib.pyplot as plt

ROOT_PATH = "/home/yxlin/github/swift/swiftLLM/dummys/benchmark/offline"
OUTPUT_PATH = ROOT_PATH + "/figures/"
vllm = pd.read_csv(ROOT_PATH + "/vllm_result/results.csv")
swift = pd.read_csv(ROOT_PATH + "/swiftllm_result/results.csv")

for df in (vllm, swift):
    df["batch_size"] = df["batch_size"].astype(int)
    df["input_len"] = df["input_len"].astype(int)
    df["label"] = df.apply(
        lambda r: f"bs={r['batch_size']}, in={r['input_len']}", axis=1
    )

# 按 vllm 的顺序对齐
order = vllm[["batch_size", "input_len", "label"]].copy()
swift = swift.merge(order[["batch_size", "input_len"]], on=["batch_size", "input_len"], how="right")

x = range(len(order))
labels = order["label"].tolist()

def plot_compare(v_col, s_col, title, ylabel, out_file):
    plt.figure(figsize=(14, 6))
    plt.plot(x, vllm[v_col].values, marker="o", label="vLLM")
    plt.plot(x, swift[s_col].values, marker="o", label="swiftLLM")
    plt.xticks(list(x), labels, rotation=45, ha="right")
    plt.xlabel("batch_size, input_len")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()

plot_compare("prefill_s_mean", "prefill_s_mean",
             "Prefill Time Usage Comparison", "time_usage (s)",
             OUTPUT_PATH + "prefill_time_comparison.png")

plot_compare("prefill_tokens_per_s", "prefill_tokens_per_s",
             "Prefill Throughput Comparison", "tokens_per_s",
             OUTPUT_PATH + "prefill_tokens_per_s_comparison.png")

plot_compare("decode_s_mean", "decode_s_mean",
             "Decode Time Usage Comparison", "time_usage (s)",
             OUTPUT_PATH + "decode_time_comparison.png")

plot_compare("decode_tokens_per_s", "decode_tokens_per_s",
             "Decode Throughput Comparison", "tokens_per_s",
             OUTPUT_PATH + "decode_tokens_per_s_comparison.png")