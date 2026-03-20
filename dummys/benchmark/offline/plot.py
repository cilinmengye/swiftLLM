from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ResultRow:
    batch_size: int
    input_len: int
    prefill_ms_mean: float
    prefill_ms_std: float
    decode_ms_mean: float
    decode_ms_std: float
    prefill_tokens_per_s: float
    decode_tokens_per_s: float


def read_results(csv_path: Path) -> list[ResultRow]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        ResultRow(
            batch_size=int(row["batch_size"]),
            input_len=int(row["input_len"]),
            prefill_ms_mean=float(row["prefill_ms_mean"]),
            prefill_ms_std=float(row["prefill_ms_std"]),
            decode_ms_mean=float(row["decode_ms_mean"]),
            decode_ms_std=float(row["decode_ms_std"]),
            prefill_tokens_per_s=float(row["prefill_tokens_per_s"]),
            decode_tokens_per_s=float(row["decode_tokens_per_s"]),
        )
        for row in rows
    ]


def plot_metric(rows: list[ResultRow], metric: str, output_path: Path, ylabel: str, title: str) -> None:
    import matplotlib.pyplot as plt

    ordered_rows = sorted(rows, key=lambda row: row.input_len)
    x_values = [row.input_len for row in ordered_rows]
    y_values = [getattr(row, metric) for row in ordered_rows]

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker="o")
    for row in ordered_rows:
        plt.annotate(
            f"B={row.batch_size}",
            (row.input_len, getattr(row, metric)),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )
    plt.xscale("log", base=2)
    plt.xlabel("input_len")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    default_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Plot SwiftLLM offline benchmark results.")
    parser.add_argument("--results-csv", type=Path, default=default_dir / "results.csv", help="CSV file produced by benchmark.py.")
    parser.add_argument("--prefill-output", type=Path, default=default_dir / "prefill.png", help="Output path for prefill plot.")
    parser.add_argument("--decode-output", type=Path, default=default_dir / "decode.png", help="Output path for decode plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_results(args.results_csv)
    plot_metric(rows, "prefill_ms_mean", args.prefill_output, "Prefill time (ms)", "SwiftLLM offline prefill")
    plot_metric(rows, "decode_ms_mean", args.decode_output, "Decode time (ms)", "SwiftLLM offline single-step decode")
    print(f"Wrote plots to {args.prefill_output} and {args.decode_output}")


if __name__ == "__main__":
    main()
