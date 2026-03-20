from __future__ import annotations

import argparse
from pathlib import Path

from benchmark import (
    BenchmarkCase,
    benchmark_single_case,
    write_single_result_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single SwiftLLM offline benchmark case in its own process.")
    parser.add_argument("--model-path", type=Path, required=True, help="Local Hugging Face model directory.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing generated case input data.")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size for this benchmark case.")
    parser.add_argument("--input-len", type=int, required=True, help="Input length for this benchmark case.")
    parser.add_argument("--warmup-runs", type=int, required=True, help="Number of warmup runs before timing.")
    parser.add_argument("--measure-runs", type=int, required=True, help="Number of timed runs for this case.")
    parser.add_argument("--output-path", type=Path, required=True, help="Path to write the single-case JSON result.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case = BenchmarkCase(batch_size=args.batch_size, input_len=args.input_len)
    result = benchmark_single_case(
        model_path=args.model_path,
        data_dir=args.data_dir,
        case=case,
        warmup_runs=args.warmup_runs,
        measure_runs=args.measure_runs,
    )
    write_single_result_json(args.output_path, result)
    print(f"Wrote case result to {args.output_path}")


if __name__ == "__main__":
    main()
