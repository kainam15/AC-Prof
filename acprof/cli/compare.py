"""Compare latency and energy across independent repetitions of two experiments."""
import argparse
import json
from pathlib import Path

from acprof.analysis.independent_comparison import compare_experiments
from acprof.artifacts import atomic_write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, action="append", required=True, help="左组独立实验，可重复")
    parser.add_argument("--right", type=Path, action="append", required=True, help="右组独立实验，可重复")
    parser.add_argument("--metric", action="append")
    parser.add_argument("--purpose", choices=("same-hardware", "cross-hardware"), default="same-hardware")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error("output must be a new file")
    try:
        report = compare_experiments(args.left, args.right,
            metrics=args.metric or ["latency_app_s", "container_attributed_energy_eff_j"],
            purpose=args.purpose, confidence=args.confidence, resamples=args.resamples, seed=args.seed)
    except (OSError, ValueError) as error:
        parser.exit(1, f"比较失败：{error}\n")
    if args.output:
        atomic_write_json(args.output, report)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["status"] == "compatible" else 1
