"""Compare latency and energy across independent repetitions of two experiments."""
import argparse
import json
from pathlib import Path

from acprof.analysis.conditions import PURPOSES
from acprof.artifacts import atomic_write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, action="append", required=True, help="左组独立实验，可重复")
    parser.add_argument("--right", type=Path, action="append", required=True, help="右组独立实验，可重复")
    parser.add_argument("--metric", action="append")
    parser.add_argument("--purpose", choices=PURPOSES, default="same-hardware")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv-output", type=Path, help="独立比较 CSV 导出")
    parser.add_argument("--html-output", type=Path, help="统一离线 HTML 报告")
    args = parser.parse_args(argv)
    from acprof.host.run_state import RunStateError, offline_analysis_lock
    destinations = [path for path in (args.output, args.csv_output, args.html_output) if path is not None]
    if len({path.resolve() for path in destinations}) != len(destinations):
        parser.error("output destinations must be distinct")
    if any(path.exists() or path.is_symlink() for path in destinations):
        parser.error("output must be a new file")
    try:
        with offline_analysis_lock():
            from acprof.analysis.independent_comparison import compare_experiments

            report = compare_experiments(args.left, args.right,
                metrics=args.metric or ["latency_app_s", "container_attributed_energy_eff_j"],
                purpose=args.purpose, confidence=args.confidence, resamples=args.resamples, seed=args.seed)
            if args.html_output:
                from acprof.analysis.model import load_analysis
                from acprof.plotting.report import write_report
                write_report(load_analysis([*args.left, *args.right]), args.html_output,
                             comparison_purpose=args.purpose, independent_report=report,
                             statistics_options={"metrics": args.metric or ["latency_app_s", "container_attributed_energy_eff_j"],
                                                 "confidence": args.confidence, "resamples": args.resamples, "seed": args.seed})
                report["html_report_path"] = str(args.html_output.resolve())
            if args.csv_output:
                from acprof.analysis.export import export_analysis
                export_analysis(report, args.csv_output, format="csv")
            if args.output:
                atomic_write_json(args.output, report, overwrite=False)
            else:
                print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, RunStateError) as error:
        parser.exit(1, f"比较失败：{error}\n")
    return 0 if report["status"] == "compatible" else 1
