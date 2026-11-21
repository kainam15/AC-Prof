"""Compare latency and energy across independent repetitions of two experiments."""
import argparse
import json
from contextlib import nullcontext
from pathlib import Path

from acprof.analysis.conditions import PURPOSES
from acprof.artifacts import atomic_write_json
from acprof.system.cleanup import format_exception_notes


def _output_paths(args) -> dict[Path, Path]:
    from acprof.analysis.export import validate_export_format
    from acprof.results.outputs import validate_comparison_destinations

    paths = [path for path in (args.csv_output, args.output, args.html_output) if path is not None]
    validate_comparison_destinations(paths)
    if args.csv_output is not None:
        validate_export_format(args.csv_output, "csv")
    return {path: path.parent.resolve() / path.name for path in paths}


def _check_output_directories(destinations: dict[Path, Path]) -> None:
    for requested, bound in destinations.items():
        try:
            current = requested.parent.resolve()
        except (OSError, RuntimeError) as error:
            raise ValueError(f"comparison output directory changed: {requested.parent}: {error}") from error
        if current != bound.parent:
            raise ValueError(f"comparison output directory changed: {requested.parent}")


def _export_comparison(args, report, metrics, destinations):
    from acprof.analysis.publication import ComparisonPublication

    _check_output_directories(destinations)
    exported = report
    if args.html_output is not None:
        exported = {**report, "html_report_path": str(destinations[args.html_output])}
    stdout = json.dumps(exported, ensure_ascii=False, indent=2, allow_nan=False) if args.output is None else None
    publication = ComparisonPublication() if len(destinations) > 1 else nullcontext()
    with publication as batch:
        # Publish the HTML only after its associated exports and window inputs.
        paths = {requested: batch.stage(path) if batch is not None else path for requested, path in destinations.items()}
        if args.csv_output is not None:
            from acprof.analysis.export import export_analysis
            export_analysis(exported, paths[args.csv_output], format="csv")
        if args.output is not None:
            atomic_write_json(paths[args.output], exported, overwrite=False)
        if args.html_output is not None:
            from acprof.analysis.model import load_analysis
            from acprof.plotting.report import write_report
            write_report(load_analysis([*args.left, *args.right]), paths[args.html_output],
                         comparison_purpose=args.purpose, independent_report=report,
                         statistics_options={"metrics": metrics, "confidence": args.confidence,
                                             "resamples": args.resamples, "seed": args.seed})
        if batch is not None:
            _check_output_directories(destinations)
            batch.publish()
    if stdout is not None:
        print(stdout)


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
    try:
        try:
            destinations = _output_paths(args)
        except ValueError as error:
            parser.error(str(error))
        with offline_analysis_lock():
            from acprof.analysis.independent_comparison import compare_experiments
            from acprof.results.outputs import validate_analysis_destinations

            validate_analysis_destinations([*args.left, *args.right], [*destinations, *destinations.values()])
            metrics = args.metric or ["latency_app_s", "container_attributed_energy_eff_j"]
            report = compare_experiments(args.left, args.right, metrics=metrics, purpose=args.purpose,
                                         confidence=args.confidence, resamples=args.resamples, seed=args.seed)
            _export_comparison(args, report, metrics, destinations)
    except (OSError, ValueError, RunStateError) as error:
        parser.exit(1, f"比较失败：{format_exception_notes(error)}\n")
    return 0 if report["status"] == "compatible" else 1
