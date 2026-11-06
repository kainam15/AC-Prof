"""Generate formal-result comparisons or streaming load point/sweep reports."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from acprof.analysis.resource_comparison import REPORT_PURPOSES
from acprof.system.cleanup import format_exception_notes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="*", type=Path, help="实验目录、CSV、流式 load.json 或 sweep.json")
    parser.add_argument("--left", type=Path, action="append", help="基线组独立实验，可重复")
    parser.add_argument("--right", type=Path, action="append", help="对照组独立实验，可重复")
    parser.add_argument("--baseline", help="唯一 config_id（或仅有一个配置的 run_id）")
    parser.add_argument("--comparison-purpose", choices=REPORT_PURPOSES)
    parser.add_argument("--metric", action="append", help="窗口统计指标，可重复")
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--resamples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--output", type=Path, help="新 HTML 路径；默认 report.html、load_report.html 或 load_sweep_report.html")
    args = parser.parse_args(argv)
    if bool(args.left) != bool(args.right):
        parser.error("--left and --right must be provided together")
    if not args.results and not args.left:
        parser.error("provide result sources or both comparison groups")
    from acprof.host.run_state import offline_analysis_lock
    try:
        with offline_analysis_lock():
            selected = args.results or [*args.left, *args.right]
            first = selected[0]
            sweep_inputs = [
                path for path in [*selected, *(args.left or []), *(args.right or [])]
                if path.name == "sweep.json" or (path.is_dir() and (path / "sweep.json").is_file())
            ]
            comparison_requested = args.baseline or args.comparison_purpose or args.left or any(
                value is not None for value in (args.metric, args.confidence, args.resamples, args.seed, args.block_size)
            )
            load_inputs = [
                path for path in selected
                if path.name == "load.json" or (path.is_dir() and (path / "load.json").is_file())
            ]
            if sweep_inputs:
                if len(selected) != 1 or load_inputs or comparison_requested:
                    raise ValueError("sweep report supports one sweep.json without mixed inputs or comparison options")
                from acprof.plotting.load_sweep import write_load_sweep_report
                output = args.output or (first if first.is_dir() else first.parent) / "load_sweep_report.html"
                summary = write_load_sweep_report(first, output)
                print("ACPROF_REPORT " + json.dumps({"report_path": str(output.resolve()), **summary}, ensure_ascii=False))
                print(f"流式负载扫描报告已生成：{output.resolve()}（{summary['point_count']} 个负载点）")
                return 0
            if load_inputs:
                if len(selected) != 1 or comparison_requested:
                    raise ValueError("capacity report supports one load.json without baseline or comparison options")
                from acprof.plotting.load_report import write_load_report
                output = args.output or (first if first.is_dir() else first.parent) / "load_report.html"
                write_load_report(first, output)
                print("ACPROF_REPORT " + json.dumps({
                    "report_path": str(output.resolve()), "kind": "stream_load_capacity",
                }, ensure_ascii=False))
                print(f"单负载点容量报告已生成：{output.resolve()}")
                return 0
            from acprof.analysis.model import load_analysis
            from acprof.plotting.report import write_report

            output = args.output or (first if first.is_dir() else first.parent) / "report.html"
            if args.left:
                from acprof.analysis.report_comparison import comparison_sources
                selected = list(dict.fromkeys(comparison_sources([*selected, *args.left, *args.right])))
            model = load_analysis(selected)
            write_report(
                model, output, baseline=args.baseline,
                comparison_purpose=args.comparison_purpose or "within-experiment",
                statistics_options={key: value for key, value in {
                    "metrics": args.metric, "confidence": args.confidence, "resamples": args.resamples,
                    "seed": args.seed, "block_size": args.block_size,
                }.items() if value is not None},
                comparison_groups=(args.left, args.right) if args.left else None,
            )
            print("ACPROF_REPORT " + json.dumps({
                "report_path": str(output.resolve()), "config_count": len(model.configs),
                "valid_windows": sum(config["valid_windows"] for config in model.configs),
                "excluded_rows": sum(config["excluded_rows"] for config in model.configs),
                "quality_status": sorted({config["quality_status"] for config in model.configs}),
                "measurement_status": sorted({config["measurement_status"] for config in model.configs}),
            }, ensure_ascii=False))
            print(f"报告已生成：{output.resolve()}（{len(model.configs)} 个配置）")
    except (OSError, ValueError, RuntimeError, ImportError, csv.Error) as error:
        parser.exit(1, f"报告生成失败：{format_exception_notes(error)}\n")
    return 0
