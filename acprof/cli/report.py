"""Generate formal-result comparisons or a single-point streaming load report."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from acprof.analysis.conditions import PURPOSES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path, help="实验目录、CSV 或单个流式 load.json")
    parser.add_argument("--baseline", help="唯一 config_id（或仅有一个配置的 run_id）")
    parser.add_argument("--comparison-purpose", choices=PURPOSES)
    parser.add_argument("--output", type=Path, help="新 HTML 路径；默认 report.html 或 load_report.html")
    args = parser.parse_args(argv)
    from acprof.host.run_state import offline_analysis_lock
    try:
        with offline_analysis_lock():
            first = args.results[0]
            load_inputs = [
                path for path in args.results
                if path.name == "load.json" or (path.is_dir() and (path / "load.json").is_file())
            ]
            if load_inputs:
                if len(args.results) != 1 or args.baseline or args.comparison_purpose:
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
            model = load_analysis(args.results)
            write_report(
                model, output, baseline=args.baseline,
                comparison_purpose=args.comparison_purpose or "same-hardware",
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
        parser.exit(1, f"报告生成失败：{error}\n")
    return 0
