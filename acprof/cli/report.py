"""Generate Comparison Matrix, Pareto and Scaling views from existing result CSVs."""
from __future__ import annotations

import argparse
import csv
import os
from contextlib import nullcontext
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path, help="实验目录或 CSV，可指定多个")
    parser.add_argument("--baseline", help="唯一 config_id（或仅有一个配置的 run_id）")
    parser.add_argument("--output", type=Path, help="新 HTML 路径；默认首个实验目录下 report.html")
    args = parser.parse_args(argv)
    first = args.results[0]
    output = args.output or (first if first.is_dir() else first.parent) / "report.html"
    try:
        from acprof.analysis.model import load_analysis
        from acprof.plotting.report import write_report
        guard = nullcontext()
        if os.name == "posix":
            from acprof.host.run_state import MeasurementLock
            guard = MeasurementLock()
        with guard:
            model = load_analysis(args.results)
            write_report(model, output, baseline=args.baseline)
    except (OSError, ValueError, RuntimeError, ImportError, csv.Error) as error:
        parser.exit(1, f"报告生成失败：{error}\n")
    print(f"报告已生成：{output.resolve()}（{len(model.configs)} 个配置）")
    return 0
