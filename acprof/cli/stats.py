"""将结果按独立测量窗口汇总，输出均值、标准差和 bootstrap 置信区间。"""
import argparse
import csv
import json
from contextlib import ExitStack
from pathlib import Path

from acprof.artifacts import atomic_write_json
from acprof.system.cleanup import format_exception_notes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="CSV 或结果目录")
    parser.add_argument("--metric", action="append", help="可重复指定，默认两种延迟和归因能耗")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=1, help="连续窗口的循环移动块长度")
    parser.add_argument("--precision-target", type=float,
                        help="Optional within-run CI half-width/mean target (0.05 = 5%%); "
                             "choose beforehand; not a stopping rule")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output", type=Path, help="保存 JSON；省略输出选项时输出到 stdout")
    output.add_argument("--output-dir", type=Path, help="按本地日期时间保存 JSON；内容相同时复用已有报告")
    args = parser.parse_args(argv)
    from acprof.host.run_state import RunStateError, offline_analysis_lock
    with ExitStack() as stack:
        try:
            stack.enter_context(offline_analysis_lock())
        except RunStateError as error:
            parser.exit(1, f"统计失败：{format_exception_notes(error)}\n")
        from acprof.analysis.statistics_report import (
            build_statistics_report,
            save_unique_statistics_report,
        )

        path = args.source / "result_layers.json" if args.source.is_dir() else args.source
        if args.output and (args.output.exists() or args.output.is_symlink() or args.output.resolve() == path.resolve()):
            parser.error("统计输出必须为新的文件，不能覆盖输入或已有产物")
        try:
            report = build_statistics_report(path, args.metric, confidence=args.confidence,
                                             resamples=args.resamples, seed=args.seed,
                                             block_size=args.block_size, precision_target=args.precision_target)
        except (ValueError, OSError, csv.Error) as error:
            parser.exit(1, f"统计失败：{format_exception_notes(error)}\n")
        try:
            if args.output_dir:
                saved, reused = save_unique_statistics_report(args.output_dir, report)
                print("ACPROF_STATS " + json.dumps({"report_path": str(saved), "reused": reused}, ensure_ascii=False))
            elif args.output:
                atomic_write_json(args.output, report, overwrite=False)
                print(f"已保存 {len(report['groups'])} 项窗口统计：{args.output}")
            else:
                print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        except (ValueError, OSError) as error:
            parser.exit(1, f"统计保存失败：{format_exception_notes(error)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
