"""将结果按独立测量窗口汇总，输出均值、标准差和 bootstrap 置信区间。"""
import argparse
import csv
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from acprof.analysis.audit import audit_result
from acprof.analysis.uncertainty import summarize_windows
from acprof.artifacts import atomic_write_json, read_json_object
from acprof.quality import QUALITY_FIELDS
from acprof.result_csv import read_result_csv_snapshot, result_csv_snapshot_unchanged


def _save_unique_report(directory: Path, report: dict) -> tuple[Path, bool]:
    """Compare complete JSON content before publishing a timestamped report."""
    import fcntl

    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, sort_keys=True, ensure_ascii=False, allow_nan=False)
    # Lock the directory itself: concurrent statistics processes share the check
    # and publish boundary without leaving a lock file among the reports.
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        for candidate in sorted(directory.glob("window-statistics-*.json")):
            try:
                existing = read_json_object(candidate, label="statistics report")
                identical = json.dumps(existing, sort_keys=True, ensure_ascii=False, allow_nan=False) == content
            except (FileNotFoundError, IsADirectoryError, ValueError):
                # Skip unusable candidates, not permission or storage failures.
                continue
            if identical:
                return candidate, True
        timestamp = datetime.now()
        destination = directory / f"window-statistics-{timestamp:%Y%m%d-%H%M%S-%f}.json"
        while destination.exists():
            timestamp += timedelta(microseconds=1)
            destination = directory / f"window-statistics-{timestamp:%Y%m%d-%H%M%S-%f}.json"
        atomic_write_json(destination, report)
        return destination, False
    finally:
        os.close(descriptor)


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
    path = args.source / "result_all.csv" if args.source.is_dir() else args.source
    if args.output and (args.output.resolve() == path.resolve() or args.output.exists()):
        parser.error("统计输出必须为新的文件，不能覆盖输入或已有产物")
    try:
        snapshot = read_result_csv_snapshot(path)
        report = summarize_windows(snapshot.rows, args.metric or ["latency_app_s", "latency_s", "container_attributed_energy_eff_j"],
                                   confidence=args.confidence, resamples=args.resamples, seed=args.seed,
                                   block_size=args.block_size, precision_target=args.precision_target)
        report["result_sha256"] = snapshot.sha256
        report["result_csv"] = str(path.resolve())
        audit = audit_result(path, result_snapshot=snapshot, verify_result_snapshot=False)
        report.update({key: audit[key] for key in (*QUALITY_FIELDS, "run_status", "measurement_status")})
        if not result_csv_snapshot_unchanged(snapshot):
            raise ValueError("统计期间结果 CSV 发生变化，请采集结束后重试")
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(1, f"统计失败：{error}\n")
    try:
        if args.output_dir:
            saved, reused = _save_unique_report(args.output_dir, report)
            print("ACPROF_STATS " + json.dumps({"report_path": str(saved), "reused": reused}, ensure_ascii=False))
        elif args.output:
            atomic_write_json(args.output, report)
            print(f"已保存 {len(report['groups'])} 项窗口统计：{args.output}")
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError) as error:
        parser.exit(1, f"统计保存失败：{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
