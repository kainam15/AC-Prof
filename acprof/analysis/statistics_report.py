"""Window statistics and durable deduplicated publication, outside collection."""
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from acprof.analysis.audit import audit_result
from acprof.analysis.uncertainty import summarize_windows
from acprof.artifacts import atomic_write_json, read_json_object
from acprof.quality import QUALITY_FIELDS
from acprof.results.csv import read_result_csv_snapshot, result_csv_snapshot_unchanged


def build_statistics_report(path: Path, metrics=None, *, confidence=0.95, resamples=5000,
                            seed=0, block_size=1, precision_target=None) -> dict:
    """Compute and audit one immutable snapshot; callers own measurement exclusion."""
    snapshot = read_result_csv_snapshot(path)
    report = summarize_windows(snapshot.rows, metrics or ["latency_app_s", "latency_s", "container_attributed_energy_eff_j"],
                               confidence=confidence, resamples=resamples, seed=seed,
                               block_size=block_size, precision_target=precision_target)
    report["result_sha256"] = snapshot.sha256
    report["result_csv"] = str(path.resolve())
    audit = audit_result(path, result_snapshot=snapshot, verify_result_snapshot=False)
    report.update({key: audit[key] for key in (*QUALITY_FIELDS, "run_status", "measurement_status")})
    if not result_csv_snapshot_unchanged(snapshot):
        raise ValueError("统计期间结果 CSV 发生变化，请采集结束后重试")
    return report


def save_unique_statistics_report(directory: Path, report: dict) -> tuple[Path, bool]:
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
        while True:
            destination = directory / f"window-statistics-{timestamp:%Y%m%d-%H%M%S-%f}.json"
            if not (destination.exists() or destination.is_symlink()):
                try:
                    atomic_write_json(destination, report, overwrite=False)
                except FileExistsError:
                    # Writers outside this directory lock can claim the name
                    # after our check. Retry only a real destination collision.
                    if not (destination.exists() or destination.is_symlink()):
                        raise
                else:
                    return destination, False
            timestamp += timedelta(microseconds=1)
    finally:
        os.close(descriptor)
