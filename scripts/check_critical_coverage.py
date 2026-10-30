"""Fail closed when host branch coverage regresses on measurement-critical modules.

The per-file floors are conservative bounds below the reproducible 2026-10-18
focused host regression baseline. They do not assert Docker/GPU/subprocess
coverage and must not be presented as hardware validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Rounded down below the observed focused-test branch ratios.
BRANCH_FLOORS = {
    "acprof/results/csv.py": 65.0,
    "acprof/results/layers.py": 70.0,
    "acprof/host/matrix_plan.py": 60.0,
    "acprof/host/run_state.py": 65.0,
    "acprof/monitors/energy_cpu.py": 55.0,
    "acprof/host/result_cleanup.py": 75.0,
    "acprof/host/client.py": 65.0,
    # Preserve the client floor when window lifecycle and row reduction move.
    "acprof/host/measurement/execution.py": 65.0,
    "acprof/host/measurement/metrics.py": 65.0,
}


def check_report(report: dict) -> list[str]:
    files = report.get("files")
    if not isinstance(files, dict):
        return ["coverage report is missing file data"]
    errors = []
    for filename, floor in BRANCH_FLOORS.items():
        record = files.get(filename)
        summary = record.get("summary") if isinstance(record, dict) else None
        if not isinstance(summary, dict):
            errors.append(f"{filename}: missing coverage data")
            continue
        total, covered = summary.get("num_branches"), summary.get("covered_branches")
        if (type(total) is not int or type(covered) is not int
                or total <= 0 or covered < 0 or covered > total):
            errors.append(f"{filename}: invalid branch counters")
            continue
        ratio = 100 * covered / total
        if ratio < floor:
            errors.append(f"{filename}: {covered}/{total} branches ({ratio:.2f}%) below {floor:.1f}%")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        errors = check_report(json.loads(args.report.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Cannot read coverage report: {exc}\n")
    if errors:
        parser.exit(1, "Critical branch coverage gate failed:\n" + "\n".join(errors) + "\n")
    print("Critical branch coverage gate passed (host tests only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
