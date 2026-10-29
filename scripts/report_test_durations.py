"""Rank pytest evidence by cumulative file and parameterized test-group time."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def rank_durations(report: dict, *, top: int = 20) -> dict:
    """Produce cumulative rankings without assuming the evidence run succeeded."""
    if top < 1:
        raise ValueError("top must be positive")
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("expected AC-Prof pytest evidence schema v1")
    rows = report.get("tests")
    if not isinstance(rows, list):
        raise ValueError("tests must be a list")

    by_file: dict[str, list[float]] = defaultdict(list)
    by_group: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("test record has no valid ID")
        duration = row.get("duration_s")
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
            raise ValueError(f"invalid test duration for {row['id']}")
        nodeid = row["id"]
        by_file[nodeid.split("::", 1)[0]].append(duration)
        by_group[nodeid.split("[", 1)[0]].append(duration)

    def ordered(groups: dict[str, list[float]]) -> list[dict]:
        result = [{"name": key, "seconds": round(sum(values), 3), "cases": len(values)}
                  for key, values in groups.items()]
        return sorted(result, key=lambda row: (-row["seconds"], row["name"]))[:top]

    return {
        "successful": report.get("successful") is True,
        "timestamp": report.get("timestamp"),
        "duration_s": report.get("duration_s"),
        "recorded_cases": len(rows),
        "recorded_test_seconds": round(sum(sum(values) for values in by_file.values()), 3),
        "files": ordered(by_file),
        "groups": ordered(by_group),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path, help="AC-Prof pytest evidence JSON")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--json", action="store_true", help="Output machine-readable rankings")
    args = parser.parse_args(argv)
    try:
        summary = rank_durations(json.loads(args.evidence.read_text()), top=args.top)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    status = "complete" if summary["successful"] else "INCOMPLETE OR FAILED"
    print(f"Evidence: {args.evidence} ({status}, {summary['recorded_cases']} cases)")
    print(f"Recorded test phases: {summary['recorded_test_seconds']:.1f}s; "
          f"session wall time: {summary['duration_s']}s")
    for name in ("files", "groups"):
        print(f"\nSlowest {name} (cumulative phase time, seconds):")
        for row in summary[name]:
            print(f"{row['seconds']:9.2f}s  {row['cases']:5d} cases  {row['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
