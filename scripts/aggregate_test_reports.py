"""Verify host test shard coverage before declaring the CI suite complete."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from acprof.artifacts import atomic_write_json  # noqa: E402 -- 脚本先设置仓库导入路径。

OUTCOMES = {"passed": "passed", "failed": "failed", "error": "errors", "skipped": "skipped",
            "expected_failure": "expected_failures", "unexpected_success": "unexpected_successes"}


def aggregate_reports(reports, *, python_versions, shard_count):
    if shard_count < 1 or not python_versions or len(set(python_versions)) != len(python_versions):
        raise ValueError("require positive shard count and unique Python versions")
    errors, versions = [], {}
    groups = defaultdict(list)
    for report in reports:
        if not isinstance(report, dict):
            errors.append("report must be an object")
            continue
        version = ".".join(str(report.get("python", "")).split(".")[:2])
        if version not in python_versions:
            errors.append(f"unexpected Python version: {version}")
        else:
            groups[version].append(report)
    for version in python_versions:
        rows = groups[version]
        shards, records, totals = {}, [], Counter()
        signature = None
        for report in rows:
            try:
                shard = report["shard"]
                if not isinstance(shard, dict) or not isinstance(report.get("counts"), dict):
                    raise ValueError("shard and counts must be objects")
                if not isinstance(report.get("tests"), list) or not all(isinstance(item, dict) for item in report["tests"]):
                    raise ValueError("tests must be a list of records")
                index = shard["index"]
                if type(index) is not int or not 0 <= index < shard_count or index in shards:
                    raise ValueError("duplicate or invalid shard index")
                if report["schema_version"] != 1 or shard["count"] != shard_count:
                    raise ValueError("unsupported report schema or shard count")
                current = (report["python"], shard["suite_sha256"], shard["discovered"],
                           report["patterns"], report["discovery_counts"])
                if signature is not None and current != signature:
                    raise ValueError("discovery metadata differs across shards")
                signature = current
                tests = report["tests"]
                counts = Counter()
                for test in tests:
                    if not isinstance(test["id"], str) or not test["id"]:
                        raise ValueError("missing test ID")
                    counts[OUTCOMES[test["outcome"]]] += 1
                if (report["successful"] is not True or counts["failed"] or counts["errors"]
                        or counts["unexpected_successes"]):
                    raise ValueError("shard contains unsuccessful tests")
                if report.get("require_no_skips") and (counts["skipped"] or counts["expected_failures"]):
                    raise ValueError("strict shard contains skipped tests")
                if not tests or len(tests) != shard["selected"] or len(tests) != report["counts"]["run"]:
                    raise ValueError("selected/run/record counts differ or are empty")
                if any(counts[name] != report["counts"][name] for name in OUTCOMES.values()):
                    raise ValueError("outcome counts do not match test records")
                records.extend(test["id"] for test in tests)
                totals.update(counts)
                shards[index] = report
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"Python {version}: {error}")
        if set(shards) != set(range(shard_count)):
            errors.append(f"Python {version}: missing shards {sorted(set(range(shard_count)) - set(shards))}")
        ids = sorted(records)
        digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        if len(ids) != len(set(ids)):
            errors.append(f"Python {version}: duplicate test IDs")
        if signature and (len(ids) != signature[2] or digest != signature[1]):
            errors.append(f"Python {version}: union does not match discovered suite")
        for index, shard in shards.items():
            if sorted(test.get("id", "") for test in shard.get("tests", [])) != ids[index::shard_count]:
                errors.append(f"Python {version}: shard {index} has incorrect test membership")
        versions[version] = {"shards": sorted(shards), "tests": len(ids), "suite_sha256": digest,
                             "counts": {name: totals[name] for name in OUTCOMES.values()}}
    return {"schema_version": 1, "successful": not errors, "errors": errors, "versions": versions}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="downloaded artifact directories containing host.json")
    parser.add_argument("--python-versions", default="3.10,3.12")
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    reports, errors = [], []
    for path in sorted(args.source.rglob("host.json")):
        try:
            reports.append(json.loads(path.read_text()))
        except (OSError, ValueError) as error:
            errors.append(f"{path}: {error}")
    result = aggregate_reports(reports, python_versions=args.python_versions.split(","), shard_count=args.shard_count)
    result["errors"].extend(errors)
    result["successful"] = not result["errors"]
    atomic_write_json(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
