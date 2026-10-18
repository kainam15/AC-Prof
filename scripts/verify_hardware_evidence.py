"""Release gate for exact-commit native CPU/GPU hardware evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


class InvalidEvidence(ValueError):
    """Retained hardware report does not satisfy publication requirements."""


def validate_evidence(report_path: Path, matrix_path: Path, commit: str) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if report.get("kind") != "hardware_matrix" or report.get("ci_commit") != commit:
        raise InvalidEvidence("wrong hardware run kind or commit")
    if report.get("successful") is not True:
        raise InvalidEvidence("hardware matrix failed")
    source = report.get("source")
    if not isinstance(source, dict) or (
        source.get("environment_class") != "native_linux"
        or source.get("collection_tier") != "full"
        or not isinstance(source.get("platform"), dict)
        or source["platform"].get("native") is not True
    ):
        raise InvalidEvidence("missing Native Linux provenance")
    expected, recorded = matrix.get("cases"), report.get("cases")
    if not isinstance(expected, list) or not expected or not isinstance(recorded, list):
        raise InvalidEvidence("missing matrix cases")
    if len(expected) != len(recorded):
        raise InvalidEvidence("partial or extra hardware cases")
    for planned, actual in zip(expected, recorded, strict=True):
        config = actual.get("config") if isinstance(actual, dict) else None
        audit = actual.get("audit") if isinstance(actual, dict) else None
        if not isinstance(config, dict) or not isinstance(audit, dict):
            raise InvalidEvidence("missing case configuration or audit")
        if planned["name"] != actual.get("name") or any(
            config.get(key) != planned[key] for key in ("model", "revision", "task", "input_scales")
        ):
            raise InvalidEvidence("case differs from pinned matrix")
        if config.get("gpus") != "off,on":
            raise InvalidEvidence("CPU and GPU modes both required")
        if actual.get("status") != "passed" or audit.get("successful") is not True:
            raise InvalidEvidence("case did not pass")
        if (audit.get("valid") is not True or audit.get("completion") != "complete"
                or audit.get("exit_code") != 0 or audit.get("required_metrics_missing") != []):
            raise InvalidEvidence("case audit or RAPL/perf/NVML metrics incomplete")
        counts = audit.get("counts")
        if not isinstance(counts, dict) or type(counts.get("formal_ok")) is not int or counts["formal_ok"] <= 0:
            raise InvalidEvidence("no valid measured rows")
        if set(audit.get("validated_gpu_modes", [])) != {"off", "on"}:
            raise InvalidEvidence("not both GPU modes have valid measurements")
    return {"status": "verified", "commit": commit,
            "scope": "Native Linux pinned CPU/GPU matrix with RAPL, perf, NVML",
            "cases": [item["name"] for item in expected]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        proof = validate_evidence(args.report, args.matrix, args.commit)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Invalid release hardware evidence: {exc}\n")
    args.output.write_text(json.dumps({**proof, "run_url": args.run_url}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
