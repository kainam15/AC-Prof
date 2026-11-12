"""Fail closed on required GitHub Actions jobs for the changed-file CI scope."""

from __future__ import annotations

import json
import os

FULL_JOBS = (
    "lint", "types", "contracts", "host-summary", "wheel",
    "report-browser", "tui-snapshots", "onnx-cpu", "runtime",
)


def check_gate(needs: object) -> list[str]:
    """Check all applicable jobs without requiring deliberately skipped paths."""
    if not isinstance(needs, dict):
        return ["missing GitHub Actions needs data"]
    changes = needs.get("changes")
    if not isinstance(changes, dict):
        return ["missing changes job"]
    outputs = changes.get("outputs")
    if not isinstance(outputs, dict):
        return ["missing changes outputs"]
    full = outputs.get("full")
    if full not in ("true", "false"):
        return [f"invalid full CI scope: {full!r}"]

    required = ["changes", "docs-check"]
    if full == "true":
        required.extend(FULL_JOBS)
    else:
        for path_key, job_name in (("readme", "wheel"), ("metric", "metric-docs")):
            flag = outputs.get(path_key)
            if flag not in ("true", "false"):
                return [f"invalid {path_key} CI scope: {flag!r}"]
            if flag == "true":
                required.append(job_name)

    failures = []
    for name in required:
        job = needs.get(name)
        result = job.get("result") if isinstance(job, dict) else "missing"
        if result != "success":
            failures.append(f"{name}: expected success, got {result!r}")
    return failures


def main() -> int:
    try:
        needs = json.loads(os.environ["CI_NEEDS_JSON"])
    except (KeyError, ValueError) as exc:
        print(f"CI gate evidence missing or invalid: {exc}")
        return 1
    errors = check_gate(needs)
    if errors:
        for error in errors:
            print(f"CI gate failed: {error}")
        return 1
    print("CI gate passed: every required job for this change scope succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
