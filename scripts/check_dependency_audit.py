"""Validate pip-audit JSON output and summarize known dependency vulnerabilities.

Security audits report findings without automatically mutating locks or blocking
an ordinary code merge. A missing/malformed scan is still a CI infrastructure error.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def summarize_report(path: Path) -> tuple[int, int, list[str]]:
    """Require real audit evidence; report vulnerable package and finding totals."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("dependencies"), list):
        raise ValueError("invalid pip-audit report: missing dependencies")
    dependencies = data["dependencies"]
    if not dependencies:
        raise ValueError("invalid pip-audit report: empty dependencies")
    total_findings = 0
    summaries: list[str] = []
    for dep in dependencies:
        if not isinstance(dep, dict) or not isinstance(dep.get("name"), str):
            raise ValueError("invalid pip-audit report: malformed dependency")
        findings = dep.get("vulns")
        if not isinstance(findings, list):
            raise ValueError("invalid pip-audit report: missing vulnerability list")
        if not findings:
            continue
        for finding in findings:
            if not isinstance(finding, dict) or not isinstance(finding.get("id"), str):
                raise ValueError("invalid pip-audit report: malformed vulnerability")
        total_findings += len(findings)
        summaries.append(f"{dep['name']}: {len(findings)}")
    return len(dependencies), total_findings, summaries


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: check_dependency_audit.py <report.json> <lock>", file=sys.stderr)
        return 2
    path, lock_name = Path(args[0]), args[1]
    try:
        packages, findings, summaries = summarize_report(path)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"audit evidence unavailable/invalid: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"{lock_name}: audited {packages} locked packages; {findings} known vulnerability records")
    if summaries:
        print("Affected packages: " + "; ".join(summaries))
        # Nonblocking findings appear as a warning, not a success claim.
        print(f"::warning::Dependency audit found {findings} vulnerability records in {lock_name}.")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as out:
            out.write(f"### {lock_name}: {packages} packages, {findings} vulnerability records\n\n")
            for summary in summaries:
                out.write(f"- {summary}\n")
            if not summaries:
                out.write("No known vulnerabilities in scanned dependencies.\n")
            out.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
