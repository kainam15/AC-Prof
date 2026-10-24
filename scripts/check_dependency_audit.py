"""Validate pip-audit JSON output and summarize known dependency vulnerabilities.

Security audits report findings without automatically mutating locks or blocking
an ordinary code merge. A missing/malformed scan is still a CI infrastructure error.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def summarize_report(path: Path) -> tuple[int, int, list[str], list[str]]:
    """Count audited packages, findings and explicitly skipped packages separately."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("dependencies"), list):
        raise ValueError("invalid pip-audit report: missing dependencies")
    dependencies = data["dependencies"]
    if not dependencies:
        raise ValueError("invalid pip-audit report: empty dependencies")
    total_findings = 0
    summaries: list[str] = []
    skipped: list[str] = []
    seen: set[str] = set()
    for dep in dependencies:
        if not isinstance(dep, dict) or not isinstance(dep.get("name"), str) or not dep["name"].strip():
            raise ValueError("invalid pip-audit report: malformed dependency")
        name = dep["name"]
        if name in seen:
            raise ValueError(f"invalid pip-audit report: duplicate dependency {name}")
        seen.add(name)
        # pip-audit emits {"name": "...", "skip_reason": "..."} for
        # packages unavailable on PyPI, notably CUDA/CPU-local PyTorch wheels.
        # Treat these as NOT ASSESSED, never as clean packages.
        if "skip_reason" in dep:
            reason = dep["skip_reason"]
            if not isinstance(reason, str) or not reason.strip() or "vulns" in dep:
                raise ValueError("invalid pip-audit report: malformed skipped dependency")
            skipped.append(f"{name}: {reason}")
            continue
        findings = dep.get("vulns")
        if not isinstance(findings, list):
            raise ValueError("invalid pip-audit report: missing vulnerability list or skip reason")
        for finding in findings:
            if not isinstance(finding, dict) or not isinstance(finding.get("id"), str) or not finding["id"]:
                raise ValueError("invalid pip-audit report: malformed vulnerability")
        total_findings += len(findings)
        if findings:
            summaries.append(f"{name}: {len(findings)}")
    if len(dependencies) == len(skipped):
        raise ValueError("invalid pip-audit report: no dependencies assessed")
    return len(dependencies) - len(skipped), total_findings, summaries, skipped


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: check_dependency_audit.py <report.json> <lock>", file=sys.stderr)
        return 2
    path, lock_name = Path(args[0]), args[1]
    try:
        packages, findings, summaries, skipped = summarize_report(path)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"audit evidence unavailable/invalid: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"{lock_name}: audited {packages} packages, skipped {len(skipped)} packages; "
          f"{findings} known vulnerability records")
    if summaries:
        print("Affected packages: " + "; ".join(summaries))
        # Nonblocking findings appear as a warning, not a success claim.
        print(f"::warning::Dependency audit found {findings} vulnerability records in {lock_name}.")
    if skipped:
        print("NOT ASSESSED: " + "; ".join(skipped))
        print(f"::warning::Dependency audit INCOMPLETE for {lock_name}: "
              f"{len(skipped)} unassessed packages. Do not report a clean lock.")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as out:
            out.write(f"### {lock_name}: {packages} assessed, {len(skipped)} unassessed, "
                      f"{findings} known vulnerability records\n\n")
            for summary in summaries:
                out.write(f"- {summary}\n")
            if not summaries:
                out.write("No known vulnerabilities in **assessed** dependencies.\n")
            for item in skipped:
                out.write(f"- **NOT ASSESSED**: {item}\n")
            out.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
