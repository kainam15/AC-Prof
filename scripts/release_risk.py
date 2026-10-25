"""Audit release dependency snapshots and enforce an identity-bound risk decision.

pip-audit does not establish severity or reachability: all findings and skipped
packages require an explicit, scoped, unexpired waiver. Ordinary CI stays report-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acprof.dependency_locks import read_python_lock  # noqa: E402
from acprof.runtime_profiles import ENVIRONMENTS, PLATFORMS  # noqa: E402
from scripts.check_dependency_audit import summarize_report, vulnerability_groups  # noqa: E402


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def package_name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError("invalid dependency name")
    return re.sub(r"[-_.]+", "-", value).lower()


def pins(records: list[dict]) -> dict[str, str]:
    result = {}
    for record in records:
        name, version = package_name(record["name"]), record["version"]
        if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+!_-]*", version):
            raise ValueError(f"invalid pinned version: {name}")
        if name in result:
            raise ValueError(f"duplicate dependency: {name}")
        result[name] = version
    if not result:
        raise ValueError("empty dependency snapshot")
    return result


@dataclass(frozen=True)
class AuditSource:
    scope: str
    packages: dict[str, str]
    source_sha256: str


def runtime_locks(root: Path) -> list[Path]:
    expected = {spec.requirements_lock for spec in (*PLATFORMS.values(), *ENVIRONMENTS.values())}
    found = {path.relative_to(root).as_posix() for path in (root / "dockerfiles/locks").glob("*.txt")}
    if not expected or expected != found:
        raise ValueError("runtime lock audit coverage differs from registered environments")
    return [root / relative for relative in sorted(expected)]


def audit_sources(root: Path, distribution_evidence: Path, artifacts: dict[str, str]) -> list[AuditSource]:
    sources = []
    wheels = [value for name, value in artifacts.items() if name.endswith(".whl")]
    sdists = [value for name, value in artifacts.items() if name.endswith(".tar.gz")]
    if (len(artifacts) != 3 or len(wheels) != 1 or len(sdists) != 1
            or "acprof-linux-x86_64" not in artifacts
            or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in artifacts.values())):
        raise ValueError("invalid release artifact identity")
    expected_artifacts = {
        "wheel": wheels[0],
        "sdist": sdists[0],
        "standalone": artifacts["acprof-linux-x86_64"],
    }
    for scope in ("wheel", "sdist", "standalone"):
        path = distribution_evidence / f"{scope}.json"
        evidence = json.loads(path.read_text())
        if evidence.get("successful") is not True or not isinstance(evidence.get("dependencies"), list):
            raise ValueError(f"missing verified dependency snapshot: {scope}")
        if evidence.get("artifact_sha256", {}).get(scope) != expected_artifacts[scope]:
            raise ValueError(f"distribution dependency snapshot artifact mismatch: {scope}")
        packages = pins(evidence["dependencies"])
        if "acprof" not in packages:
            raise ValueError(f"installed release missing from dependency snapshot: {scope}")
        del packages["acprof"]  # The first-party artifact is identified separately.
        if not packages:
            raise ValueError(f"empty release dependency closure: {scope}")
        sources.append(AuditSource(scope, packages, digest(path)))
    for path in runtime_locks(root):
        sources.append(AuditSource("runtime-" + path.stem, pins(read_python_lock(path)), digest(path)))
    return sources


def lock_identity(root: Path) -> dict[str, str]:
    paths = (root / "dockerfiles/locks").rglob("*")
    return {"uv.lock": digest(root / "uv.lock"),
            **{path.relative_to(root).as_posix(): digest(path) for path in sorted(paths) if path.is_file()}}


def artifact_identity(dist: Path, *, python_only: bool = False) -> dict[str, str]:
    wheels, sdists = list(dist.glob("acprof-*.whl")), list(dist.glob("acprof-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("release must contain exactly one wheel and one sdist")
    paths = [*wheels, *sdists]
    if not python_only:
        paths.append(dist / "acprof-linux-x86_64")
    return {path.name: digest(path) for path in paths}


def read_policy(path: Path, *, today: date) -> list[dict]:
    policy = json.loads(path.read_text())
    if policy.get("schema_version") != 1 or not isinstance(policy.get("waivers"), list):
        raise ValueError("invalid release risk policy")
    fields = {"scope", "package", "version", "finding", "decision", "reason", "owner", "expires", "evidence"}
    seen = set()
    for waiver in policy["waivers"]:
        if (not isinstance(waiver, dict) or set(waiver) != fields
                or any(not isinstance(value, str) or not value.strip() for value in waiver.values())):
            raise ValueError("waiver requires exact scope/package/version/finding, decision, reason, owner, expires and evidence")
        if waiver["decision"] not in {"risk_accepted", "not_affected"}:
            raise ValueError("waiver decision must be risk_accepted or not_affected")
        if waiver["finding"] == "unassessed" and waiver["decision"] != "risk_accepted":
            raise ValueError("unassessed applicability cannot be declared not_affected")
        if date.fromisoformat(waiver["expires"]) < today:
            raise ValueError("expired release risk waiver")
        identity = (waiver["scope"], package_name(waiver["package"]), waiver["version"], waiver["finding"])
        if identity in seen:
            raise ValueError("duplicate release risk waiver")
        seen.add(identity)
    return policy["waivers"]


def assess_report(source: AuditSource, path: Path, waivers: list[dict]) -> dict:
    assessed, _, _, skipped = summarize_report(path)
    dependencies = json.loads(path.read_text())["dependencies"]
    observed = [package_name(dep["name"]) for dep in dependencies]
    if len(set(observed)) != len(observed) or set(observed) != set(source.packages):
        raise ValueError(f"audit package coverage mismatch: {source.scope}")
    decisions = []
    for dep in dependencies:
        name = package_name(dep["name"])
        version = source.packages[name]
        if "skip_reason" not in dep and dep.get("version") != version:
            raise ValueError(f"audit version mismatch: {source.scope}/{name}")
        groups = [["unassessed"]] if "skip_reason" in dep else vulnerability_groups(dep["vulns"])
        for identifiers in groups:
            matches = [waiver for waiver in waivers
                       if waiver["scope"] == source.scope and package_name(waiver["package"]) == name
                       and waiver["version"] == version and waiver["finding"] in identifiers]
            if len(matches) > 1:
                raise ValueError("ambiguous waivers for one advisory group")
            decisions.append({
                "package": name, "version": version, "identifiers": identifiers,
                "severity": "unknown", "applicability": "unknown",
                "status": "waived" if matches else "blocked",
                "waiver": matches[0] if matches else None,
                "skip_reason": dep.get("skip_reason"),
            })
    return {"scope": source.scope, "source_sha256": source.source_sha256,
            "report_sha256": digest(path), "assessed_packages": assessed,
            "unassessed_packages": len(skipped), "decisions": decisions}


def evaluate(sources: list[AuditSource], reports: Path, policy: Path, *, commit: str,
             root: Path, dist: Path, today: date) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("release risk requires an exact commit SHA")
    waivers = read_policy(policy, today=today)
    scopes = [source.scope for source in sources]
    expected = {"wheel", "sdist", "standalone", *("runtime-" + path.stem for path in runtime_locks(root))}
    if len(set(scopes)) != len(scopes) or set(scopes) != expected:
        raise ValueError("release risk requires complete audit coverage")
    assessments = [assess_report(source, reports / f"{source.scope}.json", waivers) for source in sources]
    if not assessments:
        raise ValueError("release risk requires complete audit coverage")
    blocked = any(item["status"] == "blocked" for assessment in assessments for item in assessment["decisions"])
    return {"schema_version": 1, "status": "blocked" if blocked else "approved", "commit": commit,
            "assessed_at": today.isoformat(), "artifacts": artifact_identity(dist), "locks": lock_identity(root),
            "policy_sha256": digest(policy), "assessments": assessments,
            "scope": "validated wheel/sdist installations, standalone build environment and all registered Python runtime locks",
            "not_assessed_by_this_gate": ["OCI/system libraries", "future installer resolutions", "model checkpoints"],
            "applicability_policy": "unknown blocks; no distinction between new and existing vulnerabilities"}


def verify(evidence: dict, *, commit: str, root: Path, dist: Path, policy: Path, evidence_dir: Path,
           python_only: bool = False, today: date | None = None) -> None:
    today = today or datetime.now(UTC).date()
    if evidence.get("schema_version") != 1 or evidence.get("status") != "approved" or evidence.get("commit") != commit:
        raise ValueError("release risk proof is missing, blocked or from another commit")
    if evidence.get("locks") != lock_identity(root) or evidence.get("policy_sha256") != digest(policy):
        raise ValueError("release risk lock or policy identity mismatch")
    waivers = read_policy(policy, today=today)  # A waiver may expire between scan and publication.
    for name, sha256 in artifact_identity(dist, python_only=python_only).items():
        if evidence.get("artifacts", {}).get(name) != sha256:
            raise ValueError(f"release risk artifact identity mismatch: {name}")
    sources = audit_sources(root, evidence_dir / "snapshots", evidence["artifacts"])
    assessments = [assess_report(source, evidence_dir / "reports" / f"{source.scope}.json", waivers)
                   for source in sources]
    if evidence.get("assessments") != assessments or any(
        decision["status"] != "waived" for assessment in assessments for decision in assessment["decisions"]
    ):
        raise ValueError("release risk audit evidence differs or contains unresolved findings")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("scan", "verify"))
    parser.add_argument("--commit", required=True)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=ROOT / ".github/release-risk.json")
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("release-risk"))
    parser.add_argument("--proof", type=Path)
    parser.add_argument("--python-only", action="store_true")
    args = parser.parse_args(argv)
    today = datetime.now(UTC).date()
    if args.action == "verify":
        if args.proof is None:
            parser.error("verify requires --proof")
        verify(json.loads(args.proof.read_text()), commit=args.commit, root=ROOT, dist=args.dist,
               policy=args.policy, evidence_dir=args.proof.parent, python_only=args.python_only, today=today)
        return 0
    if args.evidence_dir is None:
        parser.error("scan requires --evidence-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = args.output_dir / "reports"
    reports.mkdir()
    try:
        snapshots = args.output_dir / "snapshots"
        snapshots.mkdir()
        for scope in ("wheel", "sdist", "standalone"):
            shutil.copyfile(args.evidence_dir / f"{scope}.json", snapshots / f"{scope}.json")
        sources = audit_sources(ROOT, snapshots, artifact_identity(args.dist))
        for source in sources:
            requirements = args.output_dir / f"{source.scope}.txt"
            requirements.write_text("".join(f"{name}=={version}\n" for name, version in sorted(source.packages.items())))
            result = subprocess.run(
                [sys.executable, "-m", "pip_audit", "--disable-pip", "--no-deps", "--aliases", "on",
                 "--progress-spinner", "off", "--format", "json", "-r", str(requirements),
                 "-o", str(reports / f"{source.scope}.json")], timeout=600, check=False,
            )
            if result.returncode not in (0, 1):
                raise RuntimeError(f"audit infrastructure failed: {source.scope}, exit={result.returncode}")
        evidence = evaluate(sources, reports, args.policy, commit=args.commit, root=ROOT,
                            dist=args.dist, today=today)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.TimeoutExpired) as exc:
        evidence = {"schema_version": 1, "status": "blocked", "commit": args.commit,
                    "error": f"{type(exc).__name__}: {exc}"}
    (args.output_dir / "release-risk.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"Release dependency risk: {evidence['status']}")
    return 0 if evidence["status"] == "approved" else 1


if __name__ == "__main__":
    raise SystemExit(main())
