import json
import subprocess
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import release_risk
from scripts.release_risk import (
    AuditSource,
    artifact_identity,
    assess_report,
    audit_sources,
    evaluate,
    read_policy,
    verify,
)

TODAY = date(2026, 10, 25)
COMMIT = "a" * 40


@pytest.fixture
def risk_fixture(tmp_path, monkeypatch):
    (tmp_path / "uv.lock").write_text("locked dependencies")
    dist = tmp_path / "dist"
    dist.mkdir()
    for name in ("acprof-1.0-py3-none-any.whl", "acprof-1.0.tar.gz", "acprof-linux-x86_64"):
        (dist / name).write_bytes(name.encode())
    reports = tmp_path / "reports"
    reports.mkdir()
    for scope in ("wheel", "sdist", "standalone", "runtime-fixture"):
        (reports / f"{scope}.json").write_text(json.dumps({
            "dependencies": [{"name": "fixture", "version": "1.0", "vulns": []}],
        }))
    lock = tmp_path / "dockerfiles/locks/fixture.txt"
    lock.parent.mkdir(parents=True)
    lock.write_text("fixture @ https://example.invalid/fixture-1.0-py3-none-any.whl --hash=sha256:" + "0" * 64 + "\n")
    monkeypatch.setattr(release_risk, "PLATFORMS", {"fixture": SimpleNamespace(requirements_lock="dockerfiles/locks/fixture.txt")})
    monkeypatch.setattr(release_risk, "ENVIRONMENTS", {})
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    artifacts = artifact_identity(dist)
    for scope, name in zip(("wheel", "sdist", "standalone"), artifacts, strict=True):
        (snapshots / f"{scope}.json").write_text(json.dumps({
            "successful": True, "artifact_sha256": {scope: artifacts[name]},
            "dependencies": [{"name": "acprof", "version": "1.0"}, {"name": "fixture", "version": "1.0"}],
        }))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"schema_version": 1, "waivers": []}))
    return tmp_path, dist, reports, policy, audit_sources(tmp_path, snapshots, artifacts)[0]


def assessment(fixture):
    root, dist, reports, policy, _ = fixture
    sources = audit_sources(root, root / "snapshots", artifact_identity(dist))
    return evaluate(sources, reports, policy, commit=COMMIT, root=root, dist=dist, today=TODAY)


def test_approved_proof_is_bound_to_commit_locks_and_actual_artifact(risk_fixture):
    root, dist, _, policy, _ = risk_fixture
    proof = assessment(risk_fixture)
    assert proof["status"] == "approved"
    verify(proof, commit=COMMIT, root=root, dist=dist, policy=policy, evidence_dir=root, today=TODAY)
    with pytest.raises(ValueError, match="another commit"):
        verify(proof, commit="c" * 40, root=root, dist=dist, policy=policy, evidence_dir=root, today=TODAY)
    (dist / "acprof-1.0-py3-none-any.whl").write_bytes(b"replaced wheel")
    with pytest.raises(ValueError, match="artifact identity"):
        verify(proof, commit=COMMIT, root=root, dist=dist, policy=policy, evidence_dir=root, today=TODAY)
    (root / "uv.lock").write_text("different lock")
    with pytest.raises(ValueError, match="lock or policy identity"):
        verify(proof, commit=COMMIT, root=root, dist=dist, policy=policy, evidence_dir=root, today=TODAY)


def test_existing_vulnerability_with_unknown_applicability_blocks(risk_fixture):
    _, _, reports, _, _ = risk_fixture
    (reports / "wheel.json").write_text(json.dumps({"dependencies": [
        {"name": "fixture", "version": "1.0", "vulns": [{"id": "GHSA-fixture", "aliases": ["CVE-2020-1234"]}]},
    ]}))
    proof = assessment(risk_fixture)
    assert proof["status"] == "blocked"
    decision = proof["assessments"][0]["decisions"][0]
    assert decision["applicability"] == "unknown"
    assert decision["severity"] == "unknown"


def test_scoped_alias_waiver_is_preserved_and_expiry_rechecked(risk_fixture):
    root, dist, reports, policy, _ = risk_fixture
    (reports / "wheel.json").write_text(json.dumps({"dependencies": [
        {"name": "fixture", "version": "1.0", "vulns": [{"id": "GHSA-fixture", "aliases": ["CVE-2020-1234"]}]},
    ]}))
    waiver = {"scope": "wheel", "package": "fixture", "version": "1.0", "finding": "CVE-2020-1234",
              "decision": "risk_accepted", "reason": "test-only scoped decision", "owner": "test-maintainer",
              "expires": "2026-10-25", "evidence": "https://example.invalid/review/1"}
    policy.write_text(json.dumps({"schema_version": 1, "waivers": [waiver]}))
    proof = assessment(risk_fixture)
    assert proof["status"] == "approved"
    assert proof["assessments"][0]["decisions"][0]["waiver"] == waiver
    with pytest.raises(ValueError, match="expired"):
        verify(proof, commit=COMMIT, root=root, dist=dist, policy=policy, evidence_dir=root, today=date(2026, 10, 26))
    waiver["version"] = "2.0"
    policy.write_text(json.dumps({"schema_version": 1, "waivers": [waiver]}))
    assert assessment(risk_fixture)["status"] == "blocked"


@pytest.mark.parametrize("dependencies", [
    [], [{"name": "fixture", "skip_reason": "not in PyPI"}],
    [{"name": "unrelated", "version": "1.0", "vulns": []}],
    [{"name": "fixture", "version": "2.0", "vulns": []}],
    [{"name": "fixture", "version": "1.0", "vulns": "invalid"}],
])
def test_incomplete_or_mismatched_scans_are_never_approval(risk_fixture, dependencies):
    _, _, reports, _, _ = risk_fixture
    (reports / "wheel.json").write_text(json.dumps({"dependencies": dependencies}))
    with pytest.raises(ValueError, match="invalid pip-audit|coverage mismatch|version mismatch"):
        assessment(risk_fixture)


def test_partial_unassessed_is_recorded_and_blocks(risk_fixture):
    _, _, reports, _, source = risk_fixture
    source = AuditSource(source.scope, {**source.packages, "special-wheel": "2.0+cpu"}, source.source_sha256)
    path = reports / "wheel.json"
    path.write_text(json.dumps({"dependencies": [
        {"name": "fixture", "version": "1.0", "vulns": []},
        {"name": "special-wheel", "skip_reason": "version not available"},
    ]}))
    report = assess_report(source, path, [])
    assert report["unassessed_packages"] == 1
    assert report["decisions"][0]["status"] == "blocked"
    assert report["decisions"][0]["identifiers"] == ["unassessed"]


def test_unassessed_cannot_be_waived_as_unreachable(risk_fixture):
    _, _, _, policy, _ = risk_fixture
    waiver = {"scope": "wheel", "package": "fixture", "version": "1.0", "finding": "unassessed",
              "decision": "not_affected", "reason": "guess", "owner": "test", "expires": "2026-10-26",
              "evidence": "https://example.invalid"}
    policy.write_text(json.dumps({"schema_version": 1, "waivers": [waiver]}))
    with pytest.raises(ValueError, match="cannot be declared not_affected"):
        read_policy(policy, today=TODAY)


@pytest.mark.parametrize("tamper", ["scope", "report_hash", "report"])
def test_publisher_rechecks_raw_audit_evidence(risk_fixture, tamper):
    root, dist, reports, policy, _ = risk_fixture
    proof = assessment(risk_fixture)
    if tamper == "scope":
        proof["assessments"][0]["scope"] = "unrelated"
    elif tamper == "report_hash":
        proof["assessments"][0]["report_sha256"] = "0" * 64
    else:
        (reports / "wheel.json").write_text(json.dumps({"dependencies": [
            {"name": "fixture", "version": "1.0", "vulns": [{"id": "CVE-fixture"}]},
        ]}))
    with pytest.raises(ValueError, match="audit evidence"):
        verify(proof, commit=COMMIT, root=root, dist=dist, policy=policy, evidence_dir=root, today=TODAY)


def test_missing_scope_or_extra_runtime_lock_fails_closed(risk_fixture):
    root, dist, reports, policy, _ = risk_fixture
    sources = audit_sources(root, root / "snapshots", artifact_identity(dist))
    with pytest.raises(ValueError, match="complete audit coverage"):
        evaluate(sources[:-1], reports, policy, commit=COMMIT, root=root, dist=dist, today=TODAY)
    (root / "dockerfiles/locks/unregistered.txt").write_text("unregistered environment")
    with pytest.raises(ValueError, match="coverage differs"):
        audit_sources(root, root / "snapshots", artifact_identity(dist))


def test_distribution_snapshot_must_identify_the_published_artifact(risk_fixture):
    root, dist, _, _, _ = risk_fixture
    path = root / "snapshots/sdist.json"
    snapshot = json.loads(path.read_text())
    snapshot["artifact_sha256"]["sdist"] = "0" * 64
    path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="snapshot artifact mismatch"):
        audit_sources(root, root / "snapshots", artifact_identity(dist))


@pytest.mark.parametrize("scenario", ["clean", "vulnerability", "infrastructure", "timeout"])
def test_scan_archives_complete_evidence_and_gates_publication(risk_fixture, monkeypatch, scenario):
    root, dist, _, policy, _ = risk_fixture
    monkeypatch.setattr(release_risk, "ROOT", root)
    scanned = []

    def fake_audit(argv, *, timeout, check):
        assert timeout == 600
        assert check is False
        requirements = Path(argv[argv.index("-r") + 1])
        scope = requirements.stem
        scanned.append(scope)
        if scenario == "timeout":
            raise subprocess.TimeoutExpired(argv, timeout)
        if scenario == "infrastructure":
            return SimpleNamespace(returncode=2)
        packages = [line.split("==") for line in requirements.read_text().splitlines()]
        findings = [{"id": "CVE-existing"}] if scenario == "vulnerability" and scope == "wheel" else []
        report = {"dependencies": [{"name": name, "version": version, "vulns": findings}
                                   for name, version in packages]}
        Path(argv[argv.index("-o") + 1]).write_text(json.dumps(report))
        return SimpleNamespace(returncode=int(bool(findings)))

    monkeypatch.setattr(release_risk.subprocess, "run", fake_audit)
    output = root / "release-risk"
    code = release_risk.main(["scan", "--commit", COMMIT, "--dist", str(dist), "--policy", str(policy),
                              "--evidence-dir", str(root / "snapshots"), "--output-dir", str(output)])
    proof_path = output / "release-risk.json"
    proof = json.loads(proof_path.read_text())
    assert code == (0 if scenario == "clean" else 1)
    assert proof["status"] == ("approved" if scenario == "clean" else "blocked")
    assert {path.stem for path in (output / "snapshots").glob("*.json")} == {"wheel", "sdist", "standalone"}
    publish_args = ["verify", "--commit", COMMIT, "--dist", str(dist), "--policy", str(policy), "--proof", str(proof_path)]
    if scenario == "clean":
        assert set(scanned) == {"wheel", "sdist", "standalone", "runtime-fixture"}
        assert release_risk.main(publish_args) == 0
        (dist / "acprof-linux-x86_64").unlink()
        assert release_risk.main([*publish_args, "--python-only"]) == 0
    else:
        with pytest.raises(ValueError, match="blocked"):
            release_risk.main(publish_args)
