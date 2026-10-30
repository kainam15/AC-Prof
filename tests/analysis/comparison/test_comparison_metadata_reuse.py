"""One comparison uses one observation of each JSON metadata artifact."""
import hashlib
from collections import Counter
from pathlib import Path

import pytest

from acprof.analysis import comparison, model
from acprof.analysis.audit import audit_result
from acprof.analysis.independent_comparison import compare_experiments
from acprof.artifacts import MAX_JSON_ARTIFACT_BYTES
from acprof.quality import read_quality
from acprof.results.metadata import ResultMetadataSnapshot
from tests.support.comparison_fixtures import ComparisonFixture
from tests.support.independent_comparison_fixtures import IndependentComparisonFixture


@pytest.fixture
def experiment(tmp_path):
    fixture = ComparisonFixture()
    fixture.build(tmp_path)
    root = fixture.left
    fixture.change_json(root, "static_meta.json", lambda payload: payload.update(
        input_scale_plan_sha256=hashlib.sha256((root / "input_scale_plan.json").read_bytes()).hexdigest()))
    fixture.write_json(root, "runtime_validation.json", {"devices": {"off": {"quality_checks": []}}})
    return fixture, root


def test_audit_and_conditions_read_each_metadata_file_once(experiment, monkeypatch):
    _fixture, root = experiment
    names = {"static_meta.json", "run_state.json", "input_scale_plan.json", "runtime_validation.json"}
    opened = Counter()
    original = Path.open

    def observe(path, *args, **kwargs):
        if path.parent == root and path.name in names:
            opened[path.name] += 1
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observe)
    snapshot = comparison.load_comparison_snapshot(root)
    assert snapshot["valid"]
    assert snapshot["quality_status"] == "passed"
    assert opened == Counter({name: 1 for name in names})


@pytest.mark.parametrize("artifact", ["static_meta.json", "run_state.json", "input_scale_plan.json"])
def test_conditions_keep_the_metadata_that_was_audited(experiment, monkeypatch, artifact):
    fixture, root = experiment
    expected = comparison.load_comparison_snapshot(root)
    audit = comparison.audit_result

    def replace_after_audit(path, **kwargs):
        report = audit(path, **kwargs)
        def replace(payload):
            if artifact == "static_meta.json":
                payload["runtime_backend"] = "changed-backend"
            elif artifact == "run_state.json":
                payload.update(run_id="changed-run", status="interrupted")
            else:
                payload["entries"][0]["payload"]["features"] = [[999]]
        fixture.change_json(root, artifact, replace)
        return report

    with monkeypatch.context() as guard:
        guard.setattr(comparison, "audit_result", replace_after_audit)
        assert comparison.load_comparison_snapshot(root) == expected
    assert comparison.load_comparison_snapshot(root) != expected


def test_first_metadata_error_is_reused_without_retry(experiment, monkeypatch):
    _fixture, root = experiment
    target = root / "static_meta.json"
    original = Path.open
    calls = 0

    def fail_first(path, *args, **kwargs):
        nonlocal calls
        if path == target:
            calls += 1
            if calls == 1:
                raise PermissionError("original metadata read denied")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", fail_first)
        snapshot = comparison.load_comparison_snapshot(root)
    assert calls == 1
    assert not snapshot["valid"]
    assert snapshot["identity"]["runtime_backend"] is None
    assert any("original metadata read denied" in issue for issue in snapshot["issues"])
    assert comparison.load_comparison_snapshot(root)["valid"]


def test_missing_metadata_stays_missing_for_the_current_comparison(experiment, monkeypatch):
    _fixture, root = experiment
    target = root / "static_meta.json"
    raw = target.read_bytes()
    target.unlink()
    audit = comparison.audit_result

    def create_after_audit(path, **kwargs):
        report = audit(path, **kwargs)
        target.write_bytes(raw)
        return report

    with monkeypatch.context() as guard:
        guard.setattr(comparison, "audit_result", create_after_audit)
        snapshot = comparison.load_comparison_snapshot(root)
    assert snapshot["identity"]["runtime_backend"] is None
    assert comparison.load_comparison_snapshot(root)["valid"]


def test_independent_comparison_keeps_initial_and_final_metadata_hashes(tmp_path, monkeypatch):
    fixture = IndependentComparisonFixture()
    fixture.build(None, tmp_path)
    left = fixture.replicate("left", 0, [2])
    right = fixture.replicate("right", 0, [3])
    targets = {root / name for root in (left, right) for name in ("run_state.json", "static_meta.json")}
    expected = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in targets}
    counts = Counter()
    original = Path.open

    def observe(path, *args, **kwargs):
        if path in targets:
            counts[path] += 1
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observe)
    report = compare_experiments([left], [right], metrics=["latency_app_s"], resamples=10)
    assert report["status"] == "compatible"
    assert counts == Counter({path: 3 for path in targets})
    assert {path: report["source_sha256"][path] for path in expected} == expected


def test_analysis_and_device_quality_share_audited_metadata(experiment, monkeypatch):
    _fixture, root = experiment
    targets = {"static_meta.json", "run_state.json", "runtime_validation.json"}
    counts = Counter()
    original = Path.open

    def observe(path, *args, **kwargs):
        if path.parent == root and path.name in targets:
            counts[path.name] += 1
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observe)
    data = model.load_analysis([root])
    assert data.sources[0]["quality_status"] == "passed"
    assert data.raw_rows[0]["quality_status"] == "passed"
    assert counts == Counter({name: 1 for name in targets})


def test_device_quality_keeps_the_evidence_used_by_the_audit(experiment, monkeypatch):
    fixture, root = experiment
    original = model.load_comparison_snapshot

    def change_after_comparison(path, **kwargs):
        snapshot = original(path, **kwargs)
        fixture.write_json(root, "quality_checks.json", {"schema_version": 1, "checks": [{
            "code": "changed", "severity": "error", "detail": "new quality result", "evidence": {},
        }]})
        return snapshot

    with monkeypatch.context() as guard:
        guard.setattr(model, "load_comparison_snapshot", change_after_comparison)
        data = model.load_analysis([root])
    assert data.sources[0]["quality_status"] == "passed"
    assert data.raw_rows[0]["quality_status"] == "passed"
    assert model.load_analysis([root]).raw_rows[0]["quality_status"] == "blocked"


def test_shared_metadata_rejects_a_different_source(experiment):
    fixture, root = experiment
    snapshot = ResultMetadataSnapshot(root)
    for reader in (comparison.load_comparison_snapshot, read_quality):
        with pytest.raises(ValueError, match="different path"):
            reader(fixture.right, metadata_snapshot=snapshot)
    report = audit_result(fixture.right, metadata_snapshot=snapshot)
    assert not report["valid"]
    assert any(issue["code"] == "invalid_layout" and "different path" in issue["message"]
               for issue in report["issues"])
    assert comparison.load_comparison_snapshot(root / "result_layers.json", metadata_snapshot=snapshot)["valid"]


def test_comparison_does_not_mutate_shared_invalid_hardware(experiment):
    fixture, root = experiment
    fixture.write_json(root, "hardware_conditions.json", {"schema_version": 1, "cases": {"1c_4g_off": None}})
    metadata = ResultMetadataSnapshot(root)
    hardware = metadata.read("hardware_conditions.json").payload
    first = comparison.load_comparison_snapshot(root, metadata_snapshot=metadata)
    assert not first["valid"]
    assert hardware["cases"]["1c_4g_off"] is None
    assert comparison.load_comparison_snapshot(root, metadata_snapshot=metadata) == first


@pytest.mark.parametrize("content", [b'[]', b'{"value": NaN}', b'{"value": 1e9999}', b'\xff',
                                     b'{"value":"' + b'x' * MAX_JSON_ARTIFACT_BYTES + b'"}'])
def test_matching_hash_never_bypasses_plan_json_validation(experiment, content):
    fixture, root = experiment
    (root / "input_scale_plan.json").write_bytes(content)
    fixture.change_json(root, "static_meta.json", lambda payload: payload.update(
        input_scale_plan_sha256=hashlib.sha256(content).hexdigest()))
    report = audit_result(root)
    assert not report["valid"]
    assert any(issue["code"] == "input_plan_hash" for issue in report["issues"])
    assert not comparison.load_comparison_snapshot(root)["valid"]


def test_bom_input_plan_hash_uses_original_bytes(experiment):
    fixture, root = experiment
    plan = root / "input_scale_plan.json"
    content = b'\xef\xbb\xbf' + plan.read_bytes() + b'\n'
    plan.write_bytes(content)
    fixture.change_json(root, "static_meta.json", lambda payload: payload.update(
        input_scale_plan_sha256=hashlib.sha256(content).hexdigest()))
    assert audit_result(root)["valid"]
    assert comparison.load_comparison_snapshot(root)["valid"]
