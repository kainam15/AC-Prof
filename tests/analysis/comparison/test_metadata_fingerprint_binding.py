"""Comparison provenance must identify the metadata actually consumed."""
import hashlib
import json
from pathlib import Path

import pytest

from acprof.analysis.independent_comparison import compare_experiments
from acprof.artifacts import read_json_object_snapshot
from acprof.hardware_conditions import conditions_path
from acprof.results.layout import ArtifactLayout
from acprof.results.metadata import ResultMetadataSnapshot
from tests.analysis.comparison.test_comparison_source_integrity import incremental_copy
from tests.support.independent_comparison_fixtures import IndependentComparisonFixture


@pytest.fixture(params=["v2", "v3"])
def pair(request, tmp_path):
    fixture = IndependentComparisonFixture()
    fixture.build(None, tmp_path)
    roots = [fixture.replicate(side, 0, [value]) for side, value in (("left", 2), ("right", 3))]
    return [incremental_copy(root) for root in roots] if request.param == "v3" else roots


def metadata_path(root, name):
    layout = ArtifactLayout.discover(root)
    return conditions_path(layout) if name == "hardware_conditions.json" else layout.path(name)


def replace_during_read(monkeypatch, target, transform):
    """Restore the baseline bytes before the final on-disk fingerprint check."""
    original = target.read_bytes()
    replacement = json.loads(original)
    transform(replacement)
    consumed = []

    def read_changed(path, **kwargs):
        if Path(path) != target:
            return read_json_object_snapshot(path, **kwargs)
        target.write_text(json.dumps(replacement), encoding="utf-8")
        try:
            snapshot = read_json_object_snapshot(path, **kwargs)
            consumed.append(snapshot.payload)
            return snapshot
        finally:
            target.write_bytes(original)

    monkeypatch.setattr("acprof.results.metadata.read_json_object_snapshot", read_changed)
    return original, consumed


def test_transient_run_identity_cannot_be_reported_under_baseline_hash(pair, monkeypatch):
    target = metadata_path(pair[0], "run_state.json")
    original, consumed = replace_during_read(
        monkeypatch, target, lambda state: state.update(run_id="transient-run"))
    with pytest.raises(ValueError, match="experiment changed.*run_state"):
        compare_experiments([pair[0]], [pair[1]], metrics=["latency_app_s"], resamples=10)
    assert consumed[0]["run_id"] == "transient-run"
    assert target.read_bytes() == original


@pytest.mark.parametrize("name", [
    "static_meta.json", "input_scale_plan.json", "hardware_conditions.json",
    "runtime_validation.json", "quality_checks.json", "model_resolution.json", "runtime_failures.json",
])
def test_transient_metadata_bytes_cannot_pass_final_disk_check(pair, monkeypatch, name):
    target = metadata_path(pair[0], name)
    if not target.exists():
        target.write_text("{}", encoding="utf-8")
    # Even an ignored field changes the source bytes claimed by source_sha256.
    original, consumed = replace_during_read(
        monkeypatch, target, lambda payload: payload.update(transient_evidence=True))
    with pytest.raises(ValueError, match=f"experiment changed.*{name}"):
        compare_experiments([pair[0]], [pair[1]], metrics=["latency_app_s"], resamples=10)
    assert consumed
    assert consumed[0]["transient_evidence"] is True
    assert target.read_bytes() == original


def test_stable_bom_and_whitespace_keep_their_original_fingerprint(pair):
    target = metadata_path(pair[0], "run_state.json")
    original = b"\xef\xbb\xbf" + target.read_bytes() + b" \r\n"
    target.write_bytes(original)
    report = compare_experiments([pair[0]], [pair[1]], metrics=["latency_app_s"], resamples=10)
    assert report["status"] == "compatible"
    assert report["groups"][0]["difference"] == 1
    assert report["source_sha256"][str(target)] == hashlib.sha256(original).hexdigest()


@pytest.mark.parametrize("content", [b"[]", b'{"value":NaN}', b"\xff", b"{"])
def test_invalid_metadata_retains_its_diagnostic_and_source_identity(pair, content):
    target = metadata_path(pair[0], "static_meta.json")
    target.write_bytes(content)
    report = compare_experiments([pair[0]], [pair[1]], metrics=["latency_app_s"], resamples=10)
    left = report["condition_checks"][0]["experiments"]["left"]
    assert not left["valid"]
    assert any("static_meta.json" in issue for issue in left["issues"])
    assert report["source_sha256"][str(target)] == hashlib.sha256(content).hexdigest()


def test_observed_document_requires_an_initial_fingerprint(pair):
    snapshot = ResultMetadataSnapshot(pair[0])
    snapshot.read("run_state.json")
    with pytest.raises(ValueError, match="initial fingerprint.*run_state"):
        snapshot.verify_observed_sources({})


def test_cli_does_not_publish_a_report_with_transient_metadata(pair, monkeypatch, tmp_path, capsys):
    from acprof.cli.compare import main

    target = metadata_path(pair[0], "run_state.json")
    replace_during_read(monkeypatch, target, lambda state: state.update(run_id="transient-run"))
    output = tmp_path / "comparison.json"
    with pytest.raises(SystemExit) as error:
        main(["--left", str(pair[0]), "--right", str(pair[1]), "--metric", "latency_app_s",
              "--resamples", "10", "--output", str(output)])
    assert error.value.code == 1
    assert "experiment changed" in capsys.readouterr().err
    assert not output.exists()
