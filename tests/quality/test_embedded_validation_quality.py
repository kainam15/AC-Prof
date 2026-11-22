"""Quality summaries use the recorded source and preserve evidence across devices."""
import json

import pytest

from acprof.analysis.compatibility import report_results
from acprof.quality import read_quality
from acprof.results.metadata import ResultMetadataSnapshot


def write(path, value):
    path.write_text(json.dumps(value))


def devices():
    error = {"severity": "error", "code": "observed_output_mismatch", "evidence": {"source": "output validator"},
             "detail": "observed shape differs"}
    return {"status": "ok", "devices": {"off": {"status": "ok"}, "on": {"status": "ok", "quality_checks": [error]}}}


@pytest.mark.parametrize("embedded", [True, False], ids=["embedded", "standalone"])
def test_missing_device_checks_keep_another_devices_failure(tmp_path, embedded):
    validation = devices()
    path = tmp_path / ("static_meta.json" if embedded else "runtime_validation.json")
    write(path, {"runtime_validation": validation} if embedded else validation)
    snapshot = ResultMetadataSnapshot(tmp_path)
    health = read_quality(tmp_path, metadata_snapshot=snapshot)
    assert health["quality_status"] == "blocked"
    assert health["quality_reasons"] == ["observed_output_mismatch", "quality_evidence_missing"]
    assert health["auto_selection_eligible"] is False
    check, = health["quality_checks"]
    assert check["evidence"] == {"source": "output validator", "artifact": str(path), "device": "on"}
    assert read_quality(tmp_path, device="off", metadata_snapshot=snapshot)["quality_status"] == "unknown"
    assert read_quality(tmp_path, device="on", metadata_snapshot=snapshot)["quality_status"] == "blocked"
    assert read_quality(tmp_path, device="absent", metadata_snapshot=snapshot)["quality_status"] == "unknown"
    snapshot.verify_current(tmp_path)


def test_empty_standalone_validation_and_canonical_quality_each_override_older_sources(tmp_path):
    write(tmp_path / "static_meta.json", {"runtime_validation": devices()})
    assert read_quality(tmp_path)["quality_status"] == "blocked"
    write(tmp_path / "runtime_validation.json", {})
    assert read_quality(tmp_path)["quality_status"] == "unknown"
    write(tmp_path / "runtime_validation.json", devices())
    write(tmp_path / "quality_checks.json", {"schema_version": 1, "checks": []})
    assert read_quality(tmp_path)["quality_status"] == "passed"
    write(tmp_path / "quality_checks.json", {})
    assert read_quality(tmp_path)["quality_reasons"] == ["quality_evidence_invalid"]


def test_malformed_device_checks_cannot_erase_another_devices_valid_diagnosis(tmp_path):
    validation = devices()
    validation["devices"]["off"]["quality_checks"] = "invalid"
    write(tmp_path / "runtime_validation.json", validation)
    result = read_quality(tmp_path)
    assert result["quality_status"] == "blocked"
    assert result["quality_reasons"] == ["observed_output_mismatch", "quality_evidence_invalid"]
    assert {check["evidence"]["device"] for check in result["quality_checks"]} == {"off", "on"}


def test_report_uses_recorded_quality_from_embedded_probe(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    write(root / "static_meta.json", {"runtime_validation": devices(), "capability_report": {"full_profile_complete": True}})
    row, = report_results([root], tmp_path / "out")["rows"]
    assert row["status"] == "full_success_quality_blocked"
    assert row["full_profile_complete"] is True
    assert row["quality_status"] == "blocked"


def test_embedded_quality_is_bound_to_its_operation_snapshot(tmp_path):
    meta = tmp_path / "static_meta.json"
    write(meta, {"runtime_validation": devices()})
    snapshot = ResultMetadataSnapshot(tmp_path)
    assert read_quality(tmp_path, metadata_snapshot=snapshot)["quality_status"] == "blocked"
    write(meta, {"runtime_validation": {}})
    assert read_quality(tmp_path, metadata_snapshot=snapshot)["quality_status"] == "blocked"
    assert read_quality(tmp_path)["quality_status"] == "unknown"
    with pytest.raises(ValueError, match="metadata changed"):
        snapshot.verify_current(tmp_path)


@pytest.mark.parametrize("canonical", [True, False], ids=["canonical", "device"])
def test_malformed_check_preserves_another_explicit_blocker_in_the_same_list(tmp_path, canonical):
    validation = devices()
    records = validation["devices"]["on"]["quality_checks"]
    records.insert(0, {"code": "incomplete_record"})
    if canonical:
        write(tmp_path / "quality_checks.json", {"schema_version": 1, "checks": records})
    else:
        write(tmp_path / "runtime_validation.json", validation)
    quality = read_quality(tmp_path, device="on")
    assert quality["quality_status"] == "blocked"
    assert quality["quality_reasons"] == ["observed_output_mismatch", "quality_evidence_invalid"]


def test_nonfile_quality_artifact_is_invalid_and_bound_to_the_snapshot(tmp_path):
    write(tmp_path / "runtime_validation.json", {"devices": {"on": {"quality_checks": []}}})
    evidence = tmp_path / "quality_checks.json"
    evidence.mkdir()
    snapshot = ResultMetadataSnapshot(tmp_path)
    check, = read_quality(tmp_path, metadata_snapshot=snapshot)["quality_checks"]
    assert check["code"] == "quality_evidence_invalid"
    assert check["evidence"]["artifact"] == str(evidence)
    snapshot.verify_current(tmp_path)
    evidence.rmdir()
    with pytest.raises(ValueError, match="metadata presence changed"):
        snapshot.verify_current(tmp_path)
    assert read_quality(tmp_path)["quality_status"] == "passed"


def test_invalid_device_attribution_cannot_turn_scoped_checks_into_passed(tmp_path):
    bad = {"code": "shape_error", "severity": "error", "detail": "bad shape", "evidence": {"device": ["off"]}}
    on = {"code": "weight_error", "severity": "error", "detail": "bad weights", "evidence": {"device": "on"}}
    write(tmp_path / "quality_checks.json", {"schema_version": 1, "checks": [bad, on]})
    off = read_quality(tmp_path, device="off")
    assert off["quality_status"] == "unknown"
    assert off["quality_reasons"] == ["quality_evidence_invalid"]
    assert read_quality(tmp_path, device="on")["quality_reasons"] == ["quality_evidence_invalid", "weight_error"]
    assert read_quality(tmp_path)["quality_reasons"] == ["quality_evidence_invalid", "shape_error", "weight_error"]
