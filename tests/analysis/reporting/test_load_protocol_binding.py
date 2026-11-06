"""Single-point and sweep reports share declared load configuration checks."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from acprof.analysis.load_report import analyze_load_report
from acprof.analysis.load_sweep import read_load_sweep
from acprof.cli.main import main
from tests.analysis.reporting.test_load_report import experiment
from tests.support.load_sweep_fixtures import make_sweep


def _rehash_identity(payload: dict[str, Any]) -> None:
    payload["identity_sha256"] = hashlib.sha256(
        json.dumps(payload["identity"], sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


@pytest.fixture
def full_report() -> dict[str, Any]:
    payload = experiment()
    config = {
        "scenario": "concurrent", "connections": "close", "concurrency": 2,
        "requests": 1, "rate": None, "arrival": "fixed", "seed": 0,
        "max_pending": 1024, "timeout_seconds": 300,
    }
    payload["identity"]["load_config"] = config
    payload["result"]["protocol"] = {
        **config, "kind": "openai_stream_http_load", "schema_version": 1,
        "retry": "none", "clock": "perf_counter_relative_seconds",
    }
    _rehash_identity(payload)
    return payload


@pytest.mark.parametrize(("field", "value"), [
    ("scenario", "serial"), ("connections", "reuse"), ("concurrency", 3),
    ("requests", 2), ("rate", 10.0), ("arrival", "poisson"), ("seed", 1),
    ("max_pending", 2048), ("timeout_seconds", 30),
])
def test_declared_load_settings_must_match_result_protocol(full_report, field, value):
    full_report["result"]["protocol"][field] = value
    before = json.dumps(full_report, sort_keys=True)
    with pytest.raises(ValueError, match="declared.*(?:configuration|concurrency)"):
        analyze_load_report(full_report)
    assert json.dumps(full_report, sort_keys=True) == before


@pytest.mark.parametrize("field", [
    "scenario", "connections", "concurrency", "requests", "rate", "arrival",
    "seed", "max_pending", "timeout_seconds",
])
def test_declared_load_setting_cannot_be_missing_from_protocol(full_report, field):
    # An explicit null rate is still a declaration, not an absent field.
    del full_report["result"]["protocol"][field]
    with pytest.raises(ValueError, match="declared.*(?:configuration|concurrency)"):
        analyze_load_report(full_report)


@pytest.mark.parametrize(("field", "value"), [
    ("seed", False), ("requests", True), ("requests", 1.0),
    ("concurrency", 2.0), ("timeout_seconds", 300.0),
])
def test_protocol_cannot_substitute_equal_values_of_other_json_types(full_report, field, value):
    full_report["result"]["protocol"][field] = value
    with pytest.raises(ValueError, match="configuration.*protocol"):
        analyze_load_report(full_report)


@pytest.mark.parametrize(("field", "value"), [("connections", "reuse"), ("timeout_seconds", 30)])
def test_rehashed_declaration_still_requires_matching_protocol(full_report, field, value):
    full_report["identity"]["load_config"][field] = value
    _rehash_identity(full_report)
    with pytest.raises(ValueError, match="configuration.*protocol"):
        analyze_load_report(full_report)


def test_matching_configuration_allows_additional_protocol_details(full_report):
    before = json.dumps(full_report, sort_keys=True)
    report = analyze_load_report(full_report)
    assert report["capacity"]["successful_requests_per_s"] == 0.5
    assert report["capacity"]["client_concurrency"] == 2
    assert json.dumps(full_report, sort_keys=True) == before


@pytest.mark.parametrize("version", [1, 2])
def test_existing_partial_configuration_compatibility_is_preserved(version):
    payload = experiment(version)
    assert analyze_load_report(payload)["version"] == version


def test_sweep_still_binds_consistent_child_configuration_to_its_manifest(tmp_path):
    root = make_sweep(tmp_path / "sweep")
    source = root / "points/0002/load.json"
    payload = json.loads(source.read_text())
    payload["identity"]["load_config"]["seed"] = 1
    payload["result"]["protocol"]["seed"] = 1
    _rehash_identity(payload)
    assert analyze_load_report(payload)["version"] == 2
    source.write_text(json.dumps(payload))

    sweep = read_load_sweep(root).payload
    assert sweep["valid_points"] == 2
    assert sweep["points"][1]["evidence_status"] == "invalid"
    assert any("configuration" in message for message in sweep["points"][1]["diagnostics"])


def test_single_point_cli_and_sweep_reject_the_same_conflicting_protocol(tmp_path, capsys):
    root = make_sweep(tmp_path / "sweep")
    source = root / "points/0002/load.json"
    payload = json.loads(source.read_text())
    payload["result"]["protocol"]["connections"] = "reuse"
    source.write_text(json.dumps(payload))
    before = {path: path.read_bytes() for path in root.rglob("*.json")}
    output = tmp_path / "capacity.html"

    with pytest.raises(SystemExit) as caught:
        main(["report", str(source), "--output", str(output)])
    assert caught.value.code == 1
    assert "configuration" in capsys.readouterr().err
    assert not output.exists()

    sweep = read_load_sweep(root).payload
    assert sweep["valid_points"] == 2
    point = sweep["points"][1]
    assert point["evidence_status"] == "invalid"
    assert not point["curve_eligible"]
    assert point["metrics"] == {}
    assert any("configuration" in message for message in point["diagnostics"])
    assert sweep["points"][0]["metrics"]["successful_requests_per_s"] == 1
    assert {path: path.read_bytes() for path in root.rglob("*.json")} == before
