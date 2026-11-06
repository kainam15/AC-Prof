"""Offline sweep curves keep every point, its conditions and its evidence limits."""
from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from importlib import import_module
from pathlib import Path

import pytest

from acprof.cli.main import main
from acprof.host.measurement.load import LoadConfig
from acprof.plotting.load_sweep import write_load_sweep_report
from tests.support.load_sweep_fixtures import change_manifest, change_point, make_sweep


@pytest.fixture(name="sweep_reader")
def reader_module():
    return import_module("acprof.analysis.load_sweep")


def test_public_report_accepts_a_sweep_directory_and_exclusive_default_output(tmp_path, monkeypatch, capsys):
    root = make_sweep(tmp_path / "sweep")
    monkeypatch.setattr("acprof.host.run_state.offline_analysis_lock", nullcontext)
    assert main(["report", str(root)]) == 0
    output = root / "load_sweep_report.html"
    assert output.is_file()
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("ACPROF_REPORT ")]
    summary = json.loads(lines[0].split(" ", 1)[1])
    assert summary["kind"] == "stream_load_sweep_capacity"
    assert summary["point_count"] == 3
    assert summary["valid_points"] == 3
    original = output.read_bytes()
    with pytest.raises(SystemExit) as caught:
        main(["report", str(root)])
    assert caught.value.code == 1
    assert output.read_bytes() == original


def test_sweep_retains_order_independent_metrics_and_compact_conditions(sweep_reader, tmp_path):
    root = make_sweep(tmp_path / "sweep")
    analysis = sweep_reader.read_load_sweep(root)
    data = analysis.payload
    assert data["kind"] == "stream_load_sweep_analysis"
    assert [point["point_id"] for point in data["points"]] == ["0001", "0002", "0003"]
    assert [point["metrics"]["successful_requests_per_s"] for point in data["points"]] == [1, 2, 3]
    assert [point["metrics"]["output_tokens_per_s"] for point in data["points"]] == [3, 6, 9]
    assert data["points"][1]["metrics"]["ttft_p99_s"] == pytest.approx(0.2)
    assert data["points"][0]["metrics"]["goodput_requests_per_s"] == 1
    assert len(data["conditions"]) == 1
    assert all(point["curve_eligible"] for point in data["points"])
    assert all(point["latency_samples"]["ttft_s"] == 4 for point in data["points"])
    serialized = json.dumps(data)
    assert '"request_body":' not in serialized
    assert '"result":' not in serialized
    assert "same input" not in serialized
    assert len(analysis.sources) == 4


def test_completed_request_failures_remain_curve_points(sweep_reader, tmp_path):
    root = make_sweep(tmp_path / "sweep")
    change_point(root, 2, lambda experiment: experiment["result"]["requests"][0].update(status="error"))
    change_manifest(root, lambda manifest: (manifest.update(successful=False),
                                            manifest["points"][1].update(successful=False)))
    data = sweep_reader.read_load_sweep(root).payload
    point = data["points"][1]
    assert point["curve_eligible"] is True
    assert point["counts"]["failed"] == 1
    assert point["success_rate"] == 0.75
    assert point["metrics"]["successful_requests_per_s"] == 1.5


def test_cancelled_and_unrun_points_remain_visible_without_joining_curves(sweep_reader, tmp_path):
    root = make_sweep(tmp_path / "sweep")
    change_point(root, 2, lambda point: (
        point.update(status="cancelled"),
        point["result"]["requests"][0].update(status="cancelled"),
    ))
    change_manifest(root, lambda manifest: (
        manifest.update(status="cancelled", successful=False),
        manifest["points"][1].update(status="cancelled", successful=False, run_id=None),
        manifest["points"][2].update(status="not_run", successful=None, run_id=None),
    ))
    (root / "points/0003/load.json").write_text("must not read unrun artifacts")
    points = sweep_reader.read_load_sweep(root).payload["points"]
    assert points[0]["curve_eligible"] is True
    assert points[1]["evidence_status"] == "partial"
    assert points[1]["metrics"]["successful_requests_per_s"] == 1.5
    assert points[1]["curve_eligible"] is False
    assert points[2]["evidence_status"] == "not_run"
    assert points[2]["metrics"] == {}


@pytest.mark.parametrize("change", [
    lambda point: point["identity"]["request_body"].update(temperature=0.8),
    lambda point: point["host"].update(hostname="different-host"),
    lambda point: point["identity"]["slo_thresholds_s"].update(ttft_s=1),
    lambda point: point["service_identity"]["observed"].update(version="different-version"),
    lambda point: point["service_identity"]["observed"].update(configuration={
        "source": "/server_info", "scope": "reported_startup_configuration", "values": {"tp_size": 2},
    }),
    lambda point: point["result"]["requests"][0].update(output_token_count_source="matched_local_tokenizer"),
])
def test_changed_conditions_are_separate_groups(sweep_reader, tmp_path, change):
    root = make_sweep(tmp_path / "sweep")
    change_point(root, 2, change)
    data = sweep_reader.read_load_sweep(root).payload
    assert len(data["conditions"]) == 2
    points = data["points"]
    assert points[0]["condition_id"] == points[2]["condition_id"]
    assert points[1]["condition_id"] != points[0]["condition_id"]


def test_missing_token_and_slo_evidence_is_not_zero(sweep_reader, tmp_path):
    root = make_sweep(tmp_path / "sweep")
    change_point(root, 2, lambda experiment: (
        experiment["identity"].update(slo_thresholds_s={}),
        experiment["result"]["requests"][0].update(output_tokens=None, tpot_s=None),
    ))
    point = sweep_reader.read_load_sweep(root).payload["points"][1]
    assert point["metrics"]["goodput_requests_per_s"] is None
    assert point["metrics"]["output_tokens_per_s"] is None
    assert point["output_token_coverage"] == 0.75
    assert point["latency_samples"]["tpot_s"] == 3


@pytest.mark.parametrize("corruption", ["missing", "bad-json", "run-id", "config", "protocol", "capacity", "success"])
def test_bad_child_is_diagnostic_and_does_not_erase_other_points(sweep_reader, tmp_path, corruption):
    root = make_sweep(tmp_path / "sweep")
    path = root / "points/0002/load.json"
    if corruption == "missing":
        path.unlink()
    elif corruption == "bad-json":
        path.write_text("{broken")
    else:
        payload = json.loads(path.read_text())
        if corruption == "run-id":
            payload["run_id"] = "wrong-run"
        elif corruption == "config":
            payload["identity"]["load_config"]["requests"] = 8
        elif corruption == "protocol":
            payload["result"]["protocol"]["connections"] = "reuse"
        elif corruption == "capacity":
            payload["result"]["capacity"]["successful_requests_per_s"] = 100
        else:
            payload["successful"] = False
        path.write_text(json.dumps(payload))
    data = sweep_reader.read_load_sweep(root).payload
    assert data["valid_points"] == 2
    assert data["points"][1]["evidence_status"] in {"missing", "invalid"}
    assert data["points"][1]["metrics"] == {}
    assert data["points"][1]["diagnostics"]
    assert data["points"][0]["metrics"]["successful_requests_per_s"] == 1


@pytest.mark.parametrize("change", [
    lambda manifest: manifest.update(schema_version=True),
    lambda manifest: manifest.update(kind="other"),
    lambda manifest: manifest.update(points=[]),
    lambda manifest: manifest["points"][1].update(point_id="0001"),
    lambda manifest: manifest["points"][1].update(report_path="../load.json"),
    lambda manifest: manifest["points"][1].update(report_path="/tmp/load.json"),
    lambda manifest: manifest["points"][1].update(load_config=manifest["points"][0]["load_config"]),
    lambda manifest: manifest["points"][1].update(status="unknown"),
])
def test_malformed_manifest_is_rejected(sweep_reader, tmp_path, change):
    root = make_sweep(tmp_path / "sweep")
    change_manifest(root, change)
    with pytest.raises(ValueError, match="sweep|point|schema|configuration|path"):
        sweep_reader.read_load_sweep(root)


def test_child_symlink_is_never_read(sweep_reader, tmp_path, monkeypatch):
    root = make_sweep(tmp_path / "sweep")
    outside = tmp_path / "outside.json"
    outside.write_text("private fixture")
    path = root / "points/0002/load.json"
    path.unlink()
    path.symlink_to(outside)
    def checked_open(opened: Path, *args, **kwargs):
        assert opened != outside
        assert opened != root / "points/0002/load.json"
        return open(opened, *args, **kwargs)

    monkeypatch.setattr(Path, "open", checked_open)
    data = sweep_reader.read_load_sweep(root).payload
    assert data["points"][1]["evidence_status"] == "invalid"
    assert "symlink" in " ".join(data["points"][1]["diagnostics"])


def test_source_changes_are_detected_before_publication(sweep_reader, tmp_path):
    root = make_sweep(tmp_path / "sweep")
    analysis = sweep_reader.read_load_sweep(root)
    path = root / "points/0001/load.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed"):
        sweep_reader.verify_load_sweep_sources(analysis)


def test_two_dimensional_sweep_preserves_both_axes(sweep_reader, tmp_path):
    configs = [LoadConfig(scenario="arrival-rate", concurrency=workers, rate=rate, requests=4)
               for workers in (1, 2) for rate in (20, 10)]
    root = make_sweep(tmp_path / "sweep", configs)
    data = sweep_reader.read_load_sweep(root).payload
    assert [(point["load_config"]["concurrency"], point["load_config"]["rate"]) for point in data["points"]] == [
        (1, 20), (1, 10), (2, 20), (2, 10),
    ]
    assert len(data["conditions"]) == 1


@pytest.mark.parametrize("corruption", ["body-digest", "bool-protocol", "count-budget", "old-version", "invalid-row"])
def test_inconsistent_child_evidence_is_excluded(sweep_reader, tmp_path, corruption):
    root = make_sweep(tmp_path / "sweep")
    path = root / "points/0001/load.json"
    payload = json.loads(path.read_text())
    if corruption == "body-digest":
        payload["identity"]["request_body"]["messages"][0]["content"] = "changed input"
        payload["identity_sha256"] = hashlib.sha256(json.dumps(payload["identity"], sort_keys=True).encode()).hexdigest()
    elif corruption == "bool-protocol":
        payload["result"]["protocol"]["seed"] = False
    elif corruption == "count-budget":
        payload["result"]["counts"]["offered"] = 10
    elif corruption == "old-version":
        payload["schema_version"] = payload["result"]["schema_version"] = 1
    else:
        payload["result"]["requests"][0] = 17
    path.write_text(json.dumps(payload))
    point = sweep_reader.read_load_sweep(root).payload["points"][0]
    assert point["evidence_status"] == "invalid"
    assert point["metrics"] == {}


def test_stale_manifest_exposes_child_status_without_claiming_complete(sweep_reader, tmp_path):
    root = make_sweep(tmp_path / "sweep")
    change_manifest(root, lambda manifest: (manifest.update(status="running", successful=False),
                                            manifest["points"][2].update(status="running", successful=None, run_id=None)))
    point = sweep_reader.read_load_sweep(root).payload["points"][2]
    assert point["evidence_status"] == "partial"
    assert point["report_status"] == "complete"
    assert point["status"] == "running"
    assert point["diagnostics"]
    assert point["metrics"]["successful_requests_per_s"] == 3


@pytest.mark.parametrize("mutation", ["append", "missing-appears", "symlink"])
def test_changed_sources_never_publish_html(tmp_path, monkeypatch, mutation):
    root = make_sweep(tmp_path / "sweep")
    point = root / "points/0002/load.json"
    original = point.read_bytes()
    if mutation == "missing-appears":
        point.unlink()

    def changed_plotly():
        if mutation == "append":
            point.write_bytes(original + b"\n")
        elif mutation == "missing-appears":
            point.write_bytes(original)
        else:
            point.unlink()
            point.symlink_to(root / "points/0001/load.json")
        return ""

    monkeypatch.setattr("plotly.offline.get_plotlyjs", changed_plotly)
    output = tmp_path / "report.html"
    with pytest.raises(ValueError, match="changed|symlink"):
        write_load_sweep_report(root, output)
    assert not output.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["sweep"]


def test_output_collision_during_rendering_preserves_existing_bytes(tmp_path, monkeypatch):
    root = make_sweep(tmp_path / "sweep")
    output = tmp_path / "report.html"

    def collision():
        output.write_text("existing report")
        return ""

    monkeypatch.setattr("plotly.offline.get_plotlyjs", collision)
    with pytest.raises(FileExistsError):
        write_load_sweep_report(root, output)
    assert output.read_text() == "existing report"


@pytest.mark.parametrize("not_run", [False, True])
def test_output_cannot_create_a_missing_planned_point(tmp_path, not_run):
    root = make_sweep(tmp_path / "sweep")
    output = root / "points/0002/load.json"
    output.unlink()
    if not_run:
        change_manifest(root, lambda manifest: (manifest.update(status="failed", successful=False),
                                                manifest["points"][1].update(status="not_run", successful=None, run_id=None)))
    with pytest.raises(ValueError, match="planned point"):
        write_load_sweep_report(root, output)
    assert not output.exists()


@pytest.mark.parametrize("extra", [
    ["--baseline", "other"], ["--comparison-purpose", "same-hardware"], ["load"],
    ["--metric", "latency_app_s"], ["--confidence", "0.95"], ["--resamples", "100"],
    ["--seed", "0"], ["--block-size", "1"], ["--left", "left", "--right", "right"],
])
def test_sweep_cli_rejects_mixed_inputs_or_comparison_options(tmp_path, monkeypatch, capsys, extra):
    root = make_sweep(tmp_path / "sweep")
    monkeypatch.setattr("acprof.host.run_state.offline_analysis_lock", nullcontext)
    with pytest.raises(SystemExit) as caught:
        main(["report", str(root / "sweep.json"), *extra])
    assert caught.value.code == 1
    assert "without mixed inputs or comparison options" in capsys.readouterr().err
    assert not (root / "load_sweep_report.html").exists()


def test_sweep_in_independent_groups_is_rejected_even_with_positional_sources(tmp_path, monkeypatch, capsys):
    root = make_sweep(tmp_path / "sweep")
    monkeypatch.setattr("acprof.host.run_state.offline_analysis_lock", nullcontext)
    with pytest.raises(SystemExit) as caught:
        main(["report", str(tmp_path / "formal.csv"), "--left", str(root), "--right", "other"])
    assert caught.value.code == 1
    assert "without mixed inputs or comparison options" in capsys.readouterr().err
