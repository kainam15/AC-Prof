"""Direct file writers protect the authoritative paths recorded in their input."""
import json

import pytest

from acprof.analysis.export import export_analysis
from acprof.analysis.model import load_analysis
from acprof.analysis.statistics_report import build_statistics_report, save_unique_statistics_report
from acprof.plotting.report import write_report
from tests.support.independent_comparison_fixtures import IndependentComparisonFixture


@pytest.fixture
def runs(tmp_path):
    fixture = IndependentComparisonFixture()
    fixture.build(None, tmp_path)
    left = fixture.replicate("left", 0, [1, 2, 3])
    right = fixture.replicate("right", 0, [2, 3, 4])
    return fixture, left, right


@pytest.mark.parametrize("writer", [
    pytest.param(lambda report, path: export_analysis(report, path, format="json"), id="export"),
    pytest.param(lambda report, path: save_unique_statistics_report(path, report), id="save"),
])
def test_statistics_writer_checks_recorded_csv(runs, writer):
    _, source, _ = runs
    report = build_statistics_report(source, ["latency_app_s"], resamples=100)
    destination = source / "result_manifest.json"
    with pytest.raises(ValueError, match=destination.name):
        writer(report, destination)
    assert not destination.exists()


@pytest.mark.parametrize("side", ["left", "right"])
def test_comparison_export_checks_each_experiment(runs, side):
    fixture, left, right = runs
    report = fixture.compare([left], [right])
    destination = (left if side == "left" else right) / "summary.csv"
    with pytest.raises(ValueError, match=destination.name):
        export_analysis(report, destination, format="csv")
    assert not destination.exists()


def test_html_writer_checks_each_model_source(runs):
    _, left, right = runs
    model = load_analysis([left, right])
    destination = right / "result_manifest.json"
    with pytest.raises(ValueError, match=destination.name):
        write_report(model, destination)
    assert not destination.exists()


def test_unbound_historical_payload_can_be_exported_without_experiment_discovery(tmp_path):
    report = {"groups": [{"mean": 1.5}]}
    output = tmp_path / "result_manifest.json"
    export_analysis(report, output, format="json")
    assert json.loads(output.read_text()) == report
    saved, reused = save_unique_statistics_report(tmp_path / "metadata", report)
    assert not reused
    assert json.loads(saved.read_text()) == report


def test_archived_statistics_can_export_with_unreadable_original_layout(tmp_path):
    expired = tmp_path / "old"
    expired.mkdir()
    (expired / "result_manifest.json").write_bytes(b"{")
    payload = {"result_csv": str(expired / "result_all.csv"), "groups": []}
    target = tmp_path / "unrelated" / "result_manifest.json"
    export_analysis(payload, target, format="json")
    assert json.loads(target.read_bytes()) == payload


@pytest.mark.parametrize("kind", ["point", "sweep"])
def test_stream_writers_reject_layout_markers(tmp_path, kind):
    from acprof.plotting.load_report import write_load_report
    from acprof.plotting.load_sweep import write_load_sweep_report
    from tests.support.load_sweep_fixtures import make_sweep

    root = make_sweep(tmp_path / "stream")
    point = root / "points" / "0001"
    writer, source = (write_load_sweep_report, root) if kind == "sweep" else (write_load_report, point)
    target = source / "result_manifest.json"
    with pytest.raises(ValueError, match="reserved.*result_manifest"):
        writer(source, target)
    assert not target.exists()
