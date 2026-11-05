"""Completed experiments publish statistics without starting another measurement."""
import io
import json
import re
from pathlib import Path
from unittest.mock import Mock

import pytest

from acprof.analysis import statistics_report
from acprof.application.notifications import RunNotifications
from acprof.application.publication import publish_matrix_results
from acprof.host.run_state import MeasurementLock, RunStateError
from acprof.results.csv import read_result_csv_snapshot
from acprof.results.layout import ArtifactLayout
from tests.support.run_recovery_fixtures import RunRecoveryFixture


@pytest.fixture
def experiment(request, tmp_path) -> RunRecoveryFixture:
    fixture = RunRecoveryFixture()
    fixture.build(request, tmp_path)
    return fixture


def reports(experiment: RunRecoveryFixture) -> list[Path]:
    return sorted((experiment.directory / "analysis").glob("window-statistics-*.json"))


def test_completed_run_saves_statistics_after_commit_under_the_measurement_lock(experiment, monkeypatch):
    original = statistics_report.build_statistics_report
    snapshots = []

    def build(path):
        state = json.loads((experiment.directory / ".acprof/run_state.json").read_text())
        assert state["status"] == "complete"
        assert experiment.calls == [1, 2]
        with pytest.raises(RunStateError, match="本机测量锁"), MeasurementLock():
            pytest.fail("statistics must exclude another measurement")
        snapshot = read_result_csv_snapshot(path)
        snapshots.append(snapshot.sha256)
        return original(path)

    def collect(request, **kwargs):
        assert not (experiment.directory / "analysis").exists()
        return experiment.write_case(request, **kwargs)

    monkeypatch.setattr(statistics_report, "build_statistics_report", build)
    output = io.StringIO()
    experiment.invoke(case=collect, output=output)

    assert len(reports(experiment)) == 1
    saved = reports(experiment)[0]
    assert re.fullmatch(r"window-statistics-\d{8}-\d{6}-\d{6}\.json", saved.name)
    report = json.loads(saved.read_text())
    assert snapshots == [report["result_sha256"]]
    assert report["result_csv"] == str(experiment.directory / "result_layers.json")
    assert report["run_status"] == "complete"
    assert report["confidence"] == 0.95
    assert report["resamples"] == 5000
    assert report["seed"] == 0
    assert report["block_size"] == 1
    assert len(report["groups"]) == 6
    groups = {group["metric"]: group for group in report["groups"] if group["cpu_cores"] == 1}
    assert groups["latency_app_s"]["mean"] == 0.1
    assert groups["latency_s"]["mean"] == 0.09
    assert groups["latency_app_s"]["n_windows"] == 1
    assert groups["latency_app_s"]["ci_low"] is None
    assert groups["container_attributed_energy_eff_j"]["mean"] is None
    assert str(saved) in output.getvalue()
    assert not list(experiment.directory.glob("plots/analysis/window-statistics-*.json"))
    with MeasurementLock():
        pass


@pytest.mark.parametrize("keep_report", [False, True])
def test_completed_resume_creates_or_reuses_statistics_without_recollecting(experiment, keep_report):
    experiment.invoke()
    assert len(reports(experiment)) == 1
    saved = reports(experiment)[0]
    original, modified = saved.read_bytes(), saved.stat().st_mtime_ns
    source_files = [*experiment.directory.glob("*.csv"), experiment.directory / "result_layers.json"]
    source = {path: path.read_bytes() for path in source_files}
    if not keep_report:
        saved.unlink()
    experiment.calls.clear()
    output = io.StringIO()

    experiment.invoke("--resume", output=output)

    assert experiment.calls == []
    assert len(reports(experiment)) == 1
    assert reports(experiment)[0].read_bytes() == original
    if keep_report:
        assert reports(experiment) == [saved]
        assert saved.stat().st_mtime_ns == modified
        assert "已有相同报告" in output.getvalue()
    assert str(reports(experiment)[0]) in output.getvalue()
    assert {path: path.read_bytes() for path in source} == source


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize(("stage", "error_type"), [
    ("build_statistics_report", ValueError),
    ("save_unique_statistics_report", OSError),
])
def test_statistics_failure_is_visible_preserves_results_and_can_be_retried(
        experiment, monkeypatch, resume, stage, error_type):
    if resume:
        experiment.invoke()
    previous = {path: path.read_bytes() for path in reports(experiment)}
    failure = error_type("statistics fixture failure")
    failure.add_note("statistics diagnostic context")
    stderr = io.StringIO()
    with monkeypatch.context() as scoped:
        scoped.setattr(statistics_report, stage, Mock(side_effect=failure))
        with pytest.raises(SystemExit) as caught:
            experiment.invoke(*(["--resume"] if resume else []), error_output=stderr)
    assert caught.value.code == 1
    assert "自动统计失败" in stderr.getvalue()
    assert "statistics fixture failure" in stderr.getvalue()
    assert "statistics diagnostic context" in stderr.getvalue()
    assert "acprof stats" in stderr.getvalue()
    state = json.loads((experiment.directory / ".acprof/run_state.json").read_text())
    assert state["status"] == "complete"
    assert len(read_result_csv_snapshot(experiment.directory / "result_layers.json").rows) == 2
    assert {path: path.read_bytes() for path in reports(experiment)} == previous
    experiment.calls.clear()

    experiment.invoke("--resume")

    assert experiment.calls == []
    assert len(reports(experiment)) == 1


def test_interrupted_run_generates_statistics_only_after_resume_finishes(experiment):
    experiment.interrupt_after_first()
    assert not reports(experiment)

    experiment.invoke("--resume")

    assert experiment.calls == [2]
    assert len(reports(experiment)) == 1
    assert json.loads(reports(experiment)[0].read_text())["run_status"] == "complete"


def test_required_measurement_failure_does_not_publish_statistics(experiment):
    experiment.assert_missing_required_measurements_rejected("basic")
    assert not reports(experiment)


def test_no_results_does_not_publish_statistics(tmp_path):
    layout = ArtifactLayout(tmp_path)
    outcome = publish_matrix_results([], args=Mock(), layout=layout, run_state=Mock(),
                                     capability_report=Mock(), start_time=0,
                                     notifications=Mock(spec=RunNotifications))
    assert outcome.status == "no_results"
    assert not (tmp_path / "analysis").exists()


def test_statistics_cancellation_preserves_committed_results_and_releases_locks(experiment, monkeypatch):
    with monkeypatch.context() as scoped:
        scoped.setattr(statistics_report, "build_statistics_report", Mock(side_effect=KeyboardInterrupt))
        with pytest.raises(KeyboardInterrupt):
            experiment.invoke()
    assert not reports(experiment)
    assert json.loads((experiment.directory / ".acprof/run_state.json").read_text())["status"] == "complete"
    with MeasurementLock():
        pass
