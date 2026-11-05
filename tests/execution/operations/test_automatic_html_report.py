"""Publish the offline HTML only after committed experiments, including Resume."""
import io
import json
import re
from pathlib import Path
from unittest.mock import Mock

import pytest

from acprof.analysis import model, statistics_report
from acprof.application.notifications import RunNotifications
from acprof.application.publication import publish_matrix_results
from acprof.host.run_state import MeasurementLock, RunStateError
from acprof.plotting import report
from acprof.results.csv import read_result_csv_snapshot
from acprof.results.layout import ArtifactLayout
from tests.support.run_recovery_fixtures import RunRecoveryFixture


@pytest.fixture
def experiment(request, tmp_path) -> RunRecoveryFixture:
    fixture = RunRecoveryFixture()
    fixture.build(request, tmp_path)
    return fixture


def reports(experiment: RunRecoveryFixture) -> list[Path]:
    return sorted((experiment.directory / "plots/analysis").glob("report-*.html"))


def payload(path: Path) -> dict:
    html = path.read_text(encoding="utf-8")
    match = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', html, re.S)
    assert match is not None
    assert html.rstrip().endswith("</html>")
    return json.loads(match.group(1))


def test_completed_run_publishes_html_after_statistics_under_measurement_lock(experiment, monkeypatch):
    original = report.write_report
    snapshots = []

    def write(analysis, output, **kwargs):
        assert experiment.calls == [1, 2]
        assert json.loads((experiment.directory / ".acprof/run_state.json").read_text())["status"] == "complete"
        assert len(list((experiment.directory / "analysis").glob("window-statistics-*.json"))) == 1
        with pytest.raises(RunStateError, match="本机测量锁"), MeasurementLock():
            pytest.fail("HTML generation must exclude another measurement")
        snapshots.append(read_result_csv_snapshot(experiment.directory / "result_layers.json").sha256)
        with monkeypatch.context() as scoped:
            scoped.setattr("acprof.analysis.uncertainty.bootstrap_mean_interval",
                           Mock(side_effect=AssertionError("HTML must reuse the just-published JSON")))
            return original(analysis, output, **kwargs)

    def collect(request, **kwargs):
        assert not reports(experiment)
        return experiment.write_case(request, **kwargs)

    monkeypatch.setattr(report, "write_report", write)
    stdout = io.StringIO()
    experiment.invoke(case=collect, output=stdout)

    assert len(reports(experiment)) == 1
    saved = reports(experiment)[0]
    data = payload(saved)
    assert len(data["configs"]) == 2
    statistics = data["window_statistics"][0]
    assert statistics["origin"] == "cached"
    assert Path(statistics["report_path"]).parent == experiment.directory / "analysis"
    assert snapshots == [statistics["result_sha256"]]
    assert statistics["audit"]["run_status"] == "complete"
    assert str(saved) in stdout.getvalue()
    assert stdout.getvalue().index(str(saved)) < stdout.getvalue().index("Profiling complete!")
    with MeasurementLock():
        pass


@pytest.mark.parametrize("keep_html", [False, True])
def test_completed_resume_generates_html_without_remeasuring_or_overwriting(experiment, keep_html):
    experiment.invoke()
    assert len(reports(experiment)) == 1
    saved = reports(experiment)[0]
    before, modified = saved.read_bytes(), saved.stat().st_mtime_ns
    source_files = [*experiment.directory.glob("*.csv"), experiment.directory / "result_layers.json",
                    *experiment.directory.glob("analysis/*.json")]
    source = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in source_files}
    if not keep_html:
        saved.unlink()
    experiment.calls.clear()
    stdout = io.StringIO()

    experiment.invoke("--resume", output=stdout)

    assert experiment.calls == []
    assert len(reports(experiment)) == (2 if keep_html else 1)
    fresh = next(path for path in reports(experiment) if path != saved)
    assert payload(fresh)["window_statistics"][0]["origin"] == "cached"
    assert str(fresh) in stdout.getvalue()
    if keep_html:
        assert (saved.read_bytes(), saved.stat().st_mtime_ns) == (before, modified)
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in source} == source


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize(("stage", "error_type"), [
    ("load", ValueError), ("write", OSError), ("write", ImportError),
])
def test_html_failure_is_visible_and_resume_retries_without_remeasuring(
        experiment, monkeypatch, resume, stage, error_type):
    if resume:
        experiment.invoke()
    previous = {path: path.read_bytes() for path in reports(experiment)}
    failure = error_type("HTML fixture failure")
    failure.add_note("HTML diagnostic context")
    stderr, stdout = io.StringIO(), io.StringIO()
    with monkeypatch.context() as scoped:
        scoped.setattr(model if stage == "load" else report,
                       "load_analysis" if stage == "load" else "write_report", Mock(side_effect=failure))
        with pytest.raises(SystemExit) as caught:
            experiment.invoke(*(["--resume"] if resume else []), error_output=stderr, output=stdout)
    assert caught.value.code == 1
    assert "自动 HTML 报告生成失败" in stderr.getvalue()
    assert "HTML fixture failure" in stderr.getvalue()
    assert "HTML diagnostic context" in stderr.getvalue()
    assert "acprof report" in stderr.getvalue()
    assert "--resume" in stderr.getvalue()
    assert "HTML 报告已保存" not in stdout.getvalue()
    assert "Profiling complete!" not in stdout.getvalue()
    assert json.loads((experiment.directory / ".acprof/run_state.json").read_text())["status"] == "complete"
    assert len(read_result_csv_snapshot(experiment.directory / "result_layers.json").rows) == 2
    assert {path: path.read_bytes() for path in reports(experiment)} == previous
    statistics = {path: path.read_bytes() for path in experiment.directory.glob("analysis/*.json")}
    assert len(statistics) == 1
    experiment.calls.clear()

    experiment.invoke("--resume")

    assert experiment.calls == []
    assert len(reports(experiment)) == len(previous) + 1
    assert {path: path.read_bytes() for path in statistics} == statistics


def test_interrupted_run_generates_html_only_when_resume_completes(experiment):
    experiment.interrupt_after_first()
    assert not reports(experiment)

    experiment.invoke("--resume")

    assert experiment.calls == [2]
    assert len(reports(experiment)) == 1
    assert payload(reports(experiment)[0])["window_statistics"][0]["audit"]["run_status"] == "complete"


def test_required_measurement_failure_does_not_publish_html(experiment):
    experiment.assert_missing_required_measurements_rejected("basic")
    assert not reports(experiment)


def test_no_results_does_not_publish_html(tmp_path):
    layout = ArtifactLayout(tmp_path)
    outcome = publish_matrix_results([], args=Mock(), layout=layout, run_state=Mock(),
                                     capability_report=Mock(), start_time=0,
                                     notifications=Mock(spec=RunNotifications))
    assert outcome.status == "no_results"
    assert not list(tmp_path.rglob("*.html"))


def test_statistics_failure_does_not_publish_html(experiment, monkeypatch):
    monkeypatch.setattr(statistics_report, "build_statistics_report", Mock(side_effect=ValueError("invalid statistics")))
    with pytest.raises(SystemExit) as caught:
        experiment.invoke()
    assert caught.value.code == 1
    assert not reports(experiment)


def test_html_cancellation_preserves_results_and_releases_locks(experiment, monkeypatch):
    with monkeypatch.context() as scoped:
        scoped.setattr(report, "write_report", Mock(side_effect=KeyboardInterrupt))
        with pytest.raises(KeyboardInterrupt):
            experiment.invoke()
    assert not reports(experiment)
    assert json.loads((experiment.directory / ".acprof/run_state.json").read_text())["status"] == "complete"
    with MeasurementLock():
        pass
    experiment.calls.clear()
    experiment.invoke("--resume")
    assert experiment.calls == []
    assert len(reports(experiment)) == 1


def test_competing_html_destination_is_preserved_and_failure_visible(experiment, monkeypatch):
    original = report.write_report
    competing = []

    def write(analysis, output, **kwargs):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"another writer's HTML")
        competing.append(output)
        return original(analysis, output, **kwargs)

    monkeypatch.setattr(report, "write_report", write)
    stderr = io.StringIO()
    with pytest.raises(SystemExit) as caught:
        experiment.invoke(error_output=stderr)
    assert caught.value.code == 1
    assert "自动 HTML 报告生成失败" in stderr.getvalue()
    assert len(competing) == 1
    assert competing[0].read_bytes() == b"another writer's HTML"
    assert not list(competing[0].parent.glob(".*.tmp"))
