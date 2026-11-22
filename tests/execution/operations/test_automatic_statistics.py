"""Completed experiments publish statistics without starting another measurement."""
import csv
import io
import json
import re
from pathlib import Path
from unittest.mock import Mock

import pytest

from acprof.analysis import statistics_report
from acprof.application import publication
from acprof.application.notifications import RunNotifications
from acprof.application.publication import publish_matrix_results
from acprof.artifacts import file_sha256
from acprof.host.run_state import MeasurementLock, RunStateError
from acprof.result_store import recovery
from acprof.result_store.writer import ResultWriter
from acprof.results.csv import read_result_csv_snapshot
from acprof.results.layout import ArtifactLayout
from tests.support.result_store_fixtures import assert_writer_lease_available
from tests.support.run_recovery_fixtures import RunRecoveryFixture


@pytest.fixture
def experiment(request, tmp_path) -> RunRecoveryFixture:
    fixture = RunRecoveryFixture()
    fixture.build(request, tmp_path)
    return fixture


def reports(experiment: RunRecoveryFixture) -> list[Path]:
    return sorted((experiment.directory / "analysis").glob("window-statistics-*.json"))


def test_final_matrix_shares_one_replay_and_allows_readers_until_freeze(experiment, monkeypatch):
    scan = recovery.replay
    replayed = []
    apply = publication.apply_collection_result
    root = experiment.directory
    cleanup = publication.cleanup_intermediate_results

    def observe(layout, manifest):
        replayed.append((manifest["sequence"], manifest["row_count"]))
        return scan(layout, manifest)

    def check_report(report, rows):
        layout = ArtifactLayout.discover(root)
        assert read_result_csv_snapshot(layout.result_csv).rows == rows
        with pytest.raises(BlockingIOError):
            assert_writer_lease_available(layout)
        return apply(report, rows)

    def check_cleanup(paths, directory, destination):
        assert_writer_lease_available(ArtifactLayout.discover(directory))
        return cleanup(paths, directory, destination)

    monkeypatch.setattr(recovery, "replay", observe)
    monkeypatch.setattr(publication, "apply_collection_result", check_report)
    monkeypatch.setattr(publication, "cleanup_intermediate_results", check_cleanup)
    monkeypatch.setattr(publication, "publish_run_reports", assert_writer_lease_available)
    experiment.invoke()
    assert replayed == [(0, 0), (1, 1), (2, 2)]
    state = json.loads((root / ".acprof/run_state.json").read_bytes())
    snapshot = read_result_csv_snapshot(root / "result_layers.json")
    assert state["result_sha256"] == snapshot.sha256
    assert state["row_count"] == len(snapshot.rows)


def test_history_changed_during_capability_check_cannot_be_frozen(experiment, monkeypatch):
    apply = publication.apply_collection_result
    layout = ArtifactLayout.for_new_run(experiment.directory)
    heads = []

    def damage_derivation(report, rows):
        heads.append(layout.result_csv.read_bytes())
        update = next(layout.path("raw/result_updates").glob("*.json"))
        raw = update.read_bytes()
        assert raw.endswith(b"\n")
        update.write_bytes(raw[:-1] + b" ")
        return apply(report, rows)

    monkeypatch.setattr(publication, "apply_collection_result", damage_derivation)
    with pytest.raises(SystemExit) as stopped:
        experiment.invoke()
    assert stopped.value.code == 2
    assert heads == [layout.result_csv.read_bytes()]
    assert json.loads(layout.path("run_state.json").read_bytes())["status"] == "failed"
    assert_writer_lease_available(layout)


def test_valid_case_replacement_is_rejected_before_final_publication_and_repaired_on_resume(experiment, monkeypatch):
    root = experiment.directory
    before = {}
    event = publication.emit_event

    def replace_case(name, *args, **kwargs):
        if name == "results_merging":
            before["manifest"] = (root / "result_layers.json").read_bytes()
            case = ArtifactLayout.discover(root).case("org/model", 1, 4, "off")
            with case.csv.open(newline="") as stream:
                row, = csv.DictReader(stream)
            row["latency_app_s"] = "97"
            with case.csv.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=row)
                writer.writeheader()
                writer.writerow(row)
        event(name, *args, **kwargs)

    stderr = io.StringIO()
    with monkeypatch.context() as scoped:
        scoped.setattr(publication, "emit_event", replace_case)
        with pytest.raises(SystemExit) as caught:
            experiment.invoke(error_output=stderr)
    assert caught.value.code == 2
    assert "case CSV" in stderr.getvalue()
    assert (root / "result_layers.json").read_bytes() == before["manifest"]
    assert json.loads((root / ".acprof/run_state.json").read_bytes())["status"] == "failed"
    assert not reports(experiment)
    experiment.calls.clear()
    experiment.invoke("--resume")
    assert experiment.calls == []
    assert [row["latency_app_s"] for row in read_result_csv_snapshot(root / "result_layers.json").rows] == ["0.1", "0.1"]


def test_freeze_release_error_precedes_run_state_completion(experiment, monkeypatch):
    release = ResultWriter.__exit__

    def fail_after_freeze(writer, *exc):
        result = release(writer, *exc)
        if writer.manifest["state"] == "complete":
            raise OSError("fixture final lease release failed")
        return result

    with monkeypatch.context() as scoped:
        scoped.setattr(ResultWriter, "__exit__", fail_after_freeze)
        with pytest.raises(OSError, match="final lease release failed"):
            experiment.invoke()
    layout = ArtifactLayout.discover(experiment.directory)
    assert_writer_lease_available(layout)
    assert json.loads(layout.result_csv.read_bytes())["state"] == "complete"
    assert json.loads(layout.path("run_state.json").read_bytes())["status"] == "failed"
    assert not reports(experiment)


@pytest.mark.parametrize(("error", "outcome"), [(OSError, "failed"), (KeyboardInterrupt, "interrupted")])
def test_recorded_coverage_keeps_legacy_full_capability_after_final_freeze_fails(
        experiment, monkeypatch, tmp_path, error, outcome):
    from acprof.capabilities import measurement_report
    from acprof.cli.coverage import main
    from acprof.platform import Environment

    experiment.interrupt_after_first()
    layout = ArtifactLayout.discover(experiment.directory)
    # Restore the preparation snapshot a historical full-policy run would have frozen.
    metadata_path, state_path = layout.path("static_meta.json"), layout.path("run_state.json")
    metadata, state = json.loads(metadata_path.read_bytes()), json.loads(state_path.read_bytes())
    frozen = measurement_report("full", gpu_modes=["off"], environment=Environment("native_linux")).to_dict()
    assert frozen["full_profile_complete"] is False
    metadata.update(profiling_mode="full", capability_report=frozen)
    metadata_path.write_text(json.dumps(metadata))
    state["options"]["profiling_mode"] = "full"
    state["artifacts"]["static_meta.json"] = file_sha256(metadata_path)
    state_path.write_text(json.dumps(state))

    def fail_freeze(writer):
        assert json.loads(writer.layout.path("capability_report.json").read_bytes())["full_profile_complete"] is True
        raise error("fixture final freeze failure")

    with monkeypatch.context() as scoped:
        scoped.setattr(ResultWriter, "freeze", fail_freeze)
        with pytest.raises(error, match="fixture final freeze failure"):
            experiment.invoke("--resume")
    state = json.loads(state_path.read_bytes())
    assert state["status"] == state["attempts"][-1]["outcome"] == outcome
    assert experiment.calls == [2]

    output = tmp_path / "coverage-report"
    assert main(["report", str(experiment.directory), "--output-dir", str(output)]) == 0
    recorded, = json.loads((output / "coverage.json").read_bytes())["rows"]
    with (output / "models.csv").open(newline="") as stream:
        exported, = csv.DictReader(stream)
    assert exported["status"] == outcome
    assert recorded["status"] == recorded["run_status"] == outcome
    assert recorded["full_profile_complete"] is True
    assert recorded["runtime_status"] == "ok"


def test_completed_state_identifies_its_freeze_when_another_writer_takes_over(experiment, monkeypatch):
    release = ResultWriter.__exit__
    frozen = []

    def take_over(writer, *exc):
        result = release(writer, *exc)
        if writer.manifest["state"] == "complete" and not frozen:
            frozen.append(file_sha256(writer.layout.result_csv))
            with ResultWriter(writer.layout.root) as other:
                other.freeze()
        return result

    monkeypatch.setattr(ResultWriter, "__exit__", take_over)
    monkeypatch.setattr(publication, "publish_run_reports", lambda layout: None)
    experiment.invoke()
    layout = ArtifactLayout.discover(experiment.directory)
    state = json.loads(layout.path("run_state.json").read_bytes())
    assert state["status"] == "complete"
    assert [state["result_sha256"]] == frozen
    assert state["result_sha256"] != file_sha256(layout.result_csv)


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
    layout = ArtifactLayout.discover(experiment.directory)
    assert_writer_lease_available(layout)
    assert json.loads(layout.result_csv.read_bytes())["state"] == "processing"
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
