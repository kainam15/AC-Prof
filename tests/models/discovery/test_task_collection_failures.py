"""Task manifests preserve interruption evidence when their own publication fails."""
import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from acprof import artifacts
from acprof.application import task_runs
from acprof.application.run import RunOutcome
from acprof.experiments.configuration import RunConfig
from acprof.host.run_state import ResultDirectoryLock
from tests.models.discovery.test_task_runs import selected_model


@pytest.fixture
def collection(tmp_path):
    config = replace(RunConfig.smoke("fixture/model"), output_dir=str(tmp_path))
    return task_runs.TaskCollection(config, selected_model())


def replace_failures(monkeypatch, collection, failures):
    replace_file = artifacts.os.replace
    attempted = []

    def publish(source, destination):
        if Path(destination) == collection.path:
            payload = json.loads(Path(source).read_text())
            attempted.append(payload["status"])
            if payload["status"] in failures:
                raise failures[payload["status"]]
        return replace_file(source, destination)

    monkeypatch.setattr(artifacts.os, "replace", publish)
    return attempted


def assert_released(collection):
    with ResultDirectoryLock(collection.root / "metadata/task_run_locks" / collection.run_id):
        pass
    assert list(collection.path.parent.iterdir()) == [collection.path]


@pytest.mark.parametrize("cancellation_type", [KeyboardInterrupt, asyncio.CancelledError])
@pytest.mark.parametrize("save_failure_type", [PermissionError, KeyboardInterrupt, asyncio.CancelledError])
def test_interrupted_save_preserves_original_cancel_and_last_committed_tasks(
        collection, monkeypatch, cancellation_type, save_failure_type):
    cancellation = cancellation_type("collection cancelled")
    save_failure = save_failure_type("interrupted manifest publication failed")
    save_failure.add_note("manifest replacement diagnostic")
    attempted = replace_failures(monkeypatch, collection, {"interrupted": save_failure})
    completed_csv = collection.root / "completed.csv"
    seen, committed = [], []

    def execute(config, task):
        seen.append(task)
        if len(seen) == 1:
            completed_csv.write_bytes(b"latency_app_s\n1.25\n")
            return RunOutcome("ok", str(completed_csv), 1)
        committed.append(collection.path.read_bytes())
        raise cancellation

    with pytest.raises(BaseException, match="collection cancelled") as caught:
        collection.run(execute)

    assert caught.value is cancellation
    assert "interrupted manifest publication failed" in "\n".join(cancellation.__notes__)
    assert "manifest replacement diagnostic" in "\n".join(cancellation.__notes__)
    assert attempted.count("interrupted") == 1
    assert collection.path.read_bytes() == committed[0]
    saved = json.loads(committed[0])
    assert saved["status"] == "running"
    assert [task["status"] for task in saved["tasks"]] == ["ok", "running"]
    assert saved["tasks"][0]["final_csv"] == str(completed_csv)
    assert completed_csv.read_bytes() == b"latency_app_s\n1.25\n"
    assert_released(collection)
    restored = task_runs.TaskCollection.restore(replace(collection.config, resume=True))
    assert restored.run_id == collection.run_id
    assert restored.manifest == saved
    assert restored.task.selected_tasks == collection.task.selected_tasks


@pytest.mark.parametrize("save_failure_type", [PermissionError, KeyboardInterrupt, SystemExit, asyncio.CancelledError])
def test_terminal_publication_failure_preserves_both_errors(collection, monkeypatch, save_failure_type):
    primary = OSError("terminal manifest publication failed")
    primary.add_note("terminal replacement diagnostic")
    secondary = save_failure_type("interruption publication failed")
    attempted = replace_failures(monkeypatch, collection, {"ok": primary, "interrupted": secondary})
    expected = primary if isinstance(secondary, Exception) else secondary

    with pytest.raises(BaseException, match="publication failed") as caught:
        collection.run(lambda *_args: RunOutcome("ok"))

    assert caught.value is expected
    details = "\n".join((str(expected), *expected.__notes__))
    assert "terminal manifest publication failed" in details
    assert "terminal replacement diagnostic" in details
    assert "interruption publication failed" in details
    if expected is secondary:
        assert secondary.__cause__ is primary
    assert attempted[-2:] == ["ok", "interrupted"]
    saved = json.loads(collection.path.read_text())
    assert saved["status"] == "running"
    assert [task["status"] for task in saved["tasks"]] == ["ok", "ok"]
    assert_released(collection)


@pytest.mark.parametrize("stage", ["initial_publication", "announcement", "terminal_publication"])
def test_collection_lifecycle_failures_save_an_interrupted_manifest(collection, monkeypatch, stage):
    primary = OSError(f"{stage} failed")
    executed = []
    if stage == "announcement":
        def fail_print(*_args, **_kwargs):
            raise primary
        monkeypatch.setattr(task_runs, "print", fail_print, raising=False)
    else:
        replace_failures(monkeypatch, collection, {
            "running" if stage == "initial_publication" else "ok": primary})

    with pytest.raises(OSError, match=f"{stage} failed") as caught:
        collection.run(lambda *_args: executed.append(True) or RunOutcome("ok"))

    assert caught.value is primary
    saved = json.loads(collection.path.read_text())
    assert saved["status"] == "interrupted"
    assert [task["status"] for task in saved["tasks"]] == (
        ["ok", "ok"] if stage == "terminal_publication" else ["pending", "pending"])
    assert len(executed) == (2 if stage == "terminal_publication" else 0)
    assert_released(collection)
    restored = task_runs.TaskCollection.restore(replace(collection.config, resume=True))
    assert restored.run_id == collection.run_id
    assert restored.task.selected_tasks == collection.task.selected_tasks
