"""Automatic report failures cannot mask collection errors or cancellation."""
import asyncio
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof import artifacts
from acprof.application.errors import CollectionFailed
from acprof.application.run import RunOutcome
from acprof.cli import auto, run
from acprof.host.automation import AutomaticRun


@pytest.fixture
def automatic(tmp_path, monkeypatch):
    task = SimpleNamespace(model_id="fixture/model", model_revision="a" * 40)
    instance = Mock(root=tmp_path, preparation_artifacts={}, collection_path=None)
    instance.prepare.return_value = task
    instance.finish.return_value = 2
    collect = Mock(return_value=RunOutcome("ok"))
    monkeypatch.setattr("acprof.host.automation.AutomaticRun", Mock(return_value=instance))
    monkeypatch.setattr("acprof.host.env_utils.bootstrap_project_env", Mock())
    monkeypatch.setattr("acprof.host.run_state.MeasurementLock", lambda: nullcontext())
    monkeypatch.setattr("acprof.host.run_state.ResultDirectoryLock", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(run, "main", collect)
    args = ["fixture/model", "--profiling-mode", "basic", "--output-dir", str(tmp_path), "--notify", "none"]
    return instance, collect, args, task


@pytest.mark.parametrize("cancellation_type", [KeyboardInterrupt, asyncio.CancelledError])
def test_failed_report_keeps_collection_cancellation(automatic, cancellation_type):
    instance, collect, args, _task = automatic
    cancellation = cancellation_type("collection cancelled")
    report_error = OSError("automatic report save failed")
    report_error.add_note("report directory sync failed")
    collect.side_effect = cancellation
    instance.finish.side_effect = report_error

    with pytest.raises(cancellation_type) as caught:
        auto.main(args)

    assert caught.value is cancellation
    assert "automatic report save failed" in "\n".join(cancellation.__notes__)
    assert "report directory sync failed" in "\n".join(cancellation.__notes__)
    instance.finish.assert_called_once_with("interrupted")


@pytest.mark.parametrize("failure_type", [RuntimeError, SystemExit])
@pytest.mark.parametrize("report_fails", [False, True])
def test_ordinary_failures_keep_notes_in_report_and_cli(automatic, capsys, failure_type, report_fails):
    instance, collect, args, _task = automatic
    failure = failure_type("collection failed")
    failure.add_note("worker cleanup diagnostic")
    collect.side_effect = failure
    if report_fails:
        instance.finish.side_effect = OSError("automatic report save failed")

    assert auto.main(args) == 2

    detail = instance.finish.call_args.args[0]
    assert str(failure) in detail
    assert "worker cleanup diagnostic" in detail
    output = capsys.readouterr().err
    assert "worker cleanup diagnostic" in output
    if report_fails:
        assert "automatic report save failed" in output


@pytest.mark.parametrize("body_fails", [False, True])
@pytest.mark.parametrize("cancellation_type", [KeyboardInterrupt, SystemExit, asyncio.CancelledError])
def test_report_cancellation_propagates_and_keeps_preceding_failure(automatic, body_fails, cancellation_type):
    instance, collect, args, _task = automatic
    failure = RuntimeError("collection failed")
    failure.add_note("worker cleanup diagnostic")
    cancellation = cancellation_type("report cancelled")
    if body_fails:
        collect.side_effect = failure
    instance.finish.side_effect = cancellation

    with pytest.raises(cancellation_type) as caught:
        auto.main(args)

    assert caught.value is cancellation
    if body_fails:
        assert cancellation.__cause__ is failure
        assert "collection failed" in "\n".join(cancellation.__notes__)
        assert "worker cleanup diagnostic" in "\n".join(cancellation.__notes__)
    instance.finish.assert_called_once()


@pytest.mark.parametrize("exit_code", [None, 0])
@pytest.mark.parametrize("report_fails", [False, True])
def test_successful_exit_depends_on_report_finalization(automatic, capsys, exit_code, report_fails):
    instance, collect, args, _task = automatic
    collect.side_effect = SystemExit(exit_code)
    instance.finish.return_value = 0
    if report_fails:
        instance.finish.side_effect = OSError("automatic report save failed")

    assert auto.main(args) == (2 if report_fails else 0)
    instance.finish.assert_called_once_with(None)
    if report_fails:
        assert "automatic report save failed" in capsys.readouterr().err


def test_successful_collection_cannot_hide_failed_report(automatic, capsys):
    instance, _collect, args, _task = automatic
    instance.finish.side_effect = OSError("automatic report save failed")
    assert auto.main(args) == 2
    assert "automatic report save failed" in capsys.readouterr().err
    instance.finish.assert_called_once()


def test_failed_collection_path_survives_cli_exit_translation(automatic):
    instance, collect, args, _task = automatic
    failure = CollectionFailed("one task failed", task_collection_path="fixture-collection.json")
    exit_error = SystemExit(1)
    exit_error.__context__ = failure
    collect.side_effect = exit_error
    assert auto.main(args) == 2
    assert instance.collection_path == "fixture-collection.json"


def test_cancelled_run_keeps_existing_report_when_atomic_replacement_fails(automatic, monkeypatch):
    _instance, collect, args, task = automatic
    created, previous = [], []
    cancellation = KeyboardInterrupt("collection cancelled")

    def create(options):
        instance = AutomaticRun(options)
        instance.layout.initialize()
        instance.started = True
        instance.save()
        instance.prepare = Mock(return_value=task)
        created.append(instance)
        previous.append(instance.path.read_bytes())
        return instance

    replace_file = artifacts.os.replace

    def fail_report(source, destination):
        if created and Path(destination) == created[0].path:
            assert json.loads(Path(source).read_text())["status"] == "failed"
            raise OSError("automatic report replacement failed")
        return replace_file(source, destination)

    monkeypatch.setattr("acprof.host.automation.AutomaticRun", create)
    monkeypatch.setattr(artifacts.os, "replace", fail_report)
    collect.side_effect = cancellation
    with pytest.raises(KeyboardInterrupt) as caught:
        auto.main(args)
    assert caught.value is cancellation
    assert created[0].path.read_bytes() == previous[0]
    assert "automatic report replacement failed" in "\n".join(cancellation.__notes__)
