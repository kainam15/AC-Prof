"""Interface checks own their terminal state, including when nested in image preparation."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_model_contract as contracts

from acprof.host.collection_workflow import PreparationWorkflow
from acprof.preparation_events import encode_event, parse_event
from acprof.tui.progress import RunProgressTracker


@pytest.fixture(params=["docker", "process"])
def interface_check(request, tmp_path, monkeypatch):
    task = contracts.TestModelContract().discover()
    outcome = SimpleNamespace(error=None, write_error=None, stages=["import", "signature"])
    monkeypatch.setenv("ACPROF_INTERACTIVE_PREPARATION", "1")
    monkeypatch.delenv("ACPROF_TUI", raising=False)

    def validate():
        if outcome.error is not None:
            raise outcome.error
        return {"status": "ok", "stages": [
            {"stage": stage, "status": "verified"} for stage in outcome.stages]}

    if request.param == "docker":
        from acprof.host import interface_probe as module

        def run(command, **kwargs):
            if command[:2] == ["docker", "run"]:
                Path(command[command.index("--cidfile") + 1]).write_text("c" * 64)
                return SimpleNamespace(returncode=0, stderr="", stdout=(
                    "ACPROF_INTERFACE_VALIDATION=" + json.dumps(validate())))
            assert command == ["docker", "rm", "-f", "c" * 64]
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(module, "prepare_environment_image", lambda *args: SimpleNamespace(
            image_id="sha256:" + "b" * 64))
        monkeypatch.setattr(module, "recover_abandoned_containers", lambda *args: None)
        monkeypatch.setattr(module, "run_command", run)

        def check():
            return module.probe_interface(task, tmp_path)
    else:
        from acprof.host.runtimes import validation as module

        monkeypatch.setattr(module, "run_validation", lambda *args, **kwargs: validate())

        def check():
            return module.probe_interface(task, tmp_path, {"environment_id": "fixture"}, tmp_path)

    original_write = module.atomic_write_json

    def write(path, data):
        if outcome.write_error is not None:
            raise outcome.write_error
        return original_write(path, data)

    monkeypatch.setattr(module, "atomic_write_json", write)
    return SimpleNamespace(run=check, outcome=outcome, output=tmp_path)


def replay(output):
    tracker = RunProgressTracker()
    for line in output.splitlines():
        tracker.feed(line)
    return tracker


def interface_states(output):
    return [event["status"] for line in output.splitlines()
            if (event := parse_event(line)) is not None and event["stage"] == "interface"]


@pytest.mark.parametrize("enabled_by", ["ACPROF_INTERACTIVE_PREPARATION", "ACPROF_TUI"])
def test_completed_interface_remains_passed_when_measurement_starts(interface_check, capsys, monkeypatch, enabled_by):
    monkeypatch.delenv("ACPROF_INTERACTIVE_PREPARATION")
    monkeypatch.setenv(enabled_by, "1")
    workflow = PreparationWorkflow(interactive=True)
    workflow.emit("resolution", "passed")
    workflow.run("image", interface_check.run)
    workflow.emit("runtime", "passed")
    output = capsys.readouterr().out
    states = interface_states(output)
    assert states
    assert states[0] == "running"
    assert states[-1] == "passed"
    tracker = replay(output)
    tracker.feed('ACPROF_EVENT {"version":1,"event":"case_started","case_id":"a"}')
    snapshot = tracker.feed('ACPROF_EVENT {"version":1,"event":"measurement_started","case_id":"a"}')
    assert (snapshot.interface_status, snapshot.runtime_status, snapshot.measurement_status) == (
        "passed", "passed", "running")
    assert snapshot.measurement_active
    assert json.loads((interface_check.output / "interface_validation.json").read_text())["status"] == "ok"


@pytest.mark.parametrize("interrupted", [False, True])
def test_interface_failure_or_cancel_closes_progress_and_preserves_exception(interface_check, capsys, interrupted):
    error = KeyboardInterrupt("cancelled") if interrupted else RuntimeError("interface check failed")
    interface_check.outcome.error = error
    with pytest.raises(type(error)) as caught:
        interface_check.run()
    assert caught.value is error
    output = capsys.readouterr().out
    states = interface_states(output)
    expected = "cancelled" if interrupted else "failed"
    assert states
    assert states[0] == "running"
    assert states[-1] == expected
    assert "passed" not in states
    snapshot = replay(output).snapshot
    assert snapshot.interface_status == expected
    assert snapshot.measurement_status == "not_started"
    assert not snapshot.measurement_active
    assert snapshot.errors == (0 if interrupted else 1)


def test_interface_cannot_report_passed_if_evidence_cannot_be_saved(interface_check, capsys):
    error = OSError("evidence disk unavailable")
    interface_check.outcome.write_error = error
    with pytest.raises(OSError, match="evidence disk unavailable") as caught:
        interface_check.run()
    assert caught.value is error
    states = interface_states(capsys.readouterr().out)
    assert states
    assert states[-1] == "failed"
    assert "passed" not in states


def test_incomplete_interface_report_emits_failure(interface_check, capsys):
    interface_check.outcome.stages = ["import"]
    with pytest.raises(RuntimeError):
        interface_check.run()
    assert json.loads((interface_check.output / "interface_validation.json").read_text())["status"] == "error"
    states = interface_states(capsys.readouterr().out)
    assert states
    assert states[-1] == "failed"
    assert "passed" not in states


def test_interface_retry_replaces_failure_with_real_success(interface_check, capsys):
    interface_check.outcome.error = RuntimeError("first attempt failed")
    with pytest.raises(RuntimeError, match="first attempt"):
        interface_check.run()
    tracker = replay(capsys.readouterr().out)
    assert tracker.snapshot.interface_status == "failed"
    interface_check.outcome.error = None
    interface_check.run()
    for line in capsys.readouterr().out.splitlines():
        tracker.feed(line)
    assert tracker.snapshot.interface_status == "passed"
    assert not tracker.snapshot.measurement_active


def test_noninteractive_interface_check_does_not_emit_control_events(interface_check, monkeypatch, capsys):
    monkeypatch.delenv("ACPROF_INTERACTIVE_PREPARATION")
    interface_check.run()
    assert not [line for line in capsys.readouterr().out.splitlines() if parse_event(line) is not None]


def test_terminal_interface_events_cannot_arrive_inside_measurement():
    tracker = RunProgressTracker()
    tracker.feed('ACPROF_EVENT {"version":1,"event":"case_started","case_id":"a"}')
    tracker.feed('ACPROF_EVENT {"version":1,"event":"measurement_started","case_id":"a"}')
    with pytest.raises(ValueError, match="inside measurement window"):
        tracker.feed(encode_event("interface", "passed"))


@pytest.mark.parametrize(("size", "language"), [((120, 30), "zh"), ((80, 24), "en")])
async def test_monitor_renders_real_interface_completion(interface_check, capsys, tmp_path, size, language):
    from dataclasses import replace

    from rich.console import Console
    from textual.widgets import Static
    from tui_fixtures import AcprofTui

    interface_check.run()
    tracker = replay(capsys.readouterr().out)
    snapshot = tracker.feed(encode_event("runtime", "passed"))
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=size) as pilot:
        app.activate_tab("monitor-tab")
        app.consume_process_line("", snapshot, True)
        await pilot.pause()
        widget = app.query_one("#status-preparation", Static)
        text = "".join(segment.text for segment in Console(width=widget.content_size.width).render(widget.render()))
        assert ("接口解析" if language == "zh" else "Interface") in text
        assert text.count("通过" if language == "zh" else "Passed") == 2
        assert ("进行中" if language == "zh" else "Running") not in text

        tracker.feed('ACPROF_EVENT {"version":1,"event":"case_started","case_id":"a"}')
        snapshot = tracker.feed('ACPROF_EVENT {"version":1,"event":"measurement_started","case_id":"a"}')
        app.consume_process_line("", snapshot, True)
        await pilot.pause()
        text = "".join(segment.text for segment in Console(width=widget.content_size.width).render(widget.render()))
        assert ("接口解析" if language == "zh" else "Interface") not in text
        assert text.count("进行中" if language == "zh" else "Running") == 1
        assert app.latest_snapshot.measurement_active
        assert app.latest_snapshot.interface_status == "passed"
