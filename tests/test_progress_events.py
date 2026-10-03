import json

import pytest

from acprof.tui.progress import RunProgressTracker


def event(name, case="case-a", **fields):
    return "ACPROF_EVENT " + json.dumps({"version": 1, "event": name, "case_id": case, **fields})


def test_events_control_measurement_even_when_log_wording_changes():
    tracker = RunProgressTracker()
    tracker.feed(event("case_started"))
    assert (tracker.feed(event("measurement_started")).measurement_active)
    assert (tracker.feed("[case] Done. (third party output)").measurement_active)
    assert (tracker.feed(event("measurement_stopped", case="stale-case")).measurement_active)
    assert not (tracker.feed(event("measurement_stopped")).measurement_active)
    assert (tracker.feed(event("case_finished", status="error")).stage) == ("失败")

def test_unknown_event_version_is_rejected():
    tracker = RunProgressTracker()
    with pytest.raises(ValueError, match="version"):
        tracker.feed('ACPROF_EVENT {"version":2,"event":"measurement_started","case_id":"a"}')

def test_case_cleanup_failure_cannot_emit_success():
    import tempfile
    from contextlib import ExitStack
    from types import SimpleNamespace
    from unittest.mock import patch

    from acprof.host import orchestrator
    from acprof.host.detect import TaskInfo
    from acprof.host.docker_runtime import RunningContainer
    from acprof.host.runtime_images import ImageInfo
    task = TaskInfo("fixture/model", "tabular-classification", "structured", "onnxruntime", "onnx", "abc", "manual")
    session = RunningContainer("fixture", "http://127.0.0.1:8002", 8002, 1, container_id="a" * 64)
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        for name, result in (("start_container_session", session), ("record_case_conditions", None),
                             ("_finalize_case", None)):
            stack.enter_context(patch.object(orchestrator, name, return_value=result))
        stack.enter_context(patch("acprof.host.command.run_command", return_value=SimpleNamespace(returncode=0)))
        stack.enter_context(patch.object(orchestrator, "stop_container_session", side_effect=RuntimeError("cleanup failed")))
        emitted = stack.enter_context(patch.object(orchestrator, "emit_event"))
        with pytest.raises(RuntimeError, match="cleanup failed"):
            orchestrator.run_single_case(task, 1, 4, "off", ImageInfo("fixture"), directory,
                                         directory, profiling_mode="basic", input_scales="1")
        assert (emitted.call_args.kwargs["status"]) == ("error")

def test_client_error_row_cannot_emit_success_even_when_client_exits_zero():
    import tempfile
    from contextlib import ExitStack
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import patch

    from acprof.host import orchestrator
    from acprof.host.detect import TaskInfo
    from acprof.host.docker_runtime import RunningContainer
    from acprof.host.runtime_images import ImageInfo
    task = TaskInfo("fixture/model", "tabular-classification", "structured", "onnxruntime", "onnx", "abc", "manual")
    session = RunningContainer("fixture", "http://127.0.0.1:8002", 8002, 1, container_id="a" * 64)
    def failed_client(*args, **kwargs):
        Path(kwargs["env"]["OUT_CSV"]).write_text("status,error\nerror,predict failed\n")
        return SimpleNamespace(returncode=0)
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        for name, result in (("start_container_session", session), ("record_case_conditions", None),
                             ("stop_container_session", None), ("_finalize_case", None)):
            stack.enter_context(patch.object(orchestrator, name, return_value=result))
        stack.enter_context(patch("acprof.host.command.run_command", side_effect=failed_client))
        emitted = stack.enter_context(patch.object(orchestrator, "emit_event"))
        orchestrator.run_single_case(task, 1, 4, "off", ImageInfo("fixture"), directory,
                                     directory, profiling_mode="basic", input_scales="1")
        assert (emitted.call_args.kwargs["status"]) == ("error")
