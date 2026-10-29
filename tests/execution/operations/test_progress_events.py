import json

import pytest

from acprof.tui.progress import RunProgressTracker
from acprof.tui.stages import Stage


def event(name, case="case-a", **fields):
    return "ACPROF_EVENT " + json.dumps({"version": 1, "event": name, "case_id": case, **fields})


def test_events_control_measurement_even_when_log_wording_changes():
    tracker = RunProgressTracker()
    tracker.feed(event("case_started"))
    assert (tracker.feed(event("measurement_started")).measurement_active)
    assert (tracker.feed("[case] Done. (third party output)").measurement_active)
    assert (tracker.feed(event("measurement_stopped", case="stale-case")).measurement_active)
    assert not (tracker.feed(event("measurement_stopped")).measurement_active)
    assert (tracker.feed(event("case_finished", status="error")).stage) is Stage.FAILED

def test_unknown_event_version_is_rejected():
    tracker = RunProgressTracker()
    with pytest.raises(ValueError, match="version"):
        tracker.feed('ACPROF_EVENT {"version":2,"event":"measurement_started","case_id":"a"}')


def test_checkpoint_resume_shows_postprocessing_without_entering_measurement():
    tracker = RunProgressTracker()
    tracker.feed(event("case_selected", current=2, total=3, cpu=2, mem=4, gpu="off"))
    state = tracker.feed(event("case_started", phase="postprocessing"))
    assert state.stage is Stage.MERGING_RESULTS
    assert (state.current_case, state.total_cases, state.cpu, state.mem) == (2, 3, "2", "4")
    assert not state.measurement_active
    assert state.measurement_status == "not_started"
    assert tracker.feed(event("case_finished", status="ok")).stage is Stage.CASE_COMPLETED
    assert tracker.feed(event("case_started", case="case-next")).stage is Stage.STARTING_CONTAINER


@pytest.mark.parametrize("phase", [None, "", "measuring", {}])
def test_invalid_resume_phase_is_rejected_without_changing_progress(phase):
    tracker = RunProgressTracker()
    before = tracker.snapshot
    with pytest.raises(ValueError, match="case phase"):
        tracker.feed(event("case_started", phase=phase))
    assert tracker.snapshot == before


def test_case_cleanup_failure_cannot_emit_success():
    import tempfile
    from contextlib import ExitStack
    from types import SimpleNamespace
    from unittest.mock import patch

    from acprof.host import cases, orchestrator
    from acprof.host.detect import TaskInfo
    from acprof.host.docker_runtime import RunningContainer
    from acprof.host.images.runtime import ImageInfo
    task = TaskInfo("fixture/model", "tabular-classification", "structured", "onnxruntime", "onnx", "abc", "manual")
    session = RunningContainer("fixture", "http://127.0.0.1:8002", 8002, 1, container_id="a" * 64)
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        for name, result in (("start_service_session", session), ("record_case_conditions", None),
                             ("_finalize_case", None)):
            stack.enter_context(patch.object(orchestrator, name, return_value=result))
        stack.enter_context(patch("acprof.host.command.run_command", return_value=SimpleNamespace(returncode=0)))
        stack.enter_context(patch.object(orchestrator, "stop_service_session", side_effect=RuntimeError("cleanup failed")))
        emitted = stack.enter_context(patch.object(orchestrator, "emit_event"))
        with pytest.raises(RuntimeError, match="cleanup failed"):
            orchestrator.run_single_case(
                cases.CaseRequest(
                    task_info=task,
                    cpu=1,
                    mem=4,
                    gpu='off',
                    image_info=ImageInfo('fixture'),
                    output_dir=directory,
                    project_dir=directory,
                ),
                measurement=cases.CaseMeasurement(
                    profiling_mode='basic',
                ),
                plans=cases.CasePlans(
                    input_scales='1',
                ),
            )
        assert (emitted.call_args.kwargs["status"]) == ("error")

def test_client_error_row_cannot_emit_success_even_when_client_exits_zero():
    import tempfile
    from contextlib import ExitStack
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import patch

    from acprof.host import cases, orchestrator
    from acprof.host.detect import TaskInfo
    from acprof.host.docker_runtime import RunningContainer
    from acprof.host.images.runtime import ImageInfo
    task = TaskInfo("fixture/model", "tabular-classification", "structured", "onnxruntime", "onnx", "abc", "manual")
    session = RunningContainer("fixture", "http://127.0.0.1:8002", 8002, 1, container_id="a" * 64)
    def failed_client(*args, **kwargs):
        Path(kwargs["env"]["OUT_CSV"]).write_text("status,error\nerror,predict failed\n")
        return SimpleNamespace(returncode=0)
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        for name, result in (("start_service_session", session), ("record_case_conditions", None),
                             ("stop_service_session", None), ("_finalize_case", None)):
            stack.enter_context(patch.object(orchestrator, name, return_value=result))
        stack.enter_context(patch("acprof.host.command.run_command", side_effect=failed_client))
        emitted = stack.enter_context(patch.object(orchestrator, "emit_event"))
        orchestrator.run_single_case(
            cases.CaseRequest(
                task_info=task,
                cpu=1,
                mem=4,
                gpu='off',
                image_info=ImageInfo('fixture'),
                output_dir=directory,
                project_dir=directory,
            ),
            measurement=cases.CaseMeasurement(
                profiling_mode='basic',
            ),
            plans=cases.CasePlans(
                input_scales='1',
            ),
        )
        assert (emitted.call_args.kwargs["status"]) == ("error")


def test_human_logs_cannot_change_case_or_output_identity():
    from acprof.progress_events import encode_event
    tracker = RunProgressTracker()
    tracker.feed(encode_event("matrix_planned", total_cases=2))
    selected = tracker.feed(encode_event("case_selected", "case-a", current=1, total=2,
                                        cpu=2, mem=4, gpu="off"))
    for line in ("# Case 99/99: CPU=8, MEM=16GB, GPU=on", "Profiling complete!",
                 "Merged results: /unrelated/result.csv", "[largest-probe] RESULT status=ok"):
        assert tracker.feed(line) == selected


def test_run_success_event_requires_persisted_result_verification():
    from acprof.progress_events import encode_event
    tracker = RunProgressTracker()
    tracker.feed(encode_event("result_published", final_csv="/result.csv"))
    state = tracker.feed(encode_event("run_finished", status="ok"))
    assert state.stage is Stage.VERIFYING_ARTIFACT
    assert state.measurement_status != "passed"


def test_process_progress_retains_inherited_resource_axes():
    from acprof.progress_events import encode_event
    tracker = RunProgressTracker()
    line = encode_event('case_selected', 'process-case', current=1, total=1,
                        cpu=None, mem=None, gpu='off', runtime_kind='process')
    state = tracker.feed(line)
    assert state.cpu == state.mem == 'inherited'
    assert state.current_case == 1
    for kind in ('docker', 'unknown'):
        with pytest.raises(ValueError, match='resource progress'):
            encode_event('case_selected', 'process-case', current=1, total=1,
                         cpu=None, mem=None, gpu='off', runtime_kind=kind)
    with pytest.raises(ValueError, match='resource progress'):
        encode_event('case_selected', 'process-case', current=1, total=1,
                     cpu=0, mem=None, gpu='off', runtime_kind='process')


@pytest.mark.parametrize("fields", [
    {"current": 0, "total": 2, "cpu": 1, "mem": 4, "gpu": "off"},
    {"current": 3, "total": 2, "cpu": 1, "mem": 4, "gpu": "off"},
    {"current": 1, "total": 2, "cpu": True, "mem": 4, "gpu": "off"},
])
def test_invalid_case_selection_is_rejected_before_mutating_progress(fields):
    tracker = RunProgressTracker()
    before = tracker.snapshot
    with pytest.raises(ValueError, match="resource progress"):
        tracker.feed(event("case_selected", **fields))
    assert tracker.snapshot == before


def test_control_event_cannot_end_an_active_measurement_by_claiming_a_result():
    from acprof.progress_events import encode_event
    tracker = RunProgressTracker()
    tracker.feed(event("case_started"))
    tracker.feed(event("measurement_started"))
    with pytest.raises(ValueError, match="inside measurement"):
        tracker.feed(encode_event("result_published", final_csv="/result.csv"))
    assert tracker.snapshot.measurement_active
