"""Profiler cancellation must settle container ownership before releasing model files."""
import asyncio
import json
import subprocess
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from acprof.host.container_lifecycle import ContainerCleanupError
from acprof.host.detect import TaskInfo
from acprof.host.model_store import store
from acprof.host.profiling import compute, execution
from acprof.host.profiling.commands import run_profiler_command
from acprof.host.profiling.tools import advisor, execution_environment, massif, ncu, nsys, torch
from acprof.system import command


@pytest.fixture
def profiler_engine(monkeypatch):
    state = SimpleNamespace(
        active=set(), launched=[], removed=[], cidfiles=[], commands=[],
        failure=None, cleanup_failure=False, returncode=0, publish_id=True,
        launch_error=None, auto_remove=False, raw_stream=None,
        mount=SimpleNamespace(args=[], close=Mock()), retained=Mock(),
    )

    def run(argv, **kwargs):
        state.commands.append((argv, kwargs))
        if argv[:2] == ["docker", "run"]:
            if state.launch_error:
                raise state.launch_error
            identifier = f"{len(state.launched) + 1:064x}"
            state.launched.append(identifier)
            state.active.add(identifier)
            if state.raw_stream is not None:
                state.raw_stream.write_bytes(b"active profiler output")
            if "--cidfile" in argv:
                cidfile = Path(argv[argv.index("--cidfile") + 1])
                state.cidfiles.append(cidfile)
                if state.publish_id:
                    cidfile.write_text(identifier)
            if state.failure is not None:
                raise state.failure
            if state.auto_remove:
                state.active.remove(identifier)
            return subprocess.CompletedProcess(argv, state.returncode, "", "tool failed")
        if argv[:2] == ["docker", "rm"]:
            assert not state.mount.close.called
            assert kwargs["timeout"] > 0
            if isinstance(state.cleanup_failure, BaseException):
                raise state.cleanup_failure
            if state.cleanup_failure or argv[-1] not in state.active:
                return subprocess.CompletedProcess(argv, 1, "", "daemon unavailable")
            state.active.remove(argv[-1])
            state.removed.append(argv[-1])
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[:2] == ["docker", "inspect"]:
            assert kwargs["timeout"] > 0
            if argv[-1] not in state.active:
                return subprocess.CompletedProcess(argv, 1, "", f"No such object: {argv[-1]}")
            return subprocess.CompletedProcess(
                argv, 0, json.dumps([{"Id": argv[-1], "State": {"Running": True}}]), "",
            )
        if "--query-metrics" in argv:
            return subprocess.CompletedProcess(
                argv, 0, "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", "",
            )
        if "--help" in argv:
            return subprocess.CompletedProcess(argv, 0, "--nvtx-include", "")
        if "--version" in argv:
            return subprocess.CompletedProcess(argv, 0, "fixture-version", "")
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr(command.subprocess, "run", run)
    monkeypatch.setattr(store, "acquire_mount", lambda *_: state.mount)
    monkeypatch.setattr(store, "retain_mount_for_cleanup_debt", state.retained)
    with patch("acprof.host.profiling.support.gpu_docker_args", return_value=["--gpus", "device=GPU-fixture"]):
        yield state


def collect_massif(tmp_path, *, entries=1):
    task = TaskInfo(
        model_id="fixture", model_revision="fixed", task_family="nlp",
        pipeline_tag="fill-mask", runtime_backend="transformers_pipeline",
        runtime_profile_id="fixture", library_name="transformers", detection_method="test",
    )
    return massif.profile(
        entries=[{"input_scale": float(i + 1), "payload": {}} for i in range(entries)],
        global_error="", task_info=task, derived_image="fixture", cpu=1, mem=2,
        payload_file=str(tmp_path / "payload.json"), profile_root=str(tmp_path),
        output_dir=str(tmp_path), repeat=1,
    )


@pytest.mark.parametrize("returncode", [0, 125, 137])
def test_profiler_exit_removes_its_container_before_releasing_mount(tmp_path, profiler_engine, returncode):
    profiler_engine.returncode = returncode
    collect_massif(tmp_path)
    assert not profiler_engine.active
    assert profiler_engine.removed == profiler_engine.launched
    profiler_engine.mount.close.assert_called_once_with()
    profiler_engine.retained.assert_not_called()
    assert all(not path.exists() for path in profiler_engine.cidfiles)


@pytest.mark.parametrize("failure", [
    subprocess.TimeoutExpired(["docker", "run"], 1),
    KeyboardInterrupt("cancel profiling"), SystemExit("exit profiling"),
    asyncio.CancelledError("cancel profiling"),
])
def test_profiler_interruption_reclaims_container_and_preserves_error(tmp_path, profiler_engine, failure):
    profiler_engine.failure = failure
    if isinstance(failure, Exception):
        report = collect_massif(tmp_path)
        assert "TimeoutExpired" in report["entries"][0]["error"]
    else:
        with pytest.raises(type(failure)) as caught:
            collect_massif(tmp_path)
        assert caught.value is failure
    assert not profiler_engine.active
    assert profiler_engine.removed == profiler_engine.launched
    profiler_engine.mount.close.assert_called_once_with()
    profiler_engine.retained.assert_not_called()


@pytest.mark.parametrize("failure", [None, subprocess.TimeoutExpired(["docker", "run"], 1)])
def test_cleanup_debt_aborts_remaining_scales_and_retains_lease(tmp_path, profiler_engine, failure):
    profiler_engine.failure = failure
    profiler_engine.cleanup_failure = True
    with pytest.raises(ContainerCleanupError, match="cleanup present") as caught:
        collect_massif(tmp_path, entries=2)
    assert caught.value.container_id == profiler_engine.launched[0]
    assert caught.value.run_error is failure
    assert len(profiler_engine.launched) == 1
    assert profiler_engine.active == set(profiler_engine.launched)
    profiler_engine.mount.close.assert_not_called()
    profiler_engine.retained.assert_called_once_with(profiler_engine.mount)


def test_missing_container_identity_never_releases_the_model_lease(tmp_path, profiler_engine):
    profiler_engine.publish_id = False
    with pytest.raises(ContainerCleanupError, match="cleanup unknown"):
        collect_massif(tmp_path, entries=2)
    assert len(profiler_engine.launched) == 1
    assert not profiler_engine.removed
    profiler_engine.mount.close.assert_not_called()
    profiler_engine.retained.assert_called_once_with(profiler_engine.mount)


def test_failed_command_detail_survives_missing_container_identity(tmp_path, profiler_engine):
    profiler_engine.publish_id = False
    profiler_engine.returncode = 125
    with pytest.raises(ContainerCleanupError, match="cleanup unknown") as caught:
        collect_massif(tmp_path)
    failure = caught.value.run_error
    assert isinstance(failure, subprocess.CalledProcessError)
    assert failure.returncode == 125
    assert failure.stderr == "tool failed"
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize("tool", ["torch", "ncu", "nsys", "advisor"])
def test_other_profilers_abort_on_cleanup_debt(tmp_path, profiler_engine, tool):
    task = TaskInfo(
        model_id="fixture", model_revision="fixed", task_family="nlp",
        pipeline_tag="fill-mask", runtime_backend="transformers_pipeline",
        runtime_profile_id="fixture", library_name="transformers", detection_method="test",
    )
    common = dict(
        entries=[{"input_scale": 1.0, "payload": {}}, {"input_scale": 2.0, "payload": {}}],
        task_info=task, image_tag="fixture", cpu=1, mem=2,
        payload_file=str(tmp_path / "payload.json"), profile_root=str(tmp_path), repeat=1,
    )
    profiler_engine.cleanup_failure = True
    if tool == "nsys":
        profiler_engine.raw_stream = tmp_path / "nsys_cpu_1_mem_2_scale_1.qdstrm"
    collect = {
        "torch": partial(torch.profile_torch_entries, use_gpu=False, profile_key="cpu"),
        "ncu": partial(ncu.profile_gpu_entries, ncu_bin="ncu", ncu_root=str(tmp_path)),
        "nsys": partial(nsys.profile, global_error="", nsys_bin="nsys",
                        nsys_mount_root=str(tmp_path), output_dir=str(tmp_path)),
        "advisor": partial(advisor.profile_cpu_entries, advisor_bin="advisor", advisor_root=str(tmp_path)),
    }[tool]
    with pytest.raises(ContainerCleanupError, match="cleanup present"):
        collect(**common)
    assert len(profiler_engine.launched) == 1
    profiler_engine.mount.close.assert_not_called()
    profiler_engine.retained.assert_called_once_with(profiler_engine.mount)
    if tool == "nsys":
        assert profiler_engine.raw_stream.read_bytes() == b"active profiler output"


def test_compute_plan_does_not_publish_or_start_next_tool_with_cleanup_debt(tmp_path, profiler_engine):
    from tests.execution.planning.test_execution_profile import _task_info
    from tests.measurement.profiling.test_compute_profile import _write_input_scale_plan

    plan = _write_input_scale_plan(str(tmp_path))
    profiler_engine.cleanup_failure = True
    with patch("acprof.host.profiling.compute.find_executable", return_value=None), pytest.raises(
        ContainerCleanupError, match="cleanup present",
    ):
        compute.collect_compute_profile_plan(
            task_info=_task_info(), image_tag="fixture", cpu_list=[1], mem_list=[2],
            gpu_list=["off", "on"], output_dir=str(tmp_path), input_scale_plan_file=plan,
            advisor_root=None, ncu_root=None, advisor_repeat=1, ncu_repeat=1,
            keep_profiles=False, compute_profile_tool="torch",
        )
    assert len(profiler_engine.launched) == 1
    assert not list(tmp_path.rglob("compute_profile_plan.json"))
    profiler_engine.mount.close.assert_not_called()


def test_failed_lease_close_keeps_cancellation_and_retry_handle(tmp_path, profiler_engine):
    failure = KeyboardInterrupt("cancel profiling")
    profiler_engine.failure = failure
    profiler_engine.mount.close.side_effect = OSError("lease close failed")
    with pytest.raises(KeyboardInterrupt) as caught:
        collect_massif(tmp_path)
    assert caught.value is failure
    assert "lease close failed" in "\n".join(failure.__notes__)
    assert not profiler_engine.active
    profiler_engine.retained.assert_called_once_with(profiler_engine.mount)


@pytest.mark.parametrize("error_type", [FileNotFoundError, PermissionError])
def test_failed_client_spawn_releases_mount_without_container_cleanup(tmp_path, profiler_engine, error_type):
    profiler_engine.launch_error = error_type("cannot execute Docker CLI")
    report = collect_massif(tmp_path)
    assert error_type.__name__ in report["entries"][0]["error"]
    assert not profiler_engine.launched
    assert not profiler_engine.removed
    profiler_engine.retained.assert_not_called()
    profiler_engine.mount.close.assert_called_once_with()


def test_auto_removed_container_is_confirmed_absent_before_releasing_mount(tmp_path, profiler_engine):
    profiler_engine.auto_remove = True
    collect_massif(tmp_path)
    assert not profiler_engine.active
    profiler_engine.retained.assert_not_called()
    profiler_engine.mount.close.assert_called_once_with()
    assert any(argv[:2] == ["docker", "inspect"] for argv, _kwargs in profiler_engine.commands)


def test_cleanup_interruption_keeps_original_timeout_and_stops_next_scale(tmp_path, profiler_engine):
    failure = subprocess.TimeoutExpired(["docker", "run"], 1)
    cleanup_failure = KeyboardInterrupt("interrupted cleanup")
    profiler_engine.failure = failure
    profiler_engine.cleanup_failure = cleanup_failure
    with pytest.raises(ContainerCleanupError, match="cleanup unknown") as caught:
        collect_massif(tmp_path, entries=2)
    assert caught.value.run_error is failure
    assert caught.value.__cause__ is cleanup_failure
    assert len(profiler_engine.launched) == 1
    profiler_engine.mount.close.assert_not_called()
    profiler_engine.retained.assert_called_once_with(profiler_engine.mount)


def test_repeated_tool_commands_have_separate_ids_and_leave_input_argv_unchanged(profiler_engine):
    argv = ["docker", "run", "--rm", "fixture", "tool", "--version"]
    before = list(argv)
    for _ in range(2):
        run_profiler_command(argv, check=False)
    assert argv == before
    assert len(set(profiler_engine.cidfiles)) == 2
    assert profiler_engine.removed == profiler_engine.launched
    assert not profiler_engine.active
    for command_argv, _kwargs in profiler_engine.commands:
        if command_argv[:2] == ["docker", "run"]:
            labels = dict(command_argv[i + 1].split("=", 1)
                          for i, value in enumerate(command_argv) if value == "--label")
            assert labels["org.acprof.container.lifecycle"] == "1"
            assert all(labels[f"org.acprof.owner.{key}"] for key in ("host", "uid", "boot", "pid", "start"))


def test_massif_version_probe_cannot_downgrade_cleanup_debt_to_unknown(profiler_engine):
    profiler_engine.failure = subprocess.TimeoutExpired(["docker", "run"], 1)
    profiler_engine.cleanup_failure = True
    with pytest.raises(ContainerCleanupError, match="cleanup present"):
        execution_environment.get_massif_version("fixture")
    assert len(profiler_engine.launched) == 1


def test_execution_plan_aborts_when_nsys_preflight_leaves_a_container(tmp_path, profiler_engine):
    from tests.execution.planning.test_execution_profile import _task_info, _write_input_scale_plan

    plan = _write_input_scale_plan(str(tmp_path))
    profiler_engine.cleanup_failure = True
    request = execution.ExecutionProfileRequest(
        _task_info(), "fixture", [1], [2], ["on"], str(tmp_path), plan,
    )
    options = execution.ExecutionProfileOptions(tool_mode="nsys", nsys_root=str(tmp_path))
    with (
        patch("acprof.host.profiling.execution.require_execution_image", return_value="fixture"),
        patch("acprof.host.profiling.execution.find_nsys_executable", return_value=str(tmp_path / "nsys")),
        patch("acprof.host.profiling.execution.find_nsys_mount_root", return_value=str(tmp_path)),
        patch("acprof.host.profiling.tools.execution_environment.find_nsys_importer",
              return_value=str(tmp_path / "importer")),
        pytest.raises(ContainerCleanupError, match="cleanup present"),
    ):
        execution.collect_execution_profile_plan(request, options)
    assert len(profiler_engine.launched) == 1
    assert not list(tmp_path.rglob("execution_profile_plan.json"))
