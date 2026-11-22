"""Process preflight and the CLI must resolve the same visible GPU without changing the TUI host."""
import json
import os
import subprocess
from dataclasses import replace
from unittest.mock import Mock

import pytest

from acprof.experiments.configuration import RunConfig
from acprof.host.runtimes.cgroups import ResourceDiagnostic
from acprof.tui.run.diagnostics import permission_signature, quick_preflight


@pytest.fixture
def process_project(tmp_path, monkeypatch):
    for key in ("CUDA_VISIBLE_DEVICES", "ACPROF_GPU_DEVICE", "DEVICE_INDEX", "NVIDIA_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    helper = Mock(side_effect=lambda cmd, *, env, **kw: subprocess.CompletedProcess(
        cmd, 0, json.dumps({"status": "available", "uuid": f"GPU-{env['CUDA_VISIBLE_DEVICES']}"}), ""))
    plan = Mock(return_value={"probes": {}})
    monkeypatch.setattr("acprof.host.runtimes.gpu.run_command", helper)
    monkeypatch.setattr("acprof.host.runtimes.capabilities.probe_collection_plan", plan)
    monkeypatch.setattr("acprof.host.runtimes.process_control.require_process_supervision", lambda: None)
    monkeypatch.setattr("acprof.host.runtimes.resources.preflight_resources", lambda *a: ResourceDiagnostic(code="inherited"))
    return project, helper, plan


@pytest.mark.parametrize(("scenario", "expected"), [
    ("default", "0"), ("project", "7"), ("local", "8"), ("shell", "3"), ("explicit", "4"),
    ("legacy-selector", "7"), ("out-of-range", None), ("hidden", None), ("cpu", None),
])
def test_process_quick_preflight_uses_effective_project_environment(process_project, monkeypatch, scenario, expected):
    project, helper, plan = process_project
    config = replace(RunConfig.smoke("fixture/model"), gpus="off" if scenario == "cpu" else "on", runtime="process")
    if scenario != "default":
        (project / ".env").write_text("CUDA_VISIBLE_DEVICES=4,7\nDEVICE_INDEX=0\nACPROF_GPU_DEVICE=1\n")
    if scenario == "local":
        (project / ".env.local").write_text("CUDA_VISIBLE_DEVICES=4,8\n")
    elif scenario == "shell":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    elif scenario == "explicit":
        config = replace(config, extra_options={**config.extra_options, "gpu_device": "0"})
    elif scenario == "legacy-selector":
        (project / ".env").write_text("CUDA_VISIBLE_DEVICES=4,7\nDEVICE_INDEX=1\n")
    elif scenario in {"out-of-range", "cpu"}:
        (project / ".env.local").write_text("ACPROF_GPU_DEVICE=99\n")
    elif scenario == "hidden":
        (project / ".env.local").write_text("CUDA_VISIBLE_DEVICES=\n")

    before = os.environ.copy()
    rows = [r for r in quick_preflight(config, project_dir=project) if r.check_id == "gpu_device"]
    assert os.environ.copy() == before
    if scenario == "cpu":
        assert not rows
    elif expected is None:
        row, = rows
        assert row.status == "fail"
        assert "CUDA_VISIBLE_DEVICES" in row.detail
    else:
        row, = rows
        assert row.status == "ok"
        assert row.detail == f"GPU-{expected}"
        assert helper.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == expected
    assert helper.call_count == int(expected is not None)
    gpu = plan.call_args.kwargs["gpu"]
    if expected is not None:
        assert gpu["uuid"] == f"GPU-{expected}"
        assert gpu["original_cuda_visible_devices"] == plan.call_args.kwargs["env"].get("CUDA_VISIBLE_DEVICES")
    if scenario == "cpu":
        assert gpu == {}
        assert plan.call_args.kwargs["env"]["ACPROF_GPU_DEVICE"] == "99"


@pytest.mark.parametrize("mode", ["basic", "full"])
def test_process_legacy_preflight_passes_selected_gpu_to_plan(process_project, mode):
    project, helper, plan = process_project
    (project / ".env.local").write_text("CUDA_VISIBLE_DEVICES=4,7\nACPROF_GPU_DEVICE=1\nNVIDIA_VISIBLE_DEVICES=all\n")
    config = replace(RunConfig.smoke("fixture/model"), runtime="process", profiling_mode=mode, gpus="on")
    quick_preflight(config, project_dir=project)
    gpu = plan.call_args.kwargs["gpu"]
    assert gpu["uuid"] == "GPU-7"
    assert gpu["nvidia_visible_devices"] == "all"
    assert helper.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "7"


def test_legacy_device_selector_change_invalidates_the_preflight_cache(process_project, monkeypatch):
    project, _, _ = process_project
    original = permission_signature(project)
    monkeypatch.setenv("DEVICE_INDEX", "1")
    assert permission_signature(project) != original


def test_auto_fallback_to_process_uses_the_selected_project(process_project, monkeypatch):
    project, _, _ = process_project
    (project / ".env.local").write_text("CUDA_VISIBLE_DEVICES=1,3\nDEVICE_INDEX=1\n")
    monkeypatch.setattr("acprof.tui.run.diagnostics.inspect_docker",
                        lambda *args: Mock(code="missing", summary="Docker unavailable"))
    config = replace(RunConfig.smoke("fixture/model"), runtime="auto", gpus="on")
    rows = quick_preflight(config, project_dir=project)
    assert any(row.check_id == "runtime_auto" for row in rows)
    check, = (row for row in rows if row.check_id == "gpu_device")
    assert check.status == "ok"
    assert check.detail == "GPU-3"


async def test_worker_admission_recovers_after_project_gpu_selection_is_corrected(process_project, monkeypatch):
    from textual.widgets import Button

    from acprof.tui.app import AcprofTui
    from tests.support.tui_waits import wait_for_condition

    project, _, _ = process_project
    monkeypatch.setattr("acprof.tui.app.PROJECT_DIR", project)
    env_file = project / ".env.local"
    env_file.write_text("CUDA_VISIBLE_DEVICES=4,7\nACPROF_GPU_DEVICE=99\n")
    config = replace(RunConfig.smoke("fixture/model"), runtime="process", gpus="on")
    app = AcprofTui(config, settings_path=project / "settings.json")

    def checked(status):
        return not app.preflight.check_running and any(
            row.check_id == "gpu_device" and row.status == status for row in app.preflight.checks)

    async with app.run_test(size=(120, 36)) as pilot:
        await wait_for_condition(pilot, lambda: checked("fail"), description="invalid project GPU blocks startup", timeout=6)
        assert app.query_one("#start-run", Button).disabled
        env_file.write_text("CUDA_VISIBLE_DEVICES=4,7\nACPROF_GPU_DEVICE=1\n")
        app.preflight.check_permission_changes()
        await wait_for_condition(pilot, lambda: checked("ok") and not app.query_one("#start-run", Button).disabled,
                                 description="corrected project GPU allows the configured run", timeout=6)
        row, = (row for row in app.preflight.checks if row.check_id == "gpu_device")
        assert row.detail == "GPU-7"


@pytest.mark.parametrize("mode", ["basic", "full"])
def test_legacy_gpu_selection_error_remains_a_device_row(process_project, mode):
    project, helper, plan = process_project
    (project / ".env.local").write_text("CUDA_VISIBLE_DEVICES=4,7\nACPROF_GPU_DEVICE=99\n")
    config = replace(RunConfig.smoke("fixture/model"), runtime="process", profiling_mode=mode, gpus="on")
    rows = {row.check_id: row for row in quick_preflight(config, project_dir=project)}
    assert rows["gpu_device"].status == "fail"
    assert "exceeds CUDA_VISIBLE_DEVICES" in rows["gpu_device"].detail
    assert rows["process_resources"].status == "ok"
    assert plan.call_args.kwargs["gpu"]["error"] == rows["gpu_device"].detail
    helper.assert_not_called()
