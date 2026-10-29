"""Failure labels require evidence within an individual result record."""

import pandas as pd
import pytest

from acprof.plotting.diagnostics import summarize_resource_feasibility


@pytest.mark.parametrize(("errors", "expected"), [
    (["BLOOM model architecture is not supported"], "error"),
    (["No room for an unsupported tokenizer"], "error"),
    (["TimeoutTokenizer is not supported"], "error"),
    (["OOMKilled=false"], "error"),
    (["docker_oom_killed=false; exit_code=137"], "error"),
    (['{"OOMKilled": false, "ExitCode": 137}'], "error"),
    (["container_oom_killed=false during startup"], "error"),
    (["container_start_failed: unsupported model; request_timeout_s=30"], "error"),
    (["container_start_failed: missing module", "runtime OOM"], "runtime_oom"),
    (["startup failed: missing dependency", "CUDA out of memory"], "runtime_oom"),
    (["out of", "memory"], "error"),
    (["out", "of", "memory"], "error"),
    (["not_measured_after_startup_oom_pruning", "tokenizer incompatible"], "error"),
    (["not_measured_after_startup_oom_pruning", "runtime OOM"], "runtime_oom"),
    (["not_measured_after_timeout", "tokenizer incompatible"], "error"),
    (["not_measured_after_startup_oom_pruning", "not_measured_after_timeout"], "error"),
    (["container_oom_killed during startup (exit_code=137)"], "startup_oom"),
    (["container_start_failed: CUDA out of memory"], "startup_oom"),
    (["Startup OOM"], "startup_oom"),
    (["container_runtime_oom: docker_oom_killed=true; container=startup-model"], "runtime_oom"),
    (["runtime_oom: owned cgroup memory.events oom_kill increased"], "runtime_oom"),
    (["docker_oom_killed=true"], "runtime_oom"),
    (['{"OOMKilled": true}'], "runtime_oom"),
    (["docker_oom_killed=false; CUDA out of memory."], "runtime_oom"),
    (["torch.OutOfMemoryError: allocation failed"], "runtime_oom"),
    (["MemoryError: allocation failed"], "runtime_oom"),
    (["MPS backend out of memory (MPS allocated: 8.00 GB)"], "runtime_oom"),
    (["CUDA out of\nmemory"], "runtime_oom"),
    (["out-of-memory allocation failure"], "runtime_oom"),
    (["DefaultCPUAllocator: can't allocate memory: allocation failed"], "runtime_oom"),
    (["cuDNN error: CUDNN_STATUS_NOT_SUPPORTED."], "error"),
    (["client_request_timeout: planned_request_attempted=true; request_timeout_s=30"], "timeout"),
    (["TimeoutError: prediction did not complete"], "timeout"),
    (["ReadTimeout: peer stopped responding"], "timeout"),
    (["request timed out"], "timeout"),
    (["not_measured_after_timeout: trigger_error=client_request_timeout"], "skipped"),
    (["not_measured_after_startup_oom_pruning: reason=confirmed_startup_oom_at_reference_cpu"], "pruned_startup_oom"),
])
def test_feasibility_uses_record_local_failure_evidence(errors, expected):
    frame = pd.DataFrame([
        {"cpu_cores": 1, "mem_cap_gb": 4, "gpu_mode": "off", "input_scale": 2,
         "status": "error", "error": error}
        for error in errors
    ])
    before = frame.copy(deep=True)

    assert summarize_resource_feasibility(frame)["state"].tolist() == [expected]

    pd.testing.assert_frame_equal(frame, before)


def test_failure_classification_does_not_depend_on_repeat_order():
    frame = pd.DataFrame([
        {"cpu_cores": 1, "mem_cap_gb": 4, "gpu_mode": "off", "input_scale": 2,
         "status": "error", "error": error}
        for error in ["out of", "memory", "startup failed: unsupported model"]
    ])

    for ordered in (frame, frame.iloc[::-1], frame.iloc[[2, 0, 1]]):
        assert summarize_resource_feasibility(ordered)["state"].tolist() == ["error"]


@pytest.mark.parametrize(("phase", "oom_killed", "detail", "expected"), [
    ("startup", False, "", "error"), ("startup", True, "", "startup_oom"),
    ("runtime", True, "", "runtime_oom"), ("startup", False, "CUDA out of memory.", "startup_oom"),
])
def test_feasibility_recognizes_current_docker_diagnostics(phase, oom_killed, detail, expected, monkeypatch):
    from acprof.host import container_state

    state = {"Status": "exited", "Running": False, "Restarting": False,
             "OOMKilled": oom_killed, "ExitCode": 137 if oom_killed else 1, "Error": detail}
    monkeypatch.setattr(container_state, "inspect_container_state", lambda name: state)
    if phase == "startup":
        error = container_state.container_startup_exit_error("BLOOM-model-fixture", 4)
    else:
        error = container_state.container_runtime_oom_error("BLOOM-startup-fixture", 4, 1)
    assert error is not None
    frame = pd.DataFrame([{"cpu_cores": 1, "mem_cap_gb": 4, "gpu_mode": "off", "input_scale": 2,
                           "status": "error", "error": error}])

    assert summarize_resource_feasibility(frame)["state"].tolist() == [expected]
