"""Warning diagnostics do not supply evidence of failed workload windows."""

import pandas as pd
import pytest

from acprof.plotting.diagnostics import summarize_resource_feasibility

DIMENSIONS = ["cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale"]


@pytest.mark.parametrize("message", [
    "gpu_avg_power_eff_w<0",
    "cpu_energy_eff_j<0; vcpu_energy_eff_j<0",
    "gpu monitor: NVML sampler failed",
    "cpu monitor: TimeoutError: sampler did not stop",
    "resource monitor: CUDA out of memory",
    "startup OOM",
    "not_measured_after_timeout",
    "not_measured_after_startup_oom_pruning",
])
@pytest.mark.parametrize("include_ok", [False, True])
def test_warning_text_preserves_warning_state(message, include_ok):
    rows = [(1, 4, "off", 2, " WaRN\n", message)]
    if include_ok:
        rows.append((1, 4, "off", 2, "ok", ""))
    frame = pd.DataFrame(rows, columns=[*DIMENSIONS, "status", "error"])
    before = frame.copy(deep=True)

    assert summarize_resource_feasibility(frame)["state"].tolist() == ["warn"]

    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("warning", [
    "gpu monitor: startup OOM", "cpu monitor: TimeoutError", "not_measured_after_startup_oom_pruning",
])
@pytest.mark.parametrize(("error", "expected"), [
    ("client_request_timeout", "timeout"),
    ("runtime OOM", "runtime_oom"),
    ("container_start_failed: CUDA out of memory", "startup_oom"),
    ("unsupported model", "error"),
    ("", "error"),
    ("not_measured_after_timeout", "skipped"),
    ("not_measured_after_startup_oom_pruning", "pruned_startup_oom"),
])
def test_only_failure_records_determine_the_failure_category(warning, error, expected):
    frame = pd.DataFrame([
        (1, 4, "off", 2, "warn", warning),
        (1, 4, "off", 2, "error", error),
    ], columns=[*DIMENSIONS, "status", "error"])

    assert summarize_resource_feasibility(frame)["state"].tolist() == [expected]

    # A valid successful repeat still makes real failures partial failures.
    frame.loc[len(frame)] = (1, 4, "off", 2, "ok", "")
    assert summarize_resource_feasibility(frame)["state"].tolist() == ["mixed"]


@pytest.mark.parametrize("status", [None, pd.NA, "", "unexpected status"])
def test_unrecognized_status_does_not_discard_failure_text(status):
    frame = pd.DataFrame([
        (1, 4, "off", 2, "warn", "gpu monitor: startup OOM"),
        (1, 4, "off", 2, status, "client_request_timeout"),
    ], columns=[*DIMENSIONS, "status", "error"])

    assert summarize_resource_feasibility(frame)["state"].tolist() == ["timeout"]


def test_warning_status_stays_paired_with_its_diagnostic_across_repeats():
    frame = pd.DataFrame([
        (1, 4, "off", 1, "warn", "gpu monitor: startup OOM"),
        (1, 8, "off", 1, "warn", "cpu monitor: TimeoutError"),
        (1, 4, "off", 1, "error", "client_request_timeout"),
        (1, 8, "off", 1, "ok", ""),
        (1, 4, "off", 2, "warn", "cpu_energy_eff_j<0"),
        (1, 4, "off", 2, "error", ""),
    ], columns=[*DIMENSIONS, "status", "error"], index=[7, 2, 7, 2, 7, 7])
    expected = pd.DataFrame([
        (1, 4, "off", 1, "timeout", False),
        (1, 4, "off", 2, "error", False),
        (1, 8, "off", 1, "warn", False),
    ], columns=[*DIMENSIONS, "state", "gpu_on"])

    for ordered in (frame, frame.iloc[::-1], frame.sample(frac=1, random_state=37)):
        before = ordered.copy(deep=True)
        pd.testing.assert_frame_equal(summarize_resource_feasibility(ordered), expected)
        pd.testing.assert_frame_equal(ordered, before)


@pytest.mark.parametrize("kind", ["gpu", "cpu", "resource", "negative_energy"])
def test_current_monitor_diagnostics_remain_warnings_in_feasibility(kind):
    from acprof.host.measurement.metrics import window_monitor_warnings

    gpu_metrics = dict.fromkeys(["gpu_avg_power_eff_w", "gpu_peak_power_eff_w", "gpu_energy_eff_j"], 0.)
    cpu_metrics = dict.fromkeys([
        "cpu_avg_power_eff_w", "cpu_peak_power_eff_w", "cpu_energy_eff_j",
        "vcpu_avg_power_eff_w", "vcpu_peak_power_eff_w", "vcpu_energy_eff_j",
    ], 0.)
    if kind == "negative_energy":
        gpu_metrics["gpu_avg_power_eff_w"] = -.25
        cpu_metrics["cpu_energy_eff_j"] = -.5
        cpu_metrics["vcpu_energy_eff_j"] = -.1
    warnings = window_monitor_warnings(
        resource_error="resource usage unavailable" if kind == "resource" else "",
        gpu_error="NVML sampler failed" if kind == "gpu" else "",
        cpu_error="RAPL sampler failed" if kind == "cpu" else "",
        dram_energy="optional", gpu_metrics=gpu_metrics, cpu_metrics=cpu_metrics,
    )
    assert warnings
    frame = pd.DataFrame([(1, 4, "off", 2, "warn", "; ".join(warnings))],
                         columns=[*DIMENSIONS, "status", "error"])

    assert summarize_resource_feasibility(frame)["state"].tolist() == ["warn"]
