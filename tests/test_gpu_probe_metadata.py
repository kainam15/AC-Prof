"""GPU probe evidence survives partial failures, adapters and post-hoc merging."""

import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.artifacts import read_static_metadata
from acprof.container import compute_profile_runner as runner
from acprof.gpu_probe import gpu_probe_report, record_gpu_probe_error, set_gpu_probe_value
from acprof.host.measurement import diagnostics
from acprof.host.posthoc import backfill, plans
from acprof.host.profiling import compute
from acprof.host.profiling.tools import ncu, torch as torch_profiler

CAP = "gpu_compute_capability"
SM = "gpu_sm_count"
REPORT = "gpu_compute_metadata"
UUID = "12345678-1234-1234-1234-123456789abc"


@pytest.fixture
def cuda_backend(monkeypatch):
    cuda = SimpleNamespace(
        is_available=Mock(return_value=True),
        current_device=Mock(return_value=0),
        get_device_capability=Mock(return_value=(8, 6)),
        get_device_properties=Mock(
            return_value=SimpleNamespace(
                multi_processor_count=84,
                uuid=UUID,
            )
        ),
    )
    monkeypatch.setattr(runner, "torch", SimpleNamespace(cuda=cuda))
    return cuda


def test_cpu_probe_does_not_touch_cuda(cuda_backend):
    result = runner._gpu_metadata("cpu")
    assert result[CAP] is None
    assert result[SM] is None
    assert all(field["status"] == "not_applicable" for field in result[REPORT]["fields"].values())
    assert result[REPORT]["errors"] == []
    cuda_backend.is_available.assert_not_called()
    cuda_backend.current_device.assert_not_called()


def test_cuda_probe_keeps_valid_sm_when_capability_fails(cuda_backend):
    cuda_backend.get_device_capability.side_effect = RuntimeError("capability denied")
    result = runner._gpu_metadata("cuda")
    assert result[CAP] is None
    assert result[SM] == 84
    assert result[REPORT]["device"] == {"cuda_index": 0, "cuda_uuid": UUID}
    assert result[REPORT]["fields"][SM]["source"] == "torch.cuda"
    assert result[REPORT]["errors"][0]["message"] == "capability denied"
    assert result[REPORT]["errors"][0]["field"] == CAP


def test_cuda_probe_keeps_valid_capability_when_properties_fail(cuda_backend):
    cuda_backend.get_device_properties.side_effect = RuntimeError("properties denied")
    result = runner._gpu_metadata("cuda")
    assert result[CAP] == "8.6"
    assert result[SM] is None
    assert result[REPORT]["errors"][0]["stage"] == "properties"


@pytest.mark.parametrize("failure", [False, RuntimeError("driver not loaded")])
def test_cuda_availability_failures_are_explicit(cuda_backend, failure):
    if isinstance(failure, Exception):
        cuda_backend.is_available.side_effect = failure
    else:
        cuda_backend.is_available.return_value = failure
    result = runner._gpu_metadata("cuda")
    assert result[CAP] is None
    assert result[SM] is None
    assert result[REPORT]["errors"][0]["source"] == "torch.cuda"
    assert result[REPORT]["errors"][0]["stage"] == "availability"
    cuda_backend.get_device_capability.assert_not_called()


@pytest.mark.parametrize("count", [None, 0, True, 2.5, float("inf")])
def test_cuda_invalid_sm_count_never_becomes_zero_or_truncated(cuda_backend, count):
    cuda_backend.get_device_properties.return_value.multi_processor_count = count
    result = runner._gpu_metadata("cuda")
    assert result[CAP] == "8.6"
    assert result[SM] is None
    assert result[REPORT]["errors"][0]["field"] == SM


def _record(*, capability=None, count=None, uuid=UUID, error=None):
    report = gpu_probe_report((CAP, SM))
    report["device"]["cuda_uuid"] = uuid
    for key, value in ((CAP, capability), (SM, count)):
        if value is not None:
            set_gpu_probe_value(report, key, value, "torch.cuda")
    if error:
        record_gpu_probe_error(report, "torch.cuda", "query", RuntimeError(error), field=CAP)
    return {CAP: capability, SM: count, REPORT: report, "input_scale": 1}


def test_failed_probe_then_success_is_aggregated_without_losing_errors():
    failed = _record(error="first probe failed")
    success = _record(capability="8.6", count=84)
    profiles = {
        "cpu": {torch_profiler.TORCH_PROFILER_TOOL: {CAP: "9.0"}},
        "gpu": {torch_profiler.TORCH_PROFILER_TOOL: {"entries": [failed, success]}},
    }
    compute._bind_gpu_profile_identity(profiles, f"GPU-{UUID}")
    assert compute._first_profile_value(profiles, CAP, "unknown") == "8.6"
    assert compute._first_profile_value(profiles, SM, "unknown") == 84
    reports = compute._gpu_profile_reports(profiles)
    assert len(reports) == 2
    assert reports[0]["probe"]["errors"][0]["message"] == "first probe failed"
    assert reports[1]["probe"]["device"]["identity_status"] == "verified"
    reports[0]["probe"]["errors"].clear()
    assert failed[REPORT]["errors"]


def test_different_devices_cannot_supply_one_combined_gpu_identity():
    wrong = _record(capability="9.0", count=120, uuid="another-device")
    correct = _record(count=84)
    profiles = {"gpu": {torch_profiler.TORCH_PROFILER_TOOL: {"entries": [wrong, correct]}}}
    compute._bind_gpu_profile_identity(profiles, f"GPU-{UUID}")
    assert compute._first_profile_value(profiles, CAP, "unknown") == "unknown"
    assert compute._first_profile_value(profiles, SM, "unknown") == 84
    assert wrong[REPORT]["device"]["identity_status"] == "mismatch"
    assert "gpu_identity_mismatch" in wrong["error"]
    assert wrong[REPORT]["fields"][CAP]["value"] == "9.0"


@pytest.mark.parametrize(
    "invalid", [None, "", "unknown", "unavailable", False, 0, 2.5, float("nan")]
)
def test_legacy_unknown_sm_does_not_hide_later_valid_value(invalid):
    profiles = {"gpu": {torch_profiler.TORCH_PROFILER_TOOL: {"entries": [{SM: invalid}, {SM: 84}]}}}
    assert compute._first_profile_value(profiles, SM, "unknown") == 84


def test_missing_cuda_uuid_is_explicitly_unverified():
    record = _record(capability="8.6", uuid=None)
    profiles = {"gpu": {torch_profiler.TORCH_PROFILER_TOOL: {"entries": [record]}}}
    compute._bind_gpu_profile_identity(profiles, f"GPU-{UUID}")
    assert record[REPORT]["device"]["identity_status"] == "unverified"
    assert compute._first_profile_value(profiles, CAP, "unknown") == "8.6"


def test_ncu_fallback_retains_torch_failure_and_records_actual_field_source():
    payload = _record(count=84, error="capability denied")
    before = copy.deepcopy(payload)
    result = ncu._gpu_profile_metadata(payload, {CAP: "8.6", SM: 999})
    assert result[CAP] == "8.6"
    assert result[SM] == 84
    assert result[REPORT]["fields"][CAP]["source"] == "ncu"
    assert result[REPORT]["fields"][SM]["source"] == "torch.cuda"
    assert result[REPORT]["errors"][0]["message"] == "capability denied"
    assert payload == before


@pytest.mark.parametrize(
    "report",
    [
        {"schema_version": 1},
        {"schema_version": 1, "device": {}, "fields": {CAP: {"status": []}}, "errors": []},
    ],
)
def test_malformed_transport_report_is_rejected_explicitly(report):
    with pytest.raises(ValueError, match="invalid GPU compute metadata report"):
        ncu._gpu_profile_metadata({REPORT: report}, {})


def test_ncu_parse_failure_still_returns_gpu_evidence(monkeypatch, tmp_path):
    payload = _record(capability="8.6", error="earlier failure")
    monkeypatch.setattr(ncu, "parse_ncu_profile_csv", Mock(side_effect=ValueError("bad csv")))
    result = ncu._ncu_entry_from_csv(
        entry={"input_scale": 1},
        host_csv=str(tmp_path / "ncu.csv"),
        profile_root=str(tmp_path),
        repeat=1,
        runner_payload=payload,
    )
    assert "bad csv" in result["error"]
    assert result[CAP] == "8.6"
    assert result[REPORT]["errors"][0]["message"] == "earlier failure"


@pytest.mark.parametrize("valid_flops", [True, False])
def test_torch_adapter_preserves_probe_on_success_and_parse_failure(monkeypatch, valid_flops):
    payload = {
        **_record(capability="8.6", count=84),
        "attention_implementation": "eager",
        "attention_implementation_verified": True,
    }
    if valid_flops:
        payload["model_logical_mflop_per_request_torch_profiler_eager"] = 2.0
    monkeypatch.setattr(
        torch_profiler, "profiler_container_command", lambda **kwargs: nullcontext(["docker"])
    )
    monkeypatch.setattr(torch_profiler, "profile_runner_args", lambda *args: ["runner"])
    monkeypatch.setattr(
        torch_profiler,
        "run_command",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        ),
    )
    result = torch_profiler._run_torch_profiler_for_entry(
        task_info=SimpleNamespace(),
        image_tag="test",
        cpu=1,
        mem=1,
        use_gpu=True,
        payload_file="plan.json",
        profile_root="profiles",
        entry={"input_scale": 1},
        repeat=1,
    )
    assert result[REPORT] == payload[REPORT]
    assert result[CAP] == "8.6"
    assert result[SM] == 84
    assert bool(result["error"]) is not valid_flops


def test_merge_and_backfill_preserve_all_probe_attempts_and_valid_scalars():
    first, second = _record(capability="8.6"), _record(error="second attempt failed")
    context = SimpleNamespace(
        task_info=SimpleNamespace(
            model_id="test",
            model_revision="main",
            task_family="nlp",
            pipeline_tag="fill-mask",
            runtime_backend="transformers_pipeline",
        ),
        static_meta={CAP: "8.6", REPORT: [{"probe": first[REPORT]}]},
    )
    merged = plans.merge_compute_plans(
        context,
        {
            "torch": {"static_metadata": {CAP: "8.6", REPORT: [{"probe": first[REPORT]}]}},
            "ncu": {"static_metadata": {CAP: "unknown", REPORT: [{"probe": second[REPORT]}]}},
        },
    )
    assert merged["static_metadata"][CAP] == "8.6"
    assert len(merged["static_metadata"][REPORT]) == 2
    updated = backfill.update_static_meta(
        context,
        tools=["ncu"],
        compute_plan={
            "static_metadata": {CAP: "unavailable", REPORT: [{"probe": second[REPORT]}]},
        },
        execution_plan=None,
    )
    assert updated[CAP] == "8.6"
    assert len(updated[REPORT]) == 2
    assert len(context.static_meta[REPORT]) == 1


def test_schema_v7_without_new_optional_probe_fields_remains_readable(tmp_path):
    (tmp_path / "static_meta.json").write_text(json.dumps({"schema_version": 7}), encoding="utf-8")
    metadata = read_static_metadata(tmp_path)
    assert metadata["schema_version"] == 7
    assert "gpu_metadata" not in metadata
    assert REPORT not in metadata


def test_uuid_snapshot_resolves_current_index_before_pmon(monkeypatch):
    calls = []

    def query(command):
        calls.append(command)
        return f"3,Test GPU,P8,10,200,100,100,100,100,30,0,0,50,8000,GPU-{UUID}"

    monkeypatch.setattr(diagnostics, "_run_text", query)
    pmon = Mock(return_value=[])
    apps = Mock(return_value=[])
    monkeypatch.setattr(diagnostics, "_collect_nvidia_smi_pmon", pmon)
    monkeypatch.setattr(diagnostics, "_collect_nvidia_smi_compute_apps", apps)
    result = diagnostics.collect_gpu_idle_debug_snapshot(9, device_uuid=f"GPU-{UUID}")
    assert result["nvidia_smi_gpu"]["index"] == 3
    assert f"--id=GPU-{UUID}" in calls[0]
    pmon.assert_called_once_with(3)
    apps.assert_called_once_with(f"GPU-{UUID}")


def test_uuid_mismatch_blocks_further_idle_queries(monkeypatch):
    monkeypatch.setattr(
        diagnostics,
        "_run_text",
        lambda command: "3,Test GPU,P8,10,200,100,100,100,100,30,0,0,50,8000,GPU-wrong",
    )
    pmon, apps = Mock(), Mock()
    monkeypatch.setattr(diagnostics, "_collect_nvidia_smi_pmon", pmon)
    monkeypatch.setattr(diagnostics, "_collect_nvidia_smi_compute_apps", apps)
    result = diagnostics.collect_gpu_idle_debug_snapshot(9, device_uuid=f"GPU-{UUID}")
    assert "UUID mismatch" in result["nvidia_smi_gpu_error"]
    pmon.assert_not_called()
    apps.assert_not_called()
