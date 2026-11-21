"""Exercise the real client and resume planner with hardware only at a fake port."""
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.host import client, run_state
from acprof.host.measurement.config import ClientConfig
from acprof.monitors.resource_metrics import empty_resource_result
from acprof.result_store.writer import ResultWriter
from acprof.results.layers import read_result_layers


def test_auto_warmup_requests_are_unique_across_runtime_attempts(monkeypatch):
    calls = []
    for attempt in ("a" * 32, "b" * 32):
        config = ClientConfig(result_root="experiment", case_name="case", case_attempt_id=attempt,
                              auto_warmup_requests=2, repeat_in_window=0)
        runner = client.ClientRunner(config)
        request_ids = []
        monkeypatch.setattr(runner, "_one_request", lambda *args, **kwargs: request_ids.append(kwargs["req_id"]))
        runner._prepare_repeat_window(1.0, "seq1", {})
        calls.append(set(request_ids))
    assert len(calls[0]) == len(calls[1]) == 2
    assert calls[0].isdisjoint(calls[1])


@pytest.mark.parametrize("scales", [[1.0], [10.0000625, 10.000125]], ids=["single", "same-label"])
def test_partial_client_resume_skips_receipts_and_rewarms_outside_sampling(tmp_path, monkeypatch, request, scales):
    from tests.support.platform_fixtures import native_policy
    native_policy(request)
    monkeypatch.setattr(run_state, "MEASUREMENT_LOCK_ROOT", tmp_path)
    monkeypatch.setattr(run_state, "host_identity", lambda _: {"fixture": "fixed"})
    state = run_state.RunState(tmp_path / "experiment", {
        "cpus": "1", "mems": "4", "gpus": "off", "warmup": 1, "repeat": 2,
    }, resume=False, project_dir=".")
    state.data["runtime"] = {"planned": {"scales": scales}}
    case = state.layout.case("fixture/model", 1, 4, "off")
    plan = tmp_path / "input_scale_plan.json"
    entries = [{"input_scale": scale, "scale_label": f"dur{scale:g}s", "payload": {}} for scale in scales]
    plan.write_text(json.dumps({"schema_version": 2, "entries": entries}))
    config = ClientConfig(out_csv=str(case.csv), result_root=str(state.directory),
        cpu_cores="1", mem_cap_gb="4", gpu_mode="off", profiling_mode="basic",
        run_id=state.data["run_id"], case_id=case.case_id, input_scale_plan_file=str(plan),
        input_scale_order=list(reversed(scales)),
        case_name=case.case_id, warmup=1, repeat=2, repeat_in_window=1, use_mips=False)

    def new_attempt():
        state.prepare_case("unused.csv", 1, 4, "off")
        attempt = state.case_attempt_id(str(case.csv))
        return replace(config, case_attempt_id=attempt, window_records_dir=str(case.window_records(attempt)),
                       telemetry_dir=str(case.telemetry(attempt)))
    active = False
    calls = []
    monitor = Mock()
    monitor.telemetry_snapshot.return_value = {"status": "unavailable", "reason": "fixture"}
    def start():
        nonlocal active
        assert not active
        active = True
    def stop():
        nonlocal active
        assert active
        active = False
        return empty_resource_result(), "", []
    monitor.start.side_effect = start
    monitor.stop.side_effect = stop
    monitor.close.side_effect = lambda: None
    monkeypatch.setattr(client.requests, "get", lambda *a, **k: SimpleNamespace(status_code=200))
    original_append = ResultWriter._append_bytes
    def append(writer, *args):
        assert not active, "projection write overlapped the measurement window"
        return original_append(writer, *args)
    monkeypatch.setattr(ResultWriter, "_append_bytes", append)

    def build(settings, *, legacy_entry=None):
        runner = client.ClientRunner(settings)
        runner.cpu_energy_mod = None
        runner.resource_usage_mod = SimpleNamespace(ResourceUsageMonitor=lambda **kw: monitor)
        if legacy_entry is not None:
            # Reproduce the old receipt spelling for this scale without changing the frozen plan.
            runner.input_scale_entries = [legacy_entry]
        def one_request(scale, **kwargs):
            request_id = kwargs["req_id"]
            assert active == ("resume_warmup" not in request_id)
            calls.append((scale, request_id))
            if legacy_entry is not None and "_r1:" in request_id:
                raise KeyboardInterrupt()
            return {"latency_app_s": 0.1, "effective_input_scale": scale}
        monkeypatch.setattr(runner, "_one_request", one_request)
        return runner
    try:
        old_receipts = {}
        for entry in entries:
            legacy = new_attempt()
            with pytest.raises(KeyboardInterrupt):
                build(legacy, legacy_entry=entry).main()
            paths = case.window_records(legacy.case_attempt_id).glob("*.json")
            old_receipts.update({path: path.read_bytes() for path in paths})
        old_rows = read_result_layers(state.directory)[1]
        assert len(old_receipts) == 2 * len(scales)
        assert [(float(row["input_scale"]), row["warmup"], row["repeat_idx"]) for row in old_rows] == [
            (scale, warmup, "0") for scale in scales for warmup in ("1", "0")
        ]
        resumed = new_attempt()
        assert resumed.case_attempt_id not in {row["case_attempt_id"] for row in old_rows}
        calls.clear()
        build(resumed).main()
        rows = read_result_layers(state.directory)[1]
        assert rows[:len(old_rows)] == old_rows
        assert [(float(row["input_scale"]), row["repeat_idx"], row["case_attempt_id"]) for row in rows[len(old_rows):]] == [
            (scale, "1", resumed.case_attempt_id) for scale in reversed(scales)
        ]
        assert [scale for scale, _ in calls] == [scale for scale in reversed(scales) for _ in range(2)]
        assert all("resume_warmup" in request_id for _, request_id in calls[::2])
        assert all("_r1:" in request_id for _, request_id in calls[1::2])
        assert len({request_id for _, request_id in calls}) == len(calls)
        assert all(path.read_bytes() == raw for path, raw in old_receipts.items())
        evidence = [json.loads(path.read_bytes()) for path in state.layout.path("raw/recovery_warmup").rglob("*.json")]
        assert sorted((record["input_scale"], record["request_count"]) for record in evidence) == [
            (scale, 1) for scale in scales
        ]
    finally:
        state.close()


@pytest.mark.parametrize("stage", ["service_start", "client_ready"])
def test_restart_failure_preserves_committed_windows(tmp_path, monkeypatch, request, stage):
    from acprof.host import cases, docker_runtime, orchestrator
    from acprof.host.detect import TaskInfo
    from acprof.host.images.runtime import ImageInfo
    from acprof.results.csv import read_result_csv
    from tests.results.storage.test_incremental_results import receipt
    from tests.support.platform_fixtures import native_policy

    native_policy(request)
    monkeypatch.setattr(run_state, "MEASUREMENT_LOCK_ROOT", tmp_path)
    monkeypatch.setattr(run_state, "host_identity", lambda _: {"fixture": "fixed"})
    state = run_state.RunState(tmp_path / "experiment", {
        "cpus": "1", "mems": "4", "gpus": "off", "warmup": 0, "repeat": 2,
    }, resume=False, project_dir=".")
    try:
        state.data["runtime"] = {"planned": {"scales": [1.0]}}
        state.prepare_case("unused.csv", 1, 4, "off")
        case = state.layout.case("fixture/model", 1, 4, "off")
        path = receipt(state.directory, attempt=state.case_attempt_id(str(case.csv)))
        original_receipt = path.read_bytes()
        state.prepare_case("unused.csv", 1, 4, "off")
        original_row = read_result_csv(case.csv)[1][0]
        original_requests = case.requests.read_bytes()
        group = case.sidecar("sniff_groups").read_bytes()
        session = docker_runtime.RunningContainer(name="fixture", base_url="http://127.0.0.1:8002",
                                                  host_port=8002, cold_start_s=1.0, container_id="c"*64)
        start = Mock(side_effect=RuntimeError("fixture start failed") if stage == "service_start" else None,
                     return_value=session)
        monkeypatch.setattr(orchestrator, "start_service_session", start)
        monkeypatch.setattr(orchestrator, "stop_service_session", lambda *a, **kw: None)
        monkeypatch.setattr(orchestrator, "record_case_conditions", lambda *a, **kw: None)
        monkeypatch.setattr(client.requests, "get", lambda *a, **kw: SimpleNamespace(status_code=503, text="warming"))

        def client_command(cmd, **kwargs):
            assert cmd[-2:] == ["-m", "acprof.host.client"]
            runner = client.ClientRunner(ClientConfig.from_env(kwargs["env"]))
            runner.input_scale_entries = [{"input_scale": 1.0, "scale_label": "one", "payload": {}}]
            try:
                runner.run_cli()
            except SystemExit as exit_status:
                return SimpleNamespace(returncode=exit_status.code)
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(orchestrator.host_command, "run_command", client_command)
        task = TaskInfo("fixture/model", "fill-mask", "nlp", "transformers_pipeline", "transformers", "main", "manual")
        result = orchestrator.run_single_case(
            cases.CaseRequest(task, 1, 4, "off", ImageInfo(tag="fixture"), str(state.directory), "."),
            workload=cases.CaseWorkload(warmup=0, repeat=2, repeat_in_window=1),
            measurement=cases.CaseMeasurement(profiling_mode="basic", require_packet_latency=False),
            plans=cases.CasePlans(input_scales="1"), run_state=state,
        )
        state.finish_case(result, 1, 4, "off")
        _, rows = read_result_csv(case.csv, expected=state.expected(1, 4, "off"))
        assert rows[0] == original_row
        assert rows[1]["status"] == "error"
        assert ("fixture start failed" if stage == "service_start" else "HTTP 503: warming") in rows[1]["error"]
        assert read_result_layers(state.directory)[1] == rows
        assert path.read_bytes() == original_receipt
        assert case.retained_requests.read_bytes() == original_requests
        assert case.sidecar("sniff_groups").read_bytes() == group + b'{"sniff_group_id": ""}\n'
    finally:
        state.close()
