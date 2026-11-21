"""Exercise the real client and resume planner with hardware only at a fake port."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.host import client, run_state
from acprof.host.measurement.config import ClientConfig
from acprof.monitors.resource_metrics import empty_resource_result
from acprof.results.layers import read_result_layers
from acprof.result_store.writer import ResultWriter


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


def test_partial_client_resume_skips_receipts_and_rewarms_outside_sampling(tmp_path, monkeypatch, request):
    from tests.support.platform_fixtures import native_policy
    native_policy(request)
    monkeypatch.setattr(run_state, "MEASUREMENT_LOCK_ROOT", tmp_path)
    monkeypatch.setattr(run_state, "host_identity", lambda _: {"fixture": "fixed"})
    state = run_state.RunState(tmp_path / "experiment", {
        "cpus": "1", "mems": "4", "gpus": "off", "warmup": 1, "repeat": 2,
    }, resume=False, project_dir=".")
    state.data["runtime"] = {"planned": {"scales": [1.0]}}
    state.prepare_case("unused.csv", 1, 4, "off")
    case = state.layout.case("fixture/model", 1, 4, "off")
    first_attempt = state.case_attempt_id(str(case.csv))
    config = ClientConfig(out_csv=str(case.csv), result_root=str(state.directory),
        cpu_cores="1", mem_cap_gb="4", gpu_mode="off", profiling_mode="basic",
        run_id=state.data["run_id"], case_id=case.case_id, case_attempt_id=first_attempt,
        window_records_dir=str(case.window_records(first_attempt)), telemetry_dir=str(case.telemetry(first_attempt)),
        case_name=case.case_id, warmup=1, repeat=2, repeat_in_window=1, use_mips=False)
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

    def build(settings, *, interrupt):
        runner = client.ClientRunner(settings)
        runner.cpu_energy_mod = None
        runner.resource_usage_mod = SimpleNamespace(ResourceUsageMonitor=lambda **kw: monitor)
        runner.input_scale_entries = [{"input_scale": 1.0, "scale_label": "one", "payload": {}}]
        def one_request(*args, **kwargs):
            request_id = kwargs["req_id"]
            assert active == ("resume_warmup" not in request_id)
            calls.append(request_id)
            if interrupt and "_r1:" in request_id:
                raise KeyboardInterrupt()
            return {"latency_app_s": 0.1, "effective_input_scale": 1.0}
        monkeypatch.setattr(runner, "_one_request", one_request)
        return runner
    try:
        with pytest.raises(KeyboardInterrupt):
            build(config, interrupt=True).main()
        old_rows = read_result_layers(state.directory)[1]
        assert [(row["warmup"], row["repeat_idx"]) for row in old_rows] == [("1", "0"), ("0", "0")]
        old_receipts = {path: path.read_bytes() for path in case.window_records(first_attempt).glob("*.json")}
        state.prepare_case("unused.csv", 1, 4, "off")
        second_attempt = state.case_attempt_id(str(case.csv))
        assert second_attempt != first_attempt
        resumed = replace(config, case_attempt_id=second_attempt,
                          window_records_dir=str(case.window_records(second_attempt)),
                          telemetry_dir=str(case.telemetry(second_attempt)))
        calls.clear()
        build(resumed, interrupt=False).main()
        rows = read_result_layers(state.directory)[1]
        assert len(rows) == 3
        assert rows[:2] == old_rows
        assert rows[2]["case_attempt_id"] == second_attempt
        assert len(calls) == 2
        assert "resume_warmup" in calls[0]
        assert "_r1:" in calls[1]
        assert all(path.read_bytes() == raw for path, raw in old_receipts.items())
        assert len(list(state.layout.path("raw/recovery_warmup").rglob("*.json"))) == 1
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
