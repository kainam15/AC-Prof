"""Recovery reuses committed measurements and never silently reruns inference."""
from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from acprof.config import CLIENT_REQUEST_TIMEOUT_EXIT_CODE, CSV_FIELDS, SERVER_PORT
from acprof.host import cases, orchestrator, run_state
from acprof.host.container_lifecycle import ContainerCleanupError
from acprof.host.detect import TaskInfo
from acprof.host.docker_runtime import RunningContainer
from acprof.host.images.runtime import ImageInfo
from acprof.host.measurement.capture import PacketLatencyRuntime
from acprof.host.measurement.publication import _append_row, append_request_window
from acprof.results.windows import WindowPublicationContext
from acprof.telemetry import window_evidence


@pytest.fixture
def measured_case(tmp_path, monkeypatch):
    monkeypatch.setattr(run_state, "MEASUREMENT_LOCK_ROOT", tmp_path)
    monkeypatch.setattr(run_state, "host_identity", lambda _: {"test_host": "fixed"})
    options = {"cpus": "1", "mems": "4", "gpus": "off", "warmup": 0, "repeat": 1}
    state = run_state.RunState(tmp_path / "experiment", options, resume=False, project_dir=".")
    state.data["runtime"] = {"planned": {"scales": [64.0]}}
    state.prepare_case("unused.csv", 1, 4, "off")
    case = state.layout.case("org/model", 1, 4, "off")
    calls = {"client": 0, "container": 0, "cleanup": 0, "env": {}}

    def start(**kwargs):
        calls["container"] += 1
        return RunningContainer(name=kwargs["container_name"], base_url="http://fixture", host_port=8106,
                                cold_start_s=1.0, container_id="c" * 64)

    def stop(*args, **kwargs):
        calls["cleanup"] += 1

    class Capture:
        stopped = False

        def poll(self):
            return 0 if self.stopped else None

        def terminate(self):
            self.stopped = True

        def wait(self, timeout=None):
            return 0

    def client(cmd, **kwargs):
        assert cmd[-2:] == ["-m", "acprof.host.client"]
        calls["client"] += 1
        env = kwargs["env"]
        calls["env"] = env
        context = WindowPublicationContext(
            env["ACPROF_RUN_ID"], env["ACPROF_CASE_ID"], env["ACPROF_CASE_ATTEMPT_ID"],
            env["ACPROF_WINDOW_RECORDS_DIR"], env["ACPROF_TELEMETRY_DIR"],
        )
        group = "case-scale64-repeat0"
        evidence = window_evidence(identity=context.identity(group), context={"sampling_policy": "burst"},
                                   collectors={"resource": {"status": "unavailable", "reason": "fixture"}})
        row = dict.fromkeys(CSV_FIELDS, "nan")
        row.update(cpu_cores="1", mem_cap_gb="4", gpu_mode="off", input_scale="64", warmup="0",
                   repeat_idx="0", status="ok", error="", latency_app_s="0.100000",
                   environment_class="native_linux")
        with case.csv.open("w", newline="") as output, case.requests.open("w") as requests, case.sidecar(
                "sniff_groups").open("w") as groups:
            request = append_request_window(requests, sniff_group_id=group, input_scale=64.0,
                                            warmup=0, repeat_idx=0, latencies=[0.1])
            writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
            writer.writeheader()
            _append_row(writer, row, output, groups, group, publication=context,
                        telemetry_evidence=evidence, request_record=request)
        case.pcap.write_bytes(b"fixture-captured-packets")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(orchestrator, "start_service_session", start)
    monkeypatch.setattr(orchestrator, "stop_service_session", stop)
    monkeypatch.setattr(orchestrator, "record_case_conditions", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator, "_resolve_packet_latency_runtime", lambda **k: PacketLatencyRuntime(
        mode="local", tcpdump_cmd=["fixture-capture"], parse_cmd=["fixture-parser"]))
    monkeypatch.setattr(orchestrator.subprocess, "Popen", lambda *a, **k: Capture())
    monkeypatch.setattr(orchestrator.time, "sleep", lambda *a: None)
    monkeypatch.setattr(orchestrator.host_command, "run_command", client)
    arguments = {
        "request": cases.CaseRequest(
            task_info=TaskInfo("org/model", "fill-mask", "nlp", "transformers_pipeline",
                               "transformers", "a" * 40, "manual"),
            image_info=ImageInfo(tag="sha256:" + "b" * 64), cpu=1, mem=4, gpu="off",
            output_dir=str(state.directory), project_dir=".",
        ),
        "workload": cases.CaseWorkload(warmup=0, repeat=1),
        "measurement": cases.CaseMeasurement(profiling_mode="full"),
        "plans": cases.CasePlans(input_scales="64"),
        "run_state": state,
    }
    yield state, case, calls, arguments
    state.close()


def fail_postprocessing(*args, **kwargs):
    Path(args[7]).write_text("partially merged projection\n")
    raise RuntimeError("packet parser interrupted")


@pytest.mark.parametrize("stage", ["parsing", "merge"])
def test_packet_timeout_preserves_checkpoint_cleanup_and_resume(measured_case, monkeypatch, stage):
    import subprocess

    state, case, calls, arguments = measured_case
    client = orchestrator.host_command.run_command

    def run(command, **kwargs):
        if command == ["fixture-parser"]:
            assert kwargs.get("timeout") == 30
            assert kwargs.get("kill_process_group") is True
            if stage != "parsing":
                return SimpleNamespace(returncode=0, stdout='{"requests":{"request":{}}}')
            raise subprocess.TimeoutExpired(command, 30)
        if "acprof.packet.merge_packet_latency" in command:
            assert kwargs.get("timeout") == 30
            assert kwargs.get("kill_process_group") is True
            raise subprocess.TimeoutExpired(command, 30)
        return client(command, **kwargs)

    monkeypatch.setenv("ACPROF_POSTPROCESS_TIMEOUT_S", "30")
    monkeypatch.setattr(orchestrator.host_command, "run_command", run)
    with pytest.raises(RuntimeError, match=stage + " timed out"):
        orchestrator.run_single_case(**arguments)
    record = state.data["cases"][str(case.csv.relative_to(state.directory))]
    assert record["measurement_status"] == "complete"
    assert record["processing_status"] == "failed"
    assert record["cleanup_status"] == "complete"
    snapshot = case.attempt_directory("measurements", record["case_attempt_id"]) / "measurement.csv"
    original = snapshot.read_bytes()
    state.close()
    # Opening a fresh owner proves both experiment and host locks were released.
    resumed = run_state.RunState(state.directory, state.data["options"], resume=True, project_dir=".")
    try:
        monkeypatch.setattr(orchestrator, "_finalize_case", lambda *args: None)
        assert resumed.prepare_case("unused.csv", 1, 4, "off") is None
        orchestrator.run_single_case(**{**arguments, "run_state": resumed})
        assert case.csv.read_bytes() == original
    finally:
        resumed.close()
    assert calls["client"] == calls["container"] == calls["cleanup"] == 1


def test_postprocessing_failure_resumes_without_client_or_container(measured_case, monkeypatch):
    state, case, calls, arguments = measured_case
    monkeypatch.setattr(orchestrator, "_finalize_case", fail_postprocessing)
    with pytest.raises(RuntimeError, match="packet parser interrupted"):
        orchestrator.run_single_case(**arguments)
    record = state.data["cases"][str(case.csv.relative_to(state.directory))]
    assert record["measurement_status"] == "complete"
    assert record["processing_status"] == "failed"
    assert record["cleanup_status"] == "complete"
    attempt = record["case_attempt_id"]
    snapshot = case.attempt_directory("measurements", attempt) / "measurement.csv"
    original = snapshot.read_bytes()
    assert case.csv.read_bytes() != original
    observed = []

    def derive(*args):
        assert case.csv.read_bytes() == original
        assert args[0] is True
        assert args[1].parse_cmd[-1] == str(SERVER_PORT)
        assert "fixture-parser" not in args[1].parse_cmd
        observed.append("derived")

    monkeypatch.setattr(orchestrator, "_finalize_case", derive)
    assert state.prepare_case("unused.csv", 1, 4, "off") is None
    assert orchestrator.run_single_case(**arguments) == str(case.csv)
    state.finish_case(str(case.csv), 1, 4, "off")
    assert observed == ["derived"]
    assert calls["client"] == calls["container"] == calls["cleanup"] == 1
    assert record["case_attempt_id"] == attempt
    assert snapshot.read_bytes() == original
    assert state.data["cases"][str(case.csv.relative_to(state.directory))]["processing_attempts"] == 2
    assert not list(state.layout.path("interrupted_cases").rglob("*.csv"))


def test_changed_measurement_is_rejected_before_projection_or_inference(measured_case, monkeypatch):
    state, case, calls, arguments = measured_case
    monkeypatch.setattr(orchestrator, "_finalize_case", fail_postprocessing)
    with pytest.raises(RuntimeError):
        orchestrator.run_single_case(**arguments)
    before = case.csv.read_bytes()
    attempt = state.case_attempt_id(str(case.csv))
    raw = case.attempt_directory("measurements", attempt) / "measurement.csv"
    raw.write_bytes(raw.read_bytes() + b"changed\n")
    with pytest.raises(run_state.RunStateError, match="原始证据无效"):
        state.prepare_case("unused.csv", 1, 4, "off")
    assert calls["client"] == calls["container"] == 1
    assert case.csv.read_bytes() == before
    assert raw.read_bytes().endswith(b"changed\n")


def test_persisted_checkpoint_survives_fresh_run_state(measured_case, monkeypatch):
    state, case, calls, arguments = measured_case
    monkeypatch.setattr(orchestrator, "_finalize_case", fail_postprocessing)
    with pytest.raises(RuntimeError):
        orchestrator.run_single_case(**arguments)
    attempt = state.case_attempt_id(str(case.csv))
    state.close()
    resumed = run_state.RunState(state.directory, state.data["options"], resume=True, project_dir=".")
    try:
        monkeypatch.setattr(orchestrator, "_finalize_case", lambda *args: None)
        assert resumed.prepare_case("unused.csv", 1, 4, "off") is None
        orchestrator.run_single_case(**{**arguments, "run_state": resumed})
        resumed.finish_case(str(case.csv), 1, 4, "off")
        assert resumed.case_attempt_id(str(case.csv)) == attempt
        assert calls["client"] == calls["container"] == 1
    finally:
        resumed.close()


def test_committed_checkpoint_survives_missing_run_state_pointer(measured_case, monkeypatch):
    from acprof.host import container_lifecycle
    state, case, calls, arguments = measured_case
    original_save = state.save

    def lose_pointer():
        record = state.data["cases"][str(case.csv.relative_to(state.directory))]
        if record.get("measurement_status") != "complete":
            original_save()

    monkeypatch.setattr(state, "save", lose_pointer)
    monkeypatch.setattr(orchestrator, "_finalize_case", fail_postprocessing)
    with pytest.raises(RuntimeError):
        orchestrator.run_single_case(**arguments)
    state.data = run_state.load_run_state(state.directory)
    record = state.data["cases"][str(case.csv.relative_to(state.directory))]
    assert record["measurement_status"] == "running"
    assert "measurement_checkpoint" not in record
    assert record["cleanup_container_id"] == "c" * 64
    attempt = record["case_attempt_id"]
    monkeypatch.setattr(state, "save", original_save)
    monkeypatch.setattr(container_lifecycle, "recover_cleanup_debt", lambda *a: {"status": "complete"})
    monkeypatch.setattr(orchestrator, "_finalize_case", lambda *a: None)
    state.prepare_case("unused.csv", 1, 4, "off")
    orchestrator.run_single_case(**arguments)
    assert state.case_attempt_id(str(case.csv)) == attempt
    assert calls["client"] == calls["container"] == 1
    assert record["recovered_checkpoint_at"]


def test_uncommitted_window_requires_new_whole_case_attempt(measured_case, monkeypatch):
    state, case, calls, arguments = measured_case
    client = orchestrator.host_command.run_command

    def lose_commit(*args, **kwargs):
        result = client(*args, **kwargs)
        for receipt in case.window_records(state.case_attempt_id(str(case.csv))).glob("*.json"):
            receipt.unlink()
        return result

    monkeypatch.setattr(orchestrator.host_command, "run_command", lose_commit)
    with pytest.raises(ValueError, match="without a committed window receipt"):
        orchestrator.run_single_case(**arguments)
    previous = state.case_attempt_id(str(case.csv))
    telemetry = list(case.telemetry(previous).glob("*.json"))
    assert telemetry
    assert state.prepare_case("unused.csv", 1, 4, "off") is None
    assert state.case_attempt_id(str(case.csv)) != previous
    assert all(path.is_file() for path in telemetry)
    assert list(state.layout.path("interrupted_cases").rglob("result.csv"))
    assert calls["client"] == 1


def test_timeout_keeps_partial_requests_without_reusing_them_as_committed_windows(measured_case, monkeypatch):
    state, case, calls, arguments = measured_case
    state.data["options"]["repeat"] = 2
    client = orchestrator.host_command.run_command

    def timeout_after_committed_window(*args, **kwargs):
        client(*args, **kwargs)
        with case.requests.open("a") as stream:
            append_request_window(stream, sniff_group_id="case-scale64-repeat1", input_scale=64.0,
                                  warmup=0, repeat_idx=1, latencies=[0.2], error="request timeout",
                                  failed_request_id="case-scale64-repeat1:1")
        return SimpleNamespace(returncode=CLIENT_REQUEST_TIMEOUT_EXIT_CODE)

    monkeypatch.setattr(orchestrator.host_command, "run_command", timeout_after_committed_window)
    monkeypatch.setattr(orchestrator, "service_failure", lambda *args: None)
    monkeypatch.setattr(orchestrator, "_finalize_case", fail_postprocessing)
    with pytest.raises(RuntimeError, match="packet parser interrupted"):
        orchestrator.run_single_case(**{**arguments, "workload": replace(arguments["workload"], repeat=2)})
    attempt = state.case_attempt_id(str(case.csv))
    checkpoint = state.processing_checkpoint(str(case.csv), 1, 4, "off")
    assert checkpoint["processing"]["case_incomplete"] is True
    assert len(checkpoint["window_records"]) == 1
    monkeypatch.setattr(orchestrator, "_finalize_case", lambda *args: None)
    orchestrator.run_single_case(**{**arguments, "workload": replace(arguments["workload"], repeat=2)})
    state.finish_case(str(case.csv), 1, 4, "off")
    record = state.data["cases"][str(case.csv.relative_to(state.directory))]
    assert record["case_attempt_id"] == attempt
    assert record["status"] == "complete"
    assert record["error_rows"] == 1
    assert len(case.retained_requests.read_text().splitlines()) == 2
    assert calls["client"] == calls["container"] == 1


def test_cleanup_debt_is_verified_before_retrying_only_postprocessing(measured_case, monkeypatch):
    from acprof.host import container_lifecycle
    state, case, calls, arguments = measured_case
    monkeypatch.setattr(orchestrator, "_finalize_case", fail_postprocessing)

    def cannot_stop(*args, **kwargs):
        raise ContainerCleanupError("c" * 64, [], final_state="present")

    monkeypatch.setattr(orchestrator, "stop_service_session", cannot_stop)
    with pytest.raises(ContainerCleanupError):
        orchestrator.run_single_case(**arguments)
    monkeypatch.setattr(container_lifecycle, "recover_cleanup_debt", cannot_stop)
    with pytest.raises(ContainerCleanupError):
        state.prepare_case("unused.csv", 1, 4, "off")
    assert calls["client"] == calls["container"] == 1
    checked = []

    def recovered(errors, run):
        checked.append(errors)
        return {"status": "complete", "verified_container_ids": ["c" * 64]}

    monkeypatch.setattr(container_lifecycle, "recover_cleanup_debt", recovered)
    monkeypatch.setattr(orchestrator, "_finalize_case", lambda *args: None)
    assert state.prepare_case("unused.csv", 1, 4, "off") is None
    orchestrator.run_single_case(**arguments)
    assert checked == [[{"container_id": "c" * 64}]]
    assert calls["client"] == calls["container"] == 1
    assert not case.sidecar("cleanup_error").exists()
    assert list((state.directory / "raw/cleanup").rglob("*.json"))


def test_sampling_identity_is_forwarded_before_client_start(measured_case, monkeypatch):
    state, case, calls, arguments = measured_case
    monkeypatch.setattr(orchestrator, "_finalize_case", lambda *args: None)
    orchestrator.run_single_case(**{
        **arguments,
        "measurement": replace(arguments["measurement"], sampling_policy="skip", raw_sample_limit=50),
    })
    env = calls["env"]
    assert env["ACPROF_SAMPLING_POLICY"] == "skip"
    assert env["ACPROF_RAW_SAMPLE_LIMIT"] == "50"
    assert env["ACPROF_RUN_ID"] == state.data["run_id"]
    assert env["ACPROF_CASE_ID"] == case.case_id
    assert env["ACPROF_CASE_ATTEMPT_ID"] == state.case_attempt_id(str(case.csv))


def test_client_cancellation_and_capture_failure_still_release_container(measured_case, monkeypatch):
    state, case, calls, arguments = measured_case
    client_error = KeyboardInterrupt("client interrupted")
    cleanup_error = ContainerCleanupError("c" * 64, [], final_state="present")
    capture_stops = []
    events = []

    def interrupted(*args, **kwargs):
        raise client_error

    def broken_capture(process):
        capture_stops.append(process)
        raise RuntimeError("capture close failed")

    def broken_container(*args, **kwargs):
        calls["cleanup"] += 1
        raise cleanup_error

    monkeypatch.setattr(orchestrator.host_command, "run_command", interrupted)
    monkeypatch.setattr(orchestrator, "_stop_capture", broken_capture)
    monkeypatch.setattr(orchestrator, "stop_service_session", broken_container)
    monkeypatch.setattr(orchestrator, "emit_event", lambda kind, detail, **values: events.append((kind, values)))
    with pytest.raises(ContainerCleanupError) as caught:
        orchestrator.run_single_case(**arguments)

    assert caught.value is cleanup_error
    assert caught.value.run_error is client_error
    assert len(capture_stops) == 1
    assert calls["container"] == calls["cleanup"] == 1
    record = state.data["cases"][str(case.csv.relative_to(state.directory))]
    # Persist failed cleanup even before a measurement checkpoint so recovery
    # cannot mistake an attempted, failed cleanup for untouched ownership.
    assert record["cleanup_status"] == "failed"
    assert record["cleanup_container_id"] == "c" * 64
    assert record["measurement_status"] != "complete"
    assert case.sidecar("cleanup_error").is_file()
    assert events[-1] == ("case_finished", {"status": "error"})
