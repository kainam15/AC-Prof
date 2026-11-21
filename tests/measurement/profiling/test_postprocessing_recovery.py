"""Recovery reuses committed measurements and never silently reruns inference."""
from __future__ import annotations

import csv
import json
import shutil
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
from acprof.host.input_plan import PlannedInputScales
from acprof.host.measurement.capture import PacketLatencyRuntime
from acprof.host.measurement.publication import append_request_window, append_row
from acprof.installation import module_command
from acprof.results.windows import WindowPublicationContext
from acprof.telemetry import window_evidence


@pytest.fixture
def measured_case(tmp_path, monkeypatch):
    monkeypatch.setattr(run_state, "MEASUREMENT_LOCK_ROOT", tmp_path)
    monkeypatch.setattr(run_state, "host_identity", lambda _: {"test_host": "fixed"})
    options = {"cpus": "1", "mems": "4", "gpus": "off", "warmup": 0, "repeat": 1}
    state = run_state.RunState(tmp_path / "experiment", options, resume=False, project_dir=".")
    state.bind_runtime(
        TaskInfo("org/model", "fill-mask", "nlp", "transformers_pipeline",
                 "transformers", "a" * 40, "manual"),
        ImageInfo("sha256:" + "b" * 64), PlannedInputScales([64.0], "manual"), "", "",
    )
    state.prepare_case("unused.csv", 1, 4, "off")
    case = state.layout.case("org/model", 1, 4, "off")
    calls = {"client": 0, "container": 0, "cleanup": 0, "env": {}}
    real_command = orchestrator.host_command.run_command
    real_popen = orchestrator.subprocess.Popen

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
        if cmd[-2:] != ["-m", "acprof.host.client"]:
            return real_command(cmd, **kwargs)
        calls["client"] += 1
        env = kwargs["env"]
        calls["env"] = env
        context = WindowPublicationContext(
            env["ACPROF_RUN_ID"], env["ACPROF_CASE_ID"], env["ACPROF_CASE_ATTEMPT_ID"],
            env["ACPROF_WINDOW_RECORDS_DIR"], env["ACPROF_TELEMETRY_DIR"],
        )
        group = calls.get("group", "case-scale64-repeat0")
        evidence = window_evidence(identity=context.identity(group), context={"sampling_policy": "burst"},
                                   collectors={"resource": {"status": "unavailable", "reason": "fixture"}})
        row = dict.fromkeys(CSV_FIELDS, "nan")
        row.update(cpu_cores="1", mem_cap_gb="4", gpu_mode="off", input_scale="64", warmup="0",
                   repeat_idx="0", status="ok", error="", latency_app_s="0.100000",
                   environment_class="native_linux", cpu_idle_power_w="1.0")
        with case.csv.open("w", newline="") as output, case.requests.open("w") as requests, case.sidecar(
                "sniff_groups").open("w") as groups:
            request = append_request_window(requests, sniff_group_id=group, input_scale=64.0,
                                            warmup=0, repeat_idx=0, latencies=[0.1])
            writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
            writer.writeheader()
            append_row(writer, row, output, groups, group, publication=context,
                        telemetry_evidence=evidence, request_record=request)
        capture = calls.get("capture_path")
        case.pcap.write_bytes(capture.read_bytes() if capture else b"fixture-captured-packets")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(orchestrator, "start_service_session", start)
    monkeypatch.setattr(orchestrator, "stop_service_session", stop)
    monkeypatch.setattr(orchestrator, "record_case_conditions", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator, "resolve_packet_latency_runtime", lambda **k: PacketLatencyRuntime(
        mode="local", tcpdump_cmd=["fixture-capture"], parse_cmd=calls.get("parse_cmd", ["fixture-parser"])))
    monkeypatch.setattr(orchestrator.subprocess, "Popen", lambda command, **kwargs:
                        Capture() if command == ["fixture-capture"] else real_popen(command, **kwargs))
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


@pytest.mark.skipif(shutil.which("tshark") is None, reason="offline packet integration requires tshark")
@pytest.mark.parametrize("failure_stage", [None, "parser", "merge"])
def test_real_workers_produce_the_same_result_on_normal_and_resume_paths(measured_case, monkeypatch, failure_stage):
    state, case, calls, arguments = measured_case
    calls.update(group="case", capture_path=Path(__file__).parents[2] / "fixtures/packet/single.pcap",
                 parse_cmd=[*module_command("acprof.packet.sniff_parse_pcap"), str(case.pcap), str(SERVER_PORT)])
    run = orchestrator.host_command.run_command
    fault = failure_stage
    workers = []

    def execute(command, **kwargs):
        nonlocal fault
        if "acprof.packet.sniff_parse_pcap" in command or "acprof.packet.merge_packet_latency" in command:
            stage = "parser" if "acprof.packet.sniff_parse_pcap" in command else "merge"
            workers.append(stage)
            assert kwargs["kill_process_group"] is True
            assert kwargs["timeout"] > 0
            if stage == fault:
                fault = None
                return SimpleNamespace(returncode=7, stdout="", stderr="injected worker failure")
        return run(command, **kwargs)

    monkeypatch.setattr(orchestrator.host_command, "run_command", execute)
    key = str(case.csv.relative_to(state.directory))
    if failure_stage:
        with pytest.raises(RuntimeError, match="injected worker failure"):
            orchestrator.run_single_case(**arguments)
        record = state.data["cases"][key]
        assert record["measurement_status"] == "complete"
        assert record["processing_status"] == "failed"
        assert record["cleanup_status"] == "complete"
    else:
        assert orchestrator.run_single_case(**arguments) == str(case.csv)
    attempt = state.case_attempt_id(str(case.csv))
    raw = case.attempt_directory("measurements", attempt)
    evidence = {path: path.read_bytes() for path in raw.rglob("*") if path.is_file()}
    assert evidence
    if failure_stage:
        # Recovery must restore the verified snapshot even after a partial projection write.
        case.csv.write_text("partially merged projection\n")
        state.close()
        resumed = run_state.RunState(state.directory, state.data["options"], resume=True, project_dir=".")
        try:
            assert resumed.prepare_case("unused.csv", 1, 4, "off") is None
            assert orchestrator.run_single_case(**{**arguments, "run_state": resumed}) == str(case.csv)
            resumed.finish_case(str(case.csv), 1, 4, "off")
            assert resumed.case_attempt_id(str(case.csv)) == attempt
            assert resumed.data["cases"][key]["processing_attempts"] == 2
        finally:
            resumed.close()
    else:
        state.finish_case(str(case.csv), 1, 4, "off")
    assert {path: path.read_bytes() for path in evidence} == evidence
    assert calls["client"] == calls["container"] == calls["cleanup"] == 1
    assert workers[-2:] == ["parser", "merge"]
    with case.csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["latency_s"] == "0.002000"
    assert rows[0]["latency_app_s"] == "0.100000"
    assert rows[0]["packet_total_wire_bytes_per_request"] == "546.000000"
    assert rows[0]["packet_status"] == "ok"
    assert rows[0]["status"] == "ok"
    payload = json.loads(case.latency.read_text())
    assert payload["schema_version"] == 2
    assert payload["requests"]["case:4"]["request_id"] == "case:0"
    assert json.loads(case.retained_requests.read_text())["latency_packet_s"] == pytest.approx([0.002], abs=1e-6)
    assert not list(state.layout.path("interrupted_cases").rglob("*.csv"))


@pytest.mark.skipif(shutil.which("tshark") is None, reason="offline packet integration requires tshark")
@pytest.mark.parametrize(("fixture", "message"), [
    ("missing-response", "did not find matching"), ("invalid", "pcap parsing failed"),
])
def test_failed_or_empty_real_parser_keeps_measurements_for_recovery(measured_case, tmp_path, fixture, message):
    from tests.support.pcap_fixture_builder import write_capture

    state, case, calls, arguments = measured_case
    capture = Path(__file__).parents[2] / f"fixtures/packet/{fixture}.pcap"
    if fixture == "missing-response":
        capture = tmp_path / "no-completed-requests.pcap"
        write_capture(capture, missing_response=True)
    calls.update(group="case", capture_path=capture,
                 parse_cmd=[*module_command("acprof.packet.sniff_parse_pcap"), str(case.pcap), str(SERVER_PORT)])
    with pytest.raises(RuntimeError, match=message):
        orchestrator.run_single_case(**arguments)
    record = state.data["cases"][str(case.csv.relative_to(state.directory))]
    assert record["measurement_status"] == "complete"
    assert record["processing_status"] == "failed"
    assert record["cleanup_status"] == "complete"
    assert not case.latency.exists()
    assert not Path(str(case.csv) + ".merged").exists()
    assert calls["client"] == calls["container"] == calls["cleanup"] == 1


@pytest.mark.parametrize("payload", [
    '{"case:4":0.002}', '{"schema_version":2,"requests":[]}',
    '{"schema_version":2,"requests":{"case:4":{"latency_s":NaN}}}',
])
def test_malformed_parser_output_stops_before_derivative_writes(measured_case, monkeypatch, payload):
    state, case, calls, arguments = measured_case
    client = orchestrator.host_command.run_command
    case.latency.write_text("previous derived evidence\n")

    def execute(command, **kwargs):
        if command == ["fixture-parser"]:
            return SimpleNamespace(returncode=0, stdout=payload, stderr="")
        assert "acprof.packet.merge_packet_latency" not in command
        return client(command, **kwargs)

    monkeypatch.setattr(orchestrator.host_command, "run_command", execute)
    with pytest.raises(RuntimeError, match="invalid JSON or packet schema"):
        orchestrator.run_single_case(**arguments)
    assert case.latency.read_text() == "previous derived evidence\n"
    snapshot = case.attempt_directory("measurements", state.case_attempt_id(str(case.csv))) / "measurement.csv"
    assert case.csv.read_bytes() == snapshot.read_bytes()
    assert calls["client"] == calls["container"] == calls["cleanup"] == 1


def test_zero_exit_without_new_merge_output_never_reuses_a_stale_file(measured_case, monkeypatch):
    state, case, _calls, arguments = measured_case
    stale = Path(str(case.csv) + ".merged")
    stale.write_bytes(b"stale result from an earlier attempt\n")
    client = orchestrator.host_command.run_command

    def execute(command, **kwargs):
        if command == ["fixture-parser"]:
            return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
                "schema_version": 2, "requests": {"case:4": {"latency_s": 0.002}},
            }))
        if "acprof.packet.merge_packet_latency" in command:
            return SimpleNamespace(returncode=0, stderr="", stdout="")
        return client(command, **kwargs)

    monkeypatch.setattr(orchestrator.host_command, "run_command", execute)
    with pytest.raises(RuntimeError, match="merge did not produce"):
        orchestrator.run_single_case(**arguments)
    snapshot = case.attempt_directory("measurements", state.case_attempt_id(str(case.csv))) / "measurement.csv"
    assert case.csv.read_bytes() == snapshot.read_bytes()
    assert stale.read_bytes() == b"stale result from an earlier attempt\n"
    assert not list(case.csv.parent.glob(".packet-merge-*"))


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
                return SimpleNamespace(returncode=0, stdout='{"schema_version":2,"requests":{"request":{}}}')
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
    assert state.processing_checkpoint(str(case.csv), 1, 4, "off") is not None
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



def test_optional_parser_failure_retains_app_and_journal(tmp_path, monkeypatch):
    from acprof.host.measurement.capture import PacketLatencyError
    from acprof.results.csv import read_result_csv
    from acprof.results.layout import case_sidecar
    from acprof.results.profile import metric_observation
    path = tmp_path / "case.csv"
    data = dict.fromkeys(CSV_FIELDS, "nan")
    data.update(cpu_cores=1, mem_cap_gb=4, gpu_mode="off", runtime_kind="process", input_scale=2, warmup=0,
                repeat_idx=0, repeat_in_window=1, window_id="fixture", latency_app_s=.1, status="ok", error="",
                packet_status="pending", resource_metric_quality=json.dumps({"schema_version": 1, "metrics": {},
                    "collectors": {"client_http": {"status": "available"}, "packet_latency": {"status": "available"}}}))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, CSV_FIELDS)
        writer.writeheader()
        writer.writerow(data)
    journal = case_sidecar(path, "requests")
    raw = b'{"schema_version":1,"sniff_group_id":"fixture","latency_app_s":[0.1]}\n'
    journal.write_bytes(raw)
    pcap = tmp_path / "loopback.pcap"
    pcap.write_bytes(b"invalid capture fixture")
    def fail(command):
        assert command == ["fixture-parser"]
        raise PacketLatencyError("fixture parse failed")
    monkeypatch.setattr(orchestrator, "parse_packet_output", fail)
    orchestrator._finalize_case(True, SimpleNamespace(parse_cmd=["fixture-parser"]), False, 0, "", str(pcap),
                                str(tmp_path / "latency.json"), str(path), False, "best_effort", "off")
    _, rows = read_result_csv(path)
    assert rows[0]["packet_status"] == "unavailable"
    assert metric_observation(rows[0], "latency_s")["reason"] == "fixture parse failed"
    assert rows[0]["latency_s"] == "nan"
    assert float(rows[0]["latency_app_s"]) == .1
    assert journal.read_bytes() == raw


def test_resume_keeps_good_segments_and_uses_each_service_port(tmp_path, monkeypatch):
    from acprof.host.measurement import packet_segments
    from acprof.host.measurement.capture import CaptureSegment, PacketLatencyError
    segments = [CaptureSegment(tmp_path / name, port) for name, port in (("bad", 8141), ("good", 8161))]
    commands = []
    def parse(command, *, stage):
        commands.append(command[-2:])
        if command[-1] == "8141":
            raise PacketLatencyError("fixture packet access denied")
        return {"schema_version": 2, "requests": {"old:1": {"request_id": "old:0", "latency_s": .3}}}
    monkeypatch.setattr(packet_segments, "parse_packet_output", parse)
    current = {"schema_version": 2, "requests": {"new:4": {"request_id": "new:0", "latency_s": .2}}}
    failures = []
    combined = packet_segments.merge_capture_segments(current, segments, 9000, errors=failures)
    assert commands == [[str(segment.path), str(segment.server_port)] for segment in segments]
    assert set(combined["requests"]) == {"old:1", "new:4"}
    assert len(failures) == 1
    assert "access denied" in failures[0]
    with pytest.raises(PacketLatencyError, match="duplicate request identity"):
        packet_segments.merge_capture_segments(current, [segments[1], segments[1]], 9000, errors=[])


@pytest.mark.parametrize("failed_directory", ["raw", "destination", "source"])
def test_finish_case_retries_request_directory_sync(measured_case, monkeypatch, failed_directory):
    import errno
    import os

    state, case, _, _ = measured_case
    _write_processed_case(case.csv)
    source, destination = case.requests, case.retained_requests
    requests = b'{"request_id":"one","latency_app_s":[0.125]}\r\n'
    source.write_bytes(requests)
    assert not destination.parent.exists()
    # Fail raw only after raw/requests has been created, or a move directory after rename.
    visible, directory = {
        "raw": (destination.parent, destination.parent.parent),
        "destination": (destination, destination.parent),
        "source": (destination, source.parent),
    }[failed_directory]
    head, state_file = state.layout.result_csv, state.layout.path("run_state.json")
    original = {p: p.read_bytes() for p in (head, state_file)}
    fsync = os.fsync

    def reject_sync(fd):
        if visible.exists() and os.path.samestat(os.fstat(fd), directory.stat()):
            raise OSError(errno.EIO, "request directory fsync failed")
        return fsync(fd)

    with monkeypatch.context() as fault:
        fault.setattr(os, "fsync", reject_sync)
        for _ in range(2):
            with pytest.raises(OSError, match="request directory fsync failed") as caught:
                state.finish_case(str(case.csv), 1, 4, "off")
            assert caught.value.errno == errno.EIO
            assert {p: p.read_bytes() for p in original} == original
            kept, absent = (source, destination) if failed_directory == "raw" else (destination, source)
            assert kept.is_file()
            assert kept.read_bytes() == requests
            assert not absent.exists()

    state.finish_case(str(case.csv), 1, 4, "off")
    assert destination.read_bytes() == requests
    assert not source.exists()
    assert json.loads(head.read_bytes())["row_count"] == 1
    record = json.loads(state_file.read_bytes())["cases"][str(case.csv.relative_to(state.directory))]
    assert record["status"] == "complete"
    assert record["row_count"] == 1


def test_case_completion_publishes_the_validated_snapshot(measured_case, monkeypatch):
    from hashlib import sha256

    from acprof.result_store.writer import ResultWriter
    from acprof.results.csv import read_result_csv

    state, case, _, _ = measured_case
    original = _write_processed_case(case.csv)
    checksum = sha256(original).hexdigest()
    replacement = original.replace(b"0.125000", b"0.250000")
    update = ResultWriter.update_rows

    def replace_work_csv(writer, rows, **options):
        result = update(writer, rows, **options)
        case.csv.write_bytes(replacement)
        return result

    monkeypatch.setattr(ResultWriter, "update_rows", replace_work_csv)
    state.finish_case(str(case.csv), 1, 4, "off")
    key = str(case.csv.relative_to(state.directory))
    record = json.loads(state.layout.path("run_state.json").read_bytes())["cases"][key]
    archive = case.completed_result(checksum)
    assert case.csv.read_bytes() == replacement
    assert read_result_csv(state.layout.result_csv)[1][0]["latency_app_s"] == "0.125000"
    assert record["sha256"] == checksum
    assert record["case_result"] == {"path": archive.relative_to(state.directory).as_posix(), "sha256": checksum}
    assert archive.read_bytes() == original
    assert record["status"] == "complete"
    assert record["row_count"] == 1
    assert record["error_rows"] == 0

    from acprof.host.measurement.window_recovery import repair_result_projections
    repair_result_projections(state.layout, state.data)
    assert case.csv.read_bytes() == original


def _write_processed_case(path):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, CSV_FIELDS, restval="nan")
        writer.writeheader()
        writer.writerow({"cpu_cores": "1", "mem_cap_gb": "4", "gpu_mode": "off", "input_scale": "64",
                         "warmup": "0", "repeat_idx": "0", "status": "ok", "error": "",
                         "environment_class": "native_linux", "latency_app_s": "0.125000"})
    return path.read_bytes()


@pytest.mark.parametrize("after_link", [False, True], ids=["new-ancestor", "linked-archive"])
def test_case_archive_directory_sync_failure_preserves_state(measured_case, monkeypatch, after_link):
    import errno
    import os
    from hashlib import sha256

    state, case, _, _ = measured_case
    original = _write_processed_case(case.csv)
    archive = case.completed_result(sha256(original).hexdigest())
    entry = archive if after_link else archive.parent
    failure = FileExistsError(errno.EEXIST, "archive parent sync") if after_link else OSError(errno.EIO, "ancestor sync")
    state_path = state.layout.path("run_state.json")
    authorities = {path: path.read_bytes() for path in (state.layout.result_csv, state_path)}
    fsync = os.fsync
    with monkeypatch.context() as fault:
        def reject_sync(fd):
            if entry.exists() and os.path.samestat(os.fstat(fd), entry.parent.stat()):
                if after_link:
                    # A second successful fsync cannot turn the first error into a competing claim.
                    fault.setattr(os, "fsync", fsync)
                raise failure
            return fsync(fd)

        fault.setattr(os, "fsync", reject_sync)
        for _ in range(1 if after_link else 2):
            with pytest.raises(OSError, match=failure.strerror) as caught:
                state.finish_case(str(case.csv), 1, 4, "off")
            assert caught.value is failure
            assert {path: path.read_bytes() for path in authorities} == authorities
        if after_link:
            assert archive.read_bytes() == original
    state.finish_case(str(case.csv), 1, 4, "off")
    assert archive.read_bytes() == original
    key = str(case.csv.relative_to(state.directory))
    record = json.loads(state_path.read_bytes())["cases"][key]
    assert record["status"] == "complete"
    assert record["case_result"]["sha256"] == archive.stem


def test_case_archive_reports_non_directory_parent(measured_case):
    import errno
    from hashlib import sha256

    state, case, _, _ = measured_case
    archive = case.completed_result(sha256(_write_processed_case(case.csv)).hexdigest())
    archive.parent.parent.mkdir(parents=True, exist_ok=True)
    archive.parent.write_bytes(b"not a directory")
    with pytest.raises(FileExistsError) as caught:
        state.finish_case(str(case.csv), 1, 4, "off")
    assert caught.value.errno == errno.EEXIST
    assert Path(caught.value.filename) == archive.parent
    assert archive.parent.read_bytes() == b"not a directory"


def test_invalid_case_csv_does_not_create_archive_directory(measured_case):
    from acprof.results.csv import ResultValidationError

    state, case, _, _ = measured_case
    case.csv.write_bytes(b"status\nok\n")
    with pytest.raises(ResultValidationError, match="missing identity columns"):
        state.finish_case(str(case.csv), 1, 4, "off")
    assert not state.layout.contained("raw/case_results").exists()


@pytest.mark.parametrize("appears_at_publication", [False, True], ids=["existing", "racing"])
def test_corrupt_case_archive_preserves_root_and_state(measured_case, monkeypatch, appears_at_publication):
    import os
    from hashlib import sha256

    state, case, _, _ = measured_case
    archive = case.completed_result(sha256(_write_processed_case(case.csv)).hexdigest())
    archive.parent.mkdir(parents=True, exist_ok=True)
    damage = b"another publisher's conflicting content"
    if appears_at_publication:
        # Introduce a competing file after any existence check, at the OS publication boundary.
        def race(operation):
            def publish(src, dest, *args, **kwargs):
                if Path(dest) == archive:
                    archive.write_bytes(damage)
                return operation(src, dest, *args, **kwargs)
            return publish
        monkeypatch.setattr(os, "link", race(os.link))
        monkeypatch.setattr(os, "replace", race(os.replace))
    else:
        archive.write_bytes(damage)
    artifacts = (state.layout.result_csv, state.layout.path("run_state.json"))
    before = {path: path.read_bytes() for path in artifacts}
    with pytest.raises(run_state.RunStateError, match="Immutable case result changed"):
        state.finish_case(str(case.csv), 1, 4, "off")
    assert archive.read_bytes() == damage
    assert {path: path.read_bytes() for path in artifacts} == before


def test_case_completion_retry_reuses_the_archive_after_root_publication_fails(measured_case, monkeypatch):
    from hashlib import sha256

    from acprof.result_store.writer import ResultWriter

    state, case, _, _ = measured_case
    original = _write_processed_case(case.csv)
    digest = sha256(original).hexdigest()
    archive = case.completed_result(digest)
    state_file = state.layout.path("run_state.json")
    before = state_file.read_bytes()

    def interrupt(*args, **kwargs):
        assert archive.read_bytes() == original
        raise OSError("interrupted root publication")

    with monkeypatch.context() as fault:
        fault.setattr(ResultWriter, "update_rows", interrupt)
        with pytest.raises(OSError, match="interrupted root publication"):
            state.finish_case(str(case.csv), 1, 4, "off")
    assert state_file.read_bytes() == before
    archive_identity = archive.stat()
    state.finish_case(str(case.csv), 1, 4, "off")
    record = json.loads(state_file.read_bytes())["cases"][str(case.csv.relative_to(state.directory))]
    assert record["sha256"] == digest
    assert archive.read_bytes() == original
    assert archive.stat().st_ino == archive_identity.st_ino
    assert archive.stat().st_mtime_ns == archive_identity.st_mtime_ns


def test_legacy_flat_case_completion_reads_work_csv_once(measured_case, tmp_path, monkeypatch):
    from hashlib import sha256

    state, _, _, _ = measured_case
    state.close()
    directory = tmp_path / "flat"
    directory.mkdir()
    flat_data = {**state.data, "layout_version": 1, "cases": {}}
    (directory / "run_state.json").write_text(json.dumps(flat_data))
    resumed = run_state.RunState(directory, flat_data["options"], resume=True, project_dir=".")
    try:
        case = resumed.layout.case("fixture/model", 1, 4, "off")
        original = _write_processed_case(case.csv)
        read_opens = []
        open_path = Path.open

        def count_read(path, mode="r", *args, **kwargs):
            if path == case.csv and "r" in mode:
                read_opens.append(mode)
            return open_path(path, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", count_read)
        resumed.finish_case(str(case.csv), 1, 4, "off")
        record = json.loads((directory / "run_state.json").read_bytes())["cases"][case.csv.name]
        assert len(read_opens) == 1
        assert record["sha256"] == sha256(original).hexdigest()
        assert "case_result" not in record
        assert not (directory / "raw").exists()
    finally:
        resumed.close()


def test_case_completion_rejects_another_intact_result_store(measured_case, tmp_path):
    from hashlib import sha256

    from acprof.results.csv import ResultValidationError
    from acprof.results.layout import ArtifactLayout

    state, case, _, _ = measured_case
    archive = case.completed_result(sha256(_write_processed_case(case.csv)).hexdigest())
    other = ArtifactLayout.for_new_run(tmp_path / "other")
    other.initialize()
    assert json.loads(other.result_csv.read_bytes())["run_id"] != state.run_id
    shutil.copytree(other.result_store_path(""), state.layout.result_store_path(""), dirs_exist_ok=True)
    shutil.copyfile(other.result_csv, state.layout.result_csv)
    artifacts = (state.layout.path("run_state.json"), state.layout.result_csv, case.csv)
    before = {path: path.read_bytes() for path in artifacts}
    with pytest.raises(ResultValidationError, match="result store belongs to another run"):
        state.finish_case(str(case.csv), 1, 4, "off")
    assert not archive.parent.parent.exists()
    assert {path: path.read_bytes() for path in artifacts} == before
