"""Replay frozen runtime artifacts with local HTTP and mocked service ownership."""
import hashlib
import json
import os
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.cli import load as load_cli
from acprof.host.detect import TaskInfo
from acprof.host.images.runtime import ImageInfo
from acprof.host.runtimes.contracts import ProcessInfo
from acprof.host.runtimes.docker import DockerRuntime
from acprof.host.runtimes.process import ProcessRuntime
from acprof.process_identity import same_process
from tests.execution.operations.test_load_protocol import server
from tests.execution.process.test_process_service import service as service


def replay_environment(tmp_path, monkeypatch, kind="process", *, cpus="auto", mems="auto", gpu="off"):
    source, output = tmp_path / "source", tmp_path / "load"
    source.mkdir()
    payload = {"text": "frozen prompt", "params": {"seed": 7}}
    plan = source / "input_scale_plan.json"
    plan.write_text(json.dumps({"schema_version": 2, "entries": [{"input_scale": 2, "payload": payload}]}))
    gpu_device = {"uuid": "GPU-source", "cuda_visible_devices": "1"} if gpu == "on" else {}
    process = ProcessInfo("/fixture/venv/bin/python", "/fixture/code", "/fixture/model", "fixed-environment", {
        "logs_dir": str(source / "logs"), "gpu_device": gpu_device,
    })
    artifact = process if kind == "process" else ImageInfo("sha256:" + "a" * 64)
    task = TaskInfo("fixture/model", "fill-mask", "nlp", "transformers_pipeline", "transformers", "a" * 40, "fixture")
    state = {
        "status": "complete", "run_id": "source-run",
        "options": {"cpus": cpus, "mems": mems, "gpus": gpu, "gpu_device": "GPU-source",
                    "request_timeout_seconds": 1.25, "sniff_iface": "lo", "cpuset_cpus": "",
                    "measurement_environment": {"OMP_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": "0,1"}},
        "runtime": {"kind": kind, "task": asdict(task), "image": asdict(artifact)},
        "artifacts": {plan.name: hashlib.sha256(plan.read_bytes()).hexdigest()},
    }
    source_state = source / "fixture-state.json"
    source_state.write_text(json.dumps(state))
    originals = {p: p.read_bytes() for p in source.iterdir()}
    events = []

    @contextmanager
    def measurement_lock():
        events.append("locked")
        try:
            yield
        finally:
            events.append("unlocked")

    session = SimpleNamespace(base_url="http://127.0.0.1:1", host_port=43123, gpu_device=gpu_device,
                              runtime_kind=kind)
    start, stop = Mock(return_value=session), Mock()
    restore = Mock(side_effect=lambda snapshot: ProcessInfo(**snapshot))
    monkeypatch.setattr("acprof.host.runtimes.environment.restore_process", restore)
    monkeypatch.setattr("acprof.host.images.runtime.require_image_identity", Mock())
    runtime = ProcessRuntime if kind == "process" else DockerRuntime
    monkeypatch.setattr(runtime, "start", start)
    monkeypatch.setattr(runtime, "stop", stop)
    monkeypatch.setattr(load_cli, "MeasurementLock", measurement_lock)
    monkeypatch.setattr(load_cli, "bootstrap_project_env", lambda _root: None)
    monkeypatch.setattr(load_cli, "resolve_single_result_source", lambda _source: source)
    monkeypatch.setattr(load_cli, "load_run_state", lambda _source: deepcopy(state))
    monkeypatch.setattr(load_cli, "host_identity", lambda _root: {"fixture": True})
    observe = Mock()
    monkeypatch.setattr(load_cli, "record_case_conditions", observe)
    argv = [str(source), "--gpu", gpu, "--output-dir", str(output), "--requests", "2", "--warmup", "1"]
    return SimpleNamespace(state=state, source=source, output=output, payload=payload, originals=originals,
                           events=events, session=session, start=start, stop=stop, restore=restore,
                           observe=observe, argv=argv)


@pytest.mark.parametrize("kind", ["process", "docker"])
def test_replay_restores_runtime_and_runs_frozen_payload_without_changing_source(tmp_path, monkeypatch, kind):
    replay = replay_environment(tmp_path, monkeypatch, kind, cpus="auto" if kind == "process" else "2,4", mems="3,6")
    monkeypatch.setenv("OMP_NUM_THREADS", "caller")
    with server() as (url, ports, _peak):
        replay.session.base_url = url
        assert load_cli.main(replay.argv) == 0
    report = json.loads((replay.output / "load.json").read_text())
    assert report["successful"]
    assert report["status"] == "complete"
    assert report["result"]["counts"]["succeeded"] == 2
    assert len(ports) == 3  # One warmup request, then two measured requests.
    assert report["identity"]["runtime_kind"] == kind
    assert report["identity"]["cpu_cores"] == (None if kind == "process" else 4)
    assert report["identity"]["mem_cap_gb"] == 6
    started = replay.start.call_args.kwargs
    assert started["request_timeout_seconds"] == 1.25
    assert started["gpu"] == "off"
    if kind == "process":
        assert report["identity"]["runtime_id"] == "fixed-environment"
        assert "image_id" not in report["identity"]
        assert started["image_info"].runtime_environment["logs_dir"] == str(replay.output / "logs")
        replay.restore.assert_called_once_with(replay.state["runtime"]["image"])
    else:
        assert report["identity"]["image_id"] == "sha256:" + "a" * 64
    replay.stop.assert_called_once_with(replay.session, log_prefix="[load]")
    assert replay.events == ["locked", "unlocked"]
    assert os.environ["OMP_NUM_THREADS"] == "caller"
    assert all(path.read_bytes() == value for path, value in replay.originals.items())


def test_process_replay_accepts_inherited_memory_and_selected_numeric_cpu(tmp_path, monkeypatch):
    replay = replay_environment(tmp_path, monkeypatch, cpus="2,4")
    run = Mock(return_value={"successful": True, "success_rate": 1})
    monkeypatch.setattr("acprof.cli.load.run_load", run)
    assert load_cli.main([*replay.argv, "--cpu", "2", "--warmup", "0"]) == 0
    assert replay.start.call_args.kwargs["cpu"] == 2
    assert replay.start.call_args.kwargs["mem"] is None
    assert run.call_args.args[1] == replay.payload


@pytest.mark.parametrize(("cpus", "mems", "extra"), [
    ("auto", "auto", ["--cpu", "1"]), ("auto", "auto", ["--mem", "1"]),
    ("2,4", "3,6", ["--cpu", "3"]), ("2,4", "3,6", ["--mem", "4"]),
])
def test_process_replay_rejects_resources_outside_source_before_output(tmp_path, monkeypatch, capsys, cpus, mems, extra):
    replay = replay_environment(tmp_path, monkeypatch, cpus=cpus, mems=mems)
    with pytest.raises(SystemExit) as caught:
        load_cli.main([*replay.argv, *extra])
    assert caught.value.code == 2
    assert "resources must be present" in capsys.readouterr().err
    assert not replay.output.exists()
    replay.restore.assert_not_called()
    replay.start.assert_not_called()


@pytest.mark.parametrize("stage", ["restore", "start", "observe", "warmup", "load", "stop", "cancel"])
def test_process_replay_failure_keeps_report_and_releases_owned_session(tmp_path, monkeypatch, stage):
    replay = replay_environment(tmp_path, monkeypatch)
    partial = {"successful": False, "success_rate": 0, "requests": [{"status": "cancelled"}]}
    error = KeyboardInterrupt("fixture cancel") if stage == "cancel" else RuntimeError("fixture " + stage)
    run = Mock(return_value={"successful": True, "success_rate": 1})
    if stage in {"restore", "start", "observe", "stop"}:
        getattr(replay, stage).side_effect = error
    elif stage == "warmup":
        run.return_value = {"successful": False, "success_rate": 0}
    else:
        error.load_report = partial
        run.side_effect = [{"successful": True, "success_rate": 1}, error]
    monkeypatch.setattr("acprof.cli.load.run_load", run)
    with pytest.raises(type(error), match="warmup failed" if stage == "warmup" else str(error)):
        load_cli.main(replay.argv)
    report = json.loads((replay.output / "load.json").read_text())
    assert report["successful"] is False
    assert report["status"] == ("cancelled" if stage == "cancel" else "failed")
    if stage in {"load", "cancel"}:
        assert report["result"] == partial
    assert replay.stop.call_count == (0 if stage in {"restore", "start"} else 1)
    assert replay.events == ["locked", "unlocked"]
    assert all(path.read_bytes() == value for path, value in replay.originals.items())


@pytest.mark.parametrize("kind", ["process", "docker"])
def test_replay_capture_and_validation_use_runtime_service_port(tmp_path, monkeypatch, kind):
    replay = replay_environment(tmp_path, monkeypatch, kind, cpus="1", mems="2")
    captured = []

    @contextmanager
    def capture(path, *, interface, port):
        captured.append((path, interface, port))
        yield

    monkeypatch.setattr(load_cli, "capture_packets", capture)
    validate = Mock(return_value={"successful_request_coverage": 2})
    monkeypatch.setattr(load_cli, "validate_packets", validate)
    monkeypatch.setattr("acprof.cli.load.run_load", Mock(return_value={"successful": True, "success_rate": 1}))
    assert load_cli.main([*replay.argv, "--connections", "reuse"]) == 0
    expected = 43123 if kind == "process" else 8002
    assert captured == [(replay.output / "load.pcap", "lo", expected)]
    assert validate.call_args.kwargs == {"port": expected}


@pytest.mark.parametrize("observed_uuid", ["GPU-source", "GPU-other"])
def test_process_replay_checks_recorded_cuda_device_without_nvidia_smi(tmp_path, monkeypatch, observed_uuid):
    replay = replay_environment(tmp_path, monkeypatch, gpu="on")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "caller")
    monkeypatch.setattr("acprof.host.execution_conditions.pin_gpu_device",
                        Mock(side_effect=AssertionError("Process must not require nvidia-smi")))

    def cuda_probe(selector):
        assert selector == "0"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
        return {"status": "available", "uuid": observed_uuid, "index": 0, "cuda_visible_devices": "1"}

    monkeypatch.setattr("acprof.host.runtimes.gpu.process_gpu_device", cuda_probe)
    monkeypatch.setattr("acprof.cli.load.run_load", Mock(return_value={"successful": True, "success_rate": 1}))
    if observed_uuid == "GPU-source":
        assert load_cli.main(replay.argv) == 0
        replay.start.assert_called_once()
    else:
        with pytest.raises(ValueError, match="GPU UUID differs"):
            load_cli.main(replay.argv)
        replay.start.assert_not_called()
        assert json.loads((replay.output / "load.json").read_text())["status"] == "failed"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "caller"


@pytest.mark.parametrize(("binding", "observed", "message"), [
    ({"uuid": "GPU-source"}, {}, "CUDA device binding is missing"),
    ({"uuid": "GPU-other", "cuda_visible_devices": "1"}, {}, "GPU UUID differs"),
    ({"uuid": "GPU-source", "cuda_visible_devices": "1"},
     {"status": "unavailable", "error": "fixture CUDA driver unavailable"}, "fixture CUDA driver unavailable"),
])
def test_process_gpu_failures_are_reported_before_start_and_restore_caller(tmp_path, monkeypatch, binding, observed, message):
    replay = replay_environment(tmp_path, monkeypatch, gpu="on")
    replay.state["runtime"]["image"]["runtime_environment"]["gpu_device"] = binding
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "caller")
    probe = Mock(return_value=observed)
    monkeypatch.setattr("acprof.host.runtimes.gpu.process_gpu_device", probe)
    with pytest.raises(ValueError, match=message):
        load_cli.main(replay.argv)
    replay.start.assert_not_called()
    replay.stop.assert_not_called()
    assert replay.events == ["locked", "unlocked"]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "caller"
    report = json.loads((replay.output / "load.json").read_text())
    assert message in report["error"]
    assert report["status"] == "failed"


def test_legacy_docker_source_without_kind_uses_recorded_image(tmp_path, monkeypatch):
    replay = replay_environment(tmp_path, monkeypatch, "docker", cpus="1", mems="2")
    replay.state["runtime"].pop("kind")
    monkeypatch.setattr("acprof.cli.load.run_load", Mock(return_value={"successful": True, "success_rate": 1}))
    assert load_cli.main(replay.argv) == 0
    assert replay.start.call_args.kwargs["image_info"].tag == "sha256:" + "a" * 64


@pytest.mark.parametrize("invalid", ["runtime", "plan", "status", "output"])
def test_replay_rejects_invalid_source_or_existing_output_without_launch(tmp_path, monkeypatch, invalid):
    replay = replay_environment(tmp_path, monkeypatch)
    if invalid == "runtime":
        replay.state["runtime"]["kind"] = "unknown"
    elif invalid == "plan":
        replay.state["artifacts"]["input_scale_plan.json"] = "0" * 64
    elif invalid == "status":
        replay.state["status"] = "running"
    else:
        replay.output.mkdir()
        (replay.output / "load.json").write_text("existing report")
    with pytest.raises(SystemExit) as caught:
        load_cli.main(replay.argv)
    assert caught.value.code == 2
    replay.start.assert_not_called()
    replay.restore.assert_not_called()
    if invalid == "output":
        assert (replay.output / "load.json").read_text() == "existing report"
    else:
        assert not replay.output.exists()


def test_packet_helpers_forward_dynamic_port_and_keep_request_coverage(tmp_path, monkeypatch):
    pcap = tmp_path / "load.pcap"
    pcap.write_bytes(b"fixture" * 8)
    capture = Mock(returncode=0)
    capture.poll.return_value = None
    launch = Mock(return_value=capture)
    monkeypatch.setattr(load_cli.subprocess, "Popen", launch)
    monkeypatch.setattr(load_cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr("acprof.host.measurement.capture.require_packet_latency_prerequisites", Mock())
    with load_cli.capture_packets(pcap, interface="lo", port=43123):
        pass
    assert launch.call_args.args[0][-3:] == ["tcp", "port", "43123"]
    capture.terminate.assert_called_once()
    capture.wait.assert_called_once_with(timeout=5)
    parsed = {"requests": {"first": {"request_id": "a", "tcp_stream": 1},
                           "second": {"request_id": "b", "tcp_stream": 1}}}
    parse = Mock(return_value=SimpleNamespace(stdout=json.dumps(parsed)))
    monkeypatch.setattr("acprof.cli.load.run_command", parse)
    result = {"successful": True, "protocol": {"connections": "reuse"},
              "requests": [{"request_id": name, "status": "ok"} for name in ("a", "b")]}
    validation = load_cli.validate_packets(pcap, result, port=43123)
    assert parse.call_args.args[0][-2:] == [str(pcap), "43123"]
    assert validation["successful_request_coverage"] == 2
    assert json.loads(pcap.with_suffix(".packets.json").read_text()) == parsed


def test_process_replay_owns_real_fixture_service_and_cleans_up(tmp_path, monkeypatch, service):
    real_start, real_stop = ProcessRuntime.start, ProcessRuntime.stop
    cleanup = ProcessRuntime().stop
    replay = replay_environment(tmp_path, monkeypatch)
    _task, info = service
    replay.state["runtime"]["image"] = asdict(info)
    sessions = []

    def start(runtime, **kwargs):
        started_session = real_start(runtime, **kwargs)
        sessions.append(started_session)
        return started_session

    monkeypatch.setattr(ProcessRuntime, "start", start)
    monkeypatch.setattr(ProcessRuntime, "stop", real_stop)
    try:
        assert load_cli.main(replay.argv) == 0
        assert len(sessions) == 1
        session = sessions[0]
        assert not same_process(session.process.identity)
        assert session.resource_plan["method"] == "inherited"
        report = json.loads((replay.output / "load.json").read_text())
        assert report["result"]["counts"]["succeeded"] == 2
        assert report["identity"]["cpu_cores"] is None
        assert report["identity"]["mem_cap_gb"] is None
        assert (replay.output / "logs").is_dir()
        assert not (replay.source / "logs").exists()
        assert all(row["workload_contract"]["fixture_payload"] == replay.payload
                   for row in report["result"]["requests"])
    finally:
        for session in sessions:
            if same_process(session.process.identity):
                cleanup(session)
