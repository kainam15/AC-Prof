"""Endpoint sweeps reuse single-point measurements with ordered, durable progress."""
import json
import os
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from acprof.cli import load as load_cli


@pytest.fixture(name="sweep")
def sweep_environment(tmp_path, monkeypatch):
    output = tmp_path / "sweep"
    state = SimpleNamespace(output=output, calls=[], events=[], locked=False, effects=[], hook=None, lock_error=None)

    @contextmanager
    def measurement_scope():
        assert not state.locked, "points must share one measurement lock"
        state.locked = True
        state.events.append("lock-enter")
        try:
            yield
        finally:
            state.locked = False
            state.events.append("lock-exit")
            if state.lock_error is not None:
                raise state.lock_error

    def run(url, payload, config, **kwargs):
        assert state.locked
        state.calls.append({"url": url, "payload": payload, "config": config, **kwargs})
        state.events.append("warmup" if kwargs["token"] == "warmup" else "measure")
        if state.hook is not None:
            state.hook()
        if state.effects:
            effect = state.effects.pop(0)
            if isinstance(effect, BaseException):
                raise effect
            return effect
        return {"successful": True, "success_rate": 1.0, "requests": []}

    monkeypatch.setattr("acprof.cli.streaming_load.run_load", run)
    monkeypatch.setattr("acprof.cli.streaming_load.MeasurementLock", measurement_scope)
    monkeypatch.setattr("acprof.cli.streaming_load.host_identity", lambda _root: {"hostname": "fixture"})
    state.argv = ["--base-url", "http://127.0.0.1:8999", "--protocol", "openai-chat", "--model", "fixture",
                  "--prompt", "hello", "--scenario", "concurrent", "--requests", "5", "--warmup", "1",
                  "--connections", "reuse", "--sweep-concurrency", "1,2", "--output-dir", str(output)]
    return state


def read_manifest(sweep):
    return json.loads((sweep.output / "sweep.json").read_text())


def child_reports(sweep, manifest):
    return [json.loads((sweep.output / point["report_path"]).read_text()) for point in manifest["points"]]


def test_cartesian_sweep_preserves_order_and_uses_one_lock(sweep):
    progress = []
    sweep.hook = lambda: progress.append([point["status"] for point in read_manifest(sweep)["points"]])
    assert load_cli.main([*sweep.argv, "--scenario", "arrival-rate", "--sweep-rate", "20,10", "--seed", "17"]) == 0
    assert sweep.events == ["lock-enter", *["warmup", "measure"] * 4, "lock-exit"]
    measured = [call for call in sweep.calls if call["token"] != "warmup"]
    assert [(call["config"].concurrency, call["config"].rate) for call in measured] == [(1, 20), (1, 10), (2, 20), (2, 10)]
    assert all(call["config"].seed == 17 for call in measured)
    assert all(call["include_capacity"] is False for call in sweep.calls if call["token"] == "warmup")
    manifest = read_manifest(sweep)
    assert manifest["kind"] == "stream_load_sweep"
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "complete"
    assert manifest["successful"] is True
    assert [point["point_id"] for point in manifest["points"]] == ["0001", "0002", "0003", "0004"]
    assert [point["load_config"] for point in manifest["points"]] == [asdict(call["config"]) for call in measured]
    reports = child_reports(sweep, manifest)
    assert len({report["run_id"] for report in reports}) == 4
    assert all(report["kind"] == "stream_load_experiment" and report["schema_version"] == 2 for report in reports)
    assert [point["run_id"] for point in manifest["points"]] == [report["run_id"] for report in reports]
    assert progress[0] == ["running", "not_run", "not_run", "not_run"]
    assert progress[2] == ["complete", "running", "not_run", "not_run"]
    assert not (sweep.output / "load.json").exists()


def test_normal_request_failures_do_not_skip_later_points(sweep):
    sweep.effects = [{"successful": False, "success_rate": 0.5, "requests": []},
                     {"successful": True, "success_rate": 1.0, "requests": []}]
    assert load_cli.main([*sweep.argv, "--warmup", "0"]) == 1
    manifest = read_manifest(sweep)
    assert manifest["status"] == "complete"
    assert manifest["successful"] is False
    assert [point["successful"] for point in manifest["points"]] == [False, True]
    assert [point["status"] for point in manifest["points"]] == ["complete", "complete"]
    assert len(sweep.calls) == 2


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_exception_preserves_finished_and_partial_points_and_leaves_rest_unrun(sweep, error_type):
    error = error_type("fixture measured interruption")
    partial = {"successful": False, "success_rate": 0, "requests": [{"status": "cancelled"}]}
    setattr(error, "load_report", partial)
    sweep.effects = [{"successful": True, "success_rate": 1, "requests": []}, error]
    with pytest.raises(error_type, match="measured interruption") as caught:
        load_cli.main([*sweep.argv, "--sweep-concurrency", "1,2,3", "--warmup", "0"])
    assert caught.value is error
    manifest = read_manifest(sweep)
    status = "cancelled" if error_type is KeyboardInterrupt else "failed"
    assert manifest["status"] == status
    assert manifest["successful"] is False
    assert [point["status"] for point in manifest["points"]] == ["complete", status, "not_run"]
    assert manifest["points"][2]["successful"] is None
    first = json.loads((sweep.output / manifest["points"][0]["report_path"]).read_text())
    second = json.loads((sweep.output / manifest["points"][1]["report_path"]).read_text())
    assert first["status"] == "complete"
    assert first["successful"] is True
    assert second["result"] == partial
    assert second["status"] == status
    assert not (sweep.output / "points/0003").exists()
    assert sweep.events[-1] == "lock-exit"


def test_failed_warmup_stops_the_sweep_without_measured_requests(sweep):
    warmup = {"successful": False, "success_rate": 0, "requests": [{"status": "error"}]}
    sweep.effects = [warmup]
    with pytest.raises(RuntimeError, match="warmup failed"):
        load_cli.main(sweep.argv)
    manifest = read_manifest(sweep)
    assert [point["status"] for point in manifest["points"]] == ["failed", "not_run"]
    point = json.loads((sweep.output / manifest["points"][0]["report_path"]).read_text())
    assert point["warmup"] == warmup
    assert "result" not in point
    assert len(sweep.calls) == 1


def test_dry_run_prints_all_validated_points_without_output_or_lock(sweep, capsys, monkeypatch):
    monkeypatch.setattr("acprof.cli.streaming_load.probe_service", lambda *_args, **_kwargs: pytest.fail("dry-run probed a service"))
    assert load_cli.main([*sweep.argv, "--dry-run", "--scenario", "arrival-rate", "--sweep-rate", "10,20",
                          "--service-probe", "vllm"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["kind"] == "stream_load_sweep_plan"
    assert plan["warmup_requests_per_point"] == 1
    assert [(point["concurrency"], point["rate"]) for point in plan["points"]] == [(1, 10), (1, 20), (2, 10), (2, 20)]
    assert not sweep.output.exists()
    assert sweep.calls == []
    assert sweep.events == []


@pytest.mark.parametrize(("arguments", "message"), [
    (["--sweep-concurrency", ""], "positive"),
    (["--sweep-concurrency", "1,,2"], "positive"),
    (["--sweep-concurrency", "1,1"], "duplicate"),
    (["--sweep-concurrency", "1,0"], "positive"),
    (["--sweep-concurrency", "1,2.5"], "integer"),
    (["--scenario", "serial"], "scenario"),
    (["--sweep-rate", "1,2"], "arrival-rate"),
    (["--scenario", "arrival-rate", "--sweep-rate", "1,nan"], "finite"),
    (["--scenario", "arrival-rate", "--sweep-rate", "1,inf"], "finite"),
    (["--scenario", "arrival-rate", "--sweep-rate", "1,0"], "positive"),
    (["--scenario", "arrival-rate", "--sweep-rate", "1,1.0"], "duplicate"),
    (["--scenario", "arrival-rate", "--sweep-rate", "1,1e-320"], "schedule"),
    (["--sweep-concurrency", "1,3", "--max-pending", "2"], "max_pending"),
    (["--sweep-concurrency", "1," + "9" * 350, "--connections", "close"], "max_pending"),
    (["--requests", "2"], "reuse"),
    (["--sweep-concurrency", ",".join(str(value) for value in range(1, 258))], "256"),
])
def test_invalid_point_is_rejected_before_any_output_or_request(sweep, capsys, arguments, message):
    with pytest.raises(SystemExit) as caught:
        load_cli.main([*sweep.argv, *arguments])
    assert caught.value.code == 2
    assert message in capsys.readouterr().err
    assert not sweep.output.exists()
    assert sweep.calls == []
    assert sweep.events == []


def test_nonempty_sweep_directory_is_preserved(sweep):
    sweep.output.mkdir()
    marker = sweep.output / "keep.txt"
    marker.write_text("existing evidence")
    with pytest.raises(SystemExit) as caught:
        load_cli.main(sweep.argv)
    assert caught.value.code == 2
    assert marker.read_text() == "existing evidence"
    assert list(sweep.output.iterdir()) == [marker]
    assert sweep.calls == []


def test_request_and_authorization_are_frozen_once_for_all_points(sweep, tmp_path, monkeypatch):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"model": "fixture", "messages": [{"role": "user", "content": "frozen"}]}))
    monkeypatch.setenv("ACPROF_TEST_SWEEP_KEY", "fixture-original-secret")
    argv = list(sweep.argv)
    index = argv.index("--prompt")
    del argv[index:index + 2]

    def change_inputs():
        request.write_text(json.dumps({"model": "changed", "messages": []}))
        monkeypatch.setenv("ACPROF_TEST_SWEEP_KEY", "fixture-replaced-secret")

    sweep.hook = change_inputs
    assert load_cli.main([*argv, "--request-json", str(request), "--api-key-env", "ACPROF_TEST_SWEEP_KEY"]) == 0
    assert all(call["payload"]["messages"][0]["content"] == "frozen" for call in sweep.calls)
    assert all(call["headers"] == {"Authorization": "Bearer fixture-original-secret"} for call in sweep.calls)
    reports = child_reports(sweep, read_manifest(sweep))
    assert len({report["identity"]["request_body_sha256"] for report in reports}) == 1
    assert all(report["identity"]["request_body"]["model"] == "fixture" for report in reports)
    assert all("fixture-original-secret" not in path.read_text() for path in sweep.output.rglob("*.json"))


def test_lock_cleanup_failure_marks_sweep_failed_after_complete_points(sweep):
    sweep.lock_error = RuntimeError("fixture lock cleanup failed")
    with pytest.raises(RuntimeError, match="lock cleanup failed"):
        load_cli.main(sweep.argv)
    manifest = read_manifest(sweep)
    assert manifest["status"] == "failed"
    assert manifest["successful"] is False
    assert all(point["status"] == "complete" for point in manifest["points"])


def test_replay_rejects_endpoint_sweep_before_runtime_preparation(tmp_path, capsys):
    with pytest.raises(SystemExit) as caught:
        load_cli.main([str(tmp_path / "source"), "--gpu", "off", "--sweep-concurrency", "1,2",
                       "--output-dir", str(tmp_path / "output")])
    assert caught.value.code == 2
    assert "require --base-url" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


def test_rate_only_sweep_keeps_the_requested_scalar_concurrency(sweep):
    argv = list(sweep.argv)
    index = argv.index("--sweep-concurrency")
    del argv[index:index + 2]
    assert load_cli.main([*argv, "--scenario", "arrival-rate", "--sweep-rate", "20,10",
                          "--concurrency", "3", "--warmup", "0"]) == 0
    assert [(call["config"].concurrency, call["config"].rate) for call in sweep.calls] == [(3, 20), (3, 10)]


def test_dry_run_requires_an_explicit_sweep(sweep, capsys):
    argv = list(sweep.argv)
    index = argv.index("--sweep-concurrency")
    del argv[index:index + 2]
    with pytest.raises(SystemExit) as caught:
        load_cli.main([*argv, "--dry-run"])
    assert caught.value.code == 2
    assert "--dry-run requires" in capsys.readouterr().err
    assert not sweep.output.exists()
    assert sweep.calls == []


@pytest.mark.parametrize("cancelled", [False, True])
def test_manifest_publication_failure_stops_new_points_and_preserves_primary(sweep, monkeypatch, cancelled):
    original_replace = os.replace
    writes = []
    publication = PermissionError("fixture sweep publication denied")
    cancellation = KeyboardInterrupt("fixture point cancelled")
    if cancelled:
        setattr(cancellation, "load_report", {"successful": False, "success_rate": 0, "requests": []})
        sweep.effects = [{"successful": True, "success_rate": 1, "requests": []}, cancellation]

    def replace(source, destination):
        if Path(destination) == sweep.output / "sweep.json":
            writes.append(destination)
            # Initial, first start, first complete, second start, cancelled terminal.
            if len(writes) >= (5 if cancelled else 1):
                raise publication
        original_replace(source, destination)

    monkeypatch.setattr("acprof.artifacts.os.replace", replace)
    expected = cancellation if cancelled else publication
    with pytest.raises(type(expected)) as caught:
        load_cli.main([*sweep.argv, "--warmup", "0", "--sweep-concurrency", "1,2,3"])
    assert caught.value is expected
    if cancelled:
        assert any("sweep publication denied" in note for note in cancellation.__notes__)
        assert len(sweep.calls) == 2
        assert read_manifest(sweep)["status"] == "running"
        second = json.loads((sweep.output / "points/0002/load.json").read_text())
        assert second["status"] == "cancelled"
        assert not (sweep.output / "points/0003").exists()
    else:
        assert sweep.calls == []
        assert sweep.events == []
        assert not (sweep.output / "points").exists()


@pytest.mark.parametrize("protocol", ["openai-chat", "openai-completions"])
def test_real_local_http_sweep_retains_readable_single_point_reports(tmp_path, monkeypatch, protocol):
    from contextlib import nullcontext

    from acprof.cli.main import main
    from acprof.plotting.load_report import read_load_report
    from tests.execution.operations.test_streaming_load import streaming_server

    endpoint = "/v1/chat/completions" if protocol == "openai-chat" else "/v1/completions"
    output = tmp_path / "sweep"
    monkeypatch.setattr("acprof.cli.streaming_load.MeasurementLock", nullcontext)
    monkeypatch.setattr("acprof.cli.streaming_load.host_identity", lambda _root: {"hostname": "fixture"})
    with streaming_server(endpoint=endpoint) as (url, ports, paths, bodies, _first_sent):
        assert main(["load", "--base-url", url, "--protocol", protocol, "--model", "fixture", "--prompt", "hello",
                     "--scenario", "concurrent", "--sweep-concurrency", "1,2", "--requests", "3",
                     "--warmup", "1", "--max-run-output-events", "6", "--output-dir", str(output)]) == 0
    manifest = json.loads((output / "sweep.json").read_text())
    assert manifest["successful"] is True
    assert len(ports) == 8
    assert paths == [endpoint] * 8
    assert all(body["model"] == "fixture" for body in bodies)
    for point in manifest["points"]:
        source = output / point["report_path"]
        report = json.loads(source.read_text())
        assert report["warmup"]["counts"]["succeeded"] == 1
        assert report["result"]["counts"]["succeeded"] == 3
        assert [row["output_tokens"] for row in report["result"]["requests"]] == [3, 3, 3]
        read_load_report(source)


def test_progress_publication_failure_retains_finished_point_and_stops_next(sweep, monkeypatch):
    original_replace = os.replace
    writes = []
    publication = PermissionError("fixture progress publication denied")

    def replace(source, destination):
        if Path(destination) == sweep.output / "sweep.json":
            writes.append(destination)
            if len(writes) == 3:  # Initial, first start, first complete.
                raise publication
        original_replace(source, destination)

    monkeypatch.setattr("acprof.artifacts.os.replace", replace)
    with pytest.raises(PermissionError, match="progress publication denied") as caught:
        load_cli.main([*sweep.argv, "--warmup", "0"])
    assert caught.value is publication
    manifest = read_manifest(sweep)
    assert manifest["status"] == "failed"
    assert [point["status"] for point in manifest["points"]] == ["complete", "not_run"]
    first = json.loads((sweep.output / "points/0001/load.json").read_text())
    assert first["successful"] is True
    assert manifest["points"][0]["run_id"] == first["run_id"]
    assert len(sweep.calls) == 1
    assert not (sweep.output / "points/0002").exists()


def test_unavailable_measurement_lock_leaves_all_points_unrun(sweep, monkeypatch):
    def unavailable():
        raise RuntimeError("fixture measurement lock is busy")

    monkeypatch.setattr("acprof.cli.streaming_load.MeasurementLock", unavailable)
    with pytest.raises(RuntimeError, match="measurement lock is busy"):
        load_cli.main(sweep.argv)
    manifest = read_manifest(sweep)
    assert manifest["status"] == "failed"
    assert all(point["status"] == "not_run" and point["successful"] is None for point in manifest["points"])
    assert sweep.calls == []
    assert not (sweep.output / "points").exists()


def test_service_probe_precedes_each_warmup_and_slo_reaches_each_point(sweep, monkeypatch):
    probes = []

    def probe(url, declared, mode, **kwargs):
        assert sweep.locked
        probes.append((url, declared, mode, kwargs))
        sweep.events.append("probe")
        return {"fixture_observation": len(probes)}

    monkeypatch.setattr("acprof.cli.streaming_load.probe_service", probe)
    assert load_cli.main([*sweep.argv, "--service-probe", "vllm", "--slo-ttft-s", "0.5",
                          "--slo-tpot-s", "0.05", "--slo-e2e-s", "3"]) == 0
    assert sweep.events == ["lock-enter", "probe", "warmup", "measure", "probe", "warmup", "measure", "lock-exit"]
    assert [probe[2] for probe in probes] == ["vllm", "vllm"]
    measured = [call for call in sweep.calls if call["token"] != "warmup"]
    assert [call["slo_thresholds"] for call in measured] == [
        {"ttft_s": 0.5, "tpot_s": 0.05, "e2e_s": 3}, {"ttft_s": 0.5, "tpot_s": 0.05, "e2e_s": 3},
    ]
    reports = child_reports(sweep, read_manifest(sweep))
    assert [report["service_identity"]["fixture_observation"] for report in reports] == [1, 2]
