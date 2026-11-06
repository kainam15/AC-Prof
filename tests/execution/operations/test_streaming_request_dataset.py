"""Ordered local request datasets preserve inputs through measurement and reports."""
from __future__ import annotations

import hashlib
import http.client
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest

from acprof.cli.main import main
from tests.execution.operations.test_streaming_load import streaming_server


def write_dataset(path: Path, endpoint: str = "/v1/chat/completions") -> list[dict[str, Any]]:
    bodies = []
    for index, prompt in enumerate(("00123", "解释吞吐量", "longer prompt with several words")):
        body: dict[str, Any] = {"model": "fixture", "max_tokens": index + 2, "temperature": 0}
        body.update({"messages": [{"role": "user", "content": prompt}]} if "chat" in endpoint else {"prompt": prompt})
        bodies.append(body)
    path.write_text("\n".join(json.dumps(body, ensure_ascii=False) for body in bodies) + "\n", encoding="utf-8")
    return bodies


@pytest.fixture(name="local_only")
def local_measurement_environment(monkeypatch):
    monkeypatch.setattr("acprof.cli.streaming_load.MeasurementLock", nullcontext)
    monkeypatch.setattr("acprof.host.run_state.offline_analysis_lock", nullcontext)
    monkeypatch.setattr("acprof.cli.streaming_load.host_identity", lambda _root: {"hostname": "fixture"})


@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/completions"])
@pytest.mark.parametrize("connections", ["close", "reuse"])
def test_public_cli_sends_each_dataset_body_and_reports_without_source(tmp_path, local_only, endpoint, connections):
    from acprof.analysis.load_report import read_load_report

    source, output = tmp_path / "requests.jsonl", tmp_path / "load"
    bodies = write_dataset(source, endpoint)
    raw_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    protocol = "openai-chat" if "chat" in endpoint else "openai-completions"
    with streaming_server(endpoint=endpoint) as (origin, _ports, _paths, received, _ready):
        assert main(["load", "--base-url", origin, "--protocol", protocol, "--request-jsonl", str(source),
                     "--requests", "3", "--warmup", "2", "--scenario", "concurrent", "--concurrency", "2",
                     "--connections", connections, "--output-dir", str(output)]) == 0
    assert len(received) == 5
    normalized = [{**body, "stream": True, "n": 1, "stream_options": {"include_usage": True}} for body in bodies]
    assert received[:2] == normalized[:2]
    assert sorted(received[2:], key=lambda item: item["max_tokens"]) == normalized
    payload = json.loads((output / "load.json").read_text())
    assert payload["schema_version"] == payload["result"]["schema_version"] == 3
    assert "request_body" not in payload["identity"]
    dataset = payload["identity"]["request_dataset"]
    assert dataset["bodies"] == normalized
    assert dataset["source"]["sha256"] == raw_digest
    assert dataset["selection"] == "listed_prefix"
    assert [item["input_index"] for item in payload["result"]["requests"]] == [0, 1, 2]
    assert [item["input_index"] for item in payload["warmup"]["requests"]] == [0, 1]
    assert payload["result"]["counts"]["succeeded"] == 3
    source.unlink()
    assert read_load_report(output)["version"] == 3
    assert main(["report", str(output)]) == 0
    html = (output / "load_report.html").read_text()
    assert "Request workload" in html
    assert "listed_prefix" in html


@pytest.mark.parametrize(("raw", "message"), [
    (b" \n\t", "at least one"), (b"[]\n", "line 1"), (b"{}\n{broken\n", "line 2"),
    (b'{"model":"fixture","prompt":"x","temperature":NaN}\n', "line 1"),
    (b'{"model":"fixture","prompt":"x","temperature":1e999}\n', "line 1"),
    (b'{"model":"fixture","prompt":"\\ud800"}\n', "line 1"),
    (b'\xff\n', "line 1"),
    (b'{"model":"fixture","prompt":"x","Authorization":"secret"}\n', "credentials"),
    (b'{"model":"fixture","prompt":"x"}\n{"model":"other","prompt":"y"}\n', "line 2"),
    (b'{"model":"fixture","prompt":"x"}\n{"model":"fixture","prompt":"y","n":2}\n', "line 2"),
], ids=["empty", "array", "malformed", "nan", "overflow", "surrogate", "utf8", "credentials", "mixed-models", "bad-unused-row"])
def test_invalid_dataset_is_rejected_before_output_probe_or_warmup(tmp_path, monkeypatch, capsys, raw, message):
    source, output = tmp_path / "invalid.jsonl", tmp_path / "absent"
    source.write_bytes(raw)
    calls = []
    monkeypatch.setattr("acprof.cli.streaming_load.probe_service", lambda *_a, **_k: calls.append("probe"))
    monkeypatch.setattr("acprof.cli.streaming_load.run_load", lambda *_a, **_k: calls.append("load"))
    with pytest.raises(SystemExit) as caught:
        main(["load", "--base-url", "http://127.0.0.1:9", "--protocol", "openai-completions",
              "--request-jsonl", str(source), "--requests", "1", "--warmup", "0", "--output-dir", str(output)])
    assert caught.value.code == 2
    assert message in capsys.readouterr().err
    assert calls == []
    assert not output.exists()


@pytest.mark.parametrize("extra", [
    ["--requests", "4"], ["--warmup", "4"], ["--model", "different"], ["--max-tokens", "3"],
    ["--temperature", "0"], ["--generation-seed", "1"], ["--prompt", "not combined"],
    ["--request-json", "another.json"],
])
def test_conflicting_or_insufficient_dataset_inputs_are_rejected(tmp_path, extra):
    source, output = tmp_path / "requests.jsonl", tmp_path / "absent"
    write_dataset(source)
    with pytest.raises(SystemExit) as caught:
        main(["load", "--base-url", "http://127.0.0.1:9", "--protocol", "openai-chat",
              "--request-jsonl", str(source), "--requests", "3", "--warmup", "0", "--output-dir", str(output), *extra])
    assert caught.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("limit", ["bytes", "rows", "line"])
def test_dataset_limits_stop_before_execution(tmp_path, monkeypatch, limit):
    from acprof.host.measurement import request_inputs

    source = tmp_path / "large.jsonl"
    write_dataset(source)
    name = {"bytes": "MAX_REQUEST_DATASET_BYTES", "rows": "MAX_REQUEST_DATASET_ROWS", "line": "MAX_JSON_ARTIFACT_BYTES"}[limit]
    monkeypatch.setattr(request_inputs, name, 2 if limit == "rows" else 25, raising=False)
    with pytest.raises(ValueError, match="limit"):
        request_inputs.read_request_dataset(source)


def test_blank_lines_keep_source_locations_and_numeric_looking_prompt(tmp_path):
    from acprof.host.measurement.request_inputs import read_request_dataset

    source = tmp_path / "requests.jsonl"
    source.write_bytes(b'\r\n{"model":"fixture","prompt":"00123"}\r\n  \n{"model":"fixture","prompt":"second"}')
    bodies, provenance = read_request_dataset(source)
    assert [row["prompt"] for row in bodies] == ["00123", "second"]
    assert provenance["lines"] == [2, 4]
    assert provenance["rows"] == 2
    assert provenance["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_dry_run_describes_input_prefix_without_accessing_service(tmp_path, monkeypatch, capsys):
    source, output = tmp_path / "requests.jsonl", tmp_path / "absent"
    write_dataset(source)
    effects = []
    monkeypatch.setattr("acprof.cli.streaming_load.probe_service", lambda *_a, **_k: effects.append("probe"))
    monkeypatch.setattr("acprof.cli.streaming_load.run_load", lambda *_a, **_k: effects.append("load"))
    assert main(["load", "--base-url", "http://127.0.0.1:9", "--protocol", "openai-chat",
                 "--request-jsonl", str(source), "--requests", "2", "--warmup", "1", "--scenario", "concurrent",
                 "--sweep-concurrency", "1,2", "--dry-run", "--output-dir", str(output)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["request_dataset"]["source"]["rows"] == 3
    assert plan["request_dataset"]["source"]["lines"] == [1, 2]
    assert "bodies" not in plan["request_dataset"]
    assert effects == []
    assert not output.exists()


def test_sweep_freezes_file_and_maps_bodies_by_request_id(tmp_path, monkeypatch, local_only):
    from acprof.analysis.load_sweep import read_load_sweep
    from acprof.cli import streaming_load

    source, output = tmp_path / "requests.jsonl", tmp_path / "sweep"
    bodies = write_dataset(source)
    observed = {}
    original_request, original_probe = http.client.HTTPConnection.request, streaming_load.probe_service

    def request(connection, method, url, body=None, headers=None, **kwargs):
        if method == "POST":
            observed[headers["X-Req-Id"]] = json.loads(body)
        return original_request(connection, method, url, body=body, headers=headers or {}, **kwargs)

    def probe(*args, **kwargs):
        source.unlink(missing_ok=True)
        return original_probe(*args, **kwargs)

    monkeypatch.setattr(http.client.HTTPConnection, "request", request)
    monkeypatch.setattr("acprof.cli.streaming_load.probe_service", probe)
    with streaming_server() as (origin, _ports, _paths, received, _ready):
        assert main(["load", "--base-url", origin, "--protocol", "openai-chat", "--request-jsonl", str(source),
                     "--requests", "3", "--warmup", "1", "--scenario", "concurrent", "--sweep-concurrency", "1,2",
                     "--output-dir", str(output)]) == 0
    assert len(received) == 8
    payloads = [json.loads((output / f"points/{index:04d}/load.json").read_text()) for index in (1, 2)]
    for payload in payloads:
        for record in payload["result"]["requests"]:
            assert observed[record["request_id"]]["messages"] == bodies[record["input_index"]]["messages"]
    assert payloads[0]["identity"]["request_dataset"] == payloads[1]["identity"]["request_dataset"]
    analysis = read_load_sweep(output)
    assert analysis.payload["valid_points"] == 2
    assert len(analysis.payload["conditions"]) == 1
    assert main(["report", str(output)]) == 0
    assert (output / "load_sweep_report.html").is_file()


@pytest.mark.parametrize("warmup", [0, 1])
def test_failed_responses_keep_input_mapping_and_phase(tmp_path, local_only, warmup):
    from acprof.analysis.load_report import read_load_report

    source, output = tmp_path / "requests.jsonl", tmp_path / "failed"
    write_dataset(source)
    with streaming_server(fault="http") as (origin, _ports, _paths, _bodies, _ready):
        argv = ["load", "--base-url", origin, "--protocol", "openai-chat", "--request-jsonl", str(source),
                "--requests", "3", "--warmup", str(warmup), "--output-dir", str(output)]
        if warmup:
            with pytest.raises(RuntimeError, match="streaming warmup failed"):
                main(argv)
        else:
            assert main(argv) == 1
    payload = json.loads((output / "load.json").read_text())
    phase = "warmup" if warmup else "result"
    assert payload["schema_version"] == payload[phase]["schema_version"] == 3
    assert [record["input_index"] for record in payload[phase]["requests"]] == list(range(warmup or 3))
    assert payload[phase]["counts"]["failed"] == (warmup or 3)
    if warmup:
        assert payload["status"] == "failed"
        assert "result" not in payload
    else:
        assert read_load_report(output)["capacity"]["counts"]["succeeded"] == 0


def test_dataset_source_replay_is_rejected_before_reading(tmp_path, capsys):
    with pytest.raises(SystemExit) as caught:
        main(["load", str(tmp_path / "missing"), "--gpu", "off", "--request-jsonl", str(tmp_path / "missing.jsonl"),
              "--output-dir", str(tmp_path / "absent")])
    assert caught.value.code == 2
    assert "require --base-url" in capsys.readouterr().err


def test_cancellation_retains_every_scheduled_input_index(tmp_path, monkeypatch):
    from acprof.host.measurement.load import LoadConfig, run_load
    from acprof.host.measurement.streaming import OpenAIStreamingAdapter

    bodies = tuple(write_dataset(tmp_path / "requests.jsonl"))
    with streaming_server(fault="cancel") as (origin, _ports, _paths, _received, ready):
        def interrupt(*_args, **_kwargs):
            assert ready.wait(2)
            raise KeyboardInterrupt("dataset cancelled")

        monkeypatch.setattr("acprof.host.measurement.load.wait", interrupt)
        with pytest.raises(KeyboardInterrupt, match="dataset cancelled") as caught:
            run_load(origin, bodies, LoadConfig(requests=3, timeout_seconds=2), token="cancelled",
                     adapter=OpenAIStreamingAdapter())
    report = getattr(caught.value, "load_report")
    assert report["schema_version"] == 3
    assert report["counts"]["cancelled"] == 3
    assert [row["input_index"] for row in report["requests"]] == [0, 1, 2]


def test_dropped_and_unsent_slots_do_not_shift_input_indices(tmp_path, monkeypatch):
    from acprof.host.measurement.load import LoadConfig, run_load
    from acprof.host.measurement.streaming import OpenAIStreamingAdapter

    def execute(transport, _schedule):
        transport.records[1] = {"request_id": "drop:1", "planned_s": 0.1, "sent_s": None,
                                "completed_s": None, "status": "dropped", "error": "max_pending exceeded", "connection": None}
        return 0.2, None

    monkeypatch.setattr("acprof.host.measurement.load._execute_load", execute)
    report = run_load("http://127.0.0.1:9", tuple(write_dataset(tmp_path / "requests.jsonl")),
                      LoadConfig(scenario="arrival-rate", rate=10, requests=3), token="drop", adapter=OpenAIStreamingAdapter())
    assert [(row["input_index"], row["status"]) for row in report["requests"]] == [(0, "cancelled"), (1, "dropped"), (2, "cancelled")]


@pytest.mark.parametrize(("adapter", "count"), [(False, 3), (True, 4)])
def test_low_level_dataset_preflight_never_creates_transport(tmp_path, monkeypatch, adapter, count):
    from acprof.host.measurement.load import LoadConfig, run_load
    from acprof.host.measurement.streaming import OpenAIStreamingAdapter

    calls = []
    monkeypatch.setattr("acprof.host.measurement.load.LoadTransport", lambda *_a, **_k: calls.append("transport"))
    with pytest.raises(ValueError, match="require|fewer"):
        run_load("http://127.0.0.1:9", tuple(write_dataset(tmp_path / "requests.jsonl")), LoadConfig(requests=count),
                 token="invalid", adapter=OpenAIStreamingAdapter() if adapter else None)
    assert calls == []


def test_dataset_encoding_and_hashing_stay_outside_load_duration(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from acprof.host.measurement import load as load_module
    from acprof.host.measurement.load import LoadConfig, run_load
    from acprof.host.measurement.streaming import OpenAIStreamingAdapter

    active = False
    calls = []
    execute, digest_function = load_module._execute_load, load_module.request_digest

    def timed(*args):
        nonlocal active
        active = True
        try:
            return execute(*args)
        finally:
            active = False

    def encode(*args, **kwargs):
        assert not active, "request encoding entered the load duration"
        calls.append("encode")
        return json.dumps(*args, **kwargs)

    def digest_input(value):
        assert not active, "input hashing entered the load duration"
        calls.append("digest")
        return digest_function(value)

    monkeypatch.setattr("acprof.host.measurement.load.json", SimpleNamespace(dumps=encode))
    monkeypatch.setattr("acprof.host.measurement.load.request_digest", digest_input)
    monkeypatch.setattr(load_module, "_execute_load", timed)
    with streaming_server() as (origin, _ports, _paths, _bodies, _ready):
        report = run_load(origin, tuple(write_dataset(tmp_path / "requests.jsonl")), LoadConfig(requests=3),
                          token="outside", adapter=OpenAIStreamingAdapter())
    assert report["counts"]["succeeded"] == 3
    assert "encode" in calls
    assert "digest" in calls


@pytest.mark.parametrize("warmup", [0, 3])
def test_measured_and_warmup_prefixes_have_independent_lengths(tmp_path, local_only, warmup):
    from acprof.analysis.load_report import read_load_report

    source, output = tmp_path / "requests.jsonl", tmp_path / "prefixes"
    bodies = write_dataset(source)
    with streaming_server() as (origin, _ports, _paths, received, _ready):
        assert main(["load", "--base-url", origin, "--protocol", "openai-chat", "--request-jsonl", str(source),
                     "--requests", "1", "--warmup", str(warmup), "--output-dir", str(output)]) == 0
    assert [body["max_tokens"] for body in received] == [body["max_tokens"] for body in bodies[:warmup]] + [2]
    data = read_load_report(output)
    assert len(data["experiment"]["identity"]["request_dataset"]["bodies"]) == max(1, warmup)
    assert data["capacity"]["counts"]["offered"] == 1
