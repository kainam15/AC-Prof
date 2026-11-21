"""Server-wide phase evidence surrounds a load without changing client timing."""
from __future__ import annotations

import hashlib
import json
import threading
from contextlib import contextmanager, nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from acprof.cli.main import main
from acprof.plotting.load_report import read_load_report

PHASE_NAMES = {
    "queue": "vllm:request_queue_time_seconds",
    "prefill": "vllm:request_prefill_time_seconds",
    "decode": "vllm:request_decode_time_seconds",
    "inference": "vllm:request_inference_time_seconds",
    "e2e": "vllm:e2e_request_latency_seconds",
}


def exposition(count=10, total=20.0, *, model="fixture", engine="0"):
    lines = ["process_start_time_seconds 100"]
    for name in PHASE_NAMES.values():
        lines.extend([
            f"# TYPE {name} histogram",
            f'{name}_count{{model_name={json.dumps(model)},engine="{engine}"}} {count}',
            f'{name}_sum{{model_name={json.dumps(model)},engine="{engine}"}} {total}',
        ])
    return "\n".join(lines) + "\n"


def sglang_exposition(requests):
    labels = 'model_name="fixture",engine_type="unified"'
    scheduler = labels + ',tp_rank="0",pp_rank="0"'
    families = {
        "queue_time_seconds": [(scheduler, requests, requests / 8)],
        "time_to_first_token_seconds": [(labels, requests, requests / 4)],
        "inter_token_latency_seconds": [(labels, 4 * requests, requests / 2)],
        "e2e_request_latency_seconds": [(labels, requests, requests)],
        "per_stage_req_latency_seconds": [
            (scheduler + ',stage="prefill_forward"', requests, requests / 2),
            (scheduler + ',stage="decode_loop"', 2 * requests, 1.5 * requests),
            ('model_name="other",stage="foreign"', requests, 10 * requests),
        ],
    }
    lines = []
    for family, series in families.items():
        metric = f"sglang:{family}"
        lines.append(f"# TYPE {metric} histogram")
        for tags, count, total in series:
            lines.extend([f'{metric}_bucket{{{tags},le="+Inf"}} {count}',
                          f"{metric}_sum{{{tags}}} {total}", f"{metric}_count{{{tags}}} {count}"])
    return "\n".join(lines) + "\n"


@contextmanager
def metrics_server(*, failure=None, failure_status=503, metrics_body=None, content_type="text/plain; version=0.0.4"):
    events, auth = [], []
    state = {"requests": 0, "scrapes": 0}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            pass

        def reply(self, status, body, response_type):
            self.send_response(status)
            self.send_header("Content-Type", response_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            events.append("metrics")
            auth.append(self.headers.get("Authorization"))
            assert self.path == "/metrics"
            state["scrapes"] += 1
            failed = failure == ("before" if state["scrapes"] == 1 else "after")
            raw = exposition(10 + state["requests"], 20 + 2 * state["requests"]).encode()
            if metrics_body is not None:
                raw = metrics_body(state)
            self.reply(failure_status if failed else 200, raw, content_type)

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            events.append("request")
            state["requests"] += 1
            choices = [
                {"choices": [{"delta": {"content": "hello"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}},
            ]
            raw = "".join(f"data: {json.dumps(value)}\n\n" for value in choices) + "data: [DONE]\n\n"
            self.reply(200, raw.encode(), "text/event-stream")

    server = ThreadingHTTPServer(("127.0.0.1", 0), lambda *args: Handler(*args))
    server.daemon_threads = False
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", events, auth
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture(autouse=True)
def isolated_host(monkeypatch):
    monkeypatch.setattr("acprof.cli.streaming_load.MeasurementLock", nullcontext)
    monkeypatch.setattr("acprof.cli.streaming_load.host_identity", lambda _root: {"hostname": "fixture"})


def command(origin, output, *extra):
    return ["load", "--base-url", origin, "--protocol", "openai-chat", "--model", "fixture",
            "--prompt", "hello", "--requests", "3", "--warmup", "1", "--output-dir", str(output), *extra]


@pytest.mark.parametrize("warmup", [0, 1])
def test_metrics_surround_only_measured_requests_and_roundtrip(tmp_path, monkeypatch, warmup):
    monkeypatch.setenv("TEST_METRICS_KEY", "fixture-secret")
    output = tmp_path / "load"
    with metrics_server() as (origin, events, auth):
        assert main(command(origin, output, "--server-metrics", "vllm", "--warmup", str(warmup),
                            "--api-key-env", "TEST_METRICS_KEY")) == 0
    assert events == ["request"] * warmup + ["metrics"] + ["request"] * 3 + ["metrics"]
    assert auth == ["Bearer fixture-secret"] * 2
    raw = (output / "load.json").read_text()
    assert "fixture-secret" not in raw
    report = read_load_report(output)
    metrics = report["server_metrics"]
    assert metrics["status"] == "ok"
    assert metrics["phases"]["prefill"] == {
        "status": "ok", "reason": None, "count": 3, "sum_s": 6.0, "mean_s": 2.0,
    }
    assert report["result"]["counts"]["succeeded"] == 3
    assert main(["report", str(output)]) == 0
    html = (output / "load_report.html").read_text()
    assert "Server phase timing" in html
    assert "other traffic" in html


def test_sglang_histograms_roundtrip_without_mixing_exporters_or_stages(tmp_path):
    output = tmp_path / "load"
    with metrics_server(metrics_body=lambda state: (
        exposition() + sglang_exposition(state["requests"])
    ).encode()) as (origin, events, _auth):
        assert main(command(origin, output, "--server-metrics", "sglang")) == 0
    assert events == ["request", "metrics", "request", "request", "request", "metrics"]
    report = read_load_report(output)
    experiment = report["experiment"]
    assert experiment["identity"]["server_metrics"]["kind"] == experiment["server_metrics"]["kind"] == "sglang"
    metrics = report["server_metrics"]
    assert metrics["status"] == "ok"
    assert {phase: (row["count"], row["sum_s"], row["mean_s"]) for phase, row in metrics["phases"].items()} == {
        "queue": (3, 0.375, 0.125), "ttft": (3, 0.75, 0.25), "itl": (12, 1.5, 0.125), "e2e": (3, 3.0, 1.0),
        "stage:prefill_forward": (3, 1.5, 0.5), "stage:decode_loop": (6, 4.5, 0.75),
    }
    assert main(["report", str(output)]) == 0
    html = (output / "load_report.html").read_text()
    assert "Exporter: <code>sglang</code>" in html
    assert "stage:prefill_forward" in html


@pytest.mark.parametrize("kind", ["vllm", "sglang"])
def test_dry_run_declares_metrics_without_http_or_files(tmp_path, capsys, kind):
    output = tmp_path / "absent"
    assert main(command("http://127.0.0.1:9", output, "--server-metrics", kind,
                        "--scenario", "concurrent", "--sweep-concurrency", "1,2", "--dry-run")) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["server_metrics"]["kind"] == kind
    assert plan["server_metrics"]["path"] == "/metrics"
    assert not output.exists()


def test_default_makes_no_metrics_requests(tmp_path):
    output = tmp_path / "load"
    with metrics_server() as (origin, events, _auth):
        assert main(command(origin, output)) == 0
    assert events == ["request"] * 4
    assert "server_metrics" not in json.loads((output / "load.json").read_text())
    assert read_load_report(output)["server_metrics"] is None


@pytest.mark.parametrize("failure", ["before", "after"])
def test_metrics_http_failure_preserves_evidence_and_fails_run(tmp_path, failure):
    output = tmp_path / "load"
    with metrics_server(failure=failure) as (origin, events, _auth):
        with pytest.raises(ValueError, match="HTTP 503"):
            main(command(origin, output, "--server-metrics", "vllm"))
    report = json.loads((output / "load.json").read_text())
    assert report["status"] == "failed"
    assert report["successful"] is False
    assert report["server_metrics"][failure]["status"] == "error"
    assert report["server_metrics"]["summary"]["status"] == "error"
    if failure == "before":
        assert events == ["request", "metrics"]
        assert "result" not in report
    else:
        assert report["result"]["counts"]["succeeded"] == 3
        assert read_load_report(output)["server_metrics"]["status"] == "error"


def test_incomplete_worker_after_load_preserves_client_results_not_server_means(tmp_path):
    orphan = 'vllm:request_queue_time_seconds_bucket{model_name="fixture",engine="1",le="+Inf"} 41\n'
    after = (exposition(13, 26) + orphan).encode()
    output = tmp_path / "load"
    with metrics_server(metrics_body=lambda state: exposition().encode() if state["scrapes"] == 1 else after) as (
        origin, events, _auth,
    ):
        with pytest.raises(ValueError, match="count/sum"):
            main(command(origin, output, "--server-metrics", "vllm", "--warmup", "0"))
    assert events == ["metrics", "request", "request", "request", "metrics"]
    report = read_load_report(output)
    assert report["experiment"]["status"] == "failed"
    assert report["result"]["counts"]["succeeded"] == 3
    observed = report["experiment"]["server_metrics"]
    assert observed["before"]["status"] == "ok"
    assert observed["after"]["status"] == "error"
    assert "count/sum" in observed["after"]["error"]
    assert observed["after"]["sha256"] == hashlib.sha256(after).hexdigest()
    assert observed["after"]["series"] is None
    queue = report["server_metrics"]["phases"]["queue"]
    assert queue["reason"] == "snapshot_error"
    assert queue["mean_s"] is None


def test_sweep_observes_each_point_and_exposes_server_summaries(tmp_path):
    from acprof.analysis.load_sweep import read_load_sweep

    output = tmp_path / "sweep"
    with metrics_server() as (origin, events, _auth):
        assert main(command(origin, output, "--server-metrics", "vllm", "--scenario", "concurrent",
                            "--sweep-concurrency", "1,2")) == 0
    assert events == (["request", "metrics"] + ["request"] * 3 + ["metrics"]) * 2
    data = read_load_sweep(output).payload
    assert data["valid_points"] == 2
    assert len(data["conditions"]) == 1
    assert all(point["server_metrics"]["phases"]["prefill"]["count"] == 3 for point in data["points"])
    assert main(["report", str(output)]) == 0


def test_observation_does_not_run_inside_load_duration(tmp_path, monkeypatch):
    from acprof.host.measurement import server_metrics
    from acprof.host.measurement.load import run_load

    running = False
    original_read = server_metrics._read_metrics
    observed = []

    def read(*args):
        assert not running
        observed.append("scrape")
        return original_read(*args)

    def load(*args, **kwargs):
        nonlocal running
        running = True
        try:
            return run_load(*args, **kwargs)
        finally:
            running = False

    monkeypatch.setattr("acprof.host.measurement.server_metrics._read_metrics", read)
    monkeypatch.setattr("acprof.cli.streaming_load.run_load", load)
    with metrics_server() as (origin, _events, _auth):
        assert main(command(origin, tmp_path / "load", "--server-metrics", "vllm")) == 0
    assert observed == ["scrape"] * 2


@pytest.mark.parametrize("primary", [RuntimeError("load failed"), KeyboardInterrupt()])
def test_after_snapshot_failure_never_masks_load_failure(tmp_path, monkeypatch, primary):
    def fail_load(*_args, **_kwargs):
        raise primary

    monkeypatch.setattr("acprof.cli.streaming_load.run_load", fail_load)
    output = tmp_path / "failed"
    with metrics_server(failure="after") as (origin, events, _auth):
        with pytest.raises(type(primary)) as caught:
            main(command(origin, output, "--server-metrics", "vllm", "--warmup", "0"))
    assert caught.value is primary
    assert any("server metrics" in note for note in getattr(primary, "__notes__", []))
    report = json.loads((output / "load.json").read_text())
    assert report["server_metrics"]["after"]["status"] == "error"
    assert events == ["metrics", "metrics"]


def test_replay_rejects_metrics_before_runtime_preparation(tmp_path, capsys):
    with pytest.raises(SystemExit) as caught:
        main(["load", str(tmp_path), "--gpu", "off", "--server-metrics", "vllm",
              "--output-dir", str(tmp_path / "unused")])
    assert caught.value.code == 2
    assert "OpenAI request options require" in capsys.readouterr().err


@pytest.mark.parametrize("fault", ["oversize", "utf8", "content-type", "missing-model", "invalid-body"])
def test_bad_metrics_response_stops_before_load(tmp_path, fault):
    from acprof.analysis.server_metrics import MAX_METRICS_BYTES

    raw = {
        "oversize": b"x" * (MAX_METRICS_BYTES + 1), "utf8": b"\xff",
        "content-type": exposition().encode(), "missing-model": exposition(model="other").encode(),
        "invalid-body": b"not a prometheus sample",
    }[fault]
    output = tmp_path / "load"
    with metrics_server(metrics_body=lambda _state: raw,
                        content_type="application/json" if fault == "content-type" else "text/plain") as (origin, events, _auth):
        with pytest.raises(ValueError, match="metrics|UTF-8|utf-8|Prometheus|model_name"):
            main(command(origin, output, "--server-metrics", "vllm", "--warmup", "0"))
    report = json.loads((output / "load.json").read_text())
    assert report["status"] == "failed"
    assert report["server_metrics"]["before"]["status"] == "error"
    assert "result" not in report
    assert events == ["metrics"]


@pytest.mark.parametrize("status", [404, 403])
def test_metrics_endpoint_hint_preserves_http_status_and_does_not_retry(tmp_path, status):
    output = tmp_path / "load"
    with metrics_server(failure="before", failure_status=status) as (origin, events, _auth):
        with pytest.raises(ValueError, match=f"GET /metrics returned HTTP {status}") as failed:
            main(command(origin, output, "--server-metrics", "sglang", "--warmup", "0"))
    assert events == ["metrics"]
    message = str(failed.value)
    assert ("--enable-metrics" in message) == (status == 404)
    assert "--model" not in message
    experiment = json.loads((output / "load.json").read_text())
    assert experiment["server_metrics"]["before"]["error"] == f"ValueError: {message}"
    assert experiment["server_metrics"]["before"]["sha256"] is None
    assert "result" not in experiment


def test_cold_metrics_need_an_explicit_warmup_not_an_automatic_request(tmp_path):
    cold = b"# TYPE sglang:queue_time_seconds histogram\n"
    def response(state):
        return sglang_exposition(state["requests"]).encode() if state["requests"] else cold

    output = tmp_path / "cold"
    with metrics_server(metrics_body=response) as (origin, events, _auth):
        with pytest.raises(ValueError, match="--warmup 1") as failed:
            main(command(origin, output, "--server-metrics", "sglang", "--warmup", "0"))
        assert events == ["metrics"]
        experiment = json.loads((output / "load.json").read_text())
        before = experiment["server_metrics"]["before"]
        assert before["error"] == f"ValueError: {failed.value}"
        assert before["sha256"] == hashlib.sha256(cold).hexdigest()
        assert "result" not in experiment
        warm = tmp_path / "warm"
        assert main(command(origin, warm, "--server-metrics", "sglang", "--warmup", "1")) == 0
    assert events[1:] == ["request", "metrics", "request", "request", "request", "metrics"]
    assert read_load_report(warm)["server_metrics"]["phases"]["queue"]["count"] == 3


def test_vllm_model_alias_hint_reaches_the_saved_report_without_foreign_labels(tmp_path):
    output = tmp_path / "load"
    other = exposition(model="private-alias").encode()
    with metrics_server(metrics_body=lambda state: other if state["requests"] else exposition().encode()) as (
        origin, events, _auth,
    ):
        with pytest.raises(ValueError, match="--served-model-name") as failed:
            main(command(origin, output, "--server-metrics", "vllm", "--warmup", "0"))
    assert events == ["metrics", "request", "request", "request", "metrics"]
    report = read_load_report(output)
    after = report["experiment"]["server_metrics"]["after"]
    assert after["error"] == f"ValueError: {failed.value}"
    assert "--model" in after["error"]
    assert "model_name" in after["error"]
    assert after["sha256"] == hashlib.sha256(other).hexdigest()
    assert report["experiment"]["status"] == "failed"
    assert report["result"]["counts"]["succeeded"] == 3
    assert report["server_metrics"]["phases"]["queue"]["mean_s"] is None
    assert main(["report", str(output)]) == 0
    assert "--served-model-name" in (output / "load_report.html").read_text()
    assert "private-alias" not in (output / "load.json").read_text()


def test_reset_is_visible_without_changing_request_success(tmp_path, capsys):
    output = tmp_path / "load"
    with metrics_server(metrics_body=lambda state: exposition(
        10 if state["scrapes"] == 1 else 1, 20.0 if state["scrapes"] == 1 else 2.0,
    ).encode()) as (origin, _events, _auth):
        assert main(command(origin, output, "--server-metrics", "vllm")) == 0
    assert "unavailable" in capsys.readouterr().out
    report = read_load_report(output)
    assert report["experiment"]["successful"]
    assert report["server_metrics"]["phases"]["prefill"]["reason"] == "counter_reset_or_creation_changed"


def test_dataset_and_metrics_work_together(tmp_path):
    source = tmp_path / "requests.jsonl"
    source.write_text("\n".join(json.dumps({
        "model": "fixture", "messages": [{"role": "user", "content": str(index)}],
    }) for index in range(3)))
    output = tmp_path / "load"
    with metrics_server() as (origin, _events, _auth):
        assert main(["load", "--base-url", origin, "--protocol", "openai-chat", "--requests", "3", "--warmup", "1",
                     "--request-jsonl", str(source), "--server-metrics", "vllm", "--output-dir", str(output)]) == 0
    report = read_load_report(output)
    assert report["version"] == 3
    assert report["server_metrics"]["phases"]["e2e"]["count"] == 3


def test_failed_metrics_stop_later_sweep_points(tmp_path):
    from acprof.analysis.load_sweep import read_load_sweep

    output = tmp_path / "sweep"
    with metrics_server(failure="after") as (origin, events, _auth):
        with pytest.raises(ValueError, match="HTTP 503"):
            main(command(origin, output, "--server-metrics", "vllm", "--scenario", "concurrent",
                         "--sweep-concurrency", "1,2"))
    manifest = json.loads((output / "sweep.json").read_text())
    assert [point["status"] for point in manifest["points"]] == ["failed", "not_run"]
    assert events.count("metrics") == 2
    data = read_load_sweep(output).payload
    assert data["partial_points"] == 1
    assert data["points"][0]["server_metrics"]["status"] == "error"
