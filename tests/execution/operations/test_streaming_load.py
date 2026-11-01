"""Real HTTP streaming verifies shared scheduling, cancellation and aggregation."""
from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from acprof.host.measurement.load import LoadConfig, _streaming_summary, run_load
from acprof.host.measurement.streaming import OpenAIStreamingAdapter


def event(value):
    return ("data: " + (value if isinstance(value, str) else json.dumps(value)) + "\n\n").encode()


@contextmanager
def streaming_server(*, endpoint="/v1/chat/completions", fault=None):
    ports, paths, bodies = [], [], []
    first_sent = threading.Event()
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_POST(self):
            ports.append(self.client_address[1])
            paths.append(self.path)
            bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(503 if fault == "http" else 200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            if fault == "close":
                self.send_header("Connection", "close")
            self.end_headers()

            def chunk(data):
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            def output(text):
                choice = {"index": 0, "finish_reason": None}
                choice.update({"delta": {"content": text}} if "chat" in endpoint else {"text": text})
                return {"choices": [choice]}

            try:
                chunk(event(output("Hello")))
                first_sent.set()
                if fault in {"timeout", "cancel"}:
                    release.wait(1)
                    return
                time.sleep(0.02)
                if fault == "error":
                    chunk(event({"error": {"message": "generation failed"}}))
                elif fault != "truncated":
                    chunk(event(output(" world!")))
                    time.sleep(0.02)
                    finish = {"index": 0, "finish_reason": "stop"}
                    finish.update({"delta": {}} if "chat" in endpoint else {"text": ""})
                    chunk(event({"choices": [finish]}))
                    if fault != "no_usage":
                        chunk(event({"choices": [], "usage": {
                            "prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}}))
                    chunk(event("[DONE]"))
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", ports, paths, bodies, first_sent
    finally:
        release.set()
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def payload(endpoint):
    return {"model": "fixture", **(
        {"messages": [{"role": "user", "content": "hello"}]} if "chat" in endpoint else
        {"prompt": "hello"})}


@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/completions"])
def test_streaming_reuses_existing_scheduler_and_connection(endpoint):
    with streaming_server(endpoint=endpoint) as (url, ports, paths, bodies, _):
        report = run_load(url, payload(endpoint), LoadConfig(requests=3, connections="reuse"),
                          token="stream", adapter=OpenAIStreamingAdapter(endpoint=endpoint))
    assert report["successful"]
    assert report["kind"] == "stream_load_result"
    assert report["counts"]["succeeded"] == 3
    assert len(set(ports)) == 1
    assert paths == [endpoint] * 3
    assert all(body["stream"] and body["stream_options"]["include_usage"] for body in bodies)
    assert report["protocol"]["kind"] == "openai_stream_http_load"
    for row in report["requests"]:
        assert row["sent_s"] <= row["first_output_s"] < row["last_output_s"] <= row["completed_s"]
        assert row["e2e_s"] >= row["output_e2e_s"] >= row["ttft_s"] > 0
        assert row["output_tokens"] == 3
        assert row["output_token_count_source"] == "server_usage"
        assert len(row["output_events"]) == 2
        assert row["itl_s"] is None
    assert report["streaming"]["itl_s"]["sample_count"] == 0
    assert report["streaming"]["output_event_interval_s"]["sample_count"] == 3
    assert report["streaming"]["ttft_s"]["sample_count"] == 3


@pytest.mark.parametrize(("scenario", "options"), [
    ("concurrent", {"concurrency": 2}),
    ("arrival-rate", {"concurrency": 2, "rate": 30, "arrival": "poisson", "seed": 7}),
])
def test_streaming_uses_existing_concurrent_and_seeded_arrival_modes(scenario, options):
    endpoint = "/v1/chat/completions"
    with streaming_server() as (url, _, _, _, _):
        report = run_load(url, payload(endpoint), LoadConfig(requests=4, scenario=scenario, **options),
                          token="load", adapter=OpenAIStreamingAdapter())
    assert report["successful"]
    assert report["counts"]["offered"] == report["counts"]["succeeded"] == 4
    if scenario == "arrival-rate":
        expected = LoadConfig(requests=4, scenario=scenario, **options).schedule()
        assert [row["planned_s"] for row in report["requests"]] == expected


@pytest.mark.parametrize("fault", ["error", "truncated", "http", "timeout", "close"])
def test_stream_failure_cannot_enter_success_distributions(fault):
    with streaming_server(fault=fault) as (url, _, _, _, _):
        report = run_load(
            url, payload("/v1/chat/completions"),
            LoadConfig(requests=2, timeout_seconds=0.1,
                       connections="reuse" if fault == "close" else "close"),
            token="failed", adapter=OpenAIStreamingAdapter(),
        )
    assert not report["successful"]
    assert report["counts"]["failed"] == 2
    assert report["streaming"]["ttft_s"]["mean"] is None
    assert report["streaming"]["tpot_s"]["sample_count"] == 0
    assert all(row["ttft_s"] is None and row["itl_s"] is None for row in report["requests"])
    if fault == "timeout":
        assert all(row["error_code"] == "timeout" for row in report["requests"])
    if fault in {"error", "truncated", "timeout", "close"}:
        assert all(row["output_events"] for row in report["requests"])


def test_interrupt_retains_partial_evidence_and_cancels_active_close_socket(monkeypatch):
    import acprof.host.measurement.load as load_module

    with streaming_server(fault="cancel") as (url, _, _, _, first_sent):
        def interrupt(*_args, **_kwargs):
            assert first_sent.wait(2)
            raise KeyboardInterrupt

        monkeypatch.setattr(load_module, "wait", interrupt)
        with pytest.raises(KeyboardInterrupt) as caught:
            run_load(url, payload("/v1/chat/completions"),
                     LoadConfig(requests=3, timeout_seconds=3), token="cancelled",
                     adapter=OpenAIStreamingAdapter())
    report = caught.value.load_report
    assert not report["successful"]
    assert report["counts"]["cancelled"] == 3
    assert report["streaming"]["ttft_s"]["sample_count"] == 0
    assert report["requests"][0]["status"] == "cancelled"
    assert report["requests"][0]["ttft_s"] is None


def test_cancellation_during_connect_does_not_start_waiting_for_response(monkeypatch):
    from types import SimpleNamespace

    import acprof.host.measurement.load as load_module

    cancellation = threading.Event()
    requesting = threading.Event()
    observed = {"response_calls": 0, "closed": False}

    class DelayedConnection:
        sock = None

        def __init__(self, *_args, **_kwargs):
            pass

        def request(self, *_args, **_kwargs):
            requesting.set()
            assert cancellation.wait(2)

        def getresponse(self):
            observed["response_calls"] += 1
            raise AssertionError("cancelled request must not wait for response headers")

        def close(self):
            observed["closed"] = True

    def interrupt(*_args, **_kwargs):
        assert requesting.wait(2)
        raise KeyboardInterrupt

    monkeypatch.setattr(load_module, "threading", SimpleNamespace(
        Event=lambda: cancellation, local=threading.local, Lock=threading.Lock,
        Barrier=threading.Barrier,
    ))
    monkeypatch.setattr(load_module.http.client, "HTTPConnection", DelayedConnection)
    monkeypatch.setattr(load_module, "wait", interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        run_load("http://127.0.0.1:1", payload("/v1/chat/completions"),
                 LoadConfig(requests=2), token="connect-cancel",
                 adapter=OpenAIStreamingAdapter())
    assert observed["response_calls"] == 0
    assert observed["closed"]
    assert caught.value.load_report["counts"]["cancelled"] == 2


def test_request_weighting_interval_pooling_and_linear_percentiles():
    records = [
        {"status": "ok", "ttft_s": 0.1, "tpot_s": 0.02, "e2e_s": 0.23, "output_e2e_s": 0.18,
         "itl_s": None, "output_event_intervals_s": [0.04, 0.04],
         "output_token_count_source": "server_usage"},
        {"status": "ok", "ttft_s": 0.3, "tpot_s": 0.06, "e2e_s": 0.6, "output_e2e_s": 0.5,
         "itl_s": [0.2], "output_event_intervals_s": [0.2],
         "output_token_count_source": "server_usage"},
        {"status": "ok", "ttft_s": None, "tpot_s": None, "e2e_s": 0.1, "output_e2e_s": None,
         "itl_s": None, "output_event_intervals_s": [], "output_token_count_source": "unavailable"},
        {"status": "error", "error_code": "timeout", "ttft_s": 999, "tpot_s": 999,
         "itl_s": [999], "output_event_intervals_s": [999]},
    ]
    result = _streaming_summary(records)
    assert result["ttft_s"]["mean"] == pytest.approx(0.2)
    assert result["ttft_s"]["p50"] == pytest.approx(0.2)
    assert result["ttft_s"]["p90"] == pytest.approx(0.28)
    assert result["ttft_s"]["p95"] == pytest.approx(0.29)
    assert result["ttft_s"]["p99"] == pytest.approx(0.298)
    assert result["ttft_s"]["sample_count"] == 2
    assert result["ttft_s"]["missing_requests"] == 1
    assert result["tpot_s"]["mean"] == pytest.approx(0.04)
    assert result["output_event_interval_s"]["mean"] == pytest.approx(0.09333333333333334)
    assert result["itl_s"]["sample_count"] == 1
    assert result["errors"] == {"timeout": 1}


@pytest.mark.parametrize("invalid_count", [False, True])
def test_tokenizer_counting_runs_after_all_requests_and_failure_keeps_timings(invalid_count):
    from acprof.host.measurement.streaming import MatchedTokenCounter

    calls = []
    with streaming_server(fault="no_usage") as (url, _, _, bodies, _):
        def count(text):
            calls.append((len(bodies), text))
            return True if invalid_count else 3

        counter = MatchedTokenCounter(count, "fixture", "rev", "fixture", "rev")
        report = run_load(url, payload("/v1/chat/completions"), LoadConfig(requests=3),
                          token="fallback", adapter=OpenAIStreamingAdapter(token_counter=counter))
    assert calls == [(3, "Hello world!")] * 3
    assert report["successful"]
    assert report["streaming"]["ttft_s"]["sample_count"] == 3
    assert report["streaming"]["token_count_errors"] == (3 if invalid_count else 0)
    for row in report["requests"]:
        assert row["ttft_s"] > 0
        assert row["output_tokens"] == (None if invalid_count else 3)
        if invalid_count:
            assert row["tpot_s"] is None
            assert row["missing_reasons"]["tpot_s"] == "tokenizer_error"


@pytest.mark.parametrize("headers", [{"Connection": "close"}, {"Authorization": "bad\r\nvalue"}])
def test_load_rejects_reserved_or_invalid_headers_before_network(headers):
    with pytest.raises(ValueError, match="header"):
        run_load("http://127.0.0.1:1", payload("/v1/chat/completions"),
                 LoadConfig(requests=1), token="invalid", adapter=OpenAIStreamingAdapter(),
                 headers=headers)
