"""Real local HTTP verifies concurrency, persistence and failed-response accounting."""
import hashlib
import json
import tempfile
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


@contextmanager
def server(*, fail=False, close=False, nonfinite=False):
    ports, active, peak = [], 0, 0
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_POST(self):
            nonlocal active, peak
            self.rfile.read(int(self.headers["Content-Length"]))
            with lock:
                ports.append(self.client_address[1])
                active += 1
                peak = max(peak, active)
            time.sleep(0.015)
            body = (
                b'{"workload_contract":{"score":NaN}}'
                if nonfinite
                else json.dumps({"error": "fixture failure"} if fail else {"workload_contract": {"input": {"count": 1}}}).encode()
            )
            self.send_response(500 if fail else 200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "application/json")
            if close:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            with lock:
                active -= 1

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", ports, lambda: peak
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def test_source_input_plan_preserves_identity_schema_and_exact_payload():
    from acprof.cli.load import _read_source_input_entry

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "input_scale_plan.json"
        payload = {"text": "unchanged", "params": {"seed": 7}}
        path.write_text(json.dumps({
            "schema_version": 2,
            "entries": [
                {"input_scale": 1, "payload": {"text": "first"}},
                {"input_scale": 2, "payload": payload},
            ],
        }))
        original = path.read_bytes()
        expected = hashlib.sha256(original).hexdigest()
        assert (_read_source_input_entry(path, expected, 2)["payload"]) == (payload)
        assert (path.read_bytes()) == (original)
        with pytest.raises(ValueError, match="identity mismatch"):
            _read_source_input_entry(path, "0" * 64, 2)
        path.write_text(json.dumps({"schema_version": 1, "entries": []}))
        old_schema_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        with pytest.raises(ValueError, match="schema_version"):
            _read_source_input_entry(path, old_schema_hash, None)


def test_source_input_plan_is_bounded_after_identity_verification():
    from acprof.cli.load import _read_source_input_entry

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "input_scale_plan.json"
        path.write_text(json.dumps({
            "schema_version": 2,
            "entries": [{"input_scale": 1, "payload": {"text": "unchanged"}}],
            "padding": "x" * (4 * 1024 * 1024),
        }))
        original = path.read_bytes()
        expected = hashlib.sha256(original).hexdigest()
        with pytest.raises(ValueError, match="4 MiB"):
            _read_source_input_entry(path, expected, None)
        assert (path.read_bytes()) == (original)


def test_packet_capture_keeps_reap_after_kill_bounded(tmp_path, monkeypatch):
    from acprof.cli import load as load_cli

    class HungCapture:
        returncode = None

        def __init__(self):
            self.wait_timeouts = []
            self.terminated = False
            self.killed = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.wait_timeouts.append(timeout)
            raise load_cli.subprocess.TimeoutExpired("tcpdump", timeout)

        def kill(self):
            self.killed = True

    capture = HungCapture()
    monkeypatch.setattr(
        "acprof.host.packet_capture.require_packet_latency_prerequisites",
        lambda _interface: None,
    )
    monkeypatch.setattr(load_cli.subprocess, "Popen", lambda *_args, **_kwargs: capture)
    monkeypatch.setattr(load_cli.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="tcpdump could not be reaped after kill"):
        with load_cli.capture_packets(tmp_path / "load.pcap", interface="lo"):
            pass

    assert capture.terminated
    assert capture.killed
    assert capture.wait_timeouts == [5, 5]


class TestLoadProtocol:
    def run_load(self, url, **options):
        from acprof.host.load_protocol import LoadConfig, run_load
        return run_load(url, {}, LoadConfig(requests=8, timeout_seconds=2, **options), token="fixture")

    def test_serial_close_and_reuse_have_distinct_protocol_identity_and_real_sockets(self):
        from acprof.host.load_protocol import LoadConfig
        with server() as (url, ports, _):
            report = self.run_load(url, connections="reuse")
            assert (len(set(ports))) == (1)
            assert (report["counts"]["succeeded"]) == (8)
            assert (report["connection_count"]) == (1)
        with server() as (url, ports, _):
            self.run_load(url, connections="close")
            assert (len(set(ports))) == (8)
        assert (LoadConfig().identity()) != (LoadConfig(connections="reuse").identity())

    def test_concurrent_closed_loop_respects_limit(self):
        with server() as (url, _, peak):
            report = self.run_load(url, scenario="concurrent", concurrency=3)
        assert (peak()) == (3)
        assert (report["counts"]["succeeded"]) == (8)
        assert (all(row["planned_s"] <= row["sent_s"] <= row["completed_s"] for row in report["requests"]))

    def test_open_loop_keeps_schedule_when_service_is_slow(self):
        with server() as (url, _, _):
            report = self.run_load(url, scenario="arrival-rate", rate=10000, concurrency=1, max_pending=2)
        assert (report["counts"]["dropped"]) > (0)
        assert ([row["planned_s"] for row in report["requests"]]) == ([i / 10000 for i in range(8)])
        assert (report["counts"]["offered"]) == (8)

    def test_errors_do_not_become_successful_latency_samples(self):
        with server(fail=True) as (url, _, _):
            report = self.run_load(url)
        assert (report["counts"]["succeeded"]) == (0)
        assert (report["counts"]["failed"]) == (8)
        assert (report["latency_s"]["p95"]) is None

    def test_nonfinite_json_response_is_not_a_successful_sample(self):
        with server(nonfinite=True) as (url, _, _):
            report = self.run_load(url)
        assert (report["counts"]["succeeded"]) == (0)
        assert (report["counts"]["failed"]) == (8)
        assert all("non-finite" in row["error"] for row in report["requests"])

    def test_reuse_is_not_silently_reported_when_server_closes_connections(self):
        with server(close=True) as (url, _, _):
            report = self.run_load(url, connections="reuse")
        assert not (report["successful"])
        assert ("reuse") in (report["requests"][0]["error"])

    @pytest.mark.parametrize('options', ({'concurrency': 2}, {'scenario': 'arrival-rate'}, {'rate': 2}, {'requests': 0}))
    def test_poisson_plan_is_seeded_and_invalid_combinations_are_rejected(self, options):
        from acprof.host.load_protocol import LoadConfig
        config = LoadConfig(scenario="arrival-rate", rate=10, arrival="poisson", seed=7)
        assert (config.schedule()) == (config.schedule())
        assert (config.schedule()) != (LoadConfig(scenario="arrival-rate", rate=10, arrival="poisson", seed=8).schedule())
        with pytest.raises(ValueError):
            LoadConfig(**options).validate()
