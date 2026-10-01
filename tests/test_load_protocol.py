"""Real local HTTP verifies concurrency, persistence and failed-response accounting."""
import json
import threading
import time
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@contextmanager
def server(*, fail=False, close=False):
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
            body = json.dumps({"error": "fixture failure"} if fail else {"workload_contract": {"input": {"count": 1}}}).encode()
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


class LoadProtocolTests(unittest.TestCase):
    def run_load(self, url, **options):
        from acprof.host.load_protocol import LoadConfig, run_load
        return run_load(url, {}, LoadConfig(requests=8, timeout_seconds=2, **options), token="fixture")

    def test_serial_close_and_reuse_have_distinct_protocol_identity_and_real_sockets(self):
        from acprof.host.load_protocol import LoadConfig
        with server() as (url, ports, _):
            report = self.run_load(url, connections="reuse")
            self.assertEqual(len(set(ports)), 1)
            self.assertEqual(report["counts"]["succeeded"], 8)
            self.assertEqual(report["connection_count"], 1)
        with server() as (url, ports, _):
            self.run_load(url, connections="close")
            self.assertEqual(len(set(ports)), 8)
        self.assertNotEqual(LoadConfig().identity(), LoadConfig(connections="reuse").identity())

    def test_concurrent_closed_loop_respects_limit(self):
        with server() as (url, _, peak):
            report = self.run_load(url, scenario="concurrent", concurrency=3)
        self.assertEqual(peak(), 3)
        self.assertEqual(report["counts"]["succeeded"], 8)
        self.assertTrue(all(row["planned_s"] <= row["sent_s"] <= row["completed_s"] for row in report["requests"]))

    def test_open_loop_keeps_schedule_when_service_is_slow(self):
        with server() as (url, _, _):
            report = self.run_load(url, scenario="arrival-rate", rate=10000, concurrency=1, max_pending=2)
        self.assertGreater(report["counts"]["dropped"], 0)
        self.assertEqual([row["planned_s"] for row in report["requests"]], [i / 10000 for i in range(8)])
        self.assertEqual(report["counts"]["offered"], 8)

    def test_errors_do_not_become_successful_latency_samples(self):
        with server(fail=True) as (url, _, _):
            report = self.run_load(url)
        self.assertEqual(report["counts"]["succeeded"], 0)
        self.assertEqual(report["counts"]["failed"], 8)
        self.assertIsNone(report["latency_s"]["p95"])

    def test_reuse_is_not_silently_reported_when_server_closes_connections(self):
        with server(close=True) as (url, _, _):
            report = self.run_load(url, connections="reuse")
        self.assertFalse(report["successful"])
        self.assertIn("reuse", report["requests"][0]["error"])

    def test_poisson_plan_is_seeded_and_invalid_combinations_are_rejected(self):
        from acprof.host.load_protocol import LoadConfig
        config = LoadConfig(scenario="arrival-rate", rate=10, arrival="poisson", seed=7)
        self.assertEqual(config.schedule(), config.schedule())
        self.assertNotEqual(config.schedule(), LoadConfig(scenario="arrival-rate", rate=10, arrival="poisson", seed=8).schedule())
        for options in ({"concurrency": 2}, {"scenario": "arrival-rate"}, {"rate": 2}, {"requests": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                LoadConfig(**options).validate()
