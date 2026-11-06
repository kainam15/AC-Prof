"""Completed HTTP evidence survives connection finalization failures."""
import http.client
import json
import threading
import time
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from acprof.cli import load as load_cli
from acprof.host.measurement import load
from acprof.host.measurement.streaming import OpenAIStreamingAdapter
from tests.execution.operations.test_load_protocol import server
from tests.execution.operations.test_streaming_load import streaming_server


@contextmanager
def connection_faults(monkeypatch, errors, *, workers=1, on_close=None):
    original = http.client.HTTPConnection
    owner = threading.current_thread()
    clients, closed = [], []
    guard = threading.Lock()
    rendezvous = threading.Barrier(workers) if workers > 1 else None

    class FaultyConnection(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            with guard:
                self.number = len(clients)
                clients.append(self)
            self.first_request = True

        def request(self, *args, **kwargs):
            if self.first_request and rendezvous is not None:
                self.first_request = False
                rendezvous.wait(timeout=3)
            super().request(*args, **kwargs)

        def close(self):
            super().close()
            if threading.current_thread() is owner:
                closed.append(self.number)
                if on_close is not None:
                    on_close()
                raise errors[self.number]

    monkeypatch.setattr("acprof.host.measurement.load_transport.http.client.HTTPConnection", FaultyConnection)
    try:
        yield SimpleNamespace(clients=clients, closed=closed)
    finally:
        # Failed assertions must not leave the local HTTP fixture waiting on keep-alive.
        for client in clients:
            original.close(client)


@pytest.fixture(name="endpoint", params=[False, True], ids=["json", "sse"])
def endpoint_fixture(request):
    streaming = request.param
    with streaming_server() if streaming else server() as running:
        yield SimpleNamespace(
            url=running[0], ports=running[1],
            payload={"model": "fixture", "messages": [{"role": "user", "content": "hello"}]} if streaming else {},
            adapter=OpenAIStreamingAdapter() if streaming else None,
        )


def run(endpoint, *, requests=2, workers=1):
    return load.run_load(
        endpoint.url, endpoint.payload,
        load.LoadConfig(requests=requests, connections="reuse", concurrency=workers,
                        scenario="concurrent" if workers > 1 else "serial", timeout_seconds=2),
        token="fixture", adapter=endpoint.adapter,
    )


def interrupt_after_completed_request(monkeypatch, error):
    wait = load.wait

    def interrupted_wait(*args, **kwargs):
        wait(*args, **kwargs)
        raise error

    monkeypatch.setattr("acprof.host.measurement.load.wait", interrupted_wait)


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_final_close_failure_retains_completed_requests(endpoint, monkeypatch, error_type):
    failure = error_type("fixture final close failed")
    with connection_faults(monkeypatch, [failure]) as connections:
        with pytest.raises(error_type, match="final close failed") as caught:
            run(endpoint)
        assert caught.value is failure
        assert connections.closed == [0]
        report = caught.value.load_report
        assert report["successful"] is False
        assert report["counts"] == {"offered": 2, "sent": 2, "succeeded": 2, "failed": 0, "dropped": 0, "cancelled": 0}
        assert [record["status"] for record in report["requests"]] == ["ok", "ok"]
        assert report["latency_s"]["sample_count"] == 2
        if endpoint.adapter is not None:
            assert [record["output_tokens"] for record in report["requests"]] == [3, 3]
            assert report["capacity"]["counts"]["succeeded"] == 2


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_final_close_failure_does_not_replace_scheduler_error(endpoint, monkeypatch, error_type):
    operation = error_type("fixture scheduling interrupted")
    cleanup = OSError("fixture final close failed")
    interrupt_after_completed_request(monkeypatch, operation)
    with connection_faults(monkeypatch, [cleanup]):
        with pytest.raises(error_type, match="scheduling interrupted") as caught:
            run(endpoint, requests=3)
        assert caught.value is operation
        assert any("OSError: fixture final close failed" in note for note in operation.__notes__)
        report = caught.value.load_report
        assert report["successful"] is False
        assert report["counts"] == {"offered": 3, "sent": 1, "succeeded": 1, "failed": 0, "dropped": 0, "cancelled": 2}
        assert [row["status"] for row in report["requests"]] == ["ok", "cancelled", "cancelled"]


def test_cancellation_during_close_keeps_preceding_scheduler_error(endpoint, monkeypatch):
    operation = RuntimeError("fixture scheduling interrupted")
    cleanup = KeyboardInterrupt("fixture close interrupted")
    interrupt_after_completed_request(monkeypatch, operation)
    with connection_faults(monkeypatch, [cleanup]):
        with pytest.raises(KeyboardInterrupt, match="close interrupted") as caught:
            run(endpoint, requests=3)
        assert caught.value is cleanup
        assert cleanup.__cause__ is operation
        assert any("RuntimeError: fixture scheduling interrupted" in note for note in cleanup.__notes__)
        assert getattr(caught.value, "load_report")["counts"]["succeeded"] == 1


def test_failed_final_close_stays_outside_the_load_duration(endpoint, monkeypatch):
    offset = 0.0
    clock = time.perf_counter
    monkeypatch.setattr(time, "perf_counter", lambda: clock() + offset)

    def slow_close():
        nonlocal offset
        offset += 1000

    with connection_faults(monkeypatch, [OSError("fixture close failed")], on_close=slow_close):
        with pytest.raises(OSError, match="close failed") as caught:
            run(endpoint)
        assert offset == 1000
        report = getattr(caught.value, "load_report")
        assert 0 < report["duration_s"] < 1000
        assert report["successful_requests_per_s"] == 2 / report["duration_s"]


@pytest.mark.parametrize("error_types", [(OSError, OSError), (OSError, KeyboardInterrupt), (KeyboardInterrupt, OSError)])
def test_all_connections_are_closed_despite_multiple_failures(endpoint, monkeypatch, error_types):
    errors = [kind(f"fixture close {index} failed") for index, kind in enumerate(error_types)]
    expected = errors[1] if error_types[1] is KeyboardInterrupt else errors[0]
    with connection_faults(monkeypatch, errors, workers=2) as connections:
        with pytest.raises(type(expected)) as caught:
            run(endpoint, requests=4, workers=2)
        assert connections.closed == [0, 1]
        assert caught.value is expected
        detail = "\n".join((str(expected), *expected.__notes__))
        assert all(str(error) in detail for error in errors)
        assert caught.value.load_report["counts"]["succeeded"] == 4
        assert caught.value.load_report["successful"] is False


@pytest.mark.parametrize("warmup", [0, 1])
def test_endpoint_cli_publishes_connection_failure_evidence(tmp_path, monkeypatch, warmup):
    failure = OSError("fixture final close failed")
    monkeypatch.setattr("acprof.cli.streaming_load.MeasurementLock", nullcontext)
    monkeypatch.setattr("acprof.cli.streaming_load.host_identity", lambda _root: {"hostname": "fixture"})
    output = tmp_path / "load"
    with streaming_server() as (url, *_rest):
        with connection_faults(monkeypatch, [failure]):
            with pytest.raises(OSError, match="final close failed"):
                load_cli.main(["--base-url", url, "--protocol", "openai-chat", "--model", "fixture",
                               "--prompt", "hello", "--requests", "2", "--warmup", str(warmup),
                               "--output-dir", str(output)])
    report = json.loads((output / "load.json").read_text())
    phase = "warmup" if warmup else "result"
    assert report["status"] == "failed"
    assert report["successful"] is False
    assert report[phase]["counts"]["succeeded"] == (1 if warmup else 2)
    assert "fixture final close failed" in report["phase_errors"][phase]["detail"]
    if warmup:
        assert "result" not in report
