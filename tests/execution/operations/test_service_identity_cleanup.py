"""Probe cleanup preserves failures, cancellation, and ownership of real HTTP resources."""
import http.client
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from acprof.cli import load as load_cli
from acprof.plotting.load_report import read_load_report
from tests.execution.operations.test_service_identity_framing import (
    CONFIG_PATH,
    MODES,
    observe,
    service_fixture,
)

ENDPOINTS = [
    ("vllm", "/version", "version"), ("vllm", "/v1/models", "models"),
    ("vllm-config", CONFIG_PATH, "server_info"),
    ("sglang", "/server_info", "server_info"), ("sglang", "/v1/models", "models"),
    ("sglang-legacy", "/get_server_info", "server_info"),
]


@pytest.fixture
def transport_faults(monkeypatch):
    original = http.client.HTTPConnection
    state = SimpleNamespace(read_errors={}, response_errors={}, connection_errors={}, connections=[], responses=[])

    class Response:
        def __init__(self, inner, path):
            self.inner = inner
            self.path = path
            self.close_attempted = False
            state.responses.append(self)

        @property
        def status(self):
            return self.inner.status

        @property
        def length(self):
            return self.inner.length

        def read(self, amount=None):
            if self.path in state.read_errors:
                raise state.read_errors[self.path]
            return self.inner.read(amount)

        def close(self):
            self.close_attempted = True
            self.inner.close()
            if self.path in state.response_errors:
                raise state.response_errors[self.path]

    class Connection:
        def __init__(self, *args, **kwargs):
            self.inner = original(*args, **kwargs)
            self.path = self.method = ""
            self.close_attempted = False
            state.connections.append(self)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def request(self, method, path, **kwargs):
            self.method, self.path = method, path
            self.inner.request(method, path, **kwargs)

        def getresponse(self):
            received = self.inner.getresponse()
            return Response(received, self.path) if self.method == "GET" else received

        def close(self):
            self.close_attempted = True
            self.inner.close()
            if self.path in state.connection_errors:
                raise state.connection_errors[self.path]

    monkeypatch.setattr(http.client, "HTTPConnection", Connection)
    try:
        yield state
    finally:
        # Fallback teardown must not satisfy the ownership assertions below.
        for connection in state.connections:
            connection.inner.close()


def assert_resources_closed(state):
    assert state.connections
    assert state.responses
    assert all(response.inner.isclosed() for response in state.responses)
    assert all(response.close_attempted for response in state.responses)
    assert all(connection.inner.sock is None for connection in state.connections)
    assert all(connection.close_attempted for connection in state.connections)


def assert_probe_diagnostic(result, kind, path, error_key, messages):
    assert result["probe"]["status"] == "partial"
    detail = result["probe"]["errors"][error_key]
    for message in messages:
        assert message in detail
    if kind != "vllm":
        attempt = next(row for row in result["probe"]["requests"] if row["path"] == path)
        assert attempt["http_status"] == 200
        assert attempt["error"] == detail


@pytest.mark.parametrize(("mode", "path", "error_key"), ENDPOINTS)
@pytest.mark.parametrize("failure_site", ["response", "connection", "both"])
def test_probe_attempts_all_closes_and_reports_cleanup_failures(transport_faults, mode, path, error_key, failure_site):
    state = transport_faults
    messages = []
    if failure_site in {"response", "both"}:
        state.response_errors[path] = OSError("fixture response close failed")
        messages.append("fixture response close failed")
    if failure_site in {"connection", "both"}:
        state.connection_errors[path] = RuntimeError("fixture connection close failed")
        messages.append("fixture connection close failed")
    with service_fixture(mode) as service:
        result = observe(service)
        assert_resources_closed(state)
        assert service.seen == service.paths
        assert result["declared"] == service.declared
        assert_probe_diagnostic(result, service.kind, path, error_key, messages)


@pytest.mark.parametrize(("mode", "path", "error_key"), ENDPOINTS)
def test_probe_read_failure_remains_primary_with_both_cleanup_diagnostics(transport_faults, mode, path, error_key):
    state = transport_faults
    state.read_errors[path] = TimeoutError("fixture response read failed")
    state.response_errors[path] = OSError("fixture response close failed")
    state.connection_errors[path] = RuntimeError("fixture connection close failed")
    with service_fixture(mode) as service:
        result = observe(service)
        assert_resources_closed(state)
        assert service.seen == service.paths
        assert_probe_diagnostic(result, service.kind, path, error_key, [
            "fixture response read failed", "fixture response close failed", "fixture connection close failed",
        ])
        assert result["probe"]["errors"][error_key].startswith("TimeoutError: fixture response read failed")


@pytest.mark.parametrize(("mode", "path", "_error_key"), ENDPOINTS)
@pytest.mark.parametrize("cancel_at", ["read", "response", "connection"])
def test_probe_cancellation_keeps_identity_and_all_failure_notes(transport_faults, mode, path, _error_key, cancel_at):
    state = transport_faults
    cancelled = KeyboardInterrupt("fixture probe interrupted")
    failure = TimeoutError("fixture response read failed")
    state.read_errors[path] = cancelled if cancel_at == "read" else failure
    state.response_errors[path] = cancelled if cancel_at == "response" else OSError("fixture response close failed")
    state.connection_errors[path] = cancelled if cancel_at == "connection" else RuntimeError("fixture connection close failed")
    with service_fixture(mode) as service:
        with pytest.raises(KeyboardInterrupt, match="fixture probe interrupted") as caught:
            observe(service)
        assert caught.value is cancelled
        assert_resources_closed(state)
        assert service.seen == service.paths[:service.paths.index(path) + 1]
        detail = "\n".join(getattr(cancelled, "__notes__", ()))
        if cancel_at != "read":
            assert cancelled.__cause__ is failure
            assert "fixture response read failed" in detail
        if cancel_at != "response":
            assert "fixture response close failed" in detail
        if cancel_at != "connection":
            assert "fixture connection close failed" in detail


def test_legacy_404_keeps_fallback_and_records_secondary_cleanup_failures(transport_faults):
    state = transport_faults
    state.response_errors["/server_info"] = OSError("fixture response close failed")
    state.connection_errors["/server_info"] = RuntimeError("fixture connection close failed")
    with service_fixture("sglang-legacy") as service:
        result = observe(service)
        assert_resources_closed(state)
        assert service.seen == service.paths
        assert result["probe"]["status"] == "available"
        assert result["observed"]["configuration"]["source"] == "/get_server_info"
        first = result["probe"]["requests"][0]
        assert first["http_status"] == 404
        assert "GET /server_info returned HTTP 404" in first["error"]
        assert "fixture response close failed" in first["error"]
        assert "fixture connection close failed" in first["error"]


def test_cleanup_cancellation_after_404_does_not_start_fallback(transport_faults):
    state = transport_faults
    cancelled = KeyboardInterrupt("fixture legacy cleanup interrupted")
    state.response_errors["/server_info"] = cancelled
    state.connection_errors["/server_info"] = RuntimeError("fixture connection close failed")
    with service_fixture("sglang-legacy") as service:
        with pytest.raises(KeyboardInterrupt, match="fixture legacy cleanup interrupted") as caught:
            observe(service)
        assert caught.value is cancelled
        assert_resources_closed(state)
        assert service.seen == ["/server_info"]
        detail = "\n".join(getattr(cancelled, "__notes__", ()))
        assert "HTTP 404" in detail
        assert "fixture connection close failed" in detail


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cancelled", [False, True])
def test_cli_publishes_probe_cleanup_diagnostics_and_preserves_cancellation(
    transport_faults, mode, cancelled, tmp_path, monkeypatch,
):
    state = transport_faults
    monkeypatch.setattr("acprof.cli.streaming_load.MeasurementLock", nullcontext)
    monkeypatch.setattr("acprof.cli.streaming_load.host_identity", lambda _root: {})
    failure = KeyboardInterrupt("fixture response read failed") if cancelled else TimeoutError("fixture response read failed")
    with service_fixture(mode) as service:
        state.read_errors[service.primary] = failure
        state.response_errors[service.primary] = OSError("fixture response close failed")
        state.connection_errors[service.primary] = RuntimeError("fixture connection close failed")
        output = tmp_path / "load"
        arguments = [
            "--base-url", service.origin, "--protocol", "openai-chat", "--model", "fixture",
            "--prompt", "hello", "--requests", "1", "--warmup", "1",
            "--service-probe", service.kind, "--output-dir", str(output),
        ]
        if cancelled:
            with pytest.raises(KeyboardInterrupt, match="fixture response read failed") as caught:
                load_cli.main(arguments)
            assert caught.value is failure
        else:
            assert load_cli.main(arguments) == 0
        assert_resources_closed(state)
        source = output / "load.json"
        if cancelled:
            # Probe cancellation has diagnostics but no measured capacity evidence.
            with pytest.raises(ValueError, match="no stream_load_result"):
                read_load_report(source)
        else:
            read_load_report(source)
        report = json.loads(source.read_text())
        if cancelled:
            assert report["status"] == "cancelled"
            assert report["successful"] is False
            assert "warmup" not in report
            assert "result" not in report
            assert service.ports == []
            detail = report["error"]
        else:
            assert report["successful"]
            assert report["result"]["counts"]["succeeded"] == 1
            assert report["service_identity"]["probe"]["status"] == "partial"
            assert len(service.ports) == 2
            detail = "\n".join(report["service_identity"]["probe"]["errors"].values())
        assert "fixture response read failed" in detail
        assert "fixture response close failed" in detail
        assert "fixture connection close failed" in detail
