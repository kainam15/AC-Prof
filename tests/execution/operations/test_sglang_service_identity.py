"""SGLang observations stay bounded, explicit and separate from launch declarations."""
from __future__ import annotations

import hashlib
import json
import threading
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import pytest

from acprof.cli import load as load_cli
from acprof.host.measurement.service_identity import initial_service_identity, probe_service


@pytest.fixture
def server():
    info = {
        "version": "0.5.5", "model_path": "/models/startup", "served_model_name": "startup-name",
        "revision": "launch-revision", "dtype": "auto", "kv_cache_dtype": "fp8_e4m3",
        "quantization": None, "tp_size": 2, "pp_size": 1, "dp_size": 1,
        "context_length": 4096, "max_running_requests": 32, "max_total_tokens": 65536,
        "max_prefill_tokens": 16384, "schedule_policy": "fcfs", "disable_radix_cache": False,
        "api_key": "discard-api-key", "launch_command": "discard-launch-command",
        "internal_states": [{"private": "discard-internal-state"}],
    }
    models = {"data": [{"id": "fixture", "root": "served-alias", "max_model_len": 2048}]}
    responses: dict[str, tuple[int, Any]] = {"/server_info": (200, info), "/v1/models": (200, models)}
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def get(self) -> None:
            calls.append((self.path, self.headers.get("Authorization"), self.headers.get("Connection")))
            status, payload = responses.get(self.path, (404, {}))
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # The bounded reader may close before a deliberately oversized fixture is sent.

    setattr(Handler, "do_GET", Handler.get)

    def handler_factory(request: Any, address: Any, httpd: Any) -> BaseHTTPRequestHandler:
        return Handler(request, address, httpd)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_factory)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield SimpleNamespace(origin=f"http://127.0.0.1:{httpd.server_port}", info=info,
                              models=models, responses=responses, calls=calls)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def declared(**overrides) -> dict[str, Any]:
    return {"backend": "sglang", "version": "0.5.5", "model": "fixture",
            "revision": "declared-revision", **overrides}


def probe(server, **overrides) -> dict[str, Any]:
    return probe_service(server.origin, declared(**overrides), "sglang",
                         headers={"Authorization": "Bearer fixture-secret"}, timeout_s=2)


def test_sglang_observation_uses_exact_model_and_retains_startup_configuration(server):
    result = probe(server)
    assert result["schema_version"] == 1
    assert result["probe"]["status"] == "available"
    assert result["matches"] == {"backend": True, "version": True, "model": True, "revision": None}
    observed = result["observed"]
    assert observed["backend"] == "sglang"
    assert observed["version"] == "0.5.5"
    assert observed["model"] == "fixture"
    assert observed["root"] == "served-alias"
    assert observed["max_model_len"] == 2048
    assert observed["revision"] is None
    configuration = observed["configuration"]
    assert configuration["source"] == "/server_info"
    assert configuration["scope"] == "reported_startup_configuration"
    values = configuration["values"]
    assert values["dtype"] == "auto"
    assert values["kv_cache_dtype"] == "fp8_e4m3"
    assert values["tp_size"] == 2
    assert values["disable_radix_cache"] is False
    assert values["revision"] == "launch-revision"
    assert values["served_model_name"] == "startup-name"
    assert values["attn_dp_size"] is None
    assert values["quantization"] is None
    assert server.calls == [("/server_info", "Bearer fixture-secret", "close"),
                            ("/v1/models", "Bearer fixture-secret", "close")]
    serialized = json.dumps(result)
    assert "fixture-secret" not in serialized
    assert "discard-" not in serialized
    assert "Authorization" not in serialized


def test_sglang_legacy_route_is_used_only_after_recorded_404(server):
    server.responses["/server_info"] = (404, {})
    server.responses["/get_server_info"] = (200, server.info)
    result = probe(server)
    assert result["probe"]["status"] == "available"
    assert result["observed"]["configuration"]["source"] == "/get_server_info"
    assert [(attempt["path"], attempt["http_status"]) for attempt in result["probe"]["requests"]] == [
        ("/server_info", 404), ("/get_server_info", 200), ("/v1/models", 200),
    ]
    assert [call[0] for call in server.calls] == ["/server_info", "/get_server_info", "/v1/models"]


@pytest.mark.parametrize(("status", "body", "message"), [
    (401, {}, "HTTP 401"), (403, {}, "HTTP 403"), (500, {}, "HTTP 500"),
    (302, {}, "HTTP 302"), (200, b"{bad-json", "JSONDecodeError"),
    (200, [], "JSON object"), (200, b"x" * (1024 * 1024 + 1), "1 MiB"),
    (200, {"version": "\ud800"}, "isolated surrogate"), (200, {"version": float("nan")}, "non-finite"),
], ids=["unauthorized", "forbidden", "server-error", "redirect", "invalid-json",
        "non-object", "oversized", "surrogate", "non-finite"])
def test_sglang_info_errors_do_not_silently_use_legacy_route(server, status, body, message):
    server.responses["/server_info"] = (status, body)
    server.responses["/get_server_info"] = (200, server.info)
    result = probe(server)
    assert result["probe"]["status"] == "partial"
    assert message.lower() in result["probe"]["errors"]["server_info"].lower()
    assert result["observed"]["backend"] is None
    assert result["observed"]["version"] is None
    assert result["observed"]["model"] == "fixture"
    assert result["observed"]["configuration"] is None
    assert result["probe"]["requests"][0]["http_status"] == status
    assert [call[0] for call in server.calls] == ["/server_info", "/v1/models"]


@pytest.mark.parametrize(("name", "value"), [
    ("dtype", 16), ("kv_cache_dtype", ""), ("quantization", []), ("tp_size", True),
    ("pp_size", 1.5), ("dp_size", 0), ("context_length", -1),
    ("max_running_requests", "32"), ("disable_radix_cache", 0),
])
def test_sglang_invalid_config_field_keeps_other_observations_and_reports_error(server, name, value):
    server.info[name] = value
    result = probe(server)
    assert result["probe"]["status"] == "partial"
    assert f"configuration.{name}" in result["probe"]["errors"]
    assert result["observed"]["version"] == "0.5.5"
    assert result["observed"]["model"] == "fixture"
    values = result["observed"]["configuration"]["values"]
    assert values[name] is None
    assert values["max_total_tokens"] == 65536


@pytest.mark.parametrize("value", ["", "  ", 123, None])
def test_sglang_invalid_version_cannot_establish_backend(server, value):
    server.info["version"] = value
    result = probe(server)
    assert result["observed"]["backend"] is None
    assert result["observed"]["version"] is None
    assert result["observed"]["configuration"]["values"]["tp_size"] == 2
    assert result["probe"]["status"] == "partial"
    assert "version" in result["probe"]["errors"]


@pytest.mark.parametrize("duplicate", [False, True])
def test_sglang_unmatched_or_duplicate_model_is_not_inferred_from_startup_config(server, duplicate):
    if duplicate:
        server.models["data"].append(dict(server.models["data"][0]))
    else:
        server.models["data"][0]["id"] = "other"
    result = probe(server, backend="other-backend", version="other-version")
    assert result["mismatches"] == ["backend", "version", "model"]
    assert result["observed"]["model"] is None
    assert result["observed"]["root"] is None
    assert result["observed"]["revision"] is None
    assert result["declared"]["model"] == "fixture"


def test_sglang_unavailable_endpoints_preserve_declarations(server):
    server.responses["/server_info"] = (503, {})
    server.responses["/v1/models"] = (503, {})
    result = probe(server)
    assert result["probe"]["status"] == "unavailable"
    assert result["observed"] is None
    assert result["declared"] == declared()
    assert set(result["probe"]["errors"]) == {"server_info", "models"}
    assert all(match is None for match in result["matches"].values())


def test_sglang_initial_identity_does_not_invent_observations():
    result = initial_service_identity(declared(), "sglang")
    assert result["observed"] is None
    assert result["probe"]["status"] == "pending"


@pytest.mark.parametrize("sweep", [False, True])
def test_sglang_cli_probes_before_each_warmup_and_keeps_observations_outside_identity(server, tmp_path, monkeypatch, sweep):
    output = tmp_path / "load"
    calls = []

    def run(*_args, **kwargs):
        calls.append((kwargs["token"], len(server.calls)))
        return {"successful": True, "success_rate": 1.0, "requests": []}

    monkeypatch.setattr("acprof.cli.streaming_load.run_load", run)
    monkeypatch.setattr("acprof.cli.streaming_load.MeasurementLock", nullcontext)
    monkeypatch.setattr("acprof.cli.streaming_load.host_identity", lambda _root: {})
    extra = ["--scenario", "concurrent", "--sweep-concurrency", "1,2"] if sweep else []
    assert load_cli.main(["--base-url", server.origin, "--protocol", "openai-chat", "--model", "fixture",
                          "--prompt", "hello", "--requests", "3", "--warmup", "1", "--service-probe", "sglang",
                          "--output-dir", str(output), *extra]) == 0
    assert [count for _token, count in calls] == ([2, 2, 4, 4] if sweep else [2, 2])
    paths = [output / "points/0001/load.json", output / "points/0002/load.json"] if sweep else [output / "load.json"]
    for path in paths:
        report = json.loads(path.read_text())
        assert report["service_identity"]["observed"]["configuration"]["values"]["dtype"] == "auto"
        assert report["identity"]["service"]["probe"] == "sglang"
        assert "configuration" not in report["identity"]
        assert report["identity_sha256"] == hashlib.sha256(
            json.dumps(report["identity"], sort_keys=True, allow_nan=False).encode()
        ).hexdigest()


@pytest.mark.parametrize("invalid_config", [False, True])
def test_sglang_capacity_html_displays_config_scope_and_escapes_observed_text(server, tmp_path, invalid_config):
    from acprof.plotting.load_report import read_load_report, write_load_report
    from tests.analysis.reporting.test_load_report import experiment

    server.info["model_path"] = "<script>startup</script>"
    if invalid_config:
        server.info["tp_size"] = "not-an-integer"
    identity = probe(server)
    payload: dict[str, Any] = experiment()
    payload["identity"]["service"].update(backend="sglang", version="0.5.5", probe="sglang")
    payload["service_identity"] = identity
    payload["identity_sha256"] = hashlib.sha256(
        json.dumps(payload["identity"], sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    source = tmp_path / "load.json"
    source.write_text(json.dumps(payload))
    read_load_report(source)
    output = write_load_report(source, tmp_path / "capacity.html")
    html = output.read_text()
    assert "reported_startup_configuration" in html
    assert "/server_info" in html
    assert "kv_cache_dtype" in html
    assert "fp8_e4m3" in html
    assert "tp_size" in html
    assert "<script>startup</script>" not in html
    assert "&lt;script&gt;startup&lt;/script&gt;" in html
    if invalid_config:
        assert "tp_size must be positive integer or null" in html


def test_sglang_missing_configuration_fields_remain_unknown(server):
    server.responses["/server_info"] = (200, {"version": "0.5.5"})
    result = probe(server)
    assert result["probe"]["status"] == "available"
    assert all(value is None for value in result["observed"]["configuration"]["values"].values())


def test_sglang_legacy_failure_keeps_both_http_attempts(server):
    server.responses["/server_info"] = (404, {})
    result = probe(server)
    assert result["probe"]["status"] == "partial"
    assert "GET /get_server_info returned HTTP 404" in result["probe"]["errors"]["server_info"]
    assert [(attempt["path"], attempt["http_status"]) for attempt in result["probe"]["requests"]] == [
        ("/server_info", 404), ("/get_server_info", 404), ("/v1/models", 200),
    ]


def test_sglang_invalid_model_metadata_does_not_erase_server_configuration(server):
    server.models["data"][0]["max_model_len"] = True
    result = probe(server)
    assert result["probe"]["status"] == "partial"
    assert result["observed"]["model"] is None
    assert result["observed"]["models"] is None
    assert result["matches"]["model"] is None
    assert result["observed"]["configuration"]["values"]["tp_size"] == 2
    assert "max_model_len" in result["probe"]["errors"]["models"]


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_sglang_invalid_timeout_is_rejected_before_network(server, timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        probe_service(server.origin, declared(), "sglang", timeout_s=timeout)
    assert server.calls == []


@pytest.mark.parametrize("cancelled", [False, True])
def test_sglang_transport_timeout_is_diagnostic_and_cancellation_propagates(server, monkeypatch, cancelled):
    opened, closed = [], []
    error = KeyboardInterrupt("fixture probe cancelled") if cancelled else TimeoutError("fixture probe timed out")

    class Connection:
        def request(self, _method, path, **_kwargs):
            opened.append(path)

        def getresponse(self):
            raise error

        def close(self):
            closed.append(True)

    monkeypatch.setattr("acprof.host.measurement.service_identity.http.client.HTTPConnection",
                        lambda *_args, **_kwargs: Connection())
    if cancelled:
        with pytest.raises(KeyboardInterrupt, match="probe cancelled") as caught:
            probe(server)
        assert caught.value is error
        assert opened == ["/server_info"]
        assert len(closed) == 1
    else:
        result = probe(server)
        assert result["probe"]["status"] == "unavailable"
        assert all("TimeoutError" in message for message in result["probe"]["errors"].values())
        assert opened == ["/server_info", "/v1/models"]
        assert len(closed) == 2
