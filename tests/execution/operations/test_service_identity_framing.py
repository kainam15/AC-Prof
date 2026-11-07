"""Service observations require complete HTTP framing, even for valid JSON prefixes."""
import hashlib
import json
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

from acprof.cli import load as load_cli
from acprof.host.measurement.service_identity import probe_service
from acprof.plotting.load_report import read_load_report
from tests.execution.operations.test_streaming_load import streaming_server

CONFIG_PATH = "/server_info?config_format=json"
MODES = ("vllm", "vllm-config", "sglang", "sglang-legacy")


@contextmanager
def service_fixture(mode):
    kind = "sglang" if mode == "sglang-legacy" else mode
    backend = "sglang" if kind == "sglang" else "vllm"
    declared = {"backend": backend, "version": "fixture-version", "model": "fixture", "revision": "declared-ref"}
    models = {"data": [{"id": "fixture", "root": "/models/fixture", "max_model_len": 2048}]}
    metadata: dict[str, tuple[int, dict[str, Any]]]
    if kind == "sglang":
        primary = "/get_server_info" if mode == "sglang-legacy" else "/server_info"
        metadata = {primary: (200, {"version": "fixture-version", "dtype": "float16"}),
                    "/v1/models": (200, models)}
        paths = [primary, "/v1/models"]
        if mode == "sglang-legacy":
            paths.insert(0, "/server_info")
    else:
        primary = "/version"
        metadata = {primary: (200, {"version": "fixture-version"}), "/v1/models": (200, models)}
        paths = [primary, "/v1/models"]
        if kind == "vllm-config":
            metadata[CONFIG_PATH] = (200, {"vllm_config": {"model_config": {"dtype": "float16"}}})
            paths.append(CONFIG_PATH)
    raw_metadata: dict[str, bytes] = {}
    with streaming_server(metadata=metadata, raw_metadata=raw_metadata) as (origin, ports, seen, bodies, _ready):
        yield SimpleNamespace(kind=kind, declared=declared, metadata=metadata, primary=primary,
                              paths=paths, raw=raw_metadata, origin=origin, ports=ports, seen=seen, bodies=bodies)


def framed_json(payload, framing):
    body = json.dumps(payload).encode()
    headers = b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: application/json\r\n"
    if framing in {"chunked", "incomplete-chunked"}:
        headers += b"Transfer-Encoding: chunked\r\n"
        body = f"{len(body):x}\r\n".encode() + body + b"\r\n"
        if framing == "chunked":
            body += b"0\r\n\r\n"
    elif framing != "connection-close":
        length = len(body) + (1 if framing == "short-length" else 0)
        headers += f"Content-Length: {length}\r\n".encode()
    return headers + b"\r\n" + body


def observe(service):
    return probe_service(service.origin, service.declared, service.kind, timeout_s=2)


def assert_attempts(service, result):
    assert service.seen == service.paths
    assert result["declared"] == service.declared
    if service.kind != "vllm":
        assert [(row["path"], row["http_status"]) for row in result["probe"]["requests"]] == [
            (path, 200 if path in service.metadata else 404) for path in service.paths
        ]


@pytest.mark.parametrize(("mode", "path", "error_key"), [
    ("vllm", "/version", "version"), ("vllm", "/v1/models", "models"),
    ("vllm-config", "/version", "version"), ("vllm-config", "/v1/models", "models"),
    ("vllm-config", CONFIG_PATH, "server_info"),
    ("sglang", "/server_info", "server_info"), ("sglang", "/v1/models", "models"),
    ("sglang-legacy", "/get_server_info", "server_info"),
])
def test_incomplete_content_length_rejects_only_the_affected_observation(mode, path, error_key):
    with service_fixture(mode) as service:
        service.raw[path] = framed_json(service.metadata[path][1], "short-length")
        result = observe(service)
        assert_attempts(service, result)
    assert result["probe"]["status"] == "partial"
    assert set(result["probe"]["errors"]) == {error_key}
    assert f"GET {path} ended before Content-Length" in result["probe"]["errors"][error_key]
    observed = result["observed"]
    if error_key == "models":
        assert observed["models"] is observed["model"] is result["matches"]["model"] is None
        assert observed["version"] == "fixture-version"
    elif error_key == "version" or service.kind == "sglang":
        assert observed["version"] is observed["backend"] is result["matches"]["backend"] is None
        assert observed["model"] == "fixture"
    if error_key == "server_info":
        assert observed["configuration"] is None
    if service.kind != "vllm":
        failed = next(row for row in result["probe"]["requests"] if row["path"] == path)
        assert failed["error"] == result["probe"]["errors"][error_key]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("framing", ["content-length", "connection-close", "chunked"])
def test_complete_framing_preserves_identity_and_configuration(mode, framing):
    with service_fixture(mode) as service:
        for path, (_status, payload) in service.metadata.items():
            service.raw[path] = framed_json(payload, framing)
        result = observe(service)
        assert_attempts(service, result)
    assert result["probe"]["status"] == "available"
    assert result["probe"]["errors"] == {}
    assert result["observed"]["backend"] == service.declared["backend"]
    assert result["observed"]["version"] == "fixture-version"
    assert result["observed"]["model"] == "fixture"
    assert result["observed"]["revision"] is None
    assert result["matches"] == {"backend": True, "version": True, "model": True, "revision": None}
    if service.kind != "vllm":
        name = "model_config.dtype" if service.kind == "vllm-config" else "dtype"
        assert result["observed"]["configuration"]["values"][name] == "float16"


@pytest.mark.parametrize("mode", MODES)
def test_no_incomplete_endpoint_can_establish_observed_identity(mode):
    with service_fixture(mode) as service:
        for path, (_status, payload) in service.metadata.items():
            service.raw[path] = framed_json(payload, "short-length")
        result = observe(service)
        assert_attempts(service, result)
    assert result["probe"]["status"] == "unavailable"
    assert result["observed"] is None
    assert all(value is None for value in result["matches"].values())
    assert len(result["probe"]["errors"]) == len(service.metadata)
    assert all("Content-Length" in message for message in result["probe"]["errors"].values())


@pytest.mark.parametrize("mode", MODES)
def test_missing_final_chunk_is_rejected_despite_complete_json(mode):
    with service_fixture(mode) as service:
        service.raw[service.primary] = framed_json(service.metadata[service.primary][1], "incomplete-chunked")
        result = observe(service)
        assert_attempts(service, result)
    assert result["probe"]["status"] == "partial"
    assert any("IncompleteRead" in message for message in result["probe"]["errors"].values())
    assert result["observed"]["version"] is None
    assert result["observed"]["model"] == "fixture"


@pytest.mark.parametrize("mode", MODES)
def test_cli_persists_framing_error_without_failing_load_and_recovers_on_next_probe(mode, tmp_path, monkeypatch):
    monkeypatch.setattr("acprof.cli.streaming_load.MeasurementLock", nullcontext)
    monkeypatch.setattr("acprof.cli.streaming_load.host_identity", lambda _root: {})
    with service_fixture(mode) as service:
        service.raw[service.primary] = framed_json(service.metadata[service.primary][1], "short-length")
        reports = []
        for phase, status in (("incomplete", "partial"), ("recovered", "available")):
            output = tmp_path / phase
            assert load_cli.main([
                "--base-url", service.origin, "--protocol", "openai-chat", "--model", "fixture",
                "--prompt", "hello", "--requests", "1", "--warmup", "1",
                "--service-probe", service.kind, "--output-dir", str(output),
            ]) == 0
            source = output / "load.json"
            read_load_report(source)
            report = json.loads(source.read_text())
            reports.append(report)
            assert report["successful"]
            assert report["warmup"]["successful"]
            assert report["result"]["successful"]
            assert report["result"]["counts"]["succeeded"] == 1
            assert report["service_identity"]["probe"]["status"] == status
            if status == "partial":
                assert report["service_identity"]["observed"]["version"] is None
                assert any("Content-Length" in message for message in report["service_identity"]["probe"]["errors"].values())
            else:
                assert report["service_identity"]["observed"]["version"] == "fixture-version"
            assert report["identity_sha256"] == hashlib.sha256(
                json.dumps(report["identity"], sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            service.raw.clear()
        assert reports[0]["identity_sha256"] == reports[1]["identity_sha256"]
        assert service.seen == (service.paths + ["/v1/chat/completions"] * 2) * 2
        assert len(service.ports) == len(service.bodies) == 4
