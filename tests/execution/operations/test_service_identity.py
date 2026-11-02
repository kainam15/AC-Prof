"""Bounded vLLM identity probes retain declarations and never persist credentials."""
from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from acprof.host.measurement.service_identity import probe_vllm


@contextmanager
def identity_server(
    *, version_status=200, models_status=200, invalid_models=False, model_id="fixture", duplicate=False,
):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization")))
            if self.path == "/version":
                status, payload = version_status, {"version": "0.11.2", "private": "discard"}
            elif self.path == "/v1/models":
                status = models_status
                model = {
                    "id": model_id, "root": "/models/fixture", "max_model_len": 4096,
                    "revision": "root-is-not-a-revision", "private": "discard",
                }
                payload = {"data": [model, model] if duplicate else [model]}
            else:
                status, payload = 404, {}
            body = b"{invalid" if invalid_models and self.path == "/v1/models" else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", requests
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def declared(**overrides):
    value = {"backend": "vllm", "version": "0.11.2", "model": "fixture", "revision": "user-revision"}
    value.update(overrides)
    return value


def test_vllm_probe_uses_exact_model_and_whitelisted_fields_with_reused_authorization():
    with identity_server() as (origin, requests):
        result = probe_vllm(
            origin, declared(), headers={"Authorization": "Bearer fixture-secret"}, timeout_s=2,
        )
    assert result["probe"]["status"] == "available"
    assert result["observed"]["version"] == "0.11.2"
    assert result["observed"]["model"] == "fixture"
    assert result["observed"]["root"] == "/models/fixture"
    assert result["observed"]["max_model_len"] == 4096
    assert result["observed"]["revision"] is None
    assert result["matches"] == {"backend": True, "version": True, "model": True, "revision": None}
    assert result["mismatches"] == []
    assert requests == [
        ("/version", "Bearer fixture-secret"), ("/v1/models", "Bearer fixture-secret"),
    ]
    serialized = json.dumps(result)
    assert "fixture-secret" not in serialized
    assert "Authorization" not in serialized
    assert "discard" not in serialized
    assert "root-is-not-a-revision" not in serialized


def test_vllm_probe_preserves_declaration_and_observation_on_mismatch():
    with identity_server(model_id="fixture-other") as (origin, _):
        result = probe_vllm(
            origin, declared(backend="other", version="0.10", model="fixture"), timeout_s=2,
        )
    assert result["declared"]["backend"] == "other"
    assert result["declared"]["version"] == "0.10"
    assert result["declared"]["model"] == "fixture"
    assert result["observed"]["backend"] == "vllm"
    assert result["observed"]["version"] == "0.11.2"
    assert result["observed"]["model"] is None
    assert result["observed"]["models"][0]["id"] == "fixture-other"
    assert result["mismatches"] == ["backend", "version", "model"]


def test_vllm_probe_rejects_ambiguous_duplicate_model_ids():
    with identity_server(duplicate=True) as (origin, _):
        result = probe_vllm(origin, declared(), timeout_s=2)
    assert result["observed"]["model"] is None
    assert result["observed"]["root"] is None
    assert result["matches"]["model"] is False
    assert result["mismatches"] == ["model"]


def test_vllm_probe_404_is_partial_and_does_not_invent_revision():
    with identity_server(version_status=404) as (origin, _):
        result = probe_vllm(origin, declared(), timeout_s=2)
    assert result["probe"]["status"] == "partial"
    assert "HTTP 404" in result["probe"]["errors"]["version"]
    assert result["observed"]["version"] is None
    assert result["observed"]["model"] == "fixture"
    assert result["observed"]["revision"] is None


def test_vllm_probe_invalid_json_is_partial():
    with identity_server(invalid_models=True) as (origin, _):
        result = probe_vllm(origin, declared(), timeout_s=2)
    assert result["probe"]["status"] == "partial"
    assert result["observed"]["version"] == "0.11.2"
    assert result["observed"]["models"] is None
    assert result["matches"]["model"] is None


def test_vllm_probe_unavailable_when_both_endpoints_fail():
    with identity_server(version_status=404, models_status=404) as (origin, _):
        result = probe_vllm(origin, declared(), timeout_s=2)
    assert result["probe"]["status"] == "unavailable"
    assert result["observed"] is None
    assert result["matches"]["model"] is None
