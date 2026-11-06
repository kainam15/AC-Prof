"""Explicit vLLM engine observations never replace served identity or declarations."""
from __future__ import annotations

import hashlib
import http.client
import json
from contextlib import nullcontext
from typing import Any

import pytest

from acprof.cli import load as load_cli
from acprof.host.measurement.service_identity import probe_service
from tests.execution.operations.test_sglang_service_identity import server as identity_http_server

__all__ = ["identity_http_server"]

CONFIG_PATH = "/server_info?config_format=json"


@pytest.fixture(name="server")
def vllm_configuration_server(identity_http_server):
    server = identity_http_server
    server.info = {
        "vllm_config": {
            "model_config": {"model": "/models/base", "revision": "main", "tokenizer": "/models/tokenizer",
                             "tokenizer_revision": "tokenizer-ref", "dtype": "torch.bfloat16", "quantization": None,
                             "max_model_len": 8192, "hf_token": "discard-hf-token", "hf_config": {"private": "discard-hf-config"}},
            "parallel_config": {"tensor_parallel_size": 2, "pipeline_parallel_size": 1,
                                "data_parallel_size": 1, "distributed_executor_backend": "mp"},
            "scheduler_config": {"max_num_seqs": 32, "max_num_batched_tokens": 4096, "enable_chunked_prefill": True},
            "cache_config": {"cache_dtype": "fp8", "block_size": 16, "enable_prefix_caching": False,
                             "gpu_memory_utilization": 0.9},
            "kv_transfer_config": {"private": "discard-transfer-config"},
        },
        "vllm_env": {"VLLM_API_KEY": "discard-api-key"}, "system_env": {"private": "discard-system-env"},
    }
    server.responses.clear()
    server.responses.update({"/version": (200, {"version": "0.11.2"}),
                             "/v1/models": (200, server.models), CONFIG_PATH: (200, server.info)})
    return server


def declared(**overrides) -> dict[str, Any]:
    return {"backend": "vllm", "version": "0.11.2", "model": "fixture", "revision": "declared-revision", **overrides}


def probe(server, **overrides):
    return probe_service(server.origin, declared(**overrides), "vllm-config",
                         headers={"Authorization": "Bearer fixture-secret"}, timeout_s=2)


def test_explicit_vllm_configuration_probe_keeps_only_named_fields(server):
    result = probe(server)
    assert result["schema_version"] == 1
    assert result["probe"]["kind"] == "vllm-config"
    assert result["probe"]["status"] == "available"
    observed = result["observed"]
    assert observed["backend"] == "vllm"
    assert observed["model"] == "fixture"
    assert observed["root"] == "served-alias"
    assert observed["max_model_len"] == 2048
    assert observed["revision"] is None
    assert result["matches"] == {"backend": True, "version": True, "model": True, "revision": None}
    config = observed["configuration"]
    assert config["source"] == CONFIG_PATH
    assert config["scope"] == "reported_engine_configuration"
    assert config["values"] == {
        "model_config.model": "/models/base", "model_config.revision": "main",
        "model_config.tokenizer": "/models/tokenizer", "model_config.tokenizer_revision": "tokenizer-ref",
        "model_config.dtype": "torch.bfloat16", "model_config.quantization": None,
        "model_config.max_model_len": 8192, "parallel_config.tensor_parallel_size": 2,
        "parallel_config.pipeline_parallel_size": 1, "parallel_config.data_parallel_size": 1,
        "parallel_config.distributed_executor_backend": "mp", "scheduler_config.max_num_seqs": 32,
        "scheduler_config.max_num_batched_tokens": 4096, "scheduler_config.enable_chunked_prefill": True,
        "cache_config.cache_dtype": "fp8", "cache_config.block_size": 16,
        "cache_config.enable_prefix_caching": False, "cache_config.gpu_memory_utilization": 0.9,
    }
    assert server.calls == [(path, "Bearer fixture-secret", "close") for path in ("/version", "/v1/models", CONFIG_PATH)]
    assert [(attempt["path"], attempt["http_status"]) for attempt in result["probe"]["requests"]] == [
        ("/version", 200), ("/v1/models", 200), (CONFIG_PATH, 200),
    ]
    serialized = json.dumps(result)
    assert "discard-" not in serialized
    assert "fixture-secret" not in serialized
    assert "Authorization" not in serialized


def test_basic_vllm_probe_does_not_request_development_configuration(server):
    result = probe_service(server.origin, declared(), "vllm", timeout_s=2)
    assert result["probe"]["status"] == "available"
    assert [path for path, *_ in server.calls] == ["/version", "/v1/models"]
    assert "configuration" not in result["observed"]


@pytest.mark.parametrize(("status", "body", "message"), [
    (404, {}, "HTTP 404"), (401, {}, "HTTP 401"), (403, {}, "HTTP 403"), (503, {}, "HTTP 503"),
    (302, {}, "HTTP 302"), (200, b"{broken", "JSONDecodeError"), (200, [], "JSON object"),
    (200, b"x" * (1024 * 1024 + 1), "1 MiB"),
    (200, {"vllm_config": "VllmConfig(model='do not evaluate')"}, "vllm_config"),
    (200, {"vllm_config": None}, "vllm_config"), (200, {}, "vllm_config"),
    (200, {"vllm_config": {"model_config": {"model": "\ud800"}}}, "isolated surrogate"),
], ids=["not-found", "unauthorized", "forbidden", "unavailable", "redirect", "malformed-json",
        "non-object", "oversize", "text-config", "null-config", "missing-config", "surrogate"])
def test_configuration_errors_are_visible_without_fallback_or_loss_of_identity(server, status, body, message):
    server.responses[CONFIG_PATH] = (status, body)
    result = probe(server)
    assert result["probe"]["status"] == "partial"
    assert message in result["probe"]["errors"]["server_info"]
    assert result["observed"]["version"] == "0.11.2"
    assert result["observed"]["model"] == "fixture"
    assert result["observed"]["configuration"] is None
    assert [path for path, *_ in server.calls] == ["/version", "/v1/models", CONFIG_PATH]


@pytest.mark.parametrize(("section", "field", "value"), [
    ("model_config", "dtype", " "), ("model_config", "model", 7),
    ("parallel_config", "tensor_parallel_size", True), ("parallel_config", "pipeline_parallel_size", 0),
    ("parallel_config", "data_parallel_size", 1.5), ("scheduler_config", "max_num_seqs", "32"),
    ("scheduler_config", "enable_chunked_prefill", 1), ("cache_config", "enable_prefix_caching", "false"),
    ("cache_config", "gpu_memory_utilization", True), ("cache_config", "gpu_memory_utilization", 0),
    ("cache_config", "gpu_memory_utilization", 1.1), ("cache_config", "gpu_memory_utilization", "0.9"),
])
def test_invalid_configuration_values_stay_unknown_with_field_diagnostics(server, section, field, value):
    server.info["vllm_config"][section][field] = value
    result = probe(server)
    assert result["probe"]["status"] == "partial"
    assert result["observed"]["configuration"]["values"][f"{section}.{field}"] is None
    assert f"configuration.{section}.{field}" in result["probe"]["errors"]
    assert result["observed"]["configuration"]["values"]["model_config.max_model_len"] == 8192


def test_missing_and_null_fields_are_unknown_without_inventing_defaults(server):
    server.info["vllm_config"] = {"model_config": None, "cache_config": {"enable_prefix_caching": None}}
    result = probe(server)
    assert result["probe"]["status"] == "available"
    assert all(value is None for value in result["observed"]["configuration"]["values"].values())


def test_wrong_container_shape_is_diagnostic_and_preserves_other_sections(server):
    server.info["vllm_config"]["parallel_config"] = ["not an object"]
    result = probe(server)
    assert result["probe"]["status"] == "partial"
    assert "configuration.parallel_config" in result["probe"]["errors"]
    assert result["observed"]["configuration"]["values"]["parallel_config.tensor_parallel_size"] is None
    assert result["observed"]["configuration"]["values"]["model_config.dtype"] == "torch.bfloat16"


@pytest.mark.parametrize("sweep", [False, True])
def test_cli_observes_configuration_before_warmup_and_every_load_point(server, tmp_path, monkeypatch, sweep):
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
                          "--prompt", "hello", "--requests", "3", "--warmup", "1", "--service-probe", "vllm-config",
                          "--output-dir", str(output), *extra]) == 0
    assert [count for _token, count in calls] == ([3, 3, 6, 6] if sweep else [3, 3])
    paths = [output / "points/0001/load.json", output / "points/0002/load.json"] if sweep else [output / "load.json"]
    for path in paths:
        report = json.loads(path.read_text())
        assert report["service_identity"]["observed"]["configuration"]["values"]["model_config.dtype"] == "torch.bfloat16"
        assert report["identity"]["service"]["probe"] == "vllm-config"
        assert "configuration" not in report["identity"]
        assert report["identity_sha256"] == hashlib.sha256(json.dumps(report["identity"], sort_keys=True).encode()).hexdigest()


def test_dry_run_never_accesses_configuration_or_creates_output(server, tmp_path):
    output = tmp_path / "uncreated"
    assert load_cli.main(["--base-url", server.origin, "--protocol", "openai-chat", "--model", "fixture",
                          "--prompt", "hello", "--requests", "3", "--warmup", "1", "--service-probe", "vllm-config",
                          "--scenario", "concurrent", "--sweep-concurrency", "1,2", "--dry-run",
                          "--output-dir", str(output)]) == 0
    assert server.calls == []
    assert not output.exists()


def test_configuration_survives_missing_identity_endpoints_without_filling_backend_or_revision(server):
    server.responses["/version"] = (404, {})
    server.responses["/v1/models"] = (503, {})
    result = probe(server)
    assert result["probe"]["status"] == "partial"
    observed = result["observed"]
    assert all(observed[name] is None for name in ("backend", "version", "model", "revision", "models"))
    assert observed["configuration"]["values"]["model_config.model"] == "/models/base"
    assert all(value is None for value in result["matches"].values())


def test_all_unavailable_endpoints_keep_declared_identity_and_request_diagnostics(server):
    for path in ("/version", "/v1/models", CONFIG_PATH):
        server.responses[path] = (503, {})
    result = probe(server)
    assert result["probe"]["status"] == "unavailable"
    assert result["observed"] is None
    assert result["declared"] == declared()
    assert set(result["probe"]["errors"]) == {"version", "models", "server_info"}
    assert [attempt["http_status"] for attempt in result["probe"]["requests"]] == [503, 503, 503]


def test_engine_model_and_revision_do_not_satisfy_missing_served_identity(server):
    server.models["data"][0]["id"] = "other"
    server.info["vllm_config"]["model_config"].update(model="fixture", revision="declared-revision")
    result = probe(server)
    assert result["observed"]["model"] is None
    assert result["observed"]["revision"] is None
    assert result["matches"]["model"] is False
    assert result["matches"]["revision"] is None
    assert result["mismatches"] == ["model"]


@pytest.mark.parametrize("value", [0.001, 1, 1.0])
def test_gpu_memory_fraction_preserves_valid_boundary_values(server, value):
    server.info["vllm_config"]["cache_config"]["gpu_memory_utilization"] = value
    result = probe(server)
    assert result["probe"]["status"] == "available"
    assert result["observed"]["configuration"]["values"]["cache_config.gpu_memory_utilization"] == value


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_configuration_json_is_rejected_before_field_extraction(server, value):
    server.info["vllm_config"]["cache_config"]["gpu_memory_utilization"] = value
    result = probe(server)
    assert result["probe"]["status"] == "partial"
    assert result["observed"]["configuration"] is None
    assert "non-finite" in result["probe"]["errors"]["server_info"]


@pytest.mark.parametrize("cancelled", [False, True])
def test_configuration_transport_timeout_is_diagnostic_and_cancellation_propagates(server, monkeypatch, cancelled):
    original = http.client.HTTPConnection
    error = KeyboardInterrupt("configuration cancelled") if cancelled else TimeoutError("configuration timed out")
    closed = []

    class Connection(original):
        requested_path = ""

        def request(self, method, url, *args, **kwargs):
            self.requested_path = url
            return super().request(method, url, *args, **kwargs)

        def getresponse(self):
            if self.requested_path == CONFIG_PATH:
                raise error
            return super().getresponse()

        def close(self):
            closed.append(self.requested_path)
            return super().close()

    monkeypatch.setattr(http.client, "HTTPConnection", Connection)
    if cancelled:
        with pytest.raises(KeyboardInterrupt, match="configuration cancelled") as caught:
            probe(server)
        assert caught.value is error
    else:
        result = probe(server)
        assert result["probe"]["status"] == "partial"
        assert "TimeoutError" in result["probe"]["errors"]["server_info"]
        assert result["observed"]["configuration"] is None
        assert result["probe"]["requests"][-1]["http_status"] is None
    assert CONFIG_PATH in closed


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_is_rejected_before_configuration_network_access(server, timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        probe_service(server.origin, declared(), "vllm-config", timeout_s=timeout)
    assert server.calls == []


@pytest.mark.parametrize("invalid", [False, True])
def test_point_html_retains_configuration_scope_and_escapes_values(server, tmp_path, invalid):
    from acprof.plotting.load_report import read_load_report, write_load_report
    from tests.analysis.reporting.test_load_report import experiment

    server.info["vllm_config"]["model_config"]["model"] = "<script>engine</script>"
    if invalid:
        server.info["vllm_config"]["cache_config"]["gpu_memory_utilization"] = 2
    payload = experiment()
    payload["identity"]["service"]["probe"] = "vllm-config"
    payload["service_identity"] = probe(server)
    payload["identity_sha256"] = hashlib.sha256(json.dumps(payload["identity"], sort_keys=True).encode()).hexdigest()
    source = tmp_path / "load.json"
    source.write_text(json.dumps(payload))
    read_load_report(source)
    html = write_load_report(source, tmp_path / "capacity.html").read_text()
    assert "reported_engine_configuration" in html
    assert CONFIG_PATH in html
    assert "cache_config.enable_prefix_caching" in html
    assert "torch.bfloat16" in html
    assert "<script>engine</script>" not in html
    assert "&lt;script&gt;engine&lt;/script&gt;" in html
    assert "discard-" not in html
    if invalid:
        assert "gpu_memory_utilization must be number in (0, 1] or null" in html


@pytest.mark.parametrize("unavailable", [False, True])
def test_sweep_groups_changed_or_unavailable_engine_configuration_separately(server, tmp_path, unavailable):
    from acprof.analysis.load_sweep import read_load_sweep
    from tests.support.load_sweep_fixtures import change_point, make_sweep

    root = make_sweep(tmp_path / "sweep")
    for index in range(1, 4):
        if unavailable:
            server.responses[CONFIG_PATH] = (404, {}) if index == 2 else (200, server.info)
        else:
            server.info["vllm_config"]["cache_config"]["enable_prefix_caching"] = index == 2
        observed = probe(server)
        change_point(root, index, lambda payload: (
            payload["identity"]["service"].update(probe="vllm-config"),
            payload.update(service_identity=observed),
        ))
    data = read_load_sweep(root).payload
    assert data["valid_points"] == 3
    assert len(data["conditions"]) == 2
    points = data["points"]
    assert points[0]["condition_id"] == points[2]["condition_id"]
    assert points[1]["condition_id"] != points[0]["condition_id"]


def test_source_replay_rejects_configuration_probe_before_reading_or_writing(tmp_path, capsys):
    output = tmp_path / "uncreated"
    with pytest.raises(SystemExit) as caught:
        load_cli.main([str(tmp_path / "missing-source"), "--gpu", "off", "--service-probe", "vllm-config",
                       "--output-dir", str(output)])
    assert caught.value.code == 2
    assert "require --base-url" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("configuration_status", [200, 404])
def test_real_http_sweep_with_configuration_produces_readable_capacity_reports(server, tmp_path, monkeypatch, configuration_status):
    from acprof.analysis.load_sweep import read_load_sweep
    from acprof.cli.main import main
    from tests.execution.operations.test_streaming_load import streaming_server

    server.responses[CONFIG_PATH] = (configuration_status, server.info if configuration_status == 200 else {})
    output = tmp_path / "sweep"
    monkeypatch.setattr("acprof.cli.streaming_load.MeasurementLock", nullcontext)
    monkeypatch.setattr("acprof.cli.streaming_load.host_identity", lambda _root: {"hostname": "fixture"})
    monkeypatch.setattr("acprof.host.run_state.offline_analysis_lock", nullcontext)
    with streaming_server(metadata=server.responses) as (origin, _ports, paths, _bodies, _ready):
        assert main(["load", "--base-url", origin, "--protocol", "openai-chat", "--model", "fixture",
                     "--prompt", "hello", "--requests", "2", "--warmup", "1", "--service-probe", "vllm-config",
                     "--scenario", "concurrent", "--sweep-concurrency", "1,2", "--output-dir", str(output)]) == 0
    assert paths == ["/version", "/v1/models", CONFIG_PATH, *(["/v1/chat/completions"] * 3)] * 2
    for index in (1, 2):
        payload = json.loads((output / f"points/{index:04d}/load.json").read_text())
        assert payload["warmup"]["counts"]["succeeded"] == 1
        assert payload["result"]["counts"]["succeeded"] == 2
        assert payload["result"]["capacity"]["token_throughput"]["output"]["tokens"] == 6
        assert payload["service_identity"]["probe"]["status"] == ("available" if configuration_status == 200 else "partial")
    analysis = read_load_sweep(output)
    assert analysis.payload["valid_points"] == 2
    assert len(analysis.payload["conditions"]) == 1
    assert main(["report", str(output)]) == 0
    assert (output / "load_sweep_report.html").is_file()
