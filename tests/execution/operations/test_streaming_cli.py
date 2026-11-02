"""Endpoint CLI validation, provenance and interrupted-report persistence."""
from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext

import pytest

from acprof.cli import load as load_cli, streaming_load


@pytest.fixture
def endpoint_cli(tmp_path, monkeypatch):
    calls = []

    def run(url, payload, config, *, token, adapter, headers, slo_thresholds=None, include_capacity=True):
        calls.append((url, payload, config, token, adapter, headers, {
            "slo_thresholds": slo_thresholds, "include_capacity": include_capacity,
        }))
        return {"successful": True, "success_rate": 1.0, "requests": []}

    monkeypatch.setattr(streaming_load, "run_load", run)
    monkeypatch.setattr(streaming_load, "MeasurementLock", nullcontext)
    monkeypatch.setattr(streaming_load, "host_identity", lambda _root: {"hostname": "fixture"})
    output = tmp_path / "load-output"
    base = ["--base-url", "http://127.0.0.1:8999", "--output-dir", str(output)]
    return base, output, calls


def test_prompt_mode_preserves_scheduler_and_generation_seed_and_separates_warmup(endpoint_cli):
    base, output, calls = endpoint_cli
    assert load_cli.main([
        *base, "--protocol", "openai-chat", "--model", "fixture-model", "--prompt", "hello",
        "--warmup", "2", "--requests", "4", "--scenario", "concurrent", "--concurrency", "2",
        "--connections", "reuse", "--seed", "71", "--generation-seed", "99",
        "--max-tokens", "9", "--temperature", "0.25", "--request-timeout", "7",
    ]) == 0
    assert len(calls) == 2
    assert calls[0][2].scenario == "serial"
    assert calls[0][2].connections == "close"
    assert calls[0][2].requests == 2
    assert calls[1][2].concurrency == 2
    assert calls[1][2].requests == 4
    assert calls[1][2].seed == 71
    assert calls[1][2].timeout_seconds == 7
    assert calls[1][1]["seed"] == 99
    assert calls[1][1]["messages"] == [{"role": "user", "content": "hello"}]
    assert calls[1][1]["max_tokens"] == 9
    assert calls[1][1]["temperature"] == 0.25
    assert calls[1][1]["stream"] is True
    assert calls[1][1]["stream_options"]["include_usage"] is True
    assert calls[1][4].endpoint == "/v1/chat/completions"
    report = json.loads((output / "load.json").read_text())
    assert report["successful"] is True
    assert report["status"] == "complete"
    assert "warmup" in report
    assert "result" in report
    assert report["identity"]["model_revision_source"] == "unavailable"
    assert "no GPU energy" in report["scope"]


def test_request_json_preserves_exact_generation_parameters(endpoint_cli, tmp_path):
    base, output, calls = endpoint_cli
    request = tmp_path / "request.json"
    payload = {"model": "fixture-model", "prompt": "answer", "max_tokens": 5,
               "seed": 42, "temperature": 0, "ignore_eos": True}
    request.write_text(json.dumps(payload))
    original = request.read_bytes()
    assert load_cli.main([
        *base, "--protocol", "openai-completions", "--request-json", str(request),
        "--warmup", "0", "--requests", "1", "--seed", "123",
    ]) == 0
    assert len(calls) == 1
    assert request.read_bytes() == original
    assert calls[0][1]["seed"] == 42
    assert calls[0][1]["ignore_eos"] is True
    assert calls[0][2].seed == 123
    assert calls[0][4].endpoint == "/v1/completions"
    report = json.loads((output / "load.json").read_text())
    saved = report["identity"]["request_body"]
    assert report["identity"]["request_body_sha256"] == hashlib.sha256(
        json.dumps(saved, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


@pytest.mark.parametrize("extra", [
    ["source-run"], ["--gpu", "off"], ["--cpu", "2"], ["--mem", "2"],
    ["--input-scale", "4"], ["--capture"], ["--protocol", "predict"],
    ["--warmup", "-1"], ["--request-timeout", "0"], ["--temperature", "nan"],
    ["--max-tokens", "0"], ["--tokenizer-model", "fixture-model"],
    ["--slo-ttft-s", "0"], ["--slo-tpot-s", "nan"], ["--slo-e2e-s", "inf"],
    ["--slo-e2e-s", "-1"], ["--service-backend", " "], ["--service-version", " "],
])
def test_invalid_endpoint_options_fail_before_network_or_output(endpoint_cli, extra):
    base, output, calls = endpoint_cli
    with pytest.raises(SystemExit) as exc:
        load_cli.main([
            *base, "--protocol", "openai-chat", "--model", "fixture-model", "--prompt", "hello",
            *extra,
        ])
    assert exc.value.code == 2
    assert not calls
    assert not output.exists()


@pytest.mark.parametrize("origin", [
    "file:///tmp/server", "http://u:p@example.com", "http://example.com/v1",
    "http://example.com?token=bad", "http://example.com#fragment", "http://example.com:99999",
])
def test_endpoint_rejects_ambiguous_or_credential_urls(endpoint_cli, origin):
    base, output, calls = endpoint_cli
    with pytest.raises(SystemExit):
        load_cli.main([
            *base, "--base-url", origin, "--protocol", "openai-chat",
            "--model", "fixture-model", "--prompt", "hello",
        ])
    assert not calls
    assert not output.exists()


def test_request_json_rejects_overrides_and_nonfinite_values(endpoint_cli, tmp_path):
    base, output, calls = endpoint_cli
    request = tmp_path / "request.json"
    request.write_text('{"model":"fixture-model","prompt":"hi","temperature":NaN}')
    with pytest.raises(SystemExit):
        load_cli.main([*base, "--protocol", "openai-completions", "--request-json", str(request)])
    request.write_text('{"model":"fixture-model","prompt":"hi"}')
    with pytest.raises(SystemExit):
        load_cli.main([
            *base, "--protocol", "openai-completions", "--request-json", str(request), "--max-tokens", "3",
        ])
    assert not calls
    assert not output.exists()


def test_bearer_credential_is_sent_but_not_serialized(endpoint_cli, monkeypatch):
    base, output, calls = endpoint_cli
    monkeypatch.setenv("ACPROF_TEST_STREAM_KEY", "fixture-only-secret")
    assert load_cli.main([
        *base, "--protocol", "openai-chat", "--model", "fixture-model", "--prompt", "hello",
        "--warmup", "0", "--requests", "1", "--api-key-env", "ACPROF_TEST_STREAM_KEY",
    ]) == 0
    assert calls[0][5] == {"Authorization": "Bearer fixture-only-secret"}
    text = (output / "load.json").read_text()
    assert "fixture-only-secret" not in text
    assert "ACPROF_TEST_STREAM_KEY" not in text


def test_missing_bearer_variable_fails_before_network(endpoint_cli, monkeypatch):
    base, output, calls = endpoint_cli
    monkeypatch.delenv("ACPROF_TEST_STREAM_KEY", raising=False)
    with pytest.raises(SystemExit):
        load_cli.main([
            *base, "--protocol", "openai-chat", "--model", "fixture-model", "--prompt", "hello",
            "--api-key-env", "ACPROF_TEST_STREAM_KEY",
        ])
    assert not calls
    assert not output.exists()


def test_cancellation_persists_partial_measured_result(endpoint_cli, monkeypatch):
    base, output, _ = endpoint_cli
    partial = {"successful": False, "success_rate": 0.0, "requests": [{"status": "cancelled"}]}

    def cancelled(*_args, **_kwargs):
        error = KeyboardInterrupt("fixture cancelled")
        error.load_report = partial
        raise error

    monkeypatch.setattr(streaming_load, "run_load", cancelled)
    with pytest.raises(KeyboardInterrupt):
        load_cli.main([
            *base, "--protocol", "openai-chat", "--model", "fixture-model", "--prompt", "hello",
            "--warmup", "0", "--requests", "1",
        ])
    report = json.loads((output / "load.json").read_text())
    assert report["status"] == "cancelled"
    assert report["result"] == partial
    assert report["successful"] is False


def test_failed_warmup_stops_measured_requests_and_keeps_evidence(endpoint_cli, monkeypatch):
    base, output, _ = endpoint_cli
    calls = []

    def failed(*_args, **kwargs):
        calls.append(kwargs["token"])
        return {"successful": False, "success_rate": 0.0, "requests": [{"status": "error"}]}

    monkeypatch.setattr(streaming_load, "run_load", failed)
    with pytest.raises(RuntimeError, match="warmup failed"):
        load_cli.main([
            *base, "--protocol", "openai-chat", "--model", "fixture-model", "--prompt", "hello",
            "--warmup", "1", "--requests", "1",
        ])
    report = json.loads((output / "load.json").read_text())
    assert calls == ["warmup"]
    assert report["status"] == "failed"
    assert report["warmup"]["requests"][0]["status"] == "error"
    assert "result" not in report


def test_nonempty_output_is_never_overwritten(endpoint_cli):
    base, output, calls = endpoint_cli
    output.mkdir()
    previous = output / "keep.txt"
    previous.write_text("prior evidence")
    with pytest.raises(SystemExit):
        load_cli.main([
            *base, "--protocol", "openai-chat", "--model", "fixture-model", "--prompt", "hello",
        ])
    assert not calls
    assert previous.read_text() == "prior evidence"


def test_local_tokenizer_requires_declared_matching_identity(endpoint_cli, tmp_path):
    base, output, calls = endpoint_cli
    with pytest.raises(SystemExit):
        load_cli.main([
            *base, "--protocol", "openai-chat", "--model", "fixture-model", "--prompt", "hello",
            "--tokenizer-json", str(tmp_path / "tokenizer.json"), "--model-revision", "abc",
            "--tokenizer-model", "wrong-model", "--tokenizer-revision", "abc",
        ])
    assert not calls
    assert not output.exists()


def test_local_tokenizer_counts_whole_text_without_downloads(endpoint_cli, tmp_path):
    # This optional integration test runs when the declared tokenizer extra is installed.
    tokenizers = pytest.importorskip("tokenizers")
    base, output, calls = endpoint_cli
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]",
    ))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    assert load_cli.main([
        *base, "--protocol", "openai-chat", "--model", "fixture-model", "--prompt", "hello",
        "--warmup", "0", "--requests", "1", "--tokenizer-json", str(path),
        "--model-revision", "abc", "--tokenizer-model", "fixture-model", "--tokenizer-revision", "abc",
    ]) == 0
    assert calls[0][4].token_counter.count("hello world") == 2
    evidence = json.loads((output / "load.json").read_text())["identity"]["tokenizer"]
    assert evidence["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert evidence["identity_evidence"] == "user_declared_equal_ids_and_revisions"


def test_source_replay_still_requires_gpu_before_loading_source(tmp_path):
    with pytest.raises(SystemExit) as exc:
        load_cli.main(["nonexistent-source", "--output-dir", str(tmp_path / "out")])
    assert exc.value.code == 2


def test_slo_and_service_declarations_enter_identity_but_not_request_status(endpoint_cli):
    base, output, calls = endpoint_cli
    assert load_cli.main([
        *base, "--protocol", "openai-chat", "--model", "fixture-model", "--prompt", "hello",
        "--warmup", "1", "--requests", "1", "--slo-ttft-s", "0.2", "--slo-tpot-s", "0.03",
        "--slo-e2e-s", "1.5", "--service-backend", "vllm", "--service-version", "0.11.2",
    ]) == 0
    assert calls[0][6] == {"slo_thresholds": None, "include_capacity": False}
    assert calls[1][6] == {
        "slo_thresholds": {"ttft_s": 0.2, "tpot_s": 0.03, "e2e_s": 1.5},
        "include_capacity": True,
    }
    report = json.loads((output / "load.json").read_text())
    expected_identity = hashlib.sha256(
        json.dumps(report["identity"], sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    assert report["schema_version"] == 2
    assert report["identity_sha256"] == expected_identity
    assert report["identity"]["slo_thresholds_s"]["tpot_s"] == 0.03
    assert report["identity"]["service"] == {
        "schema_version": 1, "backend": "vllm", "version": "0.11.2", "probe": "none",
    }
    assert report["successful"] is True


def test_identity_hash_ignores_service_observations_and_changes_with_slo(endpoint_cli, monkeypatch, tmp_path):
    base, output, _ = endpoint_cli
    observations = iter(("first-observed", "second-observed", "third-observed"))

    def observed_probe(*_args, **_kwargs):
        return {"schema_version": 1, "observed": {"version": next(observations)}}

    monkeypatch.setattr(streaming_load, "probe_service", observed_probe)
    common = [
        "--protocol", "openai-chat", "--model", "fixture-model", "--prompt", "hello",
        "--warmup", "0", "--requests", "1", "--slo-e2e-s", "1", "--service-probe", "vllm",
    ]
    assert load_cli.main([*base, *common]) == 0
    first = json.loads((output / "load.json").read_text())
    second_output = tmp_path / "second"
    second_base = [*base[:-1], str(second_output)]
    assert load_cli.main([*second_base, *common]) == 0
    second = json.loads((second_output / "load.json").read_text())
    assert first["service_identity"]["observed"]["version"] == "first-observed"
    assert second["service_identity"]["observed"]["version"] == "second-observed"
    assert first["identity_sha256"] == second["identity_sha256"]
    third_output = tmp_path / "third"
    third_common = [*common]
    third_common[third_common.index("--slo-e2e-s") + 1] = "2"
    assert load_cli.main([*base[:-1], str(third_output), *third_common]) == 0
    third = json.loads((third_output / "load.json").read_text())
    assert third["identity_sha256"] != first["identity_sha256"]


@pytest.mark.parametrize("options", [
    ["--slo-ttft-s", "1"], ["--slo-tpot-s", "0.1"], ["--slo-e2e-s", "2"],
    ["--service-backend", "vllm"], ["--service-version", "1"],
    ["--service-probe", "vllm"], ["--service-probe", "none"],
])
def test_source_replay_rejects_p1_options_before_source_or_output(tmp_path, monkeypatch, options):
    monkeypatch.setattr(
        load_cli, "bootstrap_project_env", lambda *_args: (_ for _ in ()).throw(AssertionError("source loaded")),
    )
    output = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        load_cli.main(["missing-source", "--gpu", "off", "--output-dir", str(output), *options])
    assert exc.value.code == 2
    assert not output.exists()
