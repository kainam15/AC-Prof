"""OpenAI-compatible endpoint mode for the existing load command."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from acprof.analysis.streaming_capacity import normalize_slo_thresholds
from acprof.artifacts import atomic_write_json, read_json_object
from acprof.host.measurement.load import LoadConfig, run_load
from acprof.host.measurement.service_identity import initial_service_identity, probe_service
from acprof.host.run_state import MeasurementLock, host_identity
from acprof.installation import resource_root

_ENDPOINTS = {
    "openai-chat": "/v1/chat/completions",
    "openai-completions": "/v1/completions",
}
_ENDPOINT_OPTIONS = (
    "model", "prompt", "request_json", "max_tokens", "temperature",
    "generation_seed", "request_timeout", "api_key_env", "model_revision",
    "tokenizer_json", "tokenizer_model", "tokenizer_revision",
    "slo_ttft_s", "slo_tpot_s", "slo_e2e_s", "service_backend", "service_version",
    "max_run_response_bytes", "max_run_output_events",
)


def add_streaming_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", help="Existing service origin, e.g. http://127.0.0.1:8000")
    parser.add_argument("--protocol", choices=("predict", *_ENDPOINTS), default="predict")
    parser.add_argument("--model", help="Served model name for OpenAI-compatible requests")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--prompt", help="Text prompt; converted to one user message in chat mode")
    inputs.add_argument("--request-json", type=Path, help="Complete JSON request body (maximum 4 MiB)")
    parser.add_argument("--max-tokens", type=int, help="Output token limit (prompt mode default: 128)")
    parser.add_argument("--temperature", type=float, help="Generation temperature (prompt mode default: 0)")
    parser.add_argument("--generation-seed", type=int, help="Generation seed; --seed controls arrivals")
    parser.add_argument("--request-timeout", type=float, help="Connect/read inactivity timeout in seconds")
    parser.add_argument("--api-key-env", help="Environment variable containing the Bearer credential")
    parser.add_argument("--model-revision", help="User-declared served model revision")
    parser.add_argument("--tokenizer-json", type=Path, help="Local tokenizer.json; never downloads code or files")
    parser.add_argument("--tokenizer-model", help="Explicit tokenizer model ID; must match --model/body model")
    parser.add_argument("--tokenizer-revision", help="Explicit tokenizer revision; must match --model-revision")
    parser.add_argument("--slo-ttft-s", type=float, help="Request TTFT SLO threshold in seconds")
    parser.add_argument("--slo-tpot-s", type=float, help="Request TPOT SLO threshold in seconds/token")
    parser.add_argument("--slo-e2e-s", type=float, help="Request E2E SLO threshold in seconds")
    parser.add_argument("--service-backend", help="User-declared inference backend")
    parser.add_argument("--service-version", help="User-declared inference backend version")
    parser.add_argument("--max-run-response-bytes", type=int,
                        help="Maximum raw SSE bytes received across one warmup or measured run")
    parser.add_argument("--max-run-output-events", type=int,
                        help="Maximum output events retained across one warmup or measured run")
    parser.add_argument("--service-probe", choices=("none", "vllm"),
                        help="Optional pre-warmup service identity probe (default: none)")


def validate_replay_arguments(args: argparse.Namespace) -> None:
    if args.source is None or args.gpu is None:
        raise ValueError("source replay requires a source experiment and --gpu")
    if (args.protocol != "predict" or args.service_probe is not None
            or any(getattr(args, name) is not None for name in _ENDPOINT_OPTIONS)):
        raise ValueError("OpenAI request options require --base-url and an OpenAI protocol")


def _validate_origin(value: str) -> str:
    if any(character.isspace() for character in value):
        raise ValueError("--base-url cannot contain whitespace")
    target = urlsplit(value)
    if (
        target.scheme not in {"http", "https"}
        or not target.hostname
        or target.username is not None
        or target.password is not None
        or target.query
        or target.fragment
        or target.path not in {"", "/"}
    ):
        raise ValueError("--base-url must be an HTTP(S) origin without path, credentials, query or fragment")
    _ = target.port
    return value.rstrip("/")


def _request_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.request_json is not None:
        if any(getattr(args, name) is not None for name in ("max_tokens", "temperature", "generation_seed")):
            raise ValueError("--request-json cannot be combined with generation overrides")
        payload = read_json_object(args.request_json, label="streaming request")
        if args.model is not None and payload.get("model") != args.model:
            raise ValueError("--model must exactly match the request JSON model")
    else:
        if args.prompt is None or not args.model:
            raise ValueError("endpoint mode requires --model and --prompt, or --request-json")
        maximum = args.max_tokens if args.max_tokens is not None else 128
        temperature = args.temperature if args.temperature is not None else 0.0
        if maximum < 1:
            raise ValueError("--max-tokens must be positive")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("--temperature must be finite and nonnegative")
        payload = {"model": args.model, "max_tokens": maximum, "temperature": temperature}
        if args.generation_seed is not None:
            payload["seed"] = args.generation_seed
        if args.protocol == "openai-chat":
            payload["messages"] = [{"role": "user", "content": args.prompt}]
        else:
            payload["prompt"] = args.prompt
    if any(str(key).lower() in {"authorization", "api_key", "apikey", "headers"} for key in payload):
        raise ValueError("credentials and headers are not request body fields; use --api-key-env")
    return payload


def _authorization(args: argparse.Namespace) -> dict[str, str]:
    if args.api_key_env is None:
        return {}
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.api_key_env):
        raise ValueError("--api-key-env must name an environment variable")
    credential = os.environ.get(args.api_key_env)
    if not credential or any(character in credential for character in "\r\n"):
        raise ValueError(f"--api-key-env variable {args.api_key_env!r} is missing, empty or invalid")
    return {"Authorization": f"Bearer {credential}"}


def _token_counter(args: argparse.Namespace, model: str) -> tuple[Any, dict[str, Any] | None]:
    if args.tokenizer_json is None:
        if args.tokenizer_model is not None or args.tokenizer_revision is not None:
            raise ValueError("tokenizer identity options require --tokenizer-json")
        return None, None
    if not all((args.model_revision, args.tokenizer_model, args.tokenizer_revision)):
        raise ValueError("--tokenizer-json requires --model-revision, --tokenizer-model and --tokenizer-revision")
    if args.tokenizer_model != model or args.tokenizer_revision != args.model_revision:
        raise ValueError("tokenizer model and revision must exactly match the declared served model")
    from acprof.host.measurement.streaming import MatchedTokenCounter
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise ValueError(
            "local tokenizer fallback needs the streaming-tokenizer extra; "
            "run uv sync --extra streaming-tokenizer or install acprof[streaming-tokenizer]"
        ) from exc
    path = args.tokenizer_json.resolve(strict=True)
    try:
        tokenizer_bytes = path.read_bytes()
        tokenizer = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"cannot load local tokenizer JSON: {type(exc).__name__}") from exc
    counter = MatchedTokenCounter(
        count=lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids),
        model_id=model, model_revision=args.model_revision,
        tokenizer_id=args.tokenizer_model, tokenizer_revision=args.tokenizer_revision,
    )
    evidence = {
        "path": str(path), "sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
        "model_id": model, "model_revision": args.model_revision,
        "tokenizer_id": args.tokenizer_model, "tokenizer_revision": args.tokenizer_revision,
        "identity_evidence": "user_declared_equal_ids_and_revisions",
        "count_scope": "concatenated_visible_generated_text_without_special_tokens",
    }
    return counter, evidence


def _slo_thresholds(args: argparse.Namespace) -> dict[str, float | None]:
    return normalize_slo_thresholds({
        "ttft_s": args.slo_ttft_s, "tpot_s": args.slo_tpot_s, "e2e_s": args.slo_e2e_s,
    })


def _validate_service_declaration(args: argparse.Namespace) -> None:
    for name in ("service_backend", "service_version"):
        value = getattr(args, name)
        if value is not None and not value.strip():
            raise ValueError(f"--{name.replace('_', '-')} must be nonempty text")


def _prepare(args: argparse.Namespace) -> tuple[
    str, dict[str, Any], LoadConfig, Any, dict[str, str], Any, dict[str, float | None],
]:
    if args.source is not None or any(getattr(args, name) is not None for name in ("gpu", "cpu", "mem", "input_scale")) or args.capture:
        raise ValueError("endpoint mode cannot use source replay, --gpu, --cpu, --mem, --input-scale or --capture")
    if args.protocol not in _ENDPOINTS:
        raise ValueError("--base-url requires --protocol openai-chat or openai-completions")
    if args.warmup < 0:
        raise ValueError("--warmup must be nonnegative")
    slo_thresholds = _slo_thresholds(args)
    _validate_service_declaration(args)
    origin = _validate_origin(args.base_url)
    config = LoadConfig(
        scenario=args.scenario, connections=args.connections, concurrency=args.concurrency,
        requests=args.requests, rate=args.rate, arrival=args.arrival, seed=args.seed,
        max_pending=args.max_pending,
        timeout_seconds=args.request_timeout if args.request_timeout is not None else 300,
    ).validate()
    config.schedule()  # A finite rate can still overflow the plan; check before warmup/output.
    payload = _request_payload(args)
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("OpenAI streaming requires a nonempty model")
    counter, tokenizer = _token_counter(args, model)
    from acprof.host.measurement.streaming import OpenAIStreamingAdapter
    limits = {
        name: value for name in ("max_run_response_bytes", "max_run_output_events")
        if (value := getattr(args, name)) is not None
    }
    adapter = OpenAIStreamingAdapter(
        endpoint=_ENDPOINTS[args.protocol], token_counter=counter, **limits,
    )
    payload = adapter.prepare_payload(payload)
    return origin, payload, config, adapter, _authorization(args), tokenizer, slo_thresholds


def run_streaming_load(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Run a bounded endpoint workload; no server lifecycle or hardware attribution."""
    try:
        origin, payload, config, adapter, headers, tokenizer, slo_thresholds = _prepare(args)
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise ValueError("load output directory must be empty")
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    service_probe = args.service_probe or "none"
    declared_service = {
        "schema_version": 1, "backend": args.service_backend,
        "version": args.service_version, "model": payload["model"],
        "revision": args.model_revision, "probe": service_probe,
    }
    identity = {
        "target_origin": origin, "endpoint": adapter.endpoint,
        "model": payload["model"], "model_revision": args.model_revision,
        "model_revision_source": "user_declared" if args.model_revision else "unavailable",
        "request_body": payload,
        "request_body_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "load_config": asdict(config), "adapter": adapter.protocol(),
        "tokenizer": tokenizer, "warmup_requests": args.warmup,
        "warmup_scenario": "serial", "warmup_connections": "close",
        "authentication": "bearer_environment" if headers else "none",
        "slo_thresholds_s": slo_thresholds,
        "service": declared_service,
    }
    report: dict[str, Any] = {
        "schema_version": 2, "kind": "stream_load_experiment", "run_id": uuid4().hex,
        "successful": False, "status": "running", "identity": identity,
        "service_identity": initial_service_identity(declared_service, service_probe),
        "identity_sha256": hashlib.sha256(
            json.dumps(identity, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "host": host_identity(resource_root()),
        "scope": "client HTTP streaming latency; external service; no GPU energy, profiler or managed runtime measurements",
    }
    atomic_write_json(output / "load.json", report)
    phase = "warmup"
    try:
        with MeasurementLock():
            report["service_identity"] = probe_service(
                origin, declared_service, service_probe,
                headers=headers, timeout_s=config.timeout_seconds,
            )
            if args.warmup:
                warmup_config = LoadConfig(requests=args.warmup, timeout_seconds=config.timeout_seconds)
                report["warmup"] = run_load(
                    origin, payload, warmup_config, token="warmup", adapter=adapter,
                    headers=headers, include_capacity=False,
                )
                if not report["warmup"]["successful"]:
                    raise RuntimeError("streaming warmup failed; inspect load.json warmup records")
            phase = "result"
            report["result"] = run_load(
                origin, payload, config, token=report["run_id"], adapter=adapter,
                headers=headers, slo_thresholds=slo_thresholds,
            )
        report.update(successful=report["result"]["successful"], status="complete")
    except BaseException as error:
        if hasattr(error, "load_report"):
            report[phase] = error.load_report
        report.update(
            successful=False,
            status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        atomic_write_json(output / "load.json", report)
    print(f"Streaming load: {output / 'load.json'}; success rate {report['result']['success_rate']:.1%}")
    return 0 if report["successful"] else 1
