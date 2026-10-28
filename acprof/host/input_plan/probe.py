"""Readiness-only HTTP probes used while planning input scales."""

import math
from typing import Any

from acprof.gpu_mode import normalize_gpu_mode
from acprof.host.detect import TaskInfo
from acprof.host.images.identity import model_token
from acprof.host.images.runtime import ImageInfo
from acprof.host.runtimes.contracts import ServiceSession
from acprof.host.runtimes.service import start_service_session


def start_probe_session(
    task_info: TaskInfo,
    image_info: ImageInfo,
    cpu_list: list[int],
    mem_list: list[int],
    gpu_list: list[str],
) -> ServiceSession:
    probe_cpu = max(cpu_list)
    probe_mem = max(mem_list)
    normalized_gpu = [normalize_gpu_mode(gpu) for gpu in gpu_list]
    probe_gpu = "on" if "on" in normalized_gpu else "off"
    model_tag = model_token(task_info.model_id)
    container_name = f"probe_{model_tag}_{probe_cpu}c_{probe_mem}g_{probe_gpu}"

    print(
        f"[scale] Starting probe container with CPU={probe_cpu}, "
        f"MEM={probe_mem}GB, GPU={probe_gpu}"
    )
    return start_service_session(
        task_info=task_info,
        cpu=probe_cpu,
        mem=probe_mem,
        gpu=probe_gpu,
        image_info=image_info,
        container_name=container_name,
        log_prefix="[probe]",
    )


def parse_probe_response(data: dict[str, Any], desc: str) -> dict[str, Any]:
    required = {"effective_input_scale", "truncated_by_limit", "reason"}
    missing = sorted(required - set(data.keys()))
    if missing:
        raise RuntimeError(f"/probe response missing fields for {desc}: {missing}")

    reason = str(data.get("reason", "")).strip()
    if not reason:
        raise RuntimeError(f"/probe returned empty reason for {desc}")

    if data.get("effective_input_scale") is None or data.get("truncated_by_limit") is None:
        raise RuntimeError(f"cannot reliably determine effective input scale for {desc}: {reason}")

    try:
        effective_scale = float(data["effective_input_scale"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"/probe returned invalid effective_input_scale for {desc}: "
            f"{data.get('effective_input_scale')!r}"
        ) from exc

    truncated_by_limit = data.get("truncated_by_limit")
    if not isinstance(truncated_by_limit, bool):
        raise RuntimeError(f"/probe returned non-boolean truncated_by_limit for {desc}: {truncated_by_limit!r}")
    limit_exceeded = data.get("limit_exceeded", False)
    if not isinstance(limit_exceeded, bool):
        raise RuntimeError(f"/probe returned non-boolean limit_exceeded for {desc}: {limit_exceeded!r}")

    return {
        "effective_input_scale": effective_scale,
        "truncated_by_limit": truncated_by_limit,
        "limit_exceeded": limit_exceeded,
        "reason": reason,
    }


def probe_exceeds_limit(result: dict[str, Any]) -> bool:
    return result["truncated_by_limit"] or result.get("limit_exceeded", False)


def post_probe_payload(
    session: ServiceSession,
    payload: dict[str, Any],
    desc: str,
) -> dict[str, Any]:
    import requests

    response = requests.post(
        session.base_url + "/probe",
        json=payload,
        timeout=300,
        headers={"Connection": "close"},
    )
    if response.status_code >= 400:
        stale_image_hint = ""
        if response.status_code == 404:
            stale_image_hint = " /probe endpoint is missing; rebuild the image with --rebuild-image."
        raise RuntimeError(
            f"/probe HTTP {response.status_code} for {desc}: {response.text[:300]}{stale_image_hint}"
        )

    try:
        data = response.json()
    except Exception as exc:
        raise RuntimeError(f"/probe returned non-JSON response for {desc}") from exc

    parsed = parse_probe_response(data, desc)
    parsed["payload"] = payload
    return parsed


def request_scale_meta(
    session: ServiceSession,
    payload: dict[str, Any],
) -> dict[str, Any]:
    import requests

    response = requests.post(
        session.base_url + "/scale_meta",
        json=payload,
        timeout=300,
        headers={"Connection": "close"},
    )
    if response.status_code >= 400:
        stale_image_hint = ""
        if response.status_code == 404:
            stale_image_hint = " /scale_meta endpoint is missing; rebuild the image with --rebuild-image."
        raise RuntimeError(
            f"/scale_meta HTTP {response.status_code}: {response.text[:300]}{stale_image_hint}"
        )

    try:
        data = response.json()
    except Exception as exc:
        raise RuntimeError("/scale_meta returned non-JSON response") from exc

    if not isinstance(data, dict):
        raise RuntimeError("/scale_meta returned a non-object response")
    return data


def request_nlp_scale_meta(
    session: ServiceSession,
    payload: dict[str, Any],
) -> dict[str, Any]:
    data = request_scale_meta(session, payload)
    reason = str(data.get("reason", "")).strip()
    if not reason:
        raise RuntimeError("/scale_meta returned empty reason")

    raw_max = data.get("max_effective_input_scale")
    if raw_max is None:
        raise RuntimeError(f"cannot reliably determine NLP max input scale: {reason}")

    try:
        max_effective_input_scale = int(float(raw_max))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"/scale_meta returned invalid max_effective_input_scale: {raw_max!r}"
        ) from exc

    if max_effective_input_scale <= 0:
        raise RuntimeError(
            f"/scale_meta returned non-positive max_effective_input_scale={max_effective_input_scale}: {reason}"
        )

    return {
        "input_scale_type": str(data.get("input_scale_type", "")).strip() or "input_scale",
        "max_effective_input_scale": max_effective_input_scale,
        "reason": reason,
    }


def request_audio_scale_meta(
    session: ServiceSession,
    payload: dict[str, Any],
) -> dict[str, Any]:
    data = request_scale_meta(session, payload)
    reason = str(data.get("reason", "")).strip()
    if not reason:
        raise RuntimeError(
            "cannot determine audio input constraints from /scale_meta; "
            "rebuild the audio image with --rebuild-image"
        )

    result: dict[str, Any] = {
        "input_scale_type": str(data.get("input_scale_type") or "duration_s"),
        "reason": reason,
    }
    numeric_fields = (
        "required_sampling_rate",
        "source_sampling_rate",
        "max_short_form_duration_s",
        "model_input_num_samples",
        "model_input_frames",
        "fixed_frontend_num_samples",
        "fixed_frontend_num_frames",
        "frontend_feature_bins",
        "encoder_positions",
        "decoder_output_token_limit",
    )
    for field_name in numeric_fields:
        raw_value = data.get(field_name)
        if raw_value is None:
            result[field_name] = None
            continue
        try:
            number = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"/scale_meta returned invalid {field_name}: {raw_value!r}"
            ) from exc
        if not math.isfinite(number) or number <= 0:
            raise RuntimeError(
                f"/scale_meta returned non-positive/non-finite "
                f"{field_name}: {raw_value!r}"
            )
        result[field_name] = int(number) if number.is_integer() else number

    result["model_type"] = str(data.get("model_type") or "unknown")
    raw_fixed_padding = data.get("short_form_fixed_padding")
    if not isinstance(raw_fixed_padding, bool):
        raise RuntimeError(
            "/scale_meta returned invalid short_form_fixed_padding: "
            f"{raw_fixed_padding!r}"
        )
    result["short_form_fixed_padding"] = raw_fixed_padding
    resampling_policy = data.get("resampling_policy")
    if resampling_policy is not None:
        if resampling_policy != "scipy.signal.resample_poly_if_required_in_preprocess":
            raise RuntimeError(f"/scale_meta returned unsupported resampling_policy: {resampling_policy!r}")
        result["resampling_policy"] = resampling_policy
    return result
