"""Task-family-specific input scale planning strategies."""

import logging
import math
from typing import Any

from acprof.config import SCALING_DIMENSIONS
from acprof.extensions import CATALOG
from acprof.host.detect import TaskInfo
from acprof.host.input_plan.materialize import (
    get_task_generator,
    materialize_scale_plan,
    scale_plan_file_path,
    write_scale_plan_file,
)
from acprof.host.input_plan.probe import (
    post_probe_payload,
    probe_exceeds_limit,
    request_audio_scale_meta,
    request_nlp_scale_meta,
    request_scale_meta,
    start_probe_session,
)
from acprof.host.input_plan.scales import (
    AUTO_INPUT_SCALE_COUNT,
    format_scale_value,
    integer_auto_scales,
    resolve_input_scales,
    serialize_input_scales,
)
from acprof.host.input_plan.text_scales import select_text_scale_probes
from acprof.host.input_plan.types import PlannedInputScales
from acprof.host.runtimes.contracts import RuntimeArtifact, ServiceSession
from acprof.host.runtimes.service import stop_service_session


def plan_manual_nlp_scales(
    task_info: TaskInfo,
    image_info: RuntimeArtifact,
    cpu_list: list[int],
    mem_list: list[int],
    gpu_list: list[str],
    scales: list[float],
    batch_size: int,
    output_dir: str,
    workload_spec_path: str | None = None,
    cpuset_cpus: str = "",
) -> PlannedInputScales:
    session: ServiceSession | None = None
    try:
        session = start_probe_session(task_info, image_info, cpu_list, mem_list, gpu_list, cpuset_cpus)
        workload_gen = get_task_generator(
            task_info, batch_size, workload_spec_path=workload_spec_path,
        )
        invalid: dict[float, str] = {}
        entries: list[dict[str, Any]] = []
        actual_scales: list[float] = []

        for scale in scales:
            payload = workload_gen.generate(scale)
            result = post_probe_payload(
                session, payload, f"manual scale {format_scale_value(scale)}",
            )
            if probe_exceeds_limit(result):
                invalid[scale] = (
                    f"{result['reason']}; effective_input_scale="
                    f"{format_scale_value(result['effective_input_scale'])}"
                )
                continue

            actual_scale = float(result["effective_input_scale"])
            actual_scales.append(actual_scale)
            entries.append({
                "input_scale": actual_scale,
                "scale_label": workload_gen.scale_label(actual_scale),
                "input_metadata": workload_gen.input_metadata(actual_scale, result["payload"]),
                "payload": result["payload"],
            })

        if invalid:
            details = ", ".join(
                f"{format_scale_value(scale)} ({reason})" for scale, reason in invalid.items()
            )
            kind = "Kokoro phoneme" if workload_gen.plan_metadata().get("input_scale_type") == "phoneme_ids" else "NLP"
            raise RuntimeError(f"manual {kind} input scales exceed the usable handler limit: {details}")
    finally:
        if session is not None:
            stop_service_session(session, log_prefix="[probe]")

    plan_file = scale_plan_file_path(output_dir)
    workload_metadata = workload_gen.plan_metadata()
    plan_sha256 = write_scale_plan_file(plan_file, task_info, entries, workload=workload_metadata)

    print(
        "[scale] Using manual input scales: "
        f"{serialize_input_scales(scales)}; effective scales: "
        f"{serialize_input_scales(actual_scales)}"
    )
    return PlannedInputScales(
        scales=actual_scales,
        source="manual",
        plan_file=plan_file,
        plan_sha256=plan_sha256,
        workload=workload_metadata,
    )


def plan_nlp_auto_scales(
    task_info: TaskInfo,
    image_info: RuntimeArtifact,
    cpu_list: list[int],
    mem_list: list[int],
    gpu_list: list[str],
    batch_size: int,
    output_dir: str,
    workload_spec_path: str | None = None,
    cpuset_cpus: str = "",
) -> PlannedInputScales:
    session: ServiceSession | None = None
    try:
        session = start_probe_session(task_info, image_info, cpu_list, mem_list, gpu_list, cpuset_cpus)
        workload_gen = get_task_generator(
            task_info, batch_size, workload_spec_path=workload_spec_path,
        )
        metadata_payload = workload_gen.generate(1.0)
        scale_meta = request_nlp_scale_meta(session, metadata_payload)
        max_effective = int(scale_meta["max_effective_input_scale"])

        def probe_word_count(word_count: int) -> dict[str, Any]:
            if hasattr(workload_gen, "generate_for_word_count"):
                payload = workload_gen.generate_for_word_count(word_count)
            else:
                payload = workload_gen.generate(float(word_count))
            return post_probe_payload(session, payload, f"word_count={word_count}")

        entries: list[dict[str, Any]] = []
        actual_scales: list[float] = []
        for candidate in select_text_scale_probes(max_effective, probe_word_count):
            actual_scale = float(candidate["effective_input_scale"])
            actual_scales.append(actual_scale)
            entries.append({
                "input_scale": actual_scale,
                "scale_label": workload_gen.scale_label(actual_scale),
                "input_metadata": workload_gen.input_metadata(actual_scale, candidate["payload"]),
                "payload": candidate["payload"],
            })

        plan_file = scale_plan_file_path(output_dir)
        workload_metadata = workload_gen.plan_metadata()
        plan_sha256 = write_scale_plan_file(
            plan_file, task_info, entries,
            workload=workload_metadata, model_constraints=scale_meta,
        )

        if len(actual_scales) < AUTO_INPUT_SCALE_COUNT:
            print(f"[scale] Found {len(actual_scales)} distinct usable text scales "
                  f"for {AUTO_INPUT_SCALE_COUNT} requested auto targets")
        print(
            f"[scale] Auto-planned text scales from {scale_meta.get('input_scale_type', 'seq_length')} planning limit "
            f"{max_effective}: {serialize_input_scales(actual_scales)}"
        )
        logging.getLogger(__name__).debug("scale metadata: %s", scale_meta["reason"])
        return PlannedInputScales(
            scales=actual_scales,
            source="auto",
            plan_file=plan_file,
            plan_sha256=plan_sha256,
            workload=workload_metadata,
        )
    finally:
        if session is not None:
            stop_service_session(session, log_prefix="[probe]")


def default_family_max_scale(task_info: TaskInfo, batch_size: int) -> float:
    if CATALOG.describe(task_info).declaration.input_plan.max_scale_probe:
        workload_gen = get_task_generator(task_info, batch_size)
        if hasattr(workload_gen, "max_input_scale"):
            max_scale = workload_gen.max_input_scale()
            if max_scale is not None:
                return float(max_scale)

    scaling_cfg = SCALING_DIMENSIONS.get(task_info.task_family)
    if scaling_cfg and scaling_cfg.values:
        return float(max(scaling_cfg.values))

    raise RuntimeError(f"cannot determine default max input scale for task_family={task_info.task_family}")


def plan_audio_scales(
    *,
    task_info: TaskInfo,
    image_info: RuntimeArtifact,
    cpu_list: list[int],
    mem_list: list[int],
    gpu_list: list[str],
    scales: list[float],
    batch_size: int,
    output_dir: str,
    source: str,
    workload_spec_path: str | None,
    cpuset_cpus: str = "",
) -> PlannedInputScales:
    """Validate real audio payloads against the loaded model before a sweep."""
    workload_gen = get_task_generator(
        task_info, batch_size, workload_spec_path=workload_spec_path,
    )
    normalized_scales = sorted(set(float(scale) for scale in scales))
    if not normalized_scales or normalized_scales[0] <= 0:
        raise RuntimeError("audio input scales must be finite positive durations")
    if any(not math.isfinite(scale) for scale in normalized_scales):
        raise RuntimeError("audio input scales must be finite positive durations")

    materialized_payloads = [
        (scale, workload_gen.generate(scale)) for scale in normalized_scales
    ]

    session: ServiceSession | None = None
    model_constraints: dict[str, Any] = {}
    try:
        session = start_probe_session(task_info, image_info, cpu_list, mem_list, gpu_list, cpuset_cpus)
        first_payload = materialized_payloads[0][1]
        model_constraints = request_audio_scale_meta(session, first_payload)

        expected_rate = model_constraints.get("required_sampling_rate")
        payload_rate = first_payload.get("sample_rate")
        if expected_rate is not None and payload_rate is not None:
            permits_resampling = (
                task_info.pipeline_tag not in {"automatic-speech-recognition", "asr", "speech-recognition"}
                and model_constraints.get("resampling_policy")
                == "scipy.signal.resample_poly_if_required_in_preprocess"
                and model_constraints.get("source_sampling_rate") == int(payload_rate)
            )
            if int(expected_rate) != int(payload_rate) and not permits_resampling:
                raise RuntimeError(
                    "audio workload sampling rate does not match the model: "
                    f"payload={payload_rate}Hz, model={expected_rate}Hz"
                )

        max_short = model_constraints.get("max_short_form_duration_s")
        if max_short is not None:
            invalid = [scale for scale in normalized_scales if scale > float(max_short) + 1e-9]
            if invalid:
                raise RuntimeError(
                    "audio input scales exceed the model short-form limit "
                    f"of {format_scale_value(float(max_short))} seconds: "
                    f"{serialize_input_scales(invalid)}. Use a separate long-form workload."
                )

        for scale, payload in materialized_payloads:
            result = post_probe_payload(
                session, payload, f"audio scale {format_scale_value(scale)}s",
            )
            if probe_exceeds_limit(result):
                raise RuntimeError(
                    "audio input scale is not valid for short-form inference: "
                    f"{format_scale_value(scale)}s ({result['reason']})"
                )
            if not math.isclose(
                float(result["effective_input_scale"]),
                float(scale),
                rel_tol=0.0,
                abs_tol=(0.5 / max(1, int(payload.get("sample_rate", 16000)))),
            ):
                raise RuntimeError(
                    "audio effective duration differs from the requested scale: "
                    f"requested={scale}, effective={result['effective_input_scale']}"
                )
    finally:
        if session is not None:
            stop_service_session(session, log_prefix="[probe]")

    print(f"[scale] Using {source} audio scales: {serialize_input_scales(normalized_scales)}")
    return materialize_scale_plan(
        task_info=task_info,
        scales=normalized_scales,
        batch_size=batch_size,
        output_dir=output_dir,
        source=source,
        workload_spec_path=workload_spec_path,
        model_constraints=model_constraints,
    )


def plan_timeseries_scales(
    task_info: TaskInfo,
    image_info: RuntimeArtifact,
    cpu_list: list[int],
    mem_list: list[int],
    gpu_list: list[str],
    batch_size: int,
    output_dir: str,
    input_scales: str | None,
    cpuset_cpus: str = "",
) -> PlannedInputScales:
    """Use the loaded forecasting model's context limit before materialization."""
    generator = get_task_generator(task_info, batch_size)
    session = start_probe_session(task_info, image_info, cpu_list, mem_list, gpu_list, cpuset_cpus)
    try:
        constraints = request_scale_meta(session, generator.generate(1))
    finally:
        stop_service_session(session, log_prefix="[probe]")
    limit = constraints.get("max_effective_input_scale")
    if (
        isinstance(limit, bool)
        or not isinstance(limit, (float, int))
        or not math.isfinite(limit)
        or limit < 1
        or int(limit) != limit
    ):
        raise RuntimeError(
            "cannot determine forecasting model context limit from /scale_meta; "
            "rebuild the timeseries image"
        )
    maximum = min(int(limit), int(generator.max_input_scale()))
    if input_scales:
        scales = resolve_input_scales(task_info.task_family, input_scales)
        if any(
            not math.isfinite(scale) or scale < 1 or int(scale) != scale or scale > maximum
            for scale in scales
        ):
            raise RuntimeError(
                f"manual input scales must be positive integer context lengths <= {maximum}"
            )
        source = "manual"
    else:
        scales = integer_auto_scales(maximum, count=min(AUTO_INPUT_SCALE_COUNT, maximum))
        source = "auto"
    return materialize_scale_plan(
        task_info=task_info,
        scales=scales,
        batch_size=batch_size,
        output_dir=output_dir,
        source=source,
        model_constraints=constraints,
    )
