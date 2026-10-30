"""Diagnostic timing probe for one largest-scale inference request."""
from __future__ import annotations

import datetime
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import requests

from acprof.artifacts import atomic_write, read_input_scale_plan, read_json_object
from acprof.gpu_mode import normalize_gpu_mode
from acprof.host import container_state
from acprof.host.detect import TaskInfo
from acprof.host.docker_runtime import (
    RunningContainer,
    start_container_session,
    stop_container_session,
)
from acprof.host.images.identity import model_token
from acprof.host.images.runtime import ImageInfo
from acprof.host.input_plan import PlannedInputScales

PROBE_SUMMARY_NAME = "largest_scale_probe.json"
PROBE_SUMMARY_SCHEMA_VERSION = 3
RETRYABLE_MEMORY_OOM_STATUSES = frozenset({"startup_oom", "runtime_oom"})


def read_probe_summary(path: str | os.PathLike[str]) -> dict:
    """Validate the existing diagnostic summary before claiming probe success."""
    summary = read_json_object(Path(path), label="largest-scale probe summary")
    if summary.get("schema_version") != PROBE_SUMMARY_SCHEMA_VERSION:
        raise ValueError("unsupported largest-scale probe summary version")
    if summary.get("status") != "ok":
        raise ValueError("largest-scale probe summary is not successful")
    model, resource, inputs, timing, cold = (summary.get(key) for key in
                                           ("model", "resource", "input", "timing", "cold_start"))
    if (not isinstance(model, dict) or not isinstance(resource, dict) or not isinstance(inputs, dict)
            or not isinstance(timing, dict) or not isinstance(cold, dict)):
        raise ValueError("incomplete largest-scale probe summary")
    if not isinstance(model.get("id"), str) or not model["id"]:
        raise ValueError("missing largest-scale probe model identity")
    if (any(type(resource.get(key)) is not int or resource[key] <= 0 for key in ("cpu_cores", "mem_gb"))
            or resource.get("gpu_mode") not in {"off", "on"}):
        raise ValueError("invalid largest-scale probe resources")
    planned, effective = inputs.get("planned_scale"), inputs.get("effective_scale")
    if (not isinstance(planned, (int, float)) or isinstance(planned, bool) or planned <= 0
            or not isinstance(effective, (int, float)) or isinstance(effective, bool) or effective <= 0
            or not math.isclose(planned, effective, rel_tol=0.0, abs_tol=1e-6)):
        raise ValueError("invalid largest-scale probe input evidence")
    for value in (timing.get("request_s"), timing.get("ready_plus_request_s"), cold.get("total_s")):
        if value is not None and (type(value) not in (int, float) or value < 0):
            raise ValueError("invalid largest-scale probe timing evidence")
    if timing.get("request_s") is None:
        raise ValueError("missing largest-scale probe request timing")
    return summary


def select_minimum_resources(
    cpu_list: Sequence[int],
    mem_list: Sequence[int],
    gpu_list: Sequence[str],
) -> tuple[int, int, str]:
    """Select the least provisioned requested resource configuration."""
    if not cpu_list or not mem_list or not gpu_list:
        raise ValueError("CPU、内存和 GPU 模式列表不能为空")
    if any(int(value) <= 0 for value in cpu_list):
        raise ValueError("CPU 列表必须全部大于 0")
    if any(int(value) <= 0 for value in mem_list):
        raise ValueError("内存列表必须全部大于 0")

    normalized_gpu = [normalize_gpu_mode(value) for value in gpu_list]
    gpu = "off" if "off" in normalized_gpu else "on"
    return min(int(value) for value in cpu_list), min(
        int(value) for value in mem_list
    ), gpu


def load_largest_scale_entry(plan_file: str | os.PathLike[str]) -> dict[str, Any]:
    """Load the materialized payload with the largest effective input scale."""
    path = Path(plan_file)
    try:
        plan = read_input_scale_plan(path)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"input scale plan 无法读取: {exc}") from exc
    entries = plan.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("input scale plan 没有可探测的 entries")

    candidates: list[tuple[float, dict[str, Any]]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise RuntimeError(f"input scale plan entry {index} 不是 object")
        raw_scale = entry.get("input_scale")
        if isinstance(raw_scale, bool) or not isinstance(raw_scale, (int, float)):
            raise RuntimeError(
                f"input scale plan entry {index} 的 input_scale 必须是 JSON number"
            )
        scale = float(raw_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise RuntimeError(
                f"input scale plan entry {index} 的 input_scale 必须大于 0"
            )
        payload = entry.get("payload")
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"input scale plan entry {index} 缺少 materialized payload"
            )
        candidates.append((scale, entry))

    scale, entry = max(candidates, key=lambda item: item[0])
    return {**entry, "input_scale": scale}


def create_probe_output_dir(
    output_root: str | os.PathLike[str],
    model_id: str,
) -> Path:
    """Create a unique per-run directory outside the formal result root."""
    root = Path(output_root).expanduser()
    from acprof.results.layout import ArtifactLayout
    model_root = ArtifactLayout.discover(root / model_id.replace("/", "--")).path("probes")
    timestamp = datetime.datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    base_name = f"largest_scale_{timestamp}_{os.getpid()}"
    for suffix in range(1000):
        name = base_name if suffix == 0 else f"{base_name}_{suffix}"
        candidate = model_root / name
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("无法创建唯一的最大尺度探测目录")


def write_probe_summary(
    path: str | os.PathLike[str],
    summary: dict[str, Any],
) -> None:
    """Atomically persist a probe result."""
    def write(handle) -> None:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")

    atomic_write(Path(path), write)


def _cold_start_payload(session: RunningContainer | None) -> dict[str, Any]:
    if session is None:
        return {
            "total_s": None,
            "container_launch_s": None,
            "server_setup_s": None,
            "cuda_init_s": None,
            "model_load_s": None,
            "ready_wait_s": None,
        }
    return {
        "total_s": _finite_or_none(session.cold_start_s),
        "container_launch_s": _finite_or_none(
            session.cold_start_container_launch_s
        ),
        "server_setup_s": _finite_or_none(session.cold_start_server_setup_s),
        "cuda_init_s": _finite_or_none(session.cold_start_cuda_init_s),
        "model_load_s": _finite_or_none(session.cold_start_model_load_s),
        "ready_wait_s": _finite_or_none(session.cold_start_ready_wait_s),
    }


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _result_marker(summary: dict[str, Any]) -> str:
    resource = summary["resource"]
    input_data = summary["input"]
    timing = summary["timing"]
    request_s = timing.get("request_s")
    cold_start_s = summary["cold_start"].get("total_s")
    ready_plus_request_s = timing.get("ready_plus_request_s")

    def value(raw: Any) -> str:
        return "nan" if raw is None else f"{float(raw):.6f}"

    marker_mem = resource.get("mem_gb")
    if marker_mem is None:
        marker_mem = resource.get("last_attempt_mem_gb")
    return (
        "[largest-probe] RESULT "
        f"status={summary['status']} "
        f"input_scale={float(input_data['planned_scale']):g} "
        f"cpu={resource['cpu_cores']} "
        f"mem={marker_mem if marker_mem is not None else 'none'} "
        f"gpu={resource['gpu_mode']} "
        f"cold_start_s={value(cold_start_s)} "
        f"request_s={value(request_s)} "
        f"ready_plus_request_s={value(ready_plus_request_s)}"
    )


def _classify_oom_failure(
    error: str,
    *,
    phase: str,
    container_name: str,
) -> str | None:
    """Return a narrow OOM classification suitable for a memory scan."""
    normalized = str(error).strip().lower()
    if any(
        marker in normalized
        for marker in (
            "cuda out of memory",
            "cuda error: out of memory",
            "cublas_status_alloc_failed",
        )
    ):
        return "cuda_oom"

    if any(
        marker in normalized
        for marker in (
            "container_oom_killed",
            "docker_oom_killed=true",
            "out of memory",
            "cannot allocate memory",
            "can't allocate memory",
            "failed to allocate memory",
            "memoryerror",
            "std::bad_alloc",
        )
    ):
        return "startup_oom" if phase == "startup" else "runtime_oom"

    state = container_state.inspect_container_state(container_name)
    if isinstance(state, dict) and bool(state.get("OOMKilled")):
        return "startup_oom" if phase == "startup" else "runtime_oom"
    return None


def _probe_memory_candidate(
    *,
    task_info: TaskInfo,
    image_info: ImageInfo,
    cpu: int,
    mem: int,
    gpu: str,
    payload: dict[str, Any],
    planned_scale: float,
    timeout_seconds: float | None,
    attempt_index: int,
    attempt_total: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run at most one largest-scale request for one memory candidate."""
    container_name = (
        f"largest_probe_{model_token(task_info.model_id)}_"
        f"{cpu}c_{mem}g_{gpu}_{os.getpid()}"
    )
    session: RunningContainer | None = None
    response_payload: dict[str, Any] | None = None
    request_s: float | None = None
    request_started: float | None = None
    attempt_started = time.perf_counter()
    phase = "startup"
    status = "error"
    error = ""

    from acprof.events.progress import emit_event
    emit_event("probe_memory_started", "largest_scale_probe", current=attempt_index,
               total=attempt_total, cpu=cpu, mem=mem, gpu=gpu, input_scale=planned_scale)
    print(
        "[largest-probe] MEMORY_TRY "
        f"current={attempt_index} total={attempt_total} "
        f"cpu={cpu} mem={mem} gpu={gpu} input_scale={planned_scale:g}"
    )
    try:
        session = start_container_session(
            task_info=task_info,
            cpu=cpu,
            mem=mem,
            gpu=gpu,
            image_info=image_info,
            container_name=container_name,
            log_prefix="[largest-probe]",
            request_timeout_seconds=timeout_seconds,
        )
        phase = "request"
        print("[largest-probe] Running one largest-scale request...")
        emit_event("probe_request_started", "largest_scale_probe")
        request_started = time.perf_counter()
        response = requests.post(
            session.base_url + "/predict",
            json=payload,
            timeout=timeout_seconds,
            headers={"Connection": "close"},
        )
        request_s = time.perf_counter() - request_started
        if response.status_code >= 400:
            raise RuntimeError(
                f"/predict HTTP {response.status_code}: {response.text[:500]}"
            )
        candidate = response.json()
        if not isinstance(candidate, dict):
            raise RuntimeError("/predict 返回值不是 JSON object")
        response_payload = candidate
        effective_scale = _finite_or_none(candidate.get("effective_input_scale"))
        if effective_scale is None:
            raise RuntimeError("/predict 未返回有效的 effective_input_scale")
        if not math.isclose(
            effective_scale,
            planned_scale,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError(
                "实际 input scale 与计划不一致："
                f"planned={planned_scale:g}, effective={effective_scale:g}"
            )
        status = "ok"
    except requests.Timeout:
        if request_started is not None:
            request_s = time.perf_counter() - request_started
        status = "timeout"
        if timeout_seconds is None:
            error = "最大尺度请求意外超时（未配置探测超时）"
        else:
            error = f"最大尺度请求超过 {timeout_seconds:g} 秒"
    except Exception as exc:
        if request_started is not None and request_s is None:
            request_s = time.perf_counter() - request_started
        error = f"{type(exc).__name__}: {exc}"
        status = _classify_oom_failure(
            error,
            phase=phase,
            container_name=session.name if session is not None else container_name,
        ) or "error"
    finally:
        if request_started is not None:
            emit_event("probe_request_stopped", "largest_scale_probe")
        if session is not None:
            stop_container_session(session, log_prefix="[largest-probe]")

    cold_start = _cold_start_payload(session)
    request_s = _finite_or_none(request_s)
    cold_start_s = _finite_or_none(cold_start.get("total_s"))
    ready_plus_request_s = (
        cold_start_s + request_s
        if cold_start_s is not None and request_s is not None
        else None
    )
    attempt = {
        "index": attempt_index,
        "mem_gb": mem,
        "status": status,
        "error": error,
        "cold_start": cold_start,
        "timing": {
            "request_s": request_s,
            "ready_plus_request_s": ready_plus_request_s,
            "attempt_wall_s": time.perf_counter() - attempt_started,
        },
    }
    print(f"[largest-probe] MEMORY_RESULT mem={mem} status={status}")
    emit_event("probe_memory_finished", "largest_scale_probe", mem=mem, status=status)
    if error:
        print(f"[largest-probe] Memory {mem}GB: {error}", file=sys.stderr)
    return attempt, response_payload


def run_largest_scale_probe(
    *,
    task_info: TaskInfo,
    image_info: ImageInfo,
    planned_input_scales: PlannedInputScales,
    cpu_list: Sequence[int],
    mem_list: Sequence[int],
    gpu_list: Sequence[str],
    batch_size: int,
    output_dir: str | os.PathLike[str],
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Find the minimum viable memory cap and time its largest-scale request."""
    if int(batch_size) <= 0:
        raise ValueError("batch_size 必须大于 0")
    if timeout_seconds is not None:
        timeout_seconds = float(timeout_seconds)
        if timeout_seconds <= 0.0 or not math.isfinite(timeout_seconds):
            raise ValueError("timeout_seconds 必须是大于 0 的有限值")
    if not planned_input_scales.plan_file:
        raise RuntimeError("最大尺度探测需要 materialized input scale plan")

    cpu, _, gpu = select_minimum_resources(cpu_list, mem_list, gpu_list)
    memory_candidates = sorted(set(int(value) for value in mem_list))
    entry = load_largest_scale_entry(planned_input_scales.plan_file)
    planned_scale = float(entry["input_scale"])
    payload = entry["payload"]
    summary_path = Path(output_dir) / PROBE_SUMMARY_NAME
    started_at = datetime.datetime.now().astimezone().isoformat()
    probe_started = time.perf_counter()
    attempts: list[dict[str, Any]] = []
    response_payload: dict[str, Any] | None = None
    selected_mem: int | None = None

    from acprof.events.progress import emit_event
    emit_event("probe_started", "largest_scale_probe", cpu=cpu, gpu=gpu,
               memory_candidates=memory_candidates, input_scale=planned_scale)
    print(
        "[largest-probe] MEMORY_SCAN "
        f"cpu={cpu} gpu={gpu} candidates="
        f"{','.join(str(value) for value in memory_candidates)} "
        f"input_scale={planned_scale:g}"
    )
    for index, mem in enumerate(memory_candidates, start=1):
        attempt, candidate_response = _probe_memory_candidate(
            task_info=task_info,
            image_info=image_info,
            cpu=cpu,
            mem=mem,
            gpu=gpu,
            payload=payload,
            planned_scale=planned_scale,
            timeout_seconds=timeout_seconds,
            attempt_index=index,
            attempt_total=len(memory_candidates),
        )
        attempts.append(attempt)
        if attempt["status"] == "ok":
            selected_mem = mem
            response_payload = candidate_response
            break
        if attempt["status"] in RETRYABLE_MEMORY_OOM_STATUSES:
            if index < len(memory_candidates):
                print(
                    f"[largest-probe] {mem}GB OOM; trying next memory candidate."
                )
            continue
        break

    last_attempt = attempts[-1]
    if selected_mem is not None:
        status = "ok"
        error = ""
    elif (
        len(attempts) == len(memory_candidates)
        and all(
            item["status"] in RETRYABLE_MEMORY_OOM_STATUSES
            for item in attempts
        )
    ):
        status = "oom"
        error = (
            "所选内存候选均无法完成最大 input scale："
            + ",".join(f"{value}GB" for value in memory_candidates)
        )
    else:
        status = str(last_attempt["status"])
        error = str(last_attempt["error"])
        if status == "cuda_oom":
            error = (
                "最大 input scale 发生 CUDA 显存 OOM；增加主机内存上限无法解决，"
                f"已停止内存扫描。原始错误：{error}"
            )
        elif status == "timeout":
            error = (
                f"{last_attempt['mem_gb']}GB 候选请求超时，无法证明该档位是否内存可行；"
                f"已停止扫描。{error}"
            )

    chosen_attempt = (
        next(item for item in attempts if item["status"] == "ok")
        if selected_mem is not None
        else last_attempt
    )
    cold_start = chosen_attempt["cold_start"]
    request_s = chosen_attempt["timing"]["request_s"]
    ready_plus_request_s = chosen_attempt["timing"]["ready_plus_request_s"]
    effective_scale = _finite_or_none(
        (response_payload or {}).get("effective_input_scale")
    )
    summary: dict[str, Any] = {
        "schema_version": PROBE_SUMMARY_SCHEMA_VERSION,
        "status": status,
        "error": error,
        "started_at": started_at,
        "completed_at": datetime.datetime.now().astimezone().isoformat(),
        "model": {
            "id": task_info.model_id,
            "revision": task_info.model_revision,
            "task_family": task_info.task_family,
            "pipeline_tag": task_info.pipeline_tag,
            "runtime_backend": task_info.runtime_backend,
            "image_tag": image_info.tag,
        },
        "resource": {
            "cpu_cores": cpu,
            "mem_gb": selected_mem,
            "last_attempt_mem_gb": last_attempt["mem_gb"],
            "requested_mem_candidates_gb": memory_candidates,
            "gpu_mode": gpu,
            "selection": (
                "minimum_cpu_and_gpu_off_if_available_then_first_memory_"
                "candidate_completing_largest_scale"
            ),
        },
        "memory_probe": {
            "minimum_viable_mem_gb": selected_mem,
            "candidate_order_gb": memory_candidates,
            "attempts": attempts,
            "success_criterion": (
                "fresh_container_ready_and_one_largest_scale_predict_completed"
            ),
        },
        "input": {
            "planned_scale": planned_scale,
            "effective_scale": effective_scale,
            "scale_source": planned_input_scales.source,
            "scale_label": str(entry.get("scale_label") or ""),
            "batch_size": int(batch_size),
            "plan_sha256": planned_input_scales.plan_sha256,
        },
        "cold_start": cold_start,
        "timing": {
            "request_s": request_s,
            "ready_plus_request_s": ready_plus_request_s,
            "probe_wall_s": time.perf_counter() - probe_started,
            "request_timeout_s": timeout_seconds,
        },
        "response": {
            "output_length": (response_payload or {}).get("output_length"),
            "output_token_count": (response_payload or {}).get(
                "output_token_count"
            ),
        },
        "artifacts": {
            "input_scale_plan": str(planned_input_scales.plan_file),
            "summary": str(summary_path),
        },
    }
    write_probe_summary(summary_path, summary)
    emit_event("probe_finished", "largest_scale_probe", status=status, cpu=cpu,
               mem=selected_mem if selected_mem is not None else last_attempt["mem_gb"],
               gpu=gpu, input_scale=planned_scale, request_s=request_s,
               cold_start_s=_finite_or_none(cold_start.get("total_s")),
               ready_plus_request_s=ready_plus_request_s)
    emit_event("probe_summary", path=str(summary_path))
    print(_result_marker(summary))
    if error:
        print(f"[largest-probe][ERROR] {error}", file=sys.stderr)
    print(f"[largest-probe] Summary JSON: {summary_path}")
    return summary
