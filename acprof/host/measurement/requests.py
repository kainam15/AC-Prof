"""Single-window request loop and progress state, independent of monitor ownership.

Progress is updated before each request and retained if a request fails. The
caller keeps the monitor finally block and publishes this state on failure.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from acprof.host.measurement.metrics import to_float_or_nan


def canonical_task_param(payload: dict[str, Any] | None) -> str:
    """Serialize the parameters that the server actually receives."""
    params: Any = {}
    if isinstance(payload, dict):
        nested_params = payload.get("params")
        if isinstance(nested_params, dict):
            params = dict(nested_params)
        elif nested_params is not None:
            params = {"params": nested_params}

        # Time-series handlers consume prediction_length at the top level,
        # rather than through the common nested params object.
        if "prediction_length" in payload:
            params["prediction_length"] = payload["prediction_length"]
    return json.dumps(
        params,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )

def merge_effective_input_scale(
    current: float | None,
    candidate: float | None,
    requested_scale: float,
) -> float:
    resolved = float(requested_scale) if candidate is None else float(candidate)
    if current is None:
        return resolved
    if not math.isclose(current, resolved, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            f"inconsistent effective_input_scale for requested_scale={requested_scale}: "
            f"{current} vs {resolved}"
        )
    return current

@dataclass
class WindowRequestProgress:
    count: int = 0
    elapsed_s: float = 0.0
    latencies: list[float] = field(default_factory=list)
    payload_sizes: list[float] = field(default_factory=list)
    output_lengths: list[float] = field(default_factory=list)
    output_token_counts: list[float] = field(default_factory=list)
    workload_contracts: list[dict | None] = field(default_factory=list)
    executed_task_param: str | None = None
    effective_input_scale: float | None = None
    pending_request_id: str = ""


def execute_requests(
    progress: WindowRequestProgress,
    *,
    should_send: Callable[[int, float, int], bool],
    one_request: Callable[..., dict[str, Any]],
    merge_scale: Callable[[float | None, Any, float], float | None],
    scale: float,
    group_id: str,
    payload_override: Any,
    request_limit: int,
) -> None:
    """Only invoke workload requests; never start/stop/publish monitors here."""
    while should_send(progress.count, progress.elapsed_s, request_limit):
        request_id = f"{group_id}:{progress.count}"
        progress.pending_request_id = request_id
        response = one_request(scale, req_id=request_id, payload_override=payload_override)
        progress.pending_request_id = ""
        latency = float(response["latency_app_s"])
        progress.elapsed_s += latency
        progress.latencies.append(latency)
        progress.payload_sizes.append(to_float_or_nan(response.get("request_payload_bytes")))
        progress.output_lengths.append(to_float_or_nan(response.get("output_length")))
        progress.output_token_counts.append(to_float_or_nan(response.get("output_token_count")))
        progress.workload_contracts.append(response.get("workload_contract"))
        parameter = response.get("task_param")
        if parameter is not None and progress.executed_task_param is None:
            progress.executed_task_param = str(parameter)
        progress.count += 1
        progress.effective_input_scale = merge_scale(
            progress.effective_input_scale, response.get("effective_input_scale"), scale
        )
