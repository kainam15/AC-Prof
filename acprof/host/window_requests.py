"""Single-window request loop and progress state, independent of monitor ownership.

Progress is updated before each request and retained if a request fails. The
caller keeps the monitor finally block and publishes this state on failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from acprof.host.client_metrics import _to_float_or_nan


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
        progress.payload_sizes.append(_to_float_or_nan(response.get("request_payload_bytes")))
        progress.output_lengths.append(_to_float_or_nan(response.get("output_length")))
        progress.output_token_counts.append(_to_float_or_nan(response.get("output_token_count")))
        progress.workload_contracts.append(response.get("workload_contract"))
        parameter = response.get("task_param")
        if parameter is not None and progress.executed_task_param is None:
            progress.executed_task_param = str(parameter)
        progress.count += 1
        progress.effective_input_scale = merge_scale(
            progress.effective_input_scale, response.get("effective_input_scale"), scale
        )
