"""Window-external CPU/GPU idle baseline validation for a finished case CSV."""
from __future__ import annotations

import csv
import math
import os
import sys
from typing import Any, Dict, List

from acprof.host.gpu_device import normalize_gpu_mode


class EnergyProfilingError(RuntimeError):
    """Raised when energy profiling cannot continue reliably."""


IDLE_POWER_RELATIVE_RANGE_THRESHOLD = 0.05


def _format_watts(values: List[float]) -> str:
    return "[" + ", ".join(f"{value:.3f}" for value in values) + "]"


def row_has_error_status(row: Dict[str, Any]) -> bool:
    return str(row.get("status") or "").strip().lower() == "error"


def check_idle_power_values_stable(
    *,
    csv_path: str,
    metric_name: str,
    idle_values: List[float],
    invalid_rows: int,
    row_count: int,
    threshold: float,
    remediation: str,
    skip_when_no_rows: bool = False,
) -> None:
    if row_count == 0 and skip_when_no_rows:
        return
    if invalid_rows or not idle_values:
        raise EnergyProfilingError(
            f"{metric_name} case validation failed: "
            f"csv={csv_path}, valid_rows={len(idle_values)}, invalid_rows={invalid_rows}. "
            f"This case's energy data is not reliable. {remediation}"
        )

    mean_idle = sum(idle_values) / len(idle_values)
    if mean_idle <= 0.0:
        raise EnergyProfilingError(
            f"{metric_name} case validation failed: idle baseline mean is not positive. "
            f"csv={csv_path}. This case's energy data is not reliable. {remediation}"
        )

    relative_range = (max(idle_values) - min(idle_values)) / mean_idle
    if relative_range >= threshold:
        from acprof.artifact_layout import case_sidecar
        from acprof.quality import QualityCheck, write_quality
        write_quality(case_sidecar(csv_path, "quality_checks"), [QualityCheck(
            "cpu_idle_baseline_unstable" if metric_name == "cpu_idle_power_w" else "gpu_idle_baseline_unstable",
            "warning", relative_range, threshold, "Idle baseline relative range exceeds the case threshold",
            {"source": csv_path, "metric": metric_name, "values_w": idle_values, "formula": "(max-min)/mean"}).to_dict()], append=True)
        print(
            f"[energy][WARN] {metric_name} case check warning: "
            f"csv={csv_path}, {metric_name}={_format_watts(idle_values)} W, "
            f"min={min(idle_values):.3f} W, max={max(idle_values):.3f} W, "
            f"mean={mean_idle:.3f} W, relative_range={relative_range * 100.0:.1f}%, "
            f"threshold={threshold * 100.0:.1f}%. This case's energy data may be "
            f"noisy; experiment will continue. {remediation}",
            file=sys.stderr,
        )
        return

    print(
        f"[energy] {metric_name} case check passed: "
        f"rows={len(idle_values)}, min={min(idle_values):.3f} W, "
        f"max={max(idle_values):.3f} W, mean={mean_idle:.3f} W, "
        f"relative_range={relative_range * 100.0:.1f}%",
    )


def check_case_gpu_idle_power_stable(
    csv_path: str,
    threshold: float = IDLE_POWER_RELATIVE_RANGE_THRESHOLD,
    ignore_error_rows: bool = False,
) -> None:
    """Validate that GPU idle baseline did not drift across a finished case CSV."""
    if not os.path.exists(csv_path):
        raise EnergyProfilingError(
            f"gpu_idle_power_w case validation failed: result CSV does not exist: {csv_path}"
        )

    gpu_rows = 0
    invalid_rows = 0
    idle_values: List[float] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if ignore_error_rows and row_has_error_status(row):
                continue
            if normalize_gpu_mode(row.get("gpu_mode", "off")) != "on":
                continue
            gpu_rows += 1
            gpu_idle_power_w = parse_csv_float(
                row.get("gpu_idle_power_w", row.get("idle_power_w"))
            )
            if not math.isfinite(gpu_idle_power_w) or gpu_idle_power_w <= 0.0:
                invalid_rows += 1
                continue
            idle_values.append(gpu_idle_power_w)

    check_idle_power_values_stable(
        csv_path=csv_path,
        metric_name="gpu_idle_power_w",
        idle_values=idle_values,
        invalid_rows=invalid_rows,
        row_count=gpu_rows,
        threshold=threshold,
        remediation=(
            "Increase --idle-seconds, close other GPU processes, wait for GPU "
            "clocks/power to stabilize, then rerun."
        ),
        skip_when_no_rows=True,
    )


def check_case_cpu_idle_power_stable(
    csv_path: str,
    threshold: float = IDLE_POWER_RELATIVE_RANGE_THRESHOLD,
    ignore_error_rows: bool = False,
) -> None:
    """Validate that CPU package idle baseline did not drift across a finished case CSV."""
    if not os.path.exists(csv_path):
        raise EnergyProfilingError(
            f"cpu_idle_power_w case validation failed: result CSV does not exist: {csv_path}"
        )

    rows = 0
    invalid_rows = 0
    idle_values: List[float] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if ignore_error_rows and row_has_error_status(row):
                continue
            rows += 1
            cpu_idle_power_w = parse_csv_float(row.get("cpu_idle_power_w"))
            if not math.isfinite(cpu_idle_power_w) or cpu_idle_power_w <= 0.0:
                invalid_rows += 1
                continue
            idle_values.append(cpu_idle_power_w)

    check_idle_power_values_stable(
        csv_path=csv_path,
        metric_name="cpu_idle_power_w",
        idle_values=idle_values,
        invalid_rows=invalid_rows,
        row_count=rows,
        threshold=threshold,
        remediation=(
            "Increase --idle-seconds, close host background processes, wait for "
            "CPU package power to stabilize, then rerun."
        ),
        skip_when_no_rows=ignore_error_rows,
    )


def parse_csv_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
