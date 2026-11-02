"""Window-external CPU/GPU idle baseline validation for a finished case CSV."""
from __future__ import annotations

import csv
import math
import os
import sys
from typing import Any

from acprof.gpu_mode import normalize_gpu_mode
from acprof.results.csv import allow_large_csv_fields


class EnergyProfilingError(RuntimeError):
    """Raised when energy profiling cannot continue reliably."""


IDLE_POWER_RELATIVE_RANGE_THRESHOLD = 0.05


def _format_watts(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:.3f}" for value in values) + "]"


def row_has_error_status(row: dict[str, Any]) -> bool:
    return str(row.get("status") or "").strip().lower() == "error"


def check_idle_power_values_stable(
    *,
    csv_path: str,
    metric_name: str,
    idle_values: list[float],
    invalid_rows: int,
    row_count: int,
    threshold: float,
    remediation: str,
    skip_when_no_rows: bool = False,
    excluded_warmup_rows: int = 0,
) -> None:
    if row_count == 0 and skip_when_no_rows and not excluded_warmup_rows:
        return
    if invalid_rows or (not idle_values and not excluded_warmup_rows):
        raise EnergyProfilingError(
            f"{metric_name} case validation failed: "
            f"csv={csv_path}, valid_rows={len(idle_values)}, invalid_rows={invalid_rows}. "
            f"This case's energy data is not reliable. {remediation}"
        )

    mean_idle = sum(idle_values) / len(idle_values) if idle_values else math.nan
    if mean_idle <= 0.0:
        raise EnergyProfilingError(
            f"{metric_name} case validation failed: idle baseline mean is not positive. "
            f"csv={csv_path}. This case's energy data is not reliable. {remediation}"
        )

    from acprof.quality import QualityCheck, write_quality
    from acprof.results.layout import case_sidecar

    device = "cpu" if metric_name == "cpu_idle_power_w" else "gpu"
    evidence = {"source": csv_path, "metric": metric_name, "scope": "formal",
                "values_w": idle_values, "sample_count": len(idle_values),
                "excluded_warmup_rows": excluded_warmup_rows}
    if len(idle_values) < 2:
        write_quality(case_sidecar(csv_path, "quality_checks"), [QualityCheck(
            f"{device}_idle_baseline_insufficient_samples", "warning", len(idle_values), 2,
            "At least two formal idle baselines are required to evaluate drift", evidence,
        ).to_dict()], append=True)
        print(
            f"[energy][WARN] {metric_name} case stability is unknown: "
            f"csv={csv_path}, formal_rows={len(idle_values)}, "
            f"excluded_warmup_rows={excluded_warmup_rows}. "
            "Collect at least two formal repeat windows; experiment will continue.",
            file=sys.stderr,
        )
        return

    relative_range = (max(idle_values) - min(idle_values)) / mean_idle
    if relative_range >= threshold:
        write_quality(case_sidecar(csv_path, "quality_checks"), [QualityCheck(
            f"{device}_idle_baseline_unstable",
            "warning", relative_range, threshold, "Formal idle baseline relative range reaches the case threshold",
            {**evidence, "formula": "(max-min)/mean"}).to_dict()], append=True)
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
        f"formal_rows={len(idle_values)}, excluded_warmup_rows={excluded_warmup_rows}, "
        f"min={min(idle_values):.3f} W, "
        f"max={max(idle_values):.3f} W, mean={mean_idle:.3f} W, "
        f"relative_range={relative_range * 100.0:.1f}%",
    )


def check_case_gpu_idle_power_stable(
    csv_path: str,
    threshold: float = IDLE_POWER_RELATIVE_RANGE_THRESHOLD,
    ignore_error_rows: bool = False,
) -> None:
    """Validate GPU idle drift across formal rows, retaining warmup only as raw evidence."""
    if not os.path.exists(csv_path):
        raise EnergyProfilingError(
            f"gpu_idle_power_w case validation failed: result CSV does not exist: {csv_path}"
        )

    gpu_rows = 0
    invalid_rows = 0
    excluded_warmup_rows = 0
    idle_values: list[float] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        allow_large_csv_fields()
        reader = csv.DictReader(f)
        for row in reader:
            if ignore_error_rows and row_has_error_status(row):
                continue
            if normalize_gpu_mode(row.get("gpu_mode", "off")) != "on":
                continue
            warmup = parse_csv_float(row.get("warmup"))
            if warmup == 1:
                excluded_warmup_rows += 1
                continue
            gpu_rows += 1
            if warmup != 0:
                invalid_rows += 1
                continue
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
        excluded_warmup_rows=excluded_warmup_rows,
    )


def check_case_cpu_idle_power_stable(
    csv_path: str,
    threshold: float = IDLE_POWER_RELATIVE_RANGE_THRESHOLD,
    ignore_error_rows: bool = False,
) -> None:
    """Validate CPU package idle drift across formal rows of a finished case CSV."""
    if not os.path.exists(csv_path):
        raise EnergyProfilingError(
            f"cpu_idle_power_w case validation failed: result CSV does not exist: {csv_path}"
        )

    rows = 0
    invalid_rows = 0
    excluded_warmup_rows = 0
    idle_values: list[float] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        allow_large_csv_fields()
        reader = csv.DictReader(f)
        for row in reader:
            if ignore_error_rows and row_has_error_status(row):
                continue
            warmup = parse_csv_float(row.get("warmup"))
            if warmup == 1:
                excluded_warmup_rows += 1
                continue
            rows += 1
            if warmup != 0:
                invalid_rows += 1
                continue
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
        excluded_warmup_rows=excluded_warmup_rows,
    )


def parse_csv_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
