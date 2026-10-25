"""Pure GPU power-sample reduction; importing it never loads NVML."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EnergyResult:
    energy_iters: int
    idle_power_w: float
    avg_power_total_w: float
    peak_power_total_w: float
    energy_total_j: float
    avg_power_eff_w: float
    peak_power_eff_w: float
    energy_eff_j: float
    energy_source: str = "unavailable"
    energy_fallback_reason: str = ""
    counter_start_mj: int | None = None
    counter_end_mj: int | None = None
    measurement_duration_s: float = float("nan")


def apply_gpu_energy_counter(result: EnergyResult) -> None:
    """Reduce an already selected cumulative source without inventing power samples.

    The monitor owns source selection and plausibility checks. Both live
    reduction and replay use the same counter arithmetic; sample-only peaks
    remain as recorded by ``reduce_gpu_samples`` (unknown without samples).
    """
    before, after = result.counter_start_mj, result.counter_end_mj
    duration = result.measurement_duration_s
    if (before is None or after is None or before < 0 or after < before
            or not math.isfinite(duration) or duration <= 0.0):
        raise ValueError("invalid retained GPU energy counters")
    result.energy_total_j = (after - before) / 1000.0
    result.avg_power_total_w = result.energy_total_j / duration
    result.energy_eff_j = result.energy_total_j - result.idle_power_w * duration
    result.avg_power_eff_w = result.avg_power_total_w - result.idle_power_w
    result.energy_source = "nvml_total_energy"


def median_power(xs):
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def time_weighted_average_power(samples: list[tuple[float, float]]) -> float:
    if not samples:
        return float("nan")
    if len(samples) == 1:
        return float(samples[0][1])

    energy_j = 0.0
    for idx in range(1, len(samples)):
        t0, p0 = samples[idx - 1]
        t1, p1 = samples[idx]
        dt = t1 - t0
        if dt > 0.0:
            energy_j += 0.5 * (p0 + p1) * dt

    duration_s = samples[-1][0] - samples[0][0]
    return energy_j / duration_s if duration_s > 0.0 else float("nan")


def unavailable_gpu_result(energy_iters: int = 0, idle_power_w: float = float("nan")) -> EnergyResult:
    return EnergyResult(
        energy_iters,
        idle_power_w,
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
    )


def reduce_gpu_samples(
    samples: list[tuple[float, float]],
    idle_power_w: float,
) -> EnergyResult:
    if len(samples) < 2:
        return unavailable_gpu_result(len(samples), idle_power_w)

    powers = [p for _, p in samples]
    peak_power_total = max(powers)

    energy_total = 0.0
    for i in range(1, len(samples)):
        t0, p0 = samples[i - 1]
        t1, p1 = samples[i]
        energy_total += 0.5 * (p0 + p1) * (t1 - t0)
    duration_s = samples[-1][0] - samples[0][0]
    if duration_s <= 0.0:
        return unavailable_gpu_result(len(samples), idle_power_w)
    avg_power_total = energy_total / duration_s

    powers_eff = [p - idle_power_w for p in powers]
    peak_power_eff = max(powers_eff)

    energy_eff = 0.0
    for i in range(1, len(samples)):
        t0, _ = samples[i - 1]
        t1, _ = samples[i]
        energy_eff += 0.5 * (powers_eff[i - 1] + powers_eff[i]) * (t1 - t0)
    avg_power_eff = energy_eff / duration_s

    return EnergyResult(
        len(samples),
        idle_power_w,
        avg_power_total,
        peak_power_total,
        energy_total,
        avg_power_eff,
        peak_power_eff,
        energy_eff,
        "power_integration",
    )
