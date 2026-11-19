"""Shared UUID-bound NVML resource queries; no power or extra sampler ownership."""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from acprof.capabilities import capability_from_error
from acprof.monitors.resource_metrics import normalize_pstate
from acprof.results.coverage import GPU_QUERIES

STATE_INTERVAL_S = 1.0


def planned_gpu_queries(plan: dict) -> frozenset[str]:
    return frozenset(name for name in GPU_QUERIES
                     if plan["probes"].get("gpu_" + name, {}).get("status") == "available")


def read_gpu_sample(nvml: Any, handle: Any, on_error: Callable[[str, Exception], None],
                    *, state: bool = True, enabled: frozenset[str] | None = None) -> dict:
    """Isolate each optional query; never replace failed readings with zero."""
    values: dict = {}
    def memory():
        info = nvml.nvmlDeviceGetMemoryInfo(handle)
        return {'gpu_mem_used_bytes': int(info.used), 'gpu_mem_total_bytes': int(info.total)}
    operations = [
        ('utilization', lambda: {'gpu_util_pct': float(nvml.nvmlDeviceGetUtilizationRates(handle).gpu)}),
        ('memory', memory),
    ]
    if state:
        operations.extend([
            ('sm_clock', lambda: {'gpu_sm_clock_mhz': float(nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM))}),
            ('memory_clock', lambda: {'gpu_memory_clock_mhz': float(nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM))}),
            ('pstate', lambda: {'gpu_pstate': normalize_pstate(nvml.nvmlDeviceGetPerformanceState(handle))}),
            ('temperature', lambda: {'gpu_temp_c': float(nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU))}),
        ])
    for name, operation in operations:
        if enabled is not None and name not in enabled:
            continue
        try:
            observed = operation()
            if any(value is None or isinstance(value, (int, float)) and (not math.isfinite(value) or value < 0)
                   for value in observed.values()):
                raise ValueError('invalid NVML reading')
            values.update(observed)
        except Exception as exc:
            # NVML backends expose driver-specific exception subclasses.
            on_error(name, exc)
    return values


def gpu_metric_evidence(values: dict, errors: dict, *, requested: bool,
                        initialization_error: str = '', sampling: dict | None = None) -> dict:
    result = {}
    for query, names in GPU_QUERIES.items():
        for name in names:
            value = values.get(name)
            present = (value not in {None, '', 'nan'} if name == 'gpu_pstate'
                       else isinstance(value, (int, float)) and math.isfinite(value))
            reason = str(errors.get(query) or initialization_error)
            quality = ('not_requested' if not requested else
                       'partial' if present and reason else
                       'complete' if present else
                       capability_from_error(reason).to_dict()['status'] if reason else 'unavailable')
            result[name] = {'source': 'nvml_selected_device', 'scope': 'selected_gpu_device',
                            'window': 'request_window', 'quality': quality,
                            'derivation': 'observed', 'reason': reason}
            if sampling is not None:
                result[name]['sampling'] = {key: value for key, value in sampling.items()
                    if query not in {'utilization', 'memory'} or key != 'state_interval_s'}
    return result
