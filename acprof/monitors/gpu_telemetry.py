"""Shared UUID-bound NVML resource queries; no power or extra sampler ownership."""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from acprof.capabilities import capability_from_error
from acprof.monitors.resource_metrics import normalize_pstate

STATE_INTERVAL_S = 1.0
GPU_QUERIES = {
    'utilization': ('gpu_util_avg_pct', 'gpu_util_peak_pct'),
    'memory': ('gpu_mem_used_avg_bytes', 'gpu_mem_used_peak_bytes', 'gpu_mem_util_avg_pct', 'gpu_mem_util_peak_pct'),
    'sm_clock': ('gpu_sm_clock_mhz',), 'memory_clock': ('gpu_memory_clock_mhz',),
    'pstate': ('gpu_pstate',), 'temperature': ('gpu_temp_c',),
}


def read_gpu_sample(nvml: Any, handle: Any, on_error: Callable[[str, Exception], None],
                    *, state: bool = True) -> dict:
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
                        initialization_error: str = '') -> dict:
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
    return result
