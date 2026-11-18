"""Read an already owned Process cgroup; never infer ownership of a parent."""
from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

from acprof.capabilities import capability_from_error
from acprof.monitors.resource_metrics import counter_delta
from acprof.monitors.resource_readers import ResourceMetricReaders, resolve_cgroup_metric_readers

FIELDS = ('service_cgroup_cpu_time_s', 'service_cgroup_mem_usage_avg_bytes',
          'service_cgroup_mem_usage_peak_bytes', 'service_cgroup_io_read_bytes', 'service_cgroup_io_write_bytes')


def reduce_cgroup_samples(item: dict) -> dict:
    """Pure, replayable reduction of a verified service-cgroup window."""
    samples = item.get('samples', [])
    if item.get('quality') not in {'complete', 'partial'} or len(samples) < 2:
        return {}
    values: dict[str, float] = {}
    first, last = samples[0], samples[-1]
    if last['timestamp'] <= first['timestamp']:
        return values
    def finite(field):
        return all(isinstance(sample.get(field), (int, float)) and math.isfinite(sample[field])
                   and sample[field] >= 0 for sample in samples)
    if finite('cpu_s'):
        # Counter resets anywhere in a sampled series invalidate the total.
        if all(after['cpu_s'] >= before['cpu_s'] for before, after in zip(samples, samples[1:], strict=False)):
            values['service_cgroup_cpu_time_s'] = last['cpu_s'] - first['cpu_s']
    if finite('memory_bytes'):
        memories = [sample['memory_bytes'] for sample in samples]
        values.update(service_cgroup_mem_usage_avg_bytes=sum(memories) / len(memories),
                      service_cgroup_mem_usage_peak_bytes=max(memories))
    for direction in ('read', 'write'):
        before, after = first.get(f'io_{direction}_bytes'), last.get(f'io_{direction}_bytes')
        value = counter_delta({'value': before}, {'value': after}, 'value') if before is not None and after is not None else math.nan
        if math.isfinite(value):
            values[f'service_cgroup_io_{direction}_bytes'] = value
    return values


class ProcessCgroupObserver:
    def __init__(self, pid: int, plan: dict):
        self.pid = pid
        self.path = (Path(plan['cgroup']) if plan.get('method') == 'cgroup_v2'
                     and plan.get('verified') is True and plan.get('cgroup') else None)
        self.quality = 'unavailable' if self.path else 'unsupported'
        self.error = '' if self.path else 'no_owned_service_cgroup'
        self.readers: ResourceMetricReaders | None = None
        self.samples: list[dict] = []
        self.errors: dict[str, str] = {}
        self.identity: tuple[int, int] | None = None

    def _verify(self):
        if self.path is None:
            raise RuntimeError('no owned service cgroup')
        stat = self.path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if self.identity is not None and identity != self.identity:
            raise RuntimeError('service cgroup identity changed')
        if str(self.pid) not in (self.path / 'cgroup.procs').read_text().split():
            raise RuntimeError('service cgroup membership changed')
        self.identity = identity

    def prepare(self):
        if self.path is None:
            return
        try:
            self._verify()
            self.readers = resolve_cgroup_metric_readers(str(self.path))
            self.quality = 'complete'
        except (OSError, RuntimeError, ValueError) as exc:
            self.quality = capability_from_error(exc).to_dict()['status']
            self.error = str(exc)

    def sample_at(self, timestamp: float, *, boundary: bool):
        if self.readers is None or self.quality not in {'complete', 'partial'}:
            return
        if boundary:
            try:
                self._verify()
            except (OSError, RuntimeError) as exc:
                self.quality, self.error = capability_from_error(exc).to_dict()['status'], str(exc)
                return
        sample = {'timestamp': timestamp}
        operations: list[tuple[str, Callable[[], float | tuple[int, int]] | None]] = [
            ('cpu_s', self.readers.cpu), ('memory_bytes', self.readers.memory)]
        if boundary:
            operations.append(('io', self.readers.io))
        for name, operation in operations:
            if operation is None:
                continue
            try:
                value = operation()
                if isinstance(value, tuple):
                    sample['io_read_bytes'], sample['io_write_bytes'] = value
                else:
                    sample[name] = value
            except (OSError, RuntimeError, ValueError) as exc:
                self.quality = 'partial'
                self.errors[name] = str(exc)
        self.samples.append(sample)

    def snapshot(self, limit: int) -> dict:
        from acprof.telemetry import capture_samples
        return {'status': 'recorded' if self.samples else 'unavailable', 'quality': self.quality,
                'source': 'service_cgroup', 'scope': 'owned_service_cgroup', 'window': 'request_window',
                'path': str(self.path) if self.path else None, 'identity': self.identity,
                'error': self.error, 'errors': self.errors,
                'units': {'timestamp': 's_monotonic', 'cpu_s': 's', 'memory_bytes/io_*_bytes': 'bytes'},
                **capture_samples(self.samples, limit)}

    def finalize(self) -> tuple[dict, dict]:
        values = reduce_cgroup_samples({'samples': self.samples, 'quality': self.quality})
        evidence = {}
        for name in FIELDS:
            raw = 'io' if '_io_' in name else 'cpu_s' if '_cpu_' in name else 'memory_bytes'
            reason = self.error or self.errors.get(raw, '')
            counter = ('cpu_s' if raw == 'cpu_s' else
                       'io_read_bytes' if name == 'service_cgroup_io_read_bytes' else
                       'io_write_bytes' if name == 'service_cgroup_io_write_bytes' else None)
            observed = [sample[counter] for sample in self.samples if counter in sample] if counter else []
            reset = any(after < before for before, after in zip(observed, observed[1:], strict=False))
            if reset:
                reason = '; '.join(filter(None, (reason, 'counter_decreased')))
            if name in values and not reason:
                quality = 'complete'
            elif self.quality not in {'complete', 'partial'}:
                quality = self.quality
            elif reset:
                quality = 'error'
            else:
                quality = capability_from_error(reason).to_dict()['status'] if reason else 'unavailable'
            evidence[name] = {'source': 'service_cgroup', 'scope': 'owned_service_cgroup',
                'window': 'request_window', 'derivation': 'observed', 'quality': quality,
                'reason': reason or ('' if name in values else 'missing_or_invalid_counter')}
        return values, evidence
