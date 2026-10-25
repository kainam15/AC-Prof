"""Per-operation capability evidence, collected outside measurement windows."""
from __future__ import annotations

import os
import socket
from pathlib import Path

from acprof.capabilities import Capability, capability_from_error


def _probe(operation, source):
    try:
        evidence = operation()
        return Capability('available', source=source, evidence=evidence or {})
    except Exception as exc:
        return capability_from_error(exc, source=source)


def probe_collection_plan(runtime_kind, *, gpu=None, profiling_mode='basic') -> dict:
    import psutil

    from acprof.host.runtimes.resources import inherited_resources

    def process(metric):
        target = psutil.Process()
        if metric == 'cpu':
            target.cpu_times()
        else:
            target.memory_info()
        target.children(recursive=True)
        return {'scope': 'observed_process_tree', 'source': 'psutil', 'pid_namespace': os.readlink('/proc/self/ns/pid')}

    def packet():
        with socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(3)):
            pass

    def cgroup():
        evidence = inherited_resources()
        if evidence['cgroup'] is None:
            raise FileNotFoundError('cgroup v2 hierarchy is not visible in this namespace')
        return evidence

    def rapl():
        paths = sorted(Path('/sys/class/powercap').glob('intel-rapl:*/energy_uj'))
        if not paths:
            raise FileNotFoundError('RAPL energy counters unavailable')
        for path in paths:
            int(path.read_text())
        return {'domains': [str(path) for path in paths], 'scope': 'host_package'}

    probes = {'process_cpu': _probe(lambda: process('cpu'), 'procfs/psutil'),
              'process_memory': _probe(lambda: process('memory'), 'procfs/psutil'),
              'cgroup': _probe(cgroup, 'visible_cgroup_namespace'),
              'cpu_energy': _probe(rapl, 'rapl'),
              'packet_latency': _probe(packet, 'packet_socket')}
    from acprof.host.preflight import probe_perf_instructions
    probes['cpu_instructions'] = probe_perf_instructions()
    gpu = gpu or {}
    for name in ('gpu_memory', 'gpu_utilization'):
        probes[name] = Capability('not_requested', source='nvml')
    if gpu:
        nvml = None
        try:
            import pynvml
            pynvml.nvmlInit()
            nvml = pynvml
            if not gpu.get('uuid'):
                raise RuntimeError('unavailable: visible CUDA device UUID could not be resolved')
            handle = nvml.nvmlDeviceGetHandleByUUID(gpu['uuid'])
            probes['gpu_memory'] = _probe(lambda: {'bytes': nvml.nvmlDeviceGetMemoryInfo(handle).used}, 'nvml')
            probes['gpu_utilization'] = _probe(lambda: {'percent': nvml.nvmlDeviceGetUtilizationRates(handle).gpu}, 'nvml')
        except Exception as exc:
            for name in ('gpu_memory', 'gpu_utilization'):
                probes[name] = capability_from_error(exc, source='nvml')
        finally:
            if nvml is not None:
                nvml.nvmlShutdown()
    return {'schema_version': 1, 'runtime_kind': runtime_kind, 'profiling_mode': profiling_mode,
            'probes': {name: value.to_dict() for name, value in probes.items()},
            'collectors': ['client_http', 'psutil_process_tree', *(['nvml_device'] if gpu else [])]
                         if runtime_kind == 'process' else ['client_http', 'cgroup'],
            'resource_scope': 'observed_process_tree' if runtime_kind == 'process' else 'container_cgroup',
            'gpu_scope': 'device', 'environment_sharing': 'unknown'}


def apply_process_plan(report, plan):
    report.requested.difference_update({'container_cpu', 'container_memory'})
    for name in ('container_cpu', 'container_memory'):
        report.measurement[name] = Capability('unsupported', 'Process uses separate process-tree metrics', 'execution_runtime')
        report.metric_support[name] = 'unsupported'
    for name in ('process_cpu', 'process_memory', 'gpu_memory', 'gpu_utilization'):
        report.measurement[name] = Capability(**plan['probes'][name])
        report.metric_support[name] = 'partial'
