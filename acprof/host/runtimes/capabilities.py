"""Per-operation capability evidence, collected outside measurement windows."""
from __future__ import annotations

import os
import subprocess

from acprof.capabilities import Capability, capability_from_error


def _probe(operation, source):
    try:
        evidence = operation()
        return Capability('available', source=source, evidence=evidence or {})
    except Exception as exc:
        return capability_from_error(exc, source=source)


def package_energy_capability(topology: dict) -> Capability:
    packages = [entry for entry in topology["domains"] if entry["kind"] == "package"]
    selected = {entry["name"] for entry in packages if entry["selected"]}
    missing = [entry for entry in packages if entry["name"] not in selected]
    if not packages and not topology.get("errors"):
        denied = any(entry.get("status") == "permission_denied" for entry in topology["domains"])
        return Capability("permission_denied" if denied else "unsupported", "No readable RAPL package domain", "rapl_topology")
    if missing or not selected or topology.get("errors"):
        denied = (any(entry["status"] == "permission_denied" for entry in missing) or
                  any("permission denied" in str(item).lower() for item in topology.get("errors", [])))
        return Capability("permission_denied" if denied else "unavailable", "Incomplete readable RAPL packages",
                          "rapl_topology", {"missing": [entry["id"] for entry in missing], "errors": topology.get("errors", [])})
    return Capability("available", source="rapl_package", evidence={"packages": sorted(selected)})


def _gpu_probes(gpu, *, hardware_energy):
    from acprof.monitors.gpu_telemetry import GPU_QUERIES, read_gpu_sample

    names = ["gpu_" + name for name in GPU_QUERIES] + ["gpu_power"]
    if not gpu:
        return {name: Capability("not_applicable", "GPU disabled", "nvml") for name in names}
    if not gpu.get("uuid"):
        detail = gpu.get("error") or "No selected GPU UUID"
        return {name: Capability("unavailable", detail, "nvml") for name in names}
    try:
        import pynvml
        pynvml.nvmlInit()
        nvml = pynvml
        try:
            handle = nvml.nvmlDeviceGetHandleByUUID(gpu["uuid"])
            outcomes = {name: Capability("available", source="nvml") for name in names}

            def failed(name, error):
                outcomes["gpu_" + name] = capability_from_error(error, source="nvml")

            read_gpu_sample(nvml, handle, failed)
            if hardware_energy:
                def reading(query, method, unit):
                    value = int(query(handle))
                    if value < 0:
                        raise ValueError("Negative NVML reading")
                    return {"method": method, unit: value}
                counter = _probe(lambda: reading(nvml.nvmlDeviceGetTotalEnergyConsumption, "nvml_total_energy", "mJ"), "nvml")
                power = _probe(lambda: reading(nvml.nvmlDeviceGetPowerUsage, "power_integration", "mW"), "nvml")
                selected = counter if counter.status == "available" else power
                outcomes["gpu_power"] = Capability(selected.status, selected.detail, "nvml",
                    {"counter": counter.to_dict(), "power": power.to_dict()})
            else:
                outcomes["gpu_power"] = Capability("not_requested", "Hardware energy disabled by protocol", "platform_policy")
        finally:
            nvml.nvmlShutdown()
        return outcomes
    except Exception as exc:
        return {name: capability_from_error(exc, source="nvml") for name in names}




def probe_collection_plan(runtime_kind, *, gpu=None, profiling_mode="best_effort", sniff_iface="docker0",
                          dram_energy="auto", environment=None, env=None, service_cgroup=None) -> dict:
    import psutil

    from acprof.host.runtimes.resources import inherited_resources
    from acprof.platform import detect_environment
    from acprof.results.coverage import planned_fields

    environment = environment or detect_environment()
    def process(operation):
        target = psutil.Process()
        getattr(target, operation)()
        target.children(recursive=True)
        return {"scope": "observed_process_tree", "pid_namespace": os.readlink("/proc/self/ns/pid")}

    def cgroup():
        evidence = inherited_resources()
        if evidence["cgroup"] is None and evidence.get("cgroup_error"):
            raise RuntimeError(evidence["cgroup_error"])
        if evidence["cgroup"] is None:
            raise FileNotFoundError("cgroup v2 hierarchy is not visible in this namespace")
        return evidence

    probes = {name: _probe(lambda op=op: process(op), "procfs/psutil") for name, op in {
        "process_cpu": "cpu_times", "process_memory": "memory_info", "process_io": "io_counters",
        "process_threads": "num_threads", "process_fds": "num_fds", "process_ctx": "num_ctx_switches"}.items()}
    probes["cgroup"] = (Capability("not_applicable", "Service inherits a shared parent cgroup", "resource_plan")
                        if runtime_kind == "process" and service_cgroup is False else _probe(cgroup, "visible_cgroup_namespace"))
    topology = {}
    advanced = ("cpu_energy", "cpu_instructions", "packet_latency", "dram_energy")
    hardware = profiling_mode != "basic" and environment.native
    if hardware:
        import shutil

        from acprof.host.measurement.capture import probe_packet_tools
        from acprof.monitors.perf_mips import probe_event_set, resolve_perf_command_prefix
        from acprof.monitors.rapl_topology import discover_rapl_topology

        try:
            topology = discover_rapl_topology()
            probes["cpu_energy"] = package_energy_capability(topology)
            dram_status = topology["dram_status"]
            if dram_status == "unavailable" and not topology.get("errors") and not any(
                    item["kind"] == "dram" for item in topology["domains"]):
                dram_status = "unsupported" if probes["cpu_energy"].status == "available" else probes["cpu_energy"].status
            probes["dram_energy"] = Capability(dram_status, "RAPL DRAM package coverage",
                "rapl_topology", {"missing_packages": topology["dram_missing_packages"]})
        except (OSError, ValueError, RuntimeError) as exc:
            for name in ("cpu_energy", "dram_energy"):
                probes[name] = capability_from_error(exc, source="rapl_topology")
        if dram_energy == "off":
            probes["dram_energy"] = Capability("not_requested", "DRAM disabled", "dram_energy")
        elif probes["dram_energy"].status == "available" and probes["cpu_energy"].status != "available":
            probes["dram_energy"] = Capability(probes["cpu_energy"].status, "Package RAPL collector unavailable: " +
                                               probes["cpu_energy"].detail, "rapl_topology")
        def perf():
            prefix = resolve_perf_command_prefix(docker_attach=runtime_kind == "docker", env=env)
            try:
                events = probe_event_set(prefix, env=env)
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                from acprof.monitors.perf_mips import PERF_OPTIONAL_EVENTS
                events = {"events": ["instructions"], "unavailable_events": list(PERF_OPTIONAL_EVENTS),
                          "event_statuses": {name: "probe_error" for name in PERF_OPTIONAL_EVENTS},
                          "optional_probe_error": str(exc)}
            return {"prefix": prefix, "target_check": "at_workload_attach", **events}
        probes["cpu_instructions"] = (_probe(perf, "perf_stat")
            if shutil.which("perf") else Capability("tool_missing", "perf executable missing", "perf_stat"))
        interface = "lo" if runtime_kind == "process" else sniff_iface
        missing = [tool for tool in ("tcpdump", "tshark") if not shutil.which(tool)]
        probes["packet_latency"] = (Capability("tool_missing", ", ".join(missing), "pcap") if missing else
            _probe(lambda: probe_packet_tools(interface), "pcap"))
    else:
        status = "not_requested" if profiling_mode == "basic" else "unsupported"
        probes.update({name: Capability(status, environment.label, "platform_policy") for name in advanced})
        if dram_energy == "off":
            probes["dram_energy"] = Capability("not_requested", "DRAM disabled", "dram_energy")
    probes.update(_gpu_probes(gpu, hardware_energy=hardware))
    if profiling_mode == "basic" and not gpu:
        # Historical Process Basic capabilities used not_requested for disabled GPU queries.
        probes.update({name: Capability("not_requested", source="nvml") for name in ("gpu_memory", "gpu_utilization")})
    fields = planned_fields(runtime_kind)
    perf_probe = probes["cpu_instructions"]
    # This process' cgroup is not the future container/owned service cgroup.
    # Target readers are resolved after attachment, and only those readings
    # can establish per-field availability for a formal window.
    field_probes = {}
    if unavailable := set(perf_probe.evidence.get("unavailable_events", ())):
        from acprof.results.coverage import PMU_EVENT_FIELDS
        status_codes = perf_probe.evidence.get("event_statuses", {})
        descriptions = {
            "combination_limited": "PMU event combination limited",
            "unsupported": "PMU event unsupported",
            "permission_denied": "PMU event permission denied",
            "probe_error": "PMU event probe failed",
        }
        for event in sorted(unavailable):
            reason = status_codes.get(event, "probe_error")
            for field in PMU_EVENT_FIELDS.get(event, ()):
                field_probes[field] = {
                    "status": reason if reason in {"unsupported", "permission_denied"} else "unavailable",
                    "reason_code": reason,
                    "detail": f"{descriptions.get(reason, 'PMU event unavailable')}: {event}",
                }

    return {"schema_version": 2, "runtime_kind": runtime_kind, "profiling_mode": profiling_mode,
            "rapl_topology": topology, "probes": {name: item.to_dict() for name, item in probes.items()},
            "collectors": ["client_http", *(name for name, item in probes.items() if item.status == "available")],
            "fields": fields, "field_probes": field_probes,
            "resource_scope": "observed_process_tree" if runtime_kind == "process" else "container_cgroup",
            "gpu_scope": "device", "environment_sharing": "unknown"}


def apply_process_plan(report, plan):
    report.requested.difference_update({'container_cpu', 'container_memory'})
    for name in ('container_cpu', 'container_memory'):
        report.measurement[name] = Capability('unsupported', 'Process uses separate process-tree metrics', 'execution_runtime')
        report.metric_support[name] = 'unsupported'
    for name in ('process_cpu', 'process_memory', 'gpu_memory', 'gpu_utilization'):
        report.measurement[name] = Capability(**plan['probes'][name])
        report.metric_support[name] = 'partial'
    if report.profiling_mode == "full":
        # Package energy is measured directly. Docker's vCPU attribution has no Process equivalent.
        report.requested.remove("cpu_energy")
        report.requested.update({"cpu_package_energy", "process_cpu", "process_memory"})
        report.metric_support["cpu_package_energy"] = report.metric_support.get("cpu_energy", "supported")
        report.measurement["cpu_energy"] = Capability("not_requested", "No cgroup attribution requested", "execution_runtime")
        for name, probe in (("cpu_package_energy", "cpu_energy"), ("cpu_instructions", "cpu_instructions"),
                            ("packet_latency", "packet_latency")):
            report.measurement[name] = Capability(**plan["probes"][probe])


def apply_collection_plan(report, plan):
    for name, probe in plan["probes"].items():
        report.measurement[name] = Capability(**probe)
    if plan["runtime_kind"] == "process":
        report.measurement["cpu_package_energy"] = report.measurement.pop("cpu_energy")
        report.requested.discard("cpu_energy")
    # Only the request protocol (and explicitly required DRAM) gates publication.
    report.requested.intersection_update({"latency", "throughput", "dram_energy", *(
        name for name, cap in report.measurement.items() if cap.source == "profiler_selection")})
