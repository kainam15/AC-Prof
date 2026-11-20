"""Execution selection is independent of model loading and measurement intensity."""
from dataclasses import replace

import pytest

from acprof.experiments.arguments import build_parser
from acprof.experiments.configuration import RunConfig
from acprof.platform import Environment, collection_policy_error


def test_process_runtime_is_a_shared_cli_and_config_option():
    args = build_parser().parse_args([
        "--model", "example/model", "--runtime", "process", "--profiling-mode", "basic",
        "--cpus", "auto", "--mems", "auto", "--gpus", "off",
    ])
    config = RunConfig.from_namespace(args).validate()
    assert config.runtime == "process"
    assert config.cpus == config.mems == "auto"


def test_restricted_linux_allows_basic_but_not_full():
    for kind in ("container_host", "vm", "cloud"):
        env = Environment(kind, "Linux")
        assert collection_policy_error(env, profiling_mode="basic") == ""
        assert collection_policy_error(env, profiling_mode="full")


def test_explicit_full_is_not_silently_changed_for_process():
    config = replace(RunConfig.smoke("example/model"), runtime="process", profiling_mode="full")
    from acprof.host.runtimes.selection import select_runtime
    assert config.validate().profiling_mode == "full"
    assert select_runtime("process", profiling_mode="full", environment=Environment("native_linux", "Linux")).selected == "process"
    with pytest.raises(ValueError, match="native Linux"):
        select_runtime("process", profiling_mode="full", environment=Environment("wsl2", "Linux"))


@pytest.mark.parametrize("runtime", ["docker", "process"])
@pytest.mark.parametrize("missing", [(), ("perf",), ("tcpdump",), ("rapl",), ("perf", "rapl", "tcpdump")])
def test_best_effort_native_probes_are_independent(monkeypatch, missing, runtime):
    from acprof.host.measurement import capture
    from acprof.host.runtimes.capabilities import probe_collection_plan
    from acprof.monitors import perf_mips, rapl_topology

    def topology():
        if "rapl" in missing:
            raise PermissionError("powercap access denied")
        return {"domains": [{"id": "pkg", "name": "package-0", "selected": True, "kind": "package"}],
                "dram_status": "unsupported", "dram_missing_packages": ["package-0"]}
    monkeypatch.setattr(rapl_topology, "discover_rapl_topology", topology)
    monkeypatch.setattr("shutil.which", lambda name: None if name in missing else f"/fixture/{name}")
    monkeypatch.setattr(perf_mips, "resolve_perf_command_prefix", lambda **kw: ["perf"])
    monkeypatch.setattr(perf_mips, "probe_event_set", lambda *a, **kw: {"events": ["instructions"]})
    monkeypatch.setattr(capture, "probe_packet_tools", lambda interface: {"interface": interface})
    plan = probe_collection_plan(runtime, environment=Environment("native_linux", "Linux"))
    assert plan["probes"]["cpu_energy"]["status"] == ("permission_denied" if "rapl" in missing else "available")
    assert plan["probes"]["cpu_instructions"]["status"] == ("tool_missing" if "perf" in missing else "available")
    packet = plan["probes"]["packet_latency"]
    assert packet["status"] == ("tool_missing" if "tcpdump" in missing else "available")
    if packet["status"] == "available":
        assert packet["evidence"]["interface"] == ("lo" if runtime == "process" else "docker0")
    assert "client_http" in plan["collectors"]
    assert plan["probes"]["gpu_power"]["status"] == "not_applicable"
    assert "latency_app_s" in plan["fields"]["client_http"]


@pytest.mark.parametrize("runtime", ["docker", "process"])
@pytest.mark.parametrize(
    ("reason", "expected_status"),
    [("combination_limited", "unavailable"), ("unsupported", "unsupported"),
     ("permission_denied", "permission_denied"), ("probe_error", "unavailable")],
)
def test_optional_pmu_field_probe_preserves_failure_reason(monkeypatch, runtime, reason, expected_status):
    from acprof.host.measurement import capture
    from acprof.host.runtimes.capabilities import probe_collection_plan
    from acprof.monitors import perf_mips, rapl_topology

    monkeypatch.setattr(
        rapl_topology, "discover_rapl_topology",
        lambda: {"domains": [], "errors": [], "dram_status": "unsupported", "dram_missing_packages": []},
    )
    monkeypatch.setattr("shutil.which", lambda name: f"/fixture/{name}")
    monkeypatch.setattr(perf_mips, "resolve_perf_command_prefix", lambda **kw: ["perf"])
    monkeypatch.setattr(
        perf_mips, "probe_event_set", lambda *args, **kwargs: {
            "events": ["instructions", "cycles", "dTLB-loads"],
            "unavailable_events": ["dTLB-load-misses"],
            "event_statuses": {"dTLB-load-misses": reason},
        },
    )
    monkeypatch.setattr(capture, "probe_packet_tools", lambda iface: {"interface": iface})
    plan = probe_collection_plan(runtime, environment=Environment("native_linux", "Linux"))
    perf = plan["probes"]["cpu_instructions"]
    assert perf["status"] == "available"
    for field in ("cpu_dtlb_load_misses_per_request", "cpu_dtlb_load_miss_rate_pct"):
        failure = plan["field_probes"][field]
        assert failure["status"] == expected_status
        assert failure["reason_code"] == reason
        assert "dTLB-load-misses" in failure["detail"]


def test_guest_fallback_skips_native_probes_and_uses_process(monkeypatch):
    from acprof.capabilities import Capability
    from acprof.host.runtimes.capabilities import probe_collection_plan
    from acprof.host.runtimes.selection import select_runtime
    environment = Environment("wsl2", "Linux")
    choice = select_runtime("auto", profiling_mode="best_effort", environment=environment,
                            docker_probe=lambda: Capability("permission_denied", "Docker socket"))
    assert choice.selected == "process"
    monkeypatch.setattr("acprof.monitors.perf_mips.resolve_perf_command_prefix", lambda **kw: pytest.fail("unexpected perf"))
    plan = probe_collection_plan(choice.selected, environment=environment)
    assert plan["probes"]["cpu_instructions"]["status"] == "unsupported"
    assert "client_http" in plan["collectors"]


def test_gpu_plan_checks_each_query_and_falls_back_to_power(monkeypatch):
    import sys
    from types import SimpleNamespace

    from acprof.host.runtimes.capabilities import _gpu_probes
    calls = []
    def denied(*args):
        raise PermissionError("counter access denied")
    nvml = SimpleNamespace(nvmlInit=lambda: calls.append("init"), nvmlShutdown=lambda: calls.append("shutdown"),
        nvmlDeviceGetHandleByUUID=lambda uuid: uuid, nvmlDeviceGetTotalEnergyConsumption=denied,
        nvmlDeviceGetPowerUsage=lambda _: 1200)
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setattr("acprof.monitors.gpu_telemetry.read_gpu_sample", lambda nv, h, fail: fail("temperature", PermissionError("temperature denied")))
    assert _gpu_probes({}, hardware_energy=True)["gpu_power"].status == "not_applicable"
    assert not calls
    probes = _gpu_probes({"uuid": "GPU-1"}, hardware_energy=True)
    assert calls == ["init", "shutdown"]
    assert probes["gpu_power"].status == "available"
    assert probes["gpu_power"].evidence["counter"]["status"] == "permission_denied"
    assert probes["gpu_power"].evidence["power"]["status"] == "available"
    assert probes["gpu_temperature"].status == "permission_denied"
    assert probes["gpu_memory"].status == "available"
