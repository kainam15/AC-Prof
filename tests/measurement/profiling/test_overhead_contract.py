"""Exercise the real full diagnostic orchestration; only hardware and HTTP are fake."""
import json
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from acprof.monitors.interfaces import CPUStopResult, GPUStopResult, ResourceStopResult
from acprof.monitors.sampling import SamplingDiagnostics
from scripts import measure_overhead as overhead
from tests.measurement.profiling.test_overhead import group


@dataclass
class PerfValue:
    instructions_total: float = 100.0


class TestOverheadContract:
    def window(self, failure=""):
        events = []
        monitors = {}

        def collector(name):
            monitor = Mock()
            monitor.sampling_diagnostics = SamplingDiagnostics("skip", 0.05)
            monitors[name] = monitor
            monitor.start.side_effect = lambda: events.append(name + ".start")

            def stop(*_args):
                events.append(name + ".stop")
                if failure == "control" and name == "cpu":
                    raise RuntimeError("control collector failed")
                if name == "mips":
                    return PerfValue()
                if name == "gpu":
                    return GPUStopResult(object(), "GPU", "", [1, 2])
                result_type = CPUStopResult if name == "cpu" else ResourceStopResult
                return result_type(object(), "", [1, 2])

            monitor.stop.side_effect = stop
            monitor.close.side_effect = lambda: events.append(name + ".close")
            monitor.apply_control_baseline.side_effect = lambda *_args, **_kw: events.append(name + ".baseline")
            return monitor

        def post(*_args, **_kw):
            events.append("request")
            if failure == "request":
                raise RuntimeError("request failed")
            response = Mock(status_code=200)
            response.json.return_value = {"workload_contract": {"input": {"actual_scale": 4}}}
            return response

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            output = Path(directory)
            pcap = output / "full.pcap"
            pcap.write_bytes(b"p" * 32)
            runtime = SimpleNamespace(tcpdump_cmd=["tcpdump"], parse_cmd=["parser"])
            capture = Mock(returncode=0)
            capture.poll.return_value = None
            capture.terminate.side_effect = lambda: events.append("capture.stop")
            stack.enter_context(patch("acprof.host.measurement.capture.resolve_packet_latency_runtime", return_value=runtime))
            stack.enter_context(patch("acprof.host.measurement.capture.tcpdump_can_capture_without_sudo", return_value=True))
            stack.enter_context(patch("shutil.which", return_value="/fixture/tcpdump"))
            stack.enter_context(patch("subprocess.Popen", return_value=capture))
            parsed = {"requests": {"7": {"request_id": "full:0", "latency_s": 0.1}}}
            stack.enter_context(patch("subprocess.run", return_value=SimpleNamespace(
                returncode=0, stdout=json.dumps(parsed), stderr="")))
            stack.enter_context(patch("time.sleep"))
            stack.enter_context(patch("requests.post", side_effect=post))
            factories = {}
            for module, kind, name in (("energy_cpu", "CPUEnergyMonitor", "cpu"),
                                       ("energy_nvml", "GPUEnergyMonitor", "gpu"),
                                       ("resource_usage", "ResourceUsageMonitor", "resource"),
                                       ("perf_mips", "PerfMIPSMonitor", "mips")):
                factories[name] = stack.enter_context(patch(
                    f"acprof.monitors.{module}.{kind}", return_value=collector(name)))
            options = {"idle_seconds": 0.1, "idle_cooldown_seconds": 0,
                       "request_timeout_seconds": 17, "sniff_iface": "docker0"}
            def call():
                return overhead.measure_profile_window(
                    SimpleNamespace(base_url="http://fixture.invalid", gpu_device={"uuid": "GPU-recorded", "index": 2}),
                    {"payload": {}}, scenario="full", rate=20, count=1, name="owned-container",
                    cpu=2, mem=4, gpu="on", token="full", output=output, options=options,
                    sampling_policy="skip")
            if failure:
                with pytest.raises(RuntimeError, match="control collector failed" if failure == "control" else "request failed"):
                    call()
            else:
                result = call()
                assert (result["packet_request_count"]) == (1)
                assert (result["request_count"]) == (1)
                assert (pcap.with_suffix(".packets.json").exists())
                assert all(item["sampling"]["policy"] == "skip" for item in result["monitors"])
            for name in ("cpu", "gpu", "resource"):
                assert factories[name].call_args.kwargs["sampling_policy"] == "skip"
            for name, monitor in monitors.items():
                monitor.close.assert_called_once()
                expected_starts = (1 if name in {"gpu", "cpu"} else 0)
                if failure != "control":
                    expected_starts += 1
                assert monitor.start.call_count == expected_starts
                assert monitor.stop.call_count == expected_starts
            capture.terminate.assert_called_once()
        return events

    def test_full_runs_control_requests_cleanup_and_packet_report(self):
        events = self.window()
        assert events[:4] == ["gpu.start", "cpu.start", "gpu.stop", "cpu.stop"]
        assert events.index("cpu.baseline") < events.index("resource.start")
        assert events.index("cpu.baseline") < events.index("request")

    def test_control_failure_closes_every_collector_and_capture_before_request(self):
        assert ("request") not in (self.window("control"))

    def test_request_failure_stops_every_collector_and_capture(self):
        assert ("request") in (self.window("request"))

    def test_partial_start_is_stopped_even_when_start_raises(self):
        cpu, resource = Mock(), Mock()
        cpu.stop.return_value = CPUStopResult(None, "", [1, 2])
        resource.stop.return_value = ResourceStopResult(None, "", [1, 2])
        resource.start.side_effect = RuntimeError("partial start")
        with pytest.raises(RuntimeError, match="partial start"):
            overhead.measure_window("http://fixture.invalid", {}, count=1,
                                    monitors=group(cpu, resource), token="fail")
        for monitor in (cpu, resource):
            monitor.stop.assert_called_once()
            monitor.close.assert_called_once()



def test_paired_best_effort_skips_unavailable_tools_and_records_coverage(monkeypatch, tmp_path):
    from acprof.monitors import energy_cpu, energy_nvml, perf_mips, resource_usage
    response = Mock(status_code=200)
    response.json.return_value = {"workload_contract": {"input": {"actual_scale": 4}}}
    requests = Mock(return_value=response)
    monkeypatch.setattr("requests.post", requests)
    never = Mock(side_effect=AssertionError("unplanned collector"))
    for module, name in ((energy_cpu, "CPUEnergyMonitor"), (energy_nvml, "GPUEnergyMonitor"),
                         (perf_mips, "PerfMIPSMonitor")):
        monkeypatch.setattr(module, name, never)
    monkeypatch.setattr("subprocess.Popen", never)
    monkeypatch.setattr("time.sleep", lambda duration: None)
    monitor = Mock()
    monitor.stop.return_value = ResourceStopResult(object(), "", [1, 2])
    factory = Mock(return_value=monitor)
    monkeypatch.setattr(resource_usage, "ResourceUsageMonitor", factory)
    plan = {"probes": {name: {"status": status, "detail": status} for name, status in (
        ("packet_latency", "tool_missing"), ("gpu_power", "not_applicable"),
        ("cpu_instructions", "permission_denied"), ("cpu_energy", "unsupported"))}}
    options = {"collection_plan": plan, "idle_seconds": 0, "idle_cooldown_seconds": 0, "request_timeout_seconds": 10}
    def run(mode):
        return overhead.measure_profile_window(SimpleNamespace(base_url="http://fixture.invalid", gpu_device={}),
            {"payload": {"input": "fixture"}}, scenario=mode, rate=20, count=1, name="container",
            cpu=1, mem=4, gpu="off", token=mode, output=tmp_path, options=options, sampling_policy="skip")
    previous, current = run("none"), run("best_effort")
    assert previous["request_count"] == current["request_count"] == 1
    assert requests.call_count == 2
    assert current["collector_coverage_complete"] is False
    assert set(current["unavailable_collectors"]) == {"cpu_energy", "cpu_instructions", "packet_latency"}
    assert factory.call_count == monitor.start.call_count == monitor.stop.call_count == 1
    never.assert_not_called()
