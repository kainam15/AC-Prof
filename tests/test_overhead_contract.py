"""Exercise the real full diagnostic orchestration; only hardware and HTTP are fake."""
import json
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_overhead import group

from scripts import measure_overhead as overhead


@dataclass
class PerfValue:
    instructions_total: float = 100.0


class OverheadContractTests(unittest.TestCase):
    def window(self, failure=""):
        events = []
        monitors = {}

        def collector(name):
            monitor = Mock()
            monitors[name] = monitor
            monitor.start.side_effect = lambda: events.append(name + ".start")

            def stop(*_args):
                events.append(name + ".stop")
                if failure == "control" and name == "resource":
                    raise RuntimeError("control collector failed")
                if name == "mips":
                    return PerfValue()
                return (object(), "GPU", "", [1, 2]) if name == "gpu" else (object(), "", [1, 2])

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
            stack.enter_context(patch("acprof.host.packet_capture._resolve_packet_latency_runtime", return_value=runtime))
            stack.enter_context(patch("acprof.host.packet_capture._tcpdump_can_capture_without_sudo", return_value=True))
            stack.enter_context(patch("shutil.which", return_value="/fixture/tcpdump"))
            stack.enter_context(patch("subprocess.Popen", return_value=capture))
            parsed = {"requests": {"full:0": {"latency_s": 0.1}}}
            stack.enter_context(patch("subprocess.run", return_value=SimpleNamespace(
                returncode=0, stdout=json.dumps(parsed), stderr="")))
            stack.enter_context(patch("time.sleep"))
            stack.enter_context(patch("requests.post", side_effect=post))
            for module, kind, name in (("energy_cpu", "CPUEnergyMonitor", "cpu"),
                                       ("energy_nvml", "GPUEnergyMonitor", "gpu"),
                                       ("resource_usage", "ResourceUsageMonitor", "resource"),
                                       ("perf_mips", "PerfMIPSMonitor", "mips")):
                stack.enter_context(patch(f"acprof.monitors.{module}.{kind}", return_value=collector(name)))
            options = {"idle_seconds": 0.1, "idle_cooldown_seconds": 0,
                       "request_timeout_seconds": 17, "sniff_iface": "docker0"}
            def call():
                return overhead.measure_profile_window(
                    SimpleNamespace(base_url="http://fixture.invalid", gpu_device={"uuid": "GPU-recorded", "index": 2}),
                    {"payload": {}}, scenario="full", rate=20, count=1, name="owned-container",
                    cpu=2, mem=4, gpu="on", token="full", output=output, options=options)
            if failure:
                with self.assertRaisesRegex(RuntimeError, "control collector failed" if failure == "control" else "request failed"):
                    call()
            else:
                result = call()
                self.assertEqual(result["packet_request_count"], 1)
                self.assertEqual(result["request_count"], 1)
                self.assertTrue(pcap.with_suffix(".packets.json").exists())
            for monitor in monitors.values():
                monitor.close.assert_called_once()
                self.assertEqual(monitor.start.call_count, 1 if failure == "control" else 2)
                self.assertEqual(monitor.stop.call_count, monitor.start.call_count)
            capture.terminate.assert_called_once()
        return events

    def test_full_runs_control_requests_cleanup_and_packet_report(self):
        events = self.window()
        self.assertEqual(events[:8], ["gpu.start", "cpu.start", "resource.start", "mips.start",
                                      "mips.stop", "resource.stop", "gpu.stop", "cpu.stop"])
        self.assertLess(events.index("cpu.baseline"), events.index("request"))

    def test_control_failure_closes_every_collector_and_capture_before_request(self):
        self.assertNotIn("request", self.window("control"))

    def test_request_failure_stops_every_collector_and_capture(self):
        self.assertIn("request", self.window("request"))

    def test_partial_start_is_stopped_even_when_start_raises(self):
        cpu, resource = Mock(), Mock()
        cpu.stop.return_value = resource.stop.return_value = (None, "", [1, 2])
        resource.start.side_effect = RuntimeError("partial start")
        with self.assertRaisesRegex(RuntimeError, "partial start"):
            overhead.measure_window("http://fixture.invalid", {}, count=1,
                                    monitors=group(cpu, resource), token="fail")
        for monitor in (cpu, resource):
            monitor.stop.assert_called_once()
            monitor.close.assert_called_once()
