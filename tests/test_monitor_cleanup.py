"""Exercise the real client loop with failures at its external monitor boundary."""
from acprof.host.client import ClientRunner
from acprof.host.client_config import ClientConfig
from client_fixtures import patch_client
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from acprof.host import client
from acprof.host.measurement_window import MonitorGroup
from acprof.monitors import energy_cpu, resource_usage


class MonitorCleanupTests(unittest.TestCase):
    def setUp(self):
        self.runner = ClientRunner(ClientConfig())

    def run_failure(self, fault, *, request_error=None, journal_error=None):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            if journal_error is not None:
                stack.enter_context(patch_client(self.runner, "_append_request_window", side_effect=journal_error))
            path = Path(temporary) / "case.csv"
            cpu = Mock()
            cpu.idle_trace = {}
            cpu.idle_power_w = 1.0
            cpu.stop.return_value = (energy_cpu._nan_result(), "", [])
            resource = Mock()
            resource.stop.return_value = (resource_usage._nan_result(), "", [])
            monitors = {"cpu": cpu, "resource": resource}
            name, operation = fault.split(".")
            getattr(monitors[name], operation).side_effect = RuntimeError(f"{fault} failed")
            settings = {
                "OUT_CSV": str(path), "PROFILING_MODE": "full", "DRAM_ENERGY": "auto",
                "IDLE_DEBUG": False, "WARMUP": 0, "REPEAT": 2, "REPEAT_IN_WINDOW": 1,
                "USE_ENERGY": False, "USE_MIPS": False, "GPU_MODE": "off", "BATCH_SIZE": 1,
                "energy_mod": None,
                "resource_usage_mod": SimpleNamespace(ResourceUsageMonitor=Mock(return_value=resource)),
                "cpu_energy_mod": SimpleNamespace(CPUEnergyMonitor=Mock(return_value=cpu)),
                "input_scale_entries": [{"input_scale": 1.0, "scale_label": "one", "payload": {}}],
            }
            for key, value in settings.items():
                stack.enter_context(patch_client(self.runner, key, value))
            stack.enter_context(patch.object(client.requests, "get", return_value=SimpleNamespace(status_code=200, text="ok")))
            stack.enter_context(patch_client(self.runner, "_run_matched_control_window"))
            stack.enter_context(patch_client(self.runner, "_sleep_before_idle_baseline"))
            request = stack.enter_context(patch_client(self.runner, "_one_request", side_effect=request_error,
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0,
                              "workload_contract": {"schema_version": 1, "actual_rows": 1}}))
            expected_error = (self.assertRaises(type(request_error))
                              if request_error is not None and not isinstance(request_error, Exception)
                              else self.assertRaisesRegex(RuntimeError, "cleanup"))
            with redirect_stdout(io.StringIO()), expected_error:
                self.runner.main()
            self.assertEqual(request.call_count, 1, "cleanup failure must stop later windows")
            cpu.stop.assert_called_once()
            cpu.close.assert_called_once()
            resource.close.assert_called_once()
            if journal_error is not None:
                return
            records = [json.loads(line) for line in Path(str(path) + ".requests.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertIn(fault, records[0]["error"])
            if request_error is not None:
                self.assertIn(str(request_error), records[0]["error"])

    def test_resource_stop_failure_closes_all_monitors_and_keeps_request_evidence(self):
        self.run_failure("resource.stop")

    def test_cpu_stop_failure_still_closes_resource_monitor(self):
        self.run_failure("cpu.stop")

    def test_close_failure_is_fatal_and_preserves_original_request_error(self):
        self.run_failure("cpu.close", request_error=RuntimeError("inference failed"))

    def test_cleanup_failure_stays_fatal_when_request_journal_also_fails(self):
        self.run_failure("resource.stop", journal_error=OSError("journal failed"))

    def test_cancellation_survives_both_cleanup_and_journal_failure(self):
        self.run_failure("resource.stop", request_error=KeyboardInterrupt(), journal_error=OSError("journal failed"))

    def test_partial_start_still_stops_and_closes_every_owned_monitor(self):
        group = MonitorGroup()
        cpu, resource = Mock(), Mock()
        cpu.start.side_effect = RuntimeError("partial start")
        group.add("cpu", cpu)
        group.add("resource", resource)
        with self.assertRaisesRegex(RuntimeError, "partial start"):
            try:
                group.start()
            finally:
                group.finish(0, float("nan"))
        cpu.stop.assert_called_once()
        resource.stop.assert_not_called()
        cpu.close.assert_called_once()
        resource.close.assert_called_once()

    def test_cancelled_stop_is_propagated_after_remaining_cleanup(self):
        group = MonitorGroup()
        cpu, resource = Mock(), Mock()
        resource.stop.side_effect = KeyboardInterrupt()
        group.add("cpu", cpu)
        group.add("resource", resource)
        group.start()
        group.finish(1, 0.1)
        with self.assertRaises(KeyboardInterrupt):
            group.raise_if_failed()
        cpu.stop.assert_called_once()
        cpu.close.assert_called_once()
        resource.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
