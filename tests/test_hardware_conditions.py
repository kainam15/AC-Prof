import tempfile
import unittest
from pathlib import Path

from acprof.run_args import build_parser


class HardwareConditionsTests(unittest.TestCase):
    def comparison_snapshot(self):
        from test_result_comparison import ResultComparisonTests

        from acprof.analysis.comparison import load_comparison_snapshot
        fixture = ResultComparisonTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return load_comparison_snapshot(fixture.left)

    def test_cpu_set_is_optional_and_equivalent_spellings_are_canonical(self):
        from acprof.cpu_affinity import normalize_cpu_set
        self.assertEqual(normalize_cpu_set(""), "")
        self.assertEqual(normalize_cpu_set("8,2,0-1,2"), "0-2,8")
        for value in ("1-0", "1,", "-1", "0-99999999999", "1.0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_cpu_set(value)

    def test_optional_cpu_set_is_part_of_requested_run_options(self):
        from acprof.host.run_state import run_options
        args = build_parser().parse_args(["--model", "fixture/model", "--cpuset-cpus", "2,0-1"])
        self.assertEqual(run_options(args)["cpuset_cpus"], "0-2")

    def test_old_results_without_hardware_are_unknown(self):
        from unittest.mock import patch

        from acprof.analysis.comparison import compare_results
        snapshot = self.comparison_snapshot()
        snapshot["hardware"] = dict.fromkeys(snapshot["hardware"])
        with tempfile.TemporaryDirectory() as directory, patch(
            "acprof.analysis.comparison.load_comparison_snapshot", return_value=snapshot
        ):
            report = compare_results(Path(directory), Path(directory), purpose="same-hardware")
        self.assertEqual(report["status"], "unknown")
        self.assertEqual(report["conditions"]["hardware_cpu_model"]["status"], "unknown")
        self.assertFalse(report["native_baseline_eligible"])

    def test_purpose_distinguishes_hardware_changes_from_missing_evidence(self):
        from copy import deepcopy
        from unittest.mock import patch

        from acprof.analysis.comparison import compare_results
        left = self.comparison_snapshot()
        right = deepcopy(left)
        right["hardware"]["cpu_model"]["1c_4g_off"] = ["different CPU"]
        for purpose, expected in (("same-hardware", "incompatible"), ("cross-hardware", "compatible")):
            with patch("acprof.analysis.comparison.load_comparison_snapshot", side_effect=[left, right]):
                report = compare_results("a", "b", purpose=purpose)
            self.assertEqual(report["status"], expected)
        right["hardware"]["gpu"] = None
        with patch("acprof.analysis.comparison.load_comparison_snapshot", side_effect=[left, right]):
            self.assertEqual(compare_results("a", "b", purpose="cross-hardware")["status"], "unknown")

    def test_requested_affinity_failure_is_recorded_and_stops_before_measurement(self):
        import json
        from unittest.mock import patch

        from acprof.host.hardware_conditions import HARDWARE_FIELDS, record_case_conditions
        record = {**dict.fromkeys(HARDWARE_FIELDS), "cpu_affinity": ["0-7"], "errors": []}
        with tempfile.TemporaryDirectory() as directory, patch(
            "acprof.host.hardware_conditions.observe_conditions", return_value=record
        ):
            with self.assertRaisesRegex(RuntimeError, "CPU set"):
                record_case_conditions(directory, "1c_4g_off", object(), cpuset_cpus="0-1")
            payload = json.loads(Path(directory, "hardware_conditions.json").read_text())
            self.assertTrue(payload["cases"]["1c_4g_off"]["errors"])

    def test_cross_hardware_does_not_hide_unknown_policy_under_a_known_difference(self):
        from copy import deepcopy
        from unittest.mock import patch

        from acprof.analysis.comparison import compare_results
        left = self.comparison_snapshot()
        right = deepcopy(left)
        left["hardware"]["gpu"]["1c_4g_off"] = {"model": "A", "power_limit_w": None}
        right["hardware"]["gpu"]["1c_4g_off"] = {"model": "B", "power_limit_w": 100}
        with patch("acprof.analysis.comparison.load_comparison_snapshot", side_effect=[left, right]):
            report = compare_results("a", "b", purpose="cross-hardware")
        self.assertEqual(report["conditions"]["hardware_gpu"]["status"], "unknown")
        self.assertEqual(report["status"], "unknown")

    def test_runtime_may_narrow_each_thread_within_the_requested_cpu_set(self):
        from unittest.mock import patch

        from acprof.host.hardware_conditions import HARDWARE_FIELDS, record_case_conditions
        record = {**dict.fromkeys(HARDWARE_FIELDS), "cpu_affinity": ["0", "1-2"], "errors": []}
        with tempfile.TemporaryDirectory() as directory, patch(
            "acprof.host.hardware_conditions.observe_conditions", return_value=record
        ):
            record_case_conditions(directory, "1c_4g_off", object(), cpuset_cpus="2,0-1")
        self.assertFalse(record["errors"])
