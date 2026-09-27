import tempfile
import unittest
from pathlib import Path

from acprof.cli.run_args import build_parser


class HardwareConditionsTests(unittest.TestCase):
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
        snapshot = {"run_id": "a", "result_csv": "a.csv", "valid": True, "issues": [],
                    "conditions": {"inputs": "same"}, "identity": {}, "hardware": {}}
        with tempfile.TemporaryDirectory() as directory, patch(
            "acprof.analysis.comparison._snapshot", return_value=snapshot
        ):
            report = compare_results(Path(directory), Path(directory), purpose="same-hardware")
        self.assertEqual(report["status"], "unknown")

    def test_purpose_distinguishes_hardware_changes_from_missing_evidence(self):
        from unittest.mock import patch
        from copy import deepcopy
        from acprof.analysis.comparison import compare_results
        from acprof.host.hardware_conditions import HARDWARE_FIELDS
        left = {"run_id": "a", "result_csv": "a.csv", "valid": True, "issues": [],
                "conditions": {"inputs": "same"}, "identity": {},
                "hardware": dict.fromkeys(HARDWARE_FIELDS, "same")}
        right = deepcopy(left)
        right["hardware"]["cpu_model"] = "different CPU"
        for purpose, expected in (("same-hardware", "incompatible"), ("cross-hardware", "compatible")):
            with patch("acprof.analysis.comparison._snapshot", side_effect=[left, right]):
                report = compare_results("a", "b", purpose=purpose)
            self.assertEqual(report["status"], expected)
        right["hardware"]["gpu"] = None
        with patch("acprof.analysis.comparison._snapshot", side_effect=[left, right]):
            self.assertEqual(compare_results("a", "b", purpose="cross-hardware")["status"], "unknown")

    def test_requested_affinity_failure_is_recorded_and_stops_before_measurement(self):
        import json
        from unittest.mock import patch
        from acprof.host.hardware_conditions import record_case_conditions, HARDWARE_FIELDS
        record = {**dict.fromkeys(HARDWARE_FIELDS), "cpu_affinity": ["0-7"], "errors": []}
        with tempfile.TemporaryDirectory() as directory, patch(
            "acprof.host.hardware_conditions.observe_conditions", return_value=record
        ):
            with self.assertRaisesRegex(RuntimeError, "CPU set"):
                record_case_conditions(directory, "1c_4g_off", object(), cpuset_cpus="0-1")
            payload = json.loads(Path(directory, "hardware_conditions.json").read_text())
            self.assertTrue(payload["cases"]["1c_4g_off"]["errors"])

    def test_cross_hardware_does_not_hide_unknown_policy_under_a_known_difference(self):
        from unittest.mock import patch
        from copy import deepcopy
        from acprof.analysis.comparison import compare_results
        from acprof.host.hardware_conditions import HARDWARE_FIELDS
        left = {"run_id": "a", "result_csv": "a.csv", "valid": True, "issues": [],
                "conditions": {"inputs": "same"}, "identity": {},
                "hardware": dict.fromkeys(HARDWARE_FIELDS, "same")}
        right = deepcopy(left)
        left["hardware"]["gpu"] = {"model": "A", "power_limit_w": None}
        right["hardware"]["gpu"] = {"model": "B", "power_limit_w": 100}
        with patch("acprof.analysis.comparison._snapshot", side_effect=[left, right]):
            report = compare_results("a", "b", purpose="cross-hardware")
        self.assertEqual(report["conditions"]["hardware_gpu"]["status"], "unknown")
        self.assertEqual(report["status"], "unknown")

    def test_runtime_may_narrow_each_thread_within_the_requested_cpu_set(self):
        from unittest.mock import patch
        from acprof.host.hardware_conditions import record_case_conditions, HARDWARE_FIELDS
        record = {**dict.fromkeys(HARDWARE_FIELDS), "cpu_affinity": ["0", "1-2"], "errors": []}
        with tempfile.TemporaryDirectory() as directory, patch(
            "acprof.host.hardware_conditions.observe_conditions", return_value=record
        ):
            record_case_conditions(directory, "1c_4g_off", object(), cpuset_cpus="2,0-1")
        self.assertFalse(record["errors"])
