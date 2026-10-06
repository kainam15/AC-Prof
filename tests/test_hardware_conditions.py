import tempfile
from pathlib import Path

import pytest

from acprof.run_args import build_parser


class TestHardwareConditions:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.fixture_root = tmp_path
    def comparison_snapshot(self):

        from comparison_fixtures import ComparisonFixture

        from acprof.analysis.comparison import load_comparison_snapshot
        fixture = ComparisonFixture()
        fixture.build(self._request.getfixturevalue("tmp_path_factory").mktemp("comparison"))
        return load_comparison_snapshot(fixture.left)

    @pytest.mark.parametrize('value', ('1-0', '1,', '-1', '0-99999999999', '1.0'))
    def test_cpu_set_is_optional_and_equivalent_spellings_are_canonical(self, value):
        from acprof.cpu_affinity import normalize_cpu_set
        assert (normalize_cpu_set("")) == ("")
        assert (normalize_cpu_set("8,2,0-1,2")) == ("0-2,8")
        with pytest.raises(ValueError):
            normalize_cpu_set(value)

    def test_optional_cpu_set_is_part_of_requested_run_options(self):
        from acprof.host.run_state import run_options
        args = build_parser().parse_args(["--model", "fixture/model", "--cpuset-cpus", "2,0-1"])
        assert (run_options(args)["cpuset_cpus"]) == ("0-2")

    def test_old_results_without_hardware_are_unknown(self):
        from unittest.mock import patch

        from acprof.analysis.comparison import compare_results
        snapshot = self.comparison_snapshot()
        snapshot["hardware"] = dict.fromkeys(snapshot["hardware"])
        with tempfile.TemporaryDirectory() as directory, patch(
            "acprof.analysis.comparison.load_comparison_snapshot", return_value=snapshot
        ):
            report = compare_results(Path(directory), Path(directory), purpose="same-hardware")
        assert (report["status"]) == ("unknown")
        assert (report["conditions"]["hardware_cpu_model"]["status"]) == ("unknown")
        assert not (report["native_baseline_eligible"])

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
            assert (report["status"]) == (expected)
        right["hardware"]["gpu"] = None
        with patch("acprof.analysis.comparison.load_comparison_snapshot", side_effect=[left, right]):
            assert (compare_results("a", "b", purpose="cross-hardware")["status"]) == ("unknown")

    def test_requested_affinity_failure_is_recorded_and_stops_before_measurement(self):
        import json
        from unittest.mock import patch

        from acprof.host.hardware_conditions import HARDWARE_FIELDS, record_case_conditions
        record = {**dict.fromkeys(HARDWARE_FIELDS), "cpu_affinity": ["0-7"], "errors": []}
        with tempfile.TemporaryDirectory() as directory, patch(
            "acprof.host.hardware_conditions.observe_conditions", return_value=record
        ):
            with pytest.raises(RuntimeError, match="CPU set"):
                record_case_conditions(directory, "1c_4g_off", object(), cpuset_cpus="0-1")
            payload = json.loads(Path(directory, "hardware_conditions.json").read_text())
            assert (payload["cases"]["1c_4g_off"]["errors"])

    def test_persisted_conditions_reject_nonfinite_json_before_observation(self):
        from unittest.mock import patch

        from acprof.host.hardware_conditions import record_case_conditions

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "hardware_conditions.json")
            path.write_text('{"schema_version":1,"cases":{},"corrupt_metric":NaN}')
            original = path.read_bytes()
            with patch(
                "acprof.host.hardware_conditions.observe_conditions",
                side_effect=AssertionError("observation started before persisted evidence validation"),
            ) as observe:
                with pytest.raises(ValueError, match="invalid hardware conditions JSON"):
                    record_case_conditions(directory, "1c_4g_off", object())
            observe.assert_not_called()
            assert path.read_bytes() == original

    def test_persisted_conditions_read_is_bounded(self):
        from unittest.mock import patch

        from acprof.host.hardware_conditions import record_case_conditions

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "hardware_conditions.json")
            path.write_text(
                '{"schema_version":1,"cases":{},"padding":"'
                + ("x" * (4 * 1024 * 1024))
                + '"}'
            )
            with patch("acprof.host.hardware_conditions.observe_conditions") as observe:
                with pytest.raises(ValueError, match="4 MiB read limit"):
                    record_case_conditions(directory, "1c_4g_off", object())
            observe.assert_not_called()

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
        assert (report["conditions"]["hardware_gpu"]["status"]) == ("unknown")
        assert (report["status"]) == ("unknown")

    def test_runtime_may_narrow_each_thread_within_the_requested_cpu_set(self):
        from unittest.mock import patch

        from acprof.host.hardware_conditions import HARDWARE_FIELDS, record_case_conditions
        record = {**dict.fromkeys(HARDWARE_FIELDS), "cpu_affinity": ["0", "1-2"], "errors": []}
        with tempfile.TemporaryDirectory() as directory, patch(
            "acprof.host.hardware_conditions.observe_conditions", return_value=record
        ):
            record_case_conditions(directory, "1c_4g_off", object(), cpuset_cpus="2,0-1")
        assert not (record["errors"])
