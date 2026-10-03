"""报告展示保留单位、真实零和统计边界；损坏报告不能伪装成有效结果。"""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from acprof.tui.i18n import translate
from acprof.tui.reports import read_report


class TestReportView:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        temporary = tmp_path
        self.path = Path(str(temporary)) / "统计.json"
        self.group = dict(cpu_cores=2.0, mem_cap_gb=8.0, gpu_mode="off", input_scale=64.0,
                          metric="latency_app_s", unit="s", n_windows=3, missing_windows=0,
                          mean=0.02, std=0.01, ci_low=0.01, ci_high=0.03, reason="")
        self.window = dict(schema_version=1, resampling_unit="csv_request_window", confidence=0.95,
                           filter="status=ok and warmup=0", groups=[self.group], result_csv="原始/结果.csv")

    def read(self, data):
        self.path.write_text(json.dumps(data), encoding="utf-8")
        before = self.path.read_bytes()
        result = read_report(self.path)
        assert (self.path.read_bytes()) == (before)
        return result

    def test_window_units_confidence_level_zero_and_missing_values_are_explicit(self):
        data = deepcopy(self.window)
        data["confidence"] = 0.9
        data["groups"] += [dict(self.group, metric="cpu_energy_total_j", unit="J/request",
                                mean=0, std=0, ci_low=0, ci_high=0),
                           dict(self.group, n_windows=1, missing_windows=2, std=None,
                                ci_low=None, ci_high=None, reason="insufficient_windows"),
                           dict(self.group, n_windows=0, missing_windows=3, mean=None, std=None,
                                ci_low=None, ci_high=None, reason="insufficient_windows")]
        view = self.read(data)
        assert ("90%") in (str(view.title))
        assert (view.rows[0].cells[2:4]) == (("20 ms", "[10, 30] ms"))
        assert (view.rows[1].cells[2:4]) == (("0 J/request", "[0, 0] J/request"))
        assert (view.rows[2].cells[2:]) == (("20 ms", "未知", "1/2", "窗口不足"))
        assert (view.rows[3].cells[2:]) == (("未知", "未知", "0/3", "无有效窗口"))
        assert ("标准差：未知") in (view.rows[2].detail)
        assert (translate(view.rows[3].cells[2], "en")) == ("Unknown")
        assert ("原始/结果.csv") in (str(view.note))

    def test_gpu_energy_without_a_gpu_is_inapplicable_but_missing_gpu_data_is_unknown(self):
        group = {**self.group, "metric": "gpu_energy_total_j", "unit": "J/request", "n_windows": 0,
                 "missing_windows": 3, "mean": None, "std": None, "ci_low": None, "ci_high": None,
                 "reason": "insufficient_windows"}
        view = self.read({**self.window, "groups": [group, {**group, "gpu_mode": "on"}]})
        assert (view.rows[0].cells[2:4]) == (("—", "—"))
        assert (view.rows[0].cells[-1]) == ("不适用")
        assert (view.rows[1].cells[2:4]) == (("未知", "未知"))

    def test_empty_formal_results_are_viewable_without_inventing_samples(self):
        view = self.read(dict(self.window, groups=[]))
        assert (view.rows) == (())
        assert ("0 项") in (str(view.title))

    def test_quality_state_and_source_remain_visible_in_both_languages(self):
        from acprof.quality import loading_quality, summarize_quality
        data = {**self.window, "run_status": "complete", "measurement_status": "incomplete",
                **summarize_quality(loading_quality({"missing_keys": ["head.weight"]}, source="loader-log"))}
        view = self.read(data)
        for language in ("zh", "en"):
            note = translate(view.note, language)
            assert ("blocked") in (note)
            assert ("incomplete") in (note)
            assert ("weights_reinitialized") in (note)
            assert ("loader-log") in (note)
        assert ("Automatic selection: Withheld") in (translate(view.note, "en"))

    def test_ui_comparison_keeps_headless_and_terminal_scopes_distinct(self):
        data = dict(schema_version=1, kind="ui_overhead", successful=True, ui="headless",
                    pairs=[{"round": 0}, {"round": 1}, {"round": 2}], paired_mean_change_pct=-0.16,
                    ci_low_pct=-0.57, ci_high_pct=0.35)
        view = self.read(data)
        assert (view.rows[0].cells[1:4]) == (("3", "-0.16%", "[-0.57%, 0.35%]"))
        assert (translate(view.rows[0].cells[-1], "en")) == ("Uncertain direction")
        assert ("excludes terminal rendering") in (translate(view.rows[0].detail, "en"))
        terminal = self.read(dict(data, ui="terminal"))
        assert ("Includes terminal rendering") in (translate(terminal.rows[0].detail, "en"))

    @pytest.mark.parametrize('changes_case', range(9), ids=["{'mean': float('nan')}", "{'ci_low': float('inf')}", "{'ci_low': 1.0}", "{'ci_low': None}", "{'n_windows': True}", "{'n_windows': -1}", "{'n_windows': 0}", "{'mean': None}", "{'n_windows': 1}"])
    def test_invalid_numbers_intervals_and_counts_are_rejected(self, changes_case):
        changes = tuple(({'mean': float('nan')}, {'ci_low': float('inf')}, {'ci_low': 1.0}, {'ci_low': None}, {'n_windows': True}, {'n_windows': -1}, {'n_windows': 0}, {'mean': None}, {'n_windows': 1}))[changes_case]
        with pytest.raises(ValueError):
            self.read(dict(self.window, groups=[dict(self.group, **changes)]))

    @pytest.mark.parametrize('data_case', range(6), ids=["dict(schema_version=1, kind='ui_overhead', successful=False, error='interrupted'", 'dict(window, schema_version=2)', 'dict(window, schema_version=True)', "dict(window, filter='warmup=1')", "{'schema_version': 1, 'kind': 'audit'}", '[]'])
    def test_failed_unknown_and_truncated_reports_are_not_successful_views(self, data_case):
        data = tuple((dict(schema_version=1, kind='ui_overhead', successful=False, error='interrupted'), dict(self.window, schema_version=2), dict(self.window, schema_version=True), dict(self.window, filter='warmup=1'), {'schema_version': 1, 'kind': 'audit'}, []))[data_case]
        with pytest.raises(ValueError):
            self.read(data)
        self.path.write_text('{"schema_version":1,')
        with pytest.raises(ValueError, match="JSON"):
            read_report(self.path)
