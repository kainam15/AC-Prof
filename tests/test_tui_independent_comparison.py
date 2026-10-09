"""The TUI renders the CLI comparison contract and preserves analysis task guards."""
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import independent_comparison_fixtures as comparison_fixture
import pytest
from textual.widgets import Button, Collapsible, DataTable, Input, Select, TabbedContent
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig
from acprof.quality import loading_quality
from acprof.tui.reports import read_report


class TestIndependentReport:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.fixture_root = tmp_path
        self.fixture = comparison_fixture.IndependentComparisonFixture()
        self.fixture.build(self._request, self.fixture_root)

    def report(self, count=3):
        left = [self.fixture.replicate("left", i, [2, 2]) for i in range(count)]
        right = [self.fixture.replicate("right", i, [3, 3]) for i in range(count)]
        report = self.fixture.compare(left, right)
        source = self.fixture.fixture.root / "comparison.json"
        source.write_text(json.dumps(report))
        return source, report

    def test_same_cli_report_shows_means_change_intervals_and_run_count(self):
        source, _ = self.report()
        view = read_report(source)
        cells = " ".join(view.rows[0].cells)
        for expected in ("2000 ms", "3000 ms", "+50%", "[1000, 1000] ms", "[1.5, 1.5]", "3/3"):
            assert (expected) in (cells)
        assert ("condition_checks") in (view.note)
        assert ("experiments") in (view.note)

    def test_resource_scaling_report_shows_both_resource_coordinates(self):
        source, report = self.report()
        report['purpose'] = 'resource-scaling'
        report['allowed_resource_dimensions'] = ['cpu', 'memory']
        report['groups'][0]['resource_coordinates'] = {
            'left': {'cpu_cores': 1, 'mem_cap_gb': 4}, 'right': {'cpu_cores': 2, 'mem_cap_gb': 8}}
        source.write_text(json.dumps(report))
        view = read_report(source)
        assert ('1c/4G → 2c/8G') in (view.rows[0].cells[1])
        assert ('allowed_resource_dimensions') in (view.note)

    def test_insufficient_independent_runs_is_the_first_visible_explanation(self):
        source, _ = self.report(1)
        assert ("insufficient_independent_runs") in (read_report(source).rows[0].cells[0])

    def test_incompatible_conditions_and_quality_evidence_remain_traceable(self):
        source, report = self.report()
        report["status"] = "incompatible"
        group = report["groups"][0]
        group.update(comparability="not comparable", reason="conditions_incompatible", difference=None,
                     ratio=None, difference_ci=None, ratio_ci=None)
        group["right"].update(run_status="complete", measurement_status="complete",
            quality_status="blocked", auto_selection_eligible=False,
            quality_reasons=["weights_reinitialized"],
            quality_checks=loading_quality({"missing_keys": ["head.weight"]}, source="weights.log"))
        source.write_text(json.dumps(report))
        view = read_report(source)
        assert ("conditions_incompatible") in (view.rows[0].cells[0])
        assert ("weights_reinitialized") in (view.rows[0].detail)
        assert ("weights.log") in (view.rows[0].detail)
        assert ('"auto_selection_eligible": false') in (view.rows[0].detail)
        assert ('"quality_reasons"') in (view.rows[0].detail)
        assert ("quality=unknown/blocked") in (view.rows[0].cells[0])

    def test_coverage_v2_is_accepted_without_relaxing_other_report_versions(self):
        source = self.fixture.fixture.root / "coverage.json"
        source.write_text(json.dumps({"schema_version": 2, "scope": "selected_sample_only; no_formal_measurement",
            "rows": [{"model_id": "fixture/model", "failure": {"reason_code": "request_timeout"}, "attempt_id": "0002",
                      "cleanup_status": "incomplete", "cleanup_errors": [{"container_id": "owned-id", "state": "running"}]}]}))
        row = read_report(source).rows[0]
        assert (row.cells[0]) == ("fixture/model")
        assert ('"cleanup_status": "incomplete"') in (row.detail)
        assert ("owned-id") in (row.detail)
        source.write_text(json.dumps({"schema_version": 2, "kind": "independent_experiment_comparison"}))
        with pytest.raises(ValueError):
            read_report(source)
        source, report = self.report()
        report.update(schema_version=2, scope="selected_sample_only; no_formal_measurement")
        source.write_text(json.dumps(report))
        with pytest.raises(ValueError):
            read_report(source)


class TestIndependentComparisonUi:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.fixture_root = tmp_path
    async def test_compare_selection_baseline_and_measurement_guard_at_narrow_width(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left, right = root / "left experiment", root / "right experiment"
            for directory in (left, right):
                directory.mkdir()
                (directory / "result_layers.json").write_text("fixture")
            app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=root / "settings.json")
            async with app.run_test(size=(80, 24)) as pilot:
                await pilot.pause()
                app._activate_tab("reports-tab")
                app.query_one("#comparison-options", Collapsible).collapsed = False
                app.query_one("#comparison-left", Input).value = str(left)
                app.query_one("#comparison-right", Input).value = str(right)
                app.query_one("#comparison-baseline", Select).value = "right"
                app.query_one("#comparison-purpose", Select).value = "resource-scaling"
                await pilot.pause()
                button = app.query_one("#report-compare", Button)
                button.scroll_visible(animate=False, immediate=True)
                await pilot.pause()
                with patch.object(app, "_launch") as launch:
                    assert (await pilot.click(button))
                    await pilot.pause()
                    launch.assert_called_once()
                    pending = launch.call_args.args[0]
                    command = pending.command
                    assert (command[command.index("--left") + 1]) == (str(right))
                    assert (command[command.index("--right") + 1]) == (str(left))
                    assert ("compare") in (command)
                    assert (command[command.index("--purpose") + 1]) == ("resource-scaling")
                    launch.reset_mock()
                    app._latest_snapshot = replace(app._latest_snapshot, measurement_active=True)
                    app.compare_experiments_button()
                    launch.assert_not_called()

    async def test_comparison_runs_public_cli_and_opens_incompatible_report(self):
        fixture = comparison_fixture.IndependentComparisonFixture()
        fixture.build(self._request, self.fixture_root)
        left = fixture.replicate("left", 0, [2, 2])
        right = fixture.replicate("right", 0, [3, 3])
        fixture.fixture.change_json(right, "input_scale_plan.json", lambda plan: plan.update(pipeline_tag="other-task"))
        app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=fixture.fixture.root / "settings.json")
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            app._activate_tab("reports-tab")
            app.query_one("#comparison-left", Input).value = str(left)
            app.query_one("#comparison-right", Input).value = str(right)
            with patch.object(app, "notify") as notify:
                app.compare_experiments_button()
                for _ in range(2):
                    await app.workers.wait_for_complete()
                    await pilot.pause()
                assert ("error") not in ([call.kwargs.get("severity") for call in notify.call_args_list])
            report = Path(app.query_one("#report-source", Input).value)
            assert (report.is_file())
            payload = json.loads(report.read_text())
            assert (payload["kind"]) == ("independent_experiment_comparison")
            assert (payload["status"]) == ("incompatible")
            assert ("not comparable") in (str(app.query_one("#report-table", DataTable).get_row_at(0)[0]))
            assert not (app._is_busy())
            assert (app._latest_snapshot.stage) == ("已完成")
            assert (app.query_one("#main-tabs", TabbedContent).active) == ("reports-tab")
