"""The TUI renders the CLI comparison contract and preserves analysis task guards."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import test_independent_comparison as comparison_fixture
from textual.widgets import Button, Collapsible, DataTable, Input, Select, TabbedContent

from acprof.experiment import RunConfig
from acprof.quality import loading_quality
from acprof.tui.app import AcprofTui
from acprof.tui.reports import read_report


class IndependentReportTests(unittest.TestCase):
    def setUp(self):
        self.fixture = comparison_fixture.IndependentComparisonTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

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
            self.assertIn(expected, cells)
        self.assertIn("condition_checks", view.note)
        self.assertIn("experiments", view.note)

    def test_resource_scaling_report_shows_both_resource_coordinates(self):
        source, report = self.report()
        report['purpose'] = 'resource-scaling'
        report['allowed_resource_dimensions'] = ['cpu', 'memory']
        report['groups'][0]['resource_coordinates'] = {
            'left': {'cpu_cores': 1, 'mem_cap_gb': 4}, 'right': {'cpu_cores': 2, 'mem_cap_gb': 8}}
        source.write_text(json.dumps(report))
        view = read_report(source)
        self.assertIn('1c/4G → 2c/8G', view.rows[0].cells[1])
        self.assertIn('allowed_resource_dimensions', view.note)

    def test_insufficient_independent_runs_is_the_first_visible_explanation(self):
        source, _ = self.report(1)
        self.assertIn("insufficient_independent_runs", read_report(source).rows[0].cells[0])

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
        self.assertIn("conditions_incompatible", view.rows[0].cells[0])
        self.assertIn("weights_reinitialized", view.rows[0].detail)
        self.assertIn("weights.log", view.rows[0].detail)
        self.assertIn('"auto_selection_eligible": false', view.rows[0].detail)
        self.assertIn('"quality_reasons"', view.rows[0].detail)
        self.assertIn("quality=unknown/blocked", view.rows[0].cells[0])

    def test_coverage_v2_is_accepted_without_relaxing_other_report_versions(self):
        source = self.fixture.fixture.root / "coverage.json"
        source.write_text(json.dumps({"schema_version": 2, "scope": "selected_sample_only; no_formal_measurement",
            "rows": [{"model_id": "fixture/model", "failure": {"reason_code": "request_timeout"}, "attempt_id": "0002",
                      "cleanup_status": "incomplete", "cleanup_errors": [{"container_id": "owned-id", "state": "running"}]}]}))
        row = read_report(source).rows[0]
        self.assertEqual(row.cells[0], "fixture/model")
        self.assertIn('"cleanup_status": "incomplete"', row.detail)
        self.assertIn("owned-id", row.detail)
        source.write_text(json.dumps({"schema_version": 2, "kind": "independent_experiment_comparison"}))
        with self.assertRaises(ValueError):
            read_report(source)
        source, report = self.report()
        report.update(schema_version=2, scope="selected_sample_only; no_formal_measurement")
        source.write_text(json.dumps(report))
        with self.assertRaises(ValueError):
            read_report(source)


class IndependentComparisonUiTests(unittest.IsolatedAsyncioTestCase):
    async def test_compare_selection_baseline_and_measurement_guard_at_narrow_width(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left, right = root / "left experiment", root / "right experiment"
            for directory in (left, right):
                directory.mkdir()
                (directory / "result_all.csv").write_text("fixture")
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
                    self.assertTrue(await pilot.click(button))
                    await pilot.pause()
                    launch.assert_called_once()
                    pending = launch.call_args.args[0]
                    command = pending.command
                    self.assertEqual(command[command.index("--left") + 1], str(right))
                    self.assertEqual(command[command.index("--right") + 1], str(left))
                    self.assertIn("compare", command)
                    self.assertEqual(command[command.index("--purpose") + 1], "resource-scaling")
                    launch.reset_mock()
                    app._latest_snapshot = replace(app._latest_snapshot, measurement_active=True)
                    app.compare_experiments_button()
                    launch.assert_not_called()

    async def test_comparison_runs_public_cli_and_opens_incompatible_report(self):
        fixture = comparison_fixture.IndependentComparisonTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
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
                self.assertNotIn("error", [call.kwargs.get("severity") for call in notify.call_args_list])
            report = Path(app.query_one("#report-source", Input).value)
            self.assertTrue(report.is_file())
            payload = json.loads(report.read_text())
            self.assertEqual(payload["kind"], "independent_experiment_comparison")
            self.assertEqual(payload["status"], "incompatible")
            self.assertIn("not comparable", str(app.query_one("#report-table", DataTable).get_row_at(0)[0]))
            self.assertFalse(app._is_busy())
            self.assertEqual(app._latest_snapshot.stage, "已完成")
            self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "reports-tab")
