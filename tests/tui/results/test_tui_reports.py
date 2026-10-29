"""验证用户从 TUI 计算和查看报告，保持原始结果与测量隔离。"""
import csv
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import (
    Button,
    Collapsible,
    ContentSwitcher,
    DataTable,
    Input,
    Static,
    TabbedContent,
    TabPane,
)
from tests.support.tui_fixtures import AcprofTui

from acprof.experiment import RunConfig
from acprof.tui.progress import ProgressSnapshot
from acprof.tui.reports import ReportView
from acprof.tui.run_results import RunResult


class TestTuiReports:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        temporary = tmp_path
        self.directory = Path(str(temporary))
        self.csv_path = self.directory / "模型结果.csv"
        fields = ["cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", "warmup", "repeat_idx",
                  "status", "error", "repeat_in_window", "latency_app_s", "latency_s",
                  "container_attributed_energy_eff_j"]
        with self.csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, latency in enumerate((0.01, 0.02, 0.03, 99, 999)):
                writer.writerow(dict(cpu_cores=2, mem_cap_gb=8, gpu_mode="off", input_scale=64,
                                     warmup=int(index == 3), repeat_idx=index,
                                     status="error" if index == 4 else "ok",
                                     error="deliberate failed request" if index == 4 else "",
                                     repeat_in_window=1000 if index == 0 else 1,
                                     latency_app_s=latency, latency_s=latency,
                                     container_attributed_energy_eff_j=0.1))
    async def test_compatibility_report_preserves_codes_at_narrow_width_in_both_languages(self):
        source = self.directory / "coverage.json"
        source.write_text(json.dumps({"schema_version": 1, "scope": "selected_sample_only; no_formal_measurement",
            "rows": [{"model_id": "fixture/sam2", "failure": {"reason_code": "compatibility_budget_exhausted", "detail": "budget exhausted"}}]}))
        original = source.read_bytes()
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await self.open_tab(app, pilot)
            app.results.open_report(str(source))
            await self.finish_workers(app, pilot)
            table = app.query_one("#report-table", DataTable)
            for language, title in (("zh", "模型"), ("en", "Model")):
                app.configuration.preferences = replace(app.configuration.preferences, language=language)
                app.configuration.apply_ui_preferences()
                await pilot.pause()
                assert (str(next(iter(table.columns.values())).label)) == (title)
                assert ([str(cell) for cell in table.get_row_at(0)]) == (["fixture/sam2", "inconclusive", "compatibility_budget_exhausted"])
                assert not (app.latest_snapshot.measurement_active)
        assert (source.read_bytes()) == (original)

    async def test_v2_statistics_uses_stdout_without_creating_analysis_directories(self):
        from acprof.artifact_layout import ArtifactLayout
        root = self.directory / "v2"
        ArtifactLayout.for_new_run(root).initialize()
        csv_path = root / "result_all.csv"
        csv_path.write_bytes(self.csv_path.read_bytes())
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.open_tab(app, pilot)
            with patch.object(app, 'launch') as launch:
                app.results.launch_stats(str(csv_path))
            launch.assert_called_once()
            command = launch.call_args.args[0].command
            assert "--output-dir" not in command
            assert "--output" not in command
            assert str(csv_path) in command
            assert not (root / "plots/analysis").exists()
            assert not (root / "analysis").exists()

    async def test_experiment_heading_displays_model_and_time_in_both_languages(self):
        from acprof.artifact_layout import ArtifactLayout

        run_id = '33dec6b68b6a4369a278d1b1744b71e4'
        model = 'timm/mobilenetv3_small_100.lamb_in1k'
        root = self.directory / run_id
        layout = ArtifactLayout.for_new_run(root)
        layout.initialize()
        layout.path('run_state.json').write_text(json.dumps({
            'schema_version': 1, 'run_id': run_id,
            'created_at': '2026-10-29T20:58:30',
            'options': {'model': model},
        }), encoding='utf-8')
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.open_tab(app, pilot)
            app.results.select_result(str(root / 'result_layers.json'), summarize=False)
            await pilot.pause()
            label = app.query_one('#report-experiment', Static)
            for language, prefix in [('zh', '当前实验：'), ('en', 'Current experiment: ')]:
                app.configuration.preferences = replace(app.configuration.preferences, language=language)
                app.configuration.apply_ui_preferences()
                await pilot.pause()
                assert str(label.content) == f'{prefix}{model} · 2026-10-29 20:58:30'
                assert run_id in str(label.tooltip)
                assert str(root / 'result_layers.json') in str(label.tooltip)
            app.results.select_result(str(self.csv_path), summarize=False)
            await pilot.pause()
            assert str(label.content) == 'Current experiment: 模型结果.csv'
            app.results.select_result('', summarize=False)
            await pilot.pause()
            assert str(label.content) == 'Current experiment: None selected'

    def make_app(self):
        return AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.directory / "tui.json")

    async def open_tab(self, app, pilot):
        await pilot.pause()
        tabs = app.query_one("#main-tabs", TabbedContent)
        assert ("results-tab") in ([pane.id for pane in tabs.query(TabPane)]), "TUI 应提供统计报告页，让用户查看已有报告和计算窗口统计"
        app.results.show_result_section("statistics-view")
        await pilot.pause()

    async def click_import(self, app, pilot):
        app.results.show_result_section("exports-view")
        app.query_one("#report-file-options", Collapsible).collapsed = False
        button = app.query_one("#report-open", Button)
        button.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert await pilot.click(button)

    async def finish_workers(self, app, pilot):
        # Complete the owned analysis process or imported-report reader.
        for _ in range(2):
            await app.workers.wait_for_complete()
            await pilot.pause()

    def overhead_report(self, name="开销.json", **changes):
        data = dict(schema_version=1, kind="monitor_overhead_diagnostic", successful=True,
                    gpu_mode="off", source_run_id="original-run", image_id="sha256:example",
                    comparisons=[dict(scenario="monitors-20", paired_rounds=5,
                                      paired_mean_change_pct=2.0, ci_low_pct=-1.0, ci_high_pct=3.0)])
        data.update(changes)
        path = self.directory / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    async def test_calculate_uses_formal_windows_without_saving_a_report_or_changing_csv(self):
        original = self.csv_path.read_bytes()
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.open_tab(app, pilot)
            app.query_one("#result-csv", Input).value = str(self.csv_path)
            assert (await pilot.click("#report-calculate"))
            await self.finish_workers(app, pilot)
            assert (app.query_one("#main-tabs", TabbedContent).active) == ("results-tab")
            assert app.query_one("#report-json-path", Input).value == ""
            assert app.query_one("#result-csv", Input).value == str(self.csv_path)
            assert not list(self.directory.rglob("window-statistics-*.json"))
            assert app.results.report_view is not None
            data = app.results.report_view.payload
            latency = next(row for row in data["groups"] if row["metric"] == "latency_app_s")
            assert (latency["n_windows"]) == (3)
            assert (latency["mean"]) == (0.02) or round(abs((latency["mean"]) - (0.02)), 7) == 0
            table = app.query_one("#report-table", DataTable)
            assert (table.row_count) == (3)
            cells = [str(cell) for cell in table.get_row_at(0)]
            assert ("20 ms") in (cells)
            assert not (app.is_busy())
            assert not (app.query_one("#start-run", Button).disabled)
        assert (self.csv_path.read_bytes()) == (original)

    async def test_repeated_statistics_stay_in_memory_and_preserve_historical_exports(self):
        original = self.csv_path.read_bytes()
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await self.open_tab(app, pilot)
            app.query_one("#result-csv", Input).value = str(self.csv_path)
            assert await pilot.click("#report-calculate")
            await self.finish_workers(app, pilot)
            assert app.results.report_view is not None
            payload = app.results.report_view.payload
            existing = self.directory / "window-statistics-historical.json"
            existing.write_text(json.dumps(payload))
            saved, modified = existing.read_bytes(), existing.stat().st_mtime_ns
            for language in ("zh", "en"):
                app.configuration.preferences = replace(app.configuration.preferences, language=language)
                app.configuration.apply_ui_preferences()
                await pilot.pause()
                assert await pilot.click("#report-calculate")
                await self.finish_workers(app, pilot)
                assert app.results.report_view.payload == payload
                assert app.query_one("#report-json-path", Input).value == ""
                assert list(self.directory.rglob("window-statistics-*.json")) == [existing]
                assert existing.read_bytes() == saved
                assert existing.stat().st_mtime_ns == modified
                assert app.query_one("#report-table", DataTable).row_count == 3
                assert not app.is_busy()
        assert self.csv_path.read_bytes() == original

    @pytest.mark.parametrize('size', [(80, 24), (120, 30), (150, 45)])
    async def test_report_empty_loading_error_and_recovery(self, size):
        report = self.overhead_report()
        app = self.make_app()
        async with app.run_test(size=size) as pilot:
            await self.open_tab(app, pilot)
            table = app.query_one("#report-table", DataTable)
            content = app.query_one("#report-content", ContentSwitcher)
            for language in ("zh", "en"):
                app.configuration.preferences = replace(app.configuration.preferences, language=language)
                app.configuration.apply_ui_preferences()
                await pilot.pause()
                assert content.current == "report-empty"
                assert table.region.width == 0, "Empty table should not occupy the report view"
            app.query_one("#report-file-options", Collapsible).collapsed = False
            json_path = app.query_one("#report-json-path", Input)
            json_path.value = str(report)
            await pilot.pause()
            await self.click_import(app, pilot)
            await self.finish_workers(app, pilot)
            assert content.current == "report-results"
            assert table.row_count == 1
            json_path.value = str(self.directory / "missing.json")
            await pilot.pause()
            assert content.current == "report-empty"
            await self.click_import(app, pilot)
            await self.finish_workers(app, pilot)
            assert content.current == "report-error"
            assert table.row_count == 0
            json_path.value = str(report)
            await self.click_import(app, pilot)
            await self.finish_workers(app, pilot)
            assert content.current == "report-results"
            assert table.row_count == 1

    @pytest.mark.parametrize('size', [(80, 24), (120, 30), (150, 45)])
    async def test_open_overhead_at_all_sizes_and_language_switch_preserves_data_and_draft(self, size):
        report = self.overhead_report()
        original = report.read_bytes()
        app = self.make_app()
        async with app.run_test(size=size) as pilot:
            await self.open_tab(app, pilot)
            files = app.query_one("#report-file-options", Collapsible)
            files.collapsed = False
            source = app.query_one("#report-json-path", Input)
            source.value = str(report)
            await pilot.pause()
            await self.click_import(app, pilot)
            await self.finish_workers(app, pilot)
            files.collapsed = True
            await pilot.pause()
            table = app.query_one("#report-table", DataTable)
            assert table.row_count == 1
            assert ("+2%") in ([str(cell) for cell in table.get_row_at(0)])
            assert ("方向不确定") in ([str(cell) for cell in table.get_row_at(0)])
            for selector in ("#report-calculate", "#report-table"):
                region = app.query_one(selector).region
                assert region.height > 0
                assert region.bottom <= (size[1] - 3)
            table.focus()
            await pilot.pause()
            await pilot.press("right", "left", "down")
            app.configuration.preferences = replace(app.configuration.preferences, language="en")
            app.configuration.apply_ui_preferences()
            await pilot.pause()
            assert source.value == str(report)
            assert ("Uncertain direction") in ([str(cell) for cell in table.get_row_at(0)])
            assert "Result" in app.query_one("#main-tabs", TabbedContent).get_tab("results-tab").label.plain
        assert report.read_bytes() == original

    async def test_invalid_or_failed_reports_clear_old_table_and_show_error_on_report_page(self):
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await self.open_tab(app, pilot)
            app.query_one("#report-file-options", Collapsible).collapsed = False
            source = app.query_one("#report-json-path", Input)
            source.value = str(self.overhead_report())
            await pilot.pause()
            await self.click_import(app, pilot)
            await self.finish_workers(app, pilot)
            assert (app.query_one("#report-table", DataTable).row_count) == (1)
            invalid = self.directory / "损坏.json"
            for content in ("{broken", '{"schema_version":99}',
                            '{"schema_version":1,"kind":"ui_overhead","successful":false,"error":"interrupted"}'):
                invalid.write_text(content)
                source.value = str(invalid)
                await pilot.pause()
                await self.click_import(app, pilot)
                await self.finish_workers(app, pilot)
                assert (app.query_one("#report-table", DataTable).row_count) == (0)
                assert ("报告读取失败") in (str(app.query_one("#report-error", Static).content))
                assert app.query_one("#report-content", ContentSwitcher).current == "report-error"
                assert not (app.is_busy())
                assert not (app.query_one("#report-open", Button).disabled)

    async def test_measurement_blocks_report_io_and_statistics_including_slash_commands(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.open_tab(app, pilot)
            app.query_one("#result-csv", Input).value = str(self.csv_path)
            await pilot.pause()
            app.query_one("#report-json-path", Input).value = str(self.overhead_report())
            await pilot.pause()
            app.process_kind = "run"
            app.latest_snapshot = ProgressSnapshot(measurement_active=True)
            app.set_busy(True)
            with patch.object(app.process, 'execute_command') as execute:
                for selector in ("#report-open", "#report-calculate", "#result-csv", "#report-json-path"):
                    assert (app.query_one(selector).disabled)
                for command in ("/stats", "/report"):
                    field = app.query_one("#slash-command", Input)
                    field.value = command
                    field.focus()
                    await pilot.pause()
                    await pilot.press("enter")
                    await pilot.pause()
                execute.assert_not_called()
                assert not (list(app.workers))
            app.process_kind = ""
            app.latest_snapshot = ProgressSnapshot()
            app.set_busy(False)
            assert not (app.query_one("#report-open", Button).disabled)

    async def test_current_result_is_shared_and_source_change_invalidates_imported_report(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.open_tab(app, pilot)
            app.run_result = RunResult(True, True, result_csv=str(self.csv_path))
            app.process_finished("run", 0, ProgressSnapshot(final_csv=str(self.csv_path)), "")
            await app.workers.wait_for_complete()
            await pilot.pause()
            source = app.query_one("#result-csv", Input)
            assert (source.value) == (str(self.csv_path))
            assert not (list(app.workers))
            source.value = str(self.directory / "explicit.csv")
            json_path = app.query_one("#report-json-path", Input)
            json_path.value = str(self.overhead_report())
            app.configuration.remember_last_used(result_csv=str(self.directory / "another.csv"))
            assert source.value.endswith("another.csv")
            assert json_path.value == ""
            assert app.results.report_view is None

    async def test_json_view_then_html_generation_keeps_original_experiment(self):
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await self.open_tab(app, pilot)
            source = app.query_one("#result-csv", Input)
            json_path = app.query_one("#report-json-path", Input)
            source.value = str(self.csv_path)
            assert await pilot.click("#report-calculate")
            await self.finish_workers(app, pilot)
            assert source.value == str(self.csv_path)
            assert json_path.value == ""
            imported = self.overhead_report()
            app.results.open_report(str(imported))
            await self.finish_workers(app, pilot)
            assert Path(json_path.value) == imported
            assert app.query_one("#report-content", ContentSwitcher).current == "report-results"
            with patch.object(app, "launch") as launch:
                app.results.launch_html_report()
            launch.assert_called_once()
            command = launch.call_args.args[0].command
            assert str(self.csv_path) in command
            assert json_path.value not in command
            assert source.value == str(self.csv_path)
            source.value = str(self.directory / "another.csv")
            await pilot.pause()
            assert json_path.value == ""
            assert app.query_one("#report-content", ContentSwitcher).current == "report-empty"

    async def test_report_without_rows_uses_empty_state_instead_of_blank_table(self):
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await self.open_tab(app, pilot)
            app.results.report_view = ReportView(
                source=self.csv_path, title="无有效统计", columns=("字段",), rows=(),
                note="本次实验没有可展示的数据行",
            )
            app.results.render_report_view()
            await pilot.pause()
            assert app.query_one("#report-content", ContentSwitcher).current == "report-empty"
            assert "没有可展示的数据行" in str(app.query_one("#report-empty", Static).content)
            assert app.query_one("#report-table", DataTable).row_count == 0

    async def test_failed_statistics_returns_to_reports_and_reenables_controls(self):
        self.csv_path.write_text("bad,csv\n1,2\n")
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await self.open_tab(app, pilot)
            app.query_one("#result-csv", Input).value = str(self.csv_path)
            await pilot.click("#report-calculate")
            await self.finish_workers(app, pilot)
            assert (app.query_one("#main-tabs", TabbedContent).active) == ("results-tab")
            assert "统计失败" in str(app.query_one("#report-error", Static).content)
            assert app.query_one("#report-content", ContentSwitcher).current == "report-error"
            assert (app.query_one("#report-table", DataTable).row_count) == (0)
            assert not (app.is_busy())
            assert not (app.query_one("#report-calculate", Button).disabled)
