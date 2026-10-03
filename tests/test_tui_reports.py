"""验证用户从 TUI 计算和查看报告，保持原始结果与测量隔离。"""
import csv
import json
import re
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Button, DataTable, Input, Static, TabbedContent, TabPane
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig
from acprof.tui.progress import ProgressSnapshot
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
            app._open_report(str(source))
            await self.finish_workers(app, pilot)
            table = app.query_one("#report-table", DataTable)
            for language, title in (("zh", "模型"), ("en", "Model")):
                app.ui_preferences = replace(app.ui_preferences, language=language)
                app._apply_ui_preferences()
                await pilot.pause()
                assert (str(next(iter(table.columns.values())).label)) == (title)
                assert ([str(cell) for cell in table.get_row_at(0)]) == (["fixture/sam2", "inconclusive", "compatibility_budget_exhausted"])
                assert not (app._latest_snapshot.measurement_active)
        assert (source.read_bytes()) == (original)

    async def test_v2_statistics_use_plots_directory(self):
        from acprof.artifact_layout import ArtifactLayout
        root = self.directory / "v2"
        ArtifactLayout.for_new_run(root).initialize()
        csv_path = root / "result_all.csv"
        csv_path.write_bytes(self.csv_path.read_bytes())
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.open_tab(app, pilot)
            with patch.object(app, "_launch") as launch:
                app._launch_stats(str(csv_path))
            launch.assert_called_once()
            command = launch.call_args.args[0].command
            assert ("--output-dir") in (command)
            assert (Path(command[command.index("--output-dir") + 1])) == (root / "plots/analysis")
            assert not ((root / "analysis").exists())

    def make_app(self):
        return AcprofTui(RunConfig.smoke("demo/model"), settings_path=self.directory / "tui.json")

    async def open_tab(self, app, pilot):
        await pilot.pause()
        tabs = app.query_one("#main-tabs", TabbedContent)
        assert ("reports-tab") in ([pane.id for pane in tabs.query(TabPane)]), "TUI 应提供统计报告页，让用户查看已有报告和计算窗口统计"
        app._activate_tab("reports-tab")
        await pilot.pause()

    async def finish_workers(self, app, pilot):
        # stats 子进程完成后会调度读取报告的 worker。
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

    async def test_calculate_uses_formal_windows_and_opens_saved_report_without_changing_csv(self):
        original = self.csv_path.read_bytes()
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.open_tab(app, pilot)
            app.query_one("#report-source", Input).value = str(self.csv_path)
            assert (await pilot.click("#report-calculate"))
            await self.finish_workers(app, pilot)
            assert (app.query_one("#main-tabs", TabbedContent).active) == ("reports-tab")
            output = Path(app.query_one("#report-source", Input).value)
            assert (output.suffix) == (".json")
            assert re.search(r"^window-statistics-\d{8}-\d{6}-\d{6}\.json$", output.name)
            data = json.loads(output.read_text())
            latency = next(row for row in data["groups"] if row["metric"] == "latency_app_s")
            assert (latency["n_windows"]) == (3)
            assert (latency["mean"]) == (0.02) or round(abs((latency["mean"]) - (0.02)), 7) == 0
            table = app.query_one("#report-table", DataTable)
            assert (table.row_count) == (3)
            cells = [str(cell) for cell in table.get_row_at(0)]
            assert ("20 ms") in (cells)
            assert not (app._is_busy())
            assert not (app.query_one("#start-run", Button).disabled)
        assert (self.csv_path.read_bytes()) == (original)

    async def test_repeated_statistics_opens_existing_report_and_notifies_in_both_languages(self):
        original = self.csv_path.read_bytes()
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await self.open_tab(app, pilot)
            source = app.query_one("#report-source", Input)
            source.value = str(self.csv_path)
            assert (await pilot.click("#report-calculate"))
            await self.finish_workers(app, pilot)
            generated = Path(str(source.value))
            existing = generated.with_name("window-statistics-1b1047da59ca46f186c93ac5ad86f8d0.json")
            generated.rename(existing)
            saved = existing.read_bytes()
            modified = existing.stat().st_mtime_ns
            for language, expected in (("zh", "已有相同报告"), ("en", "Identical report already exists")):
                app.ui_preferences = replace(app.ui_preferences, language=language)
                app._apply_ui_preferences()
                source.value = str(self.csv_path)
                await pilot.pause()
                with patch.object(app, "notify", wraps=app.notify) as notify:
                    assert (await pilot.click("#report-calculate"))
                    await self.finish_workers(app, pilot)
                assert (Path(str(source.value))) == (existing)
                assert (list(existing.parent.glob("*.json"))) == ([existing])
                assert (existing.read_bytes()) == (saved)
                assert (existing.stat().st_mtime_ns) == (modified)
                notices = [app.tr(call.args[0]) for call in notify.call_args_list]
                assert (any(expected in text and str(existing) in text for text in notices)), notices
                assert (app.query_one("#report-table", DataTable).row_count) == (3)
                assert (app.query_one("#main-tabs", TabbedContent).active) == ("reports-tab")
                assert not (app._is_busy())
        assert (self.csv_path.read_bytes()) == (original)

    @pytest.mark.parametrize('size', ((80, 24), (120, 30), (150, 45)))
    async def test_empty_header_click_before_loading_and_after_read_failure(self, size):
        report = self.overhead_report()
        app = self.make_app()
        async with app.run_test(size=size) as pilot:
            await self.open_tab(app, pilot)
            table = app.query_one("#report-table", DataTable)
            for language in ("zh", "en"):
                app.ui_preferences = replace(app.ui_preferences, language=language)
                app._apply_ui_preferences()
                await pilot.pause()
                assert (len(table.columns)) == (0)
                inset = table.content_region.offset - table.region.offset
                assert (await pilot.click(table, offset=(inset.x + 38, inset.y)))
                assert (app.is_running)
                assert (app.mouse_captured) is None
            source = app.query_one("#report-source", Input)
            source.value = str(report)
            assert (await pilot.click("#report-open"))
            await self.finish_workers(app, pilot)
            assert (table.row_count) == (1)
            source.value = str(self.directory / "missing.json")
            assert (await pilot.click("#report-open"))
            await self.finish_workers(app, pilot)
            assert (len(table.columns)) == (0)
            assert (await pilot.click(table, offset=(inset.x + 38, inset.y)))
            source.value = str(report)
            assert (await pilot.click("#report-open"))
            await self.finish_workers(app, pilot)
            assert (table.row_count) == (1), "空表头点击后仍能正常读取报告"

    @pytest.mark.parametrize('size', ((80, 24), (120, 30), (150, 45)))
    async def test_open_overhead_at_all_sizes_and_language_switch_preserves_data_and_draft(self, size):
        report = self.overhead_report()
        original = report.read_bytes()
        app = self.make_app()
        async with app.run_test(size=size) as pilot:
            await self.open_tab(app, pilot)
            source = app.query_one("#report-source", Input)
            source.value = str(report)
            assert (await pilot.click("#report-open"))
            await self.finish_workers(app, pilot)
            table = app.query_one("#report-table", DataTable)
            assert (table.row_count) == (1)
            assert ("+2%") in ([str(cell) for cell in table.get_row_at(0)])
            assert ("方向不确定") in ([str(cell) for cell in table.get_row_at(0)])
            for selector in ("#report-open", "#report-calculate", "#report-table"):
                region = app.query_one(selector).region
                assert (region.height) > (0)
                assert (region.bottom) <= (size[1] - 3)
            table.focus()
            await pilot.pause()
            await pilot.press("right", "left", "down")
            app.ui_preferences = replace(app.ui_preferences, language="en")
            app._apply_ui_preferences()
            await pilot.pause()
            assert (source.value) == (str(report))
            assert ("Uncertain direction") in ([str(cell) for cell in table.get_row_at(0)])
            assert ("Report") in (app.query_one("#main-tabs", TabbedContent).get_tab("reports-tab").label.plain)
        assert (report.read_bytes()) == (original)

    async def test_invalid_or_failed_reports_clear_old_table_and_show_error_on_report_page(self):
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await self.open_tab(app, pilot)
            source = app.query_one("#report-source", Input)
            source.value = str(self.overhead_report())
            await pilot.click("#report-open")
            await self.finish_workers(app, pilot)
            assert (app.query_one("#report-table", DataTable).row_count) == (1)
            invalid = self.directory / "损坏.json"
            for content in ("{broken", '{"schema_version":99}',
                            '{"schema_version":1,"kind":"ui_overhead","successful":false,"error":"interrupted"}'):
                invalid.write_text(content)
                source.value = str(invalid)
                await pilot.click("#report-open")
                await self.finish_workers(app, pilot)
                assert (app.query_one("#report-table", DataTable).row_count) == (0)
                assert ("报告读取失败") in (str(app.query_one("#report-status", Static).content))
                assert not (app._is_busy())
                assert not (app.query_one("#report-open", Button).disabled)

    async def test_measurement_blocks_report_io_and_statistics_including_slash_commands(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.open_tab(app, pilot)
            app.query_one("#report-source", Input).value = str(self.csv_path)
            app._process_kind = "run"
            app._latest_snapshot = ProgressSnapshot(measurement_active=True)
            app._set_busy(True)
            with patch.object(app, "_execute_command") as execute:
                for selector in ("#report-open", "#report-calculate", "#report-source"):
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
            app._process_kind = ""
            app._latest_snapshot = ProgressSnapshot()
            app._set_busy(False)
            assert not (app.query_one("#report-open", Button).disabled)

    async def test_current_result_follows_completed_run_but_preserves_an_explicit_report_path(self):
        app = self.make_app()
        async with app.run_test(size=(120, 30)) as pilot:
            await self.open_tab(app, pilot)
            app._run_result = RunResult(True, True, result_csv=str(self.csv_path))
            app._process_finished("run", 0, ProgressSnapshot(final_csv=str(self.csv_path)), "")
            await app.workers.wait_for_complete()
            await pilot.pause()
            source = app.query_one("#report-source", Input)
            assert (source.value) == (str(self.csv_path))
            assert not (list(app.workers))
            source.value = str(self.overhead_report())
            app._remember_last_used(result_csv=str(self.directory / "another.csv"))
            assert (source.value.endswith("开销.json"))

    async def test_failed_statistics_returns_to_reports_and_reenables_controls(self):
        self.csv_path.write_text("bad,csv\n1,2\n")
        app = self.make_app()
        async with app.run_test(size=(80, 24)) as pilot:
            await self.open_tab(app, pilot)
            app.query_one("#report-source", Input).value = str(self.csv_path)
            await pilot.click("#report-calculate")
            await self.finish_workers(app, pilot)
            assert (app.query_one("#main-tabs", TabbedContent).active) == ("reports-tab")
            assert ("统计未完成") in (str(app.query_one("#report-status", Static).content))
            assert (app.query_one("#report-table", DataTable).row_count) == (0)
            assert not (app._is_busy())
            assert not (app.query_one("#report-calculate", Button).disabled)
