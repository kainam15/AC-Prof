"""实验参数可独立编辑，空值与正在执行的任务有不同显示。"""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import DataTable, Input, Label, Static
from tui_fixtures import AcprofTui

from acprof.experiment import RunConfig, RunConfigError, build_run_command
from acprof.host.image_management import DockerConnection, ImageInventory, ImageLayer, ManagedImage
from acprof.tui.i18n import translate
from acprof.tui.images import ImageTree, format_image_size, image_metadata
from acprof.tui.progress import ProgressSnapshot
from acprof.tui.settings import load_settings

PROJECT_DIR = Path(__file__).resolve().parents[1]


class TestTuiPresentation:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        temporary = tmp_path
        self.directory = Path(str(temporary))
        self.app = AcprofTui(RunConfig(model="demo/model"), settings_path=self.directory / "ui.json")

    async def test_independent_numbers_reach_command_and_saved_defaults(self):
        app = self.app
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.click("#open-run-settings")
            await pilot.pause()
            assert (len(app.query("#warmup"))) == (1), "Warmup 应有独立的输入框"
            warmup, repeat = app.query_one("#warmup", Input), app.query_one("#repeat", Input)
            assert ((warmup.value, repeat.value)) == (("2", "5"))
            assert (app.query_one("#sample-hz", Input).value) == ("20")
            assert (app.query_one("#request-timeout-seconds", Input).value) == ("300")
            assert (await pilot.click(warmup))
            await pilot.press("ctrl+a", "3", "tab", "ctrl+a", "7")
            assert (app.focused) is (repeat)
            sample = app.query_one("#sample-hz", Input)
            assert (await pilot.click(sample))
            await pilot.press("ctrl+a", *"20.1256789")
            config = app._collect_config()
            assert ((config.warmup, config.repeat, config.sample_hz)) == ((3, 7, 20.1256789))
            command = build_run_command(config, project_dir=PROJECT_DIR)
            for flag, value in (("--warmup", "3"), ("--repeat", "7"), ("--sample-hz", "20.1256789")):
                assert (command[command.index(flag) + 1]) == (value)
            warmup.value = "-1"
            with pytest.raises(RunConfigError, match="Warmup"):
                app._collect_config()
            warmup.value = "3"
            repeat.value = "0"
            with pytest.raises(RunConfigError, match="Repeat"):
                app._collect_config()
            repeat.value = "7"
            app.query_one("#advanced-form").scroll_end(animate=False, immediate=True)
            await pilot.pause()
            assert (await pilot.click("#save-run-default"))
            await pilot.pause()
            saved, warning = load_settings(app.settings_path, PROJECT_DIR)
            assert (warning) == ("")
            assert (saved.run_defaults) == (config)
            assert saved.run_defaults is not None
            app._apply_config(RunConfig.smoke("demo/model"), preset="smoke")
            assert ((warmup.value, repeat.value)) == (("0", "1"))
            app._apply_config(saved.run_defaults)
            assert (sample.value) == ("20.1256789"), "显示不能丢失已保存的小数精度"

    @pytest.mark.parametrize('name_case', range(6), ids=["('warmup', '次' if language == 'zh' else 'runs')", "('repeat', '次' if language == 'zh' else 'runs')", "('sample-hz', 'Hz')", "('idle-seconds', 's')", "('idle-cooldown-seconds', 's')", "('request-timeout-seconds', 's')"])
    @pytest.mark.parametrize('language', ('zh', 'en'))
    @pytest.mark.parametrize('size', ((80, 24), (120, 30), (150, 45)))
    async def test_parameter_units_remain_outside_editable_inputs_across_resize_and_languages(self, name_case, language, size):
        app = self.app
        async with app.run_test(size=(150, 45)) as pilot:
            await pilot.click("#open-run-settings")
            await pilot.pause()
            assert (len(app.query("#repeat"))) == (1), "Repeat 应有独立的输入框"
            await pilot.resize_terminal(*size)
            app.ui_preferences = replace(app.ui_preferences, language=language)
            app._apply_ui_preferences()
            (name, unit) = tuple((('warmup', '次' if language == 'zh' else 'runs'), ('repeat', '次' if language == 'zh' else 'runs'), ('sample-hz', 'Hz'), ('idle-seconds', 's'), ('idle-cooldown-seconds', 's'), ('request-timeout-seconds', 's')))[name_case]
            field = app.query_one("#" + name, Input)
            field.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            assert field.parent is not None
            label = field.parent.query_one(".field-unit", Label)
            assert (str(label.content)) == (unit)
            assert (label.region.x) >= (field.region.right)
            assert (label.region.right) <= (app.size.width - 2)
            assert (await pilot.click(field))
            assert (app.focused) is (field)
            assert (unit) not in (field.value)
            app._set_busy(True)
            assert (app.query_one("#warmup", Input).disabled)
            assert (app.query_one("#repeat", Input).disabled)
            app._set_busy(False)
            assert not (app.query_one("#repeat", Input).disabled)

    async def test_monitor_distinguishes_no_case_from_missing_case_resources(self):
        app = self.app
        async with app.run_test() as pilot:
            app._render_snapshot(ProgressSnapshot())
            assert (str(app.query_one("#status-elapsed", Static).content)) == ("—")
            assert (str(app.query_one("#status-resource", Static).content)) == ("CPU=—  MEM=—  GPU=—")
            app._latest_snapshot = ProgressSnapshot(current_case=1, total_cases=2, cpu="2")
            app._render_snapshot(app._latest_snapshot)
            assert (str(app.query_one("#status-resource", Static).content)) == ("CPU=2  MEM=未知  GPU=未知")
            app.ui_preferences = replace(app.ui_preferences, language="en")
            app._apply_ui_preferences()
            await pilot.pause()
            assert ("MEM=Unknown") in (str(app.query_one("#status-resource", Static).content))

    async def test_loading_report_indicator_is_cleared_on_failure(self):
        app = self.app
        async with app.run_test() as pilot:
            with patch.object(app, "_execute_report_read"):
                app._open_report(str(self.directory / "report.json"))
                assert (app._is_busy())
                assert (str(app.query_one("#report-status", Static).content).startswith("…"))
                app._show_report(None, "broken JSON", app._report_request)
                await pilot.pause()
                assert not (app._is_busy())
                text = str(app.query_one("#report-status", Static).content)
                assert ("报告读取失败") in (text)
                assert ("…") not in (text)

    async def test_unknown_values_translate_in_image_tree_list_and_layers(self):
        app = self.app
        item = ManagedImage("sha256:base", ("acprof-platform-cpu:base",), 0, "", "base", acprof=True)
        layer = ImageLayer("chain", "diff", None, (item.image_id,))
        inventory = ImageInventory(DockerConnection((), "local"), "daemon", (item,), (layer,))
        with patch.object(app, "refresh_images"):
            async with app.run_test(size=(120, 30)) as pilot:
                app._activate_tab("images-tab")
                app._image_inventory = inventory
                app._render_images()
                await pilot.pause()
                for language, unknown in (("zh", "未知"), ("en", "Unknown")):
                    app.ui_preferences = replace(app.ui_preferences, language=language)
                    app._apply_ui_preferences()
                    await pilot.pause()
                    cells = [str(cell) for cell in app.query_one("#image-table", DataTable).get_row_at(0)]
                    assert (cells[2:6]) == (["0 B", unknown, "0", unknown])
                    assert (app.query_one("#image-layer-table", DataTable).get_row_at(0)[1]) == (unknown)
                    tree = app.query_one("#image-tree", ImageTree)
                    label = tree.root.children[0].label.plain
                    assert ("0 B") in (label)
                    assert (unknown) in (label)
                    assert ("?") not in (label)


def test_unknown_image_size_keeps_real_zero_and_translates():
    assert (format_image_size(None)) == ("未知")
    assert (translate(format_image_size(None), "en")) == ("Unknown")
    assert (format_image_size(0)) == ("0 B")

def test_absent_model_and_unknown_created_time_are_distinct():
    item = ManagedImage("sha256:base", (), 0, "", "base")
    # 路径与共享空间不需要 Docker；空清单保留镜像本身的信息。
    inventory = ImageInventory(DockerConnection((), "local"), "daemon", (item,))
    assert ("模型：— · 创建时间：未知") in (image_metadata(item, inventory))
