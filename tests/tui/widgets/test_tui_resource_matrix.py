"""Resource fields remain editable and discoverable inside the collapsed group."""
from dataclasses import replace

import pytest
from textual.widgets import Collapsible, Input, Select, Static
from textual.widgets._collapsible import CollapsibleTitle

from acprof.experiments.configuration import RunConfig
from acprof.tui.ui.dialogs import ConfirmActionScreen
from tests.support.tui_fixtures import AcprofTui
from tests.support.tui_waits import wait_for_condition


@pytest.mark.parametrize(("language", "size"), [("zh", (80, 24)), ("en", (120, 30)), ("zh", (150, 45))])
async def test_resource_matrix_toggle_preserves_configuration(tmp_path, language, size):
    config = replace(RunConfig.smoke("fixture/model"), runtime="docker", cpus="1,2", mems="2,4",
                     gpus="on,off", input_scales="64,128")
    app = AcprofTui(config, settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=size) as pilot:
        section = app.query_one("#resource-matrix", Collapsible)
        assert section.collapsed
        assert section.title.startswith("资源矩阵" if language == "zh" else "Resource matrix")
        assert "1,2" in section.title
        assert "64,128" in section.title
        assert not app.query_one("#cpus").is_on_screen
        assert app.configuration.collect_config() == config
        title = section.query_one(CollapsibleTitle)
        title.scroll_visible(animate=False, immediate=True)
        await pilot.pause()
        assert await pilot.click(title)
        await pilot.pause()
        assert not section.collapsed
        for identifier in ("cpus", "mems", "gpus", "input-scales"):
            control = section.query_one("#" + identifier)
            control.scroll_visible(animate=False, immediate=True)
            await pilot.pause()
            assert control.is_on_screen
            assert control.region.width > 0
            assert control.region.right <= app.query_one("#run-form").content_region.right
        app.query_one("#cpus", Input).value = "1,3"
        await wait_for_condition(
            pilot, lambda: "--cpus 1,3" in str(app.query_one("#command-preview", Static).content),
            description="resource edit updates the command preview", timeout=3,
        )
        title.scroll_visible(animate=False, immediate=True)
        title.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert section.collapsed
        assert app.configuration.collect_config() == replace(config, cpus="1,3")
        app.configuration.preferences = replace(app.configuration.preferences,
                                                language="en" if language == "zh" else "zh")
        app.configuration.apply_ui_preferences()
        await pilot.pause()
        assert section.collapsed
        assert section.title.startswith("Resource matrix" if language == "zh" else "资源矩阵")
        assert "1,3" in section.title


async def test_resource_error_reveals_collapsed_field(tmp_path):
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        section = app.query_one("#resource-matrix", Collapsible)
        assert section.collapsed
        cpu = app.query_one("#cpus", Input)
        cpu.value = "invalid-cpus"
        # The existing preset action validates the draft before applying changes.
        app.experiments.preset_main()
        await wait_for_condition(
            pilot, lambda: app.focused is cpu and cpu.is_on_screen,
            description="invalid CPU field is revealed and focused", timeout=3,
        )
        assert not section.collapsed
        assert cpu.value == "invalid-cpus"
        assert app.query_one("#cpus-error", Static).display


@pytest.mark.parametrize(("language", "expected"), [
    ("zh", {"cpus": "CPU 列表不能重复", "mems": "内存列表不能重复", "gpus": "GPU 模式不能重复"}),
    ("en", {"cpus": "CPU values must be unique", "mems": "Memory values must be unique",
            "gpus": "GPU modes must be unique"}),
])
async def test_probe_shows_localized_duplicate_resource_fields_without_pruning(tmp_path, language, expected):
    config = replace(RunConfig.smoke("fixture/model", runtime="docker"), prune_startup_oom=False)
    app = AcprofTui(config, settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=(120, 30)) as pilot:
        assert not app.configuration.checked("prune-startup-oom")
        app.query_one("#cpus", Input).value = "1,01"
        app.query_one("#mems", Input).value = "4,04"
        # A restored configuration can contribute a custom GPU list to the selector.
        gpu = app.query_one("#gpus", Select)
        gpu.set_options([("off,cpu", "off,cpu")])
        gpu.value = "off,cpu"
        app.experiments.request_probe()
        assert app.experiments.pending_launch is None
        assert app.preflight.preparation_screen is None
        for field, reason in expected.items():
            error = app.query_one(f"#{field}-error", Static)
            assert error.display
            assert str(error.content) == reason
        await wait_for_condition(
            pilot, lambda: app.focused is app.query_one("#cpus") and app.focused.is_on_screen,
            description="first duplicate resource field is revealed and focused", timeout=3,
        )


@pytest.mark.parametrize(("language", "expected"), [
    ("zh", "--massif-reference-cpu=3 必须位于 --cpus 选中的资源值 [1, 2] 中"),
    ("en", "--massif-reference-cpu=3 must be in the resources selected by --cpus: [1, 2]"),
])
async def test_probe_recovers_after_editing_inherited_profiler_reference(tmp_path, language, expected):
    config = replace(RunConfig.smoke("fixture/model", runtime="docker"), cpus="1,3",
                     execution_profile_tool="massif", extra_options={"massif_reference_cpu": 3})
    app = AcprofTui(config, settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=(120, 30)) as pilot:
        cpu = app.query_one("#cpus", Input)
        cpu.value = "1,2"
        app.experiments.request_probe()
        error = app.query_one("#cpus-error", Static)
        assert error.display
        assert str(error.content) == expected
        assert app.experiments.pending_launch is None
        assert app.preflight.preparation_screen is None
        await wait_for_condition(
            pilot, lambda: app.focused is cpu and cpu.is_on_screen,
            description="probe reveals the editable CPU field for a profiler reference mismatch", timeout=3,
        )
        cpu.value = "1,3"
        await wait_for_condition(
            pilot, lambda: not error.display, description="CPU error clears after correction", timeout=3,
        )
        assert app.configuration.collect_config() == config
        app.experiments.request_probe()
        await wait_for_condition(
            pilot, lambda: isinstance(app.screen, ConfirmActionScreen),
            description="corrected probe asks to confirm", timeout=3,
        )
        app.screen.dismiss(False)
        await wait_for_condition(
            pilot, lambda: app.experiments.pending_launch is None, description="probe confirmation dismissed", timeout=3,
        )
        assert app.preflight.preparation_screen is None


async def test_probe_confirmation_shows_the_exact_manual_maximum(tmp_path):
    maximum = "1.2345678901234567"
    config = replace(RunConfig.smoke("fixture/model", runtime="docker"), input_scales=f"1,{maximum}")
    app = AcprofTui(config, settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(120, 30)) as pilot:
        app.experiments.request_probe()
        await wait_for_condition(pilot, lambda: isinstance(app.screen, ConfirmActionScreen),
                                 description="manual probe scale confirmation", timeout=3)
        summary = str(app.screen.message).split("\n\n", 1)[0]
        assert maximum in summary
        pending = app.experiments.pending_launch
        assert pending.command[pending.command.index("--input-scales") + 1] == f"1,{maximum}"
        app.screen.dismiss(False)
