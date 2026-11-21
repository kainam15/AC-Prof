"""Collection-plan integration preserves drafts, startup facts and measurement gates."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.cells import cell_len
from textual.widgets import Button, Collapsible, ContentSwitcher, Input, Static

from acprof.experiments.configuration import RunConfig, RunConfigError
from acprof.tui.i18n import message_id
from acprof.tui.run.plan_panel import RunPlanPanel
from acprof.tui.ui.structured import ActionButtons, FactGrid, FactValue
from tests.support.tui_fixtures import AcprofTui, expand_check_source
from tests.support.tui_waits import wait_for_condition


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_collection_plan_and_original_actions_fit_short_terminal(tmp_path, language):
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=(80, 24)) as pilot:
        panel = app.query_one("#run-plan-panel", RunPlanPanel)
        assert panel.view is not None
        assert panel.view.summary.level == "warning"
        assert not app.query("#config-summary")
        assert not app.query("#run-plan-summary")
        grid = panel.query_one("#run-plan-facts", FactGrid)
        assert grid.section == panel.view.section
        section = grid.query_one(".fact-section", Collapsible)
        assert section.collapsed
        assert ("采集计划" if language == "zh" else "Collection plan") in section.title
        assert "1" in section.title
        # Expanding reveals the full structured grid and details disclosure.
        section_title = section.query_one("CollapsibleTitle")
        mode_region = app.query_one("#run-mode-advanced").region
        assert mode_region.bottom <= panel.region.y
        assert await pilot.click(section_title)
        await pilot.pause()
        assert not section.collapsed
        assert app.query_one("#run-mode-advanced").region == mode_region
        more = grid.query_one(".fact-more", Collapsible)
        assert more.collapsed
        body = panel.query_one("#run-plan-body")
        assert body.scroll_y == 0
        values = [value for value in grid.query(FactValue) if value.row in panel.view.section.rows]
        assert len(values) == 5
        for value in values:
            assert value.region.height > 0
            assert body.content_region.contains_region(value.region)
        more_title = more.query_one("CollapsibleTitle")
        assert more_title.region.height > 0
        assert body.content_region.contains_region(more_title.region)
        center = (more_title.region.x + more_title.region.width // 2,
                  more_title.region.y + more_title.region.height // 2)
        assert app.get_widget_at(*center)[0] is more_title
        assert await pilot.click(more_title)
        await pilot.pause()
        assert not more.collapsed
        assert await pilot.click(more_title)
        await pilot.pause()
        assert more.collapsed
        assert await pilot.click(section_title)
        await pilot.pause()
        assert section.collapsed
        assert not panel.query('#run-plan-status')
        assert any(item.check.check_id == 'quality_repeat' and item.group == 'warning'
                   for item in app.preflight.status.items)
        for label, row in zip(grid.query(".fact-label"), (
            *panel.view.section.rows, *panel.view.section.detail_rows,
        ), strict=True):
            assert str(label.content) == app.tr(row.label)

        def assert_actions_visible():
            actions = app.query_one("#run-actions")
            assert panel.region.bottom <= actions.region.y
            for selector in ("#probe-largest", "#start-run"):
                button = app.query_one(selector, Button)
                assert not button.disabled
                assert app.screen.region.contains_region(button.region)
                assert actions.content_region.contains_region(button.region)
                assert button.region.bottom <= app.query_one("#bottom-panel").region.y
                assert cell_len(str(button.label)) <= button.content_region.width
                center = (button.region.x + button.region.width // 2,
                          button.region.y + button.region.height // 2)
                assert app.get_widget_at(*center)[0] is button

        assert_actions_visible()
        assert [button.id for button in app.query_one("#run-actions").query(Button)] == ["probe-largest", "start-run"]
        for selector in ("#resource-matrix", "#measurement-parameters"):
            title = app.query_one(selector + " CollapsibleTitle")
            assert app.query_one("#run-form").content_region.contains_region(title.region)
            assert app.get_widget_at(*title.region.center)[0] is title
        assert await pilot.click("#run-mode-advanced")
        await pilot.pause()
        assert app.query_one("#experiment-pages", ContentSwitcher).current == "advanced-form"
        assert app.query_one("#run-mode-advanced", Button).has_class("-active")
        assert_actions_visible()


@pytest.mark.parametrize('language', ['zh', 'en'])
async def test_pending_scales_are_plan_facts_without_a_standalone_status(tmp_path, language):
    config = replace(RunConfig.smoke('fixture/model'), input_scales='', input_scale_policy='auto', repeat=3)
    app = AcprofTui(config, settings_path=tmp_path / 'settings.json')
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=(100, 30)):
        panel = app.query_one('#run-plan-panel', RunPlanPanel)
        assert panel.view is not None
        assert panel.view.summary.headline == message_id('run.plan.pending_inputs')
        assert not panel.query('#run-plan-status')
        assert not panel.query('.status-headline')
        grid = panel.query_one('#run-plan-facts', FactGrid)
        scales = next(row for row in grid.section.rows if row.label == message_id('run.plan.scales_label'))
        assert app.tr(message_id('common.state.pending')) in app.tr(scales.value)


async def test_run_ux_set_repeat_inline_reason_and_overrides(tmp_path):
    config = replace(RunConfig.smoke("fixture/model"), revision="a1b2c3d4")
    app = AcprofTui(config, settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        # Top mode bar keeps Basic/Advanced tabs without quick-fill buttons.
        assert not app.query("#run-quick-fill")
        assert not app.query("#quick-fill-smoke")
        assert not app.query("#quick-fill-main")
        # Active advanced override is visible in the basic view and mode badge.
        assert app.query_one("#advanced-overrides-banner", Button).display
        assert "1" in str(app.query_one("#run-mode-advanced", Button).label)
        # Collapsed section summaries surface key values without expanding.
        assert "Warmup 0" in app.query_one("#measurement-parameters", Collapsible).title
        assert "已关闭" in app.query_one("#profiling-options", Collapsible).title
        # The only repeat warning action lives in the global status center.
        assert not app.query('#run-plan-status')
        assert await pilot.click('#environment-warning')
        await expand_check_source(app, pilot, 'quality_repeat')
        warning = app.query_one('#preflight-signal-quality_repeat')
        warning.scroll_visible(immediate=True, force=True)
        await pilot.pause()
        assert await pilot.click(warning)
        callout = app.query_one('#preflight-callout')
        assert callout.display
        assert tuple(action.id for action in callout.content.actions) == ('plan-set-repeat-3', 'status-open')
        plan_action = callout.query_one(ActionButtons).query(Button).first()
        plan_action.scroll_visible(immediate=True, force=True)
        await pilot.pause()
        assert await pilot.click(plan_action)
        await wait_for_condition(
            pilot, lambda: app.configuration.collect_config().repeat == 3,
            description='repeat updated to 3', timeout=3,
        )
        assert not any(item.check.check_id == 'quality_repeat' for item in app.preflight.status.items)
        # Slash command /smoke still restores repeat=1 cleanly.
        app.slash.execute_command("/smoke")
        await wait_for_condition(
            pilot, lambda: app.configuration.collect_config().repeat == 1,
            description="smoke slash preset applied", timeout=3,
        )
        # Clearing model shows inline clickable blocking reason above Run actions.
        app.query_one("#model", Input).value = ""
        await wait_for_condition(
            pilot, lambda: app.query_one("#preflight-run-reason", Button).display,
            description="inline blocking reason visible", timeout=3,
        )
        assert app.query_one("#start-run", Button).disabled
        assert await pilot.click("#preflight-run-reason")
        await wait_for_condition(
            pilot, lambda: app.focused is app.query_one("#model", Input),
            description="clicking inline reason focuses invalid model field", timeout=3,
        )


@pytest.mark.parametrize(("field_id", "invalid_value"), [("mems", "oops"), ("model", "")])
async def test_invalid_draft_clears_previous_plan_facts_and_recovers(tmp_path, field_id, invalid_value):
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        panel = app.query_one("#run-plan-panel", RunPlanPanel)
        app.execution.session.config = app.configuration.collect_config()
        app.preflight.preparation_event({
            "stage": "input", "status": "passed",
            "input_plan": {"scales": [8.0, 16.0], "scale_type": "seq_length"},
        })
        await pilot.pause()
        assert panel.view is not None
        assert panel.view.section.rows
        scales = app.query_one("#input-scales", Input)
        scale_label = app.query_one("#input-scales-label", Static)
        assert scales.placeholder == "8,16 tokens"
        prepared_label = str(scale_label.content)
        assert "tokens" in prepared_label
        field = app.query_one(f"#{field_id}", Input)
        original_value = field.value
        field.value = invalid_value
        with pytest.raises(RunConfigError) as invalid:
            app.configuration.collect_config()
        assert not app.configuration.refresh_command_preview(notify=False)
        await pilot.pause()
        assert panel.view is not None
        assert not panel.query('#run-plan-status')
        assert panel.view.section.rows == ()
        assert panel.view.section.detail_rows == ()
        grid = panel.query_one("#run-plan-facts", FactGrid)
        assert not grid.display
        assert not panel.query('#run-plan-callout')
        await wait_for_condition(pilot, lambda: not app.configuration.validation_pending,
                                 description='current invalid draft validated', timeout=3)
        assert app.query_one('#start-run', Button).disabled
        if field_id == 'model':
            assert app.preflight.status.state == 'unconfigured'
            assert any(item.check.check_id == 'config_model' and item.group == 'pending'
                       for item in app.preflight.status.items)
        else:
            assert app.preflight.status.counts['error'] > 0
            assert any(item.check.check_id.startswith('config_') and item.group == 'error'
                       for item in app.preflight.status.items)
            for issue in invalid.value.issues:
                assert any(app.tr(issue.reason) in app.tr(item.check.detail)
                           for item in app.preflight.status.items
                           if item.check.check_id.startswith('config_'))
        assert str(scale_label.content) == app.tr(message_id("run.inputs.scales"))
        assert scales.placeholder == app.tr(message_id("common.state.pending"))

        field.value = original_value
        assert app.configuration.refresh_command_preview(notify=False)
        await pilot.pause()
        assert panel.view is not None
        assert panel.view.section.rows
        assert panel.view.callout is None
        assert grid.display
        assert not panel.query('#run-plan-callout')
        assert str(scale_label.content) == prepared_label
        assert scales.placeholder == "8,16 tokens"


async def test_busy_preparation_updates_plan_then_measurement_freezes_projection(tmp_path):
    config = replace(RunConfig.smoke("fixture/model"), input_scale_policy="auto", repeat=3)
    app = AcprofTui(config, settings_path=tmp_path / "settings.json")
    with patch.object(app.process, "execute_command") as execute:
        async with app.run_test(size=(120, 30)) as pilot:
            panel = app.query_one("#run-plan-panel", RunPlanPanel)
            app.execution.session.config = config
            app.execution.session.kind = "run"
            app.set_busy(True)
            assert app.is_busy()
            app.preflight.preparation_event({
                "stage": "input", "status": "passed",
                "input_plan": {"scales": [8.0, 16.0], "scale_type": "seq_length"},
            })
            await pilot.pause()
            assert panel.view is not None
            facts = {row.label: row.value for row in panel.view.section.rows}
            assert facts[message_id("run.plan.scales_label")] == message_id(
                "run.plan.scale_count_unit", 2, "tokens",
            )
            assert facts[message_id("run.plan.windows_label")] == message_id(
                "run.plan.window_totals", 0, 6,
            )
            scales = app.query_one("#input-scales", Input)
            assert scales.placeholder == "8,16 tokens"
            scale_label = app.query_one("#input-scales-label", Static)
            before_view = panel.view
            before_label = str(scale_label.content)
            before_placeholder = scales.placeholder
            app.execution.session.snapshot = replace(
                app.execution.session.snapshot, measurement_active=True,
            )
            app.set_busy(True)
            with patch.object(panel, "show_plan") as project, patch(
                "acprof.tui.run.planning.run_plan_view",
            ) as build_view, patch(
                "acprof.tui.run.planning.invalid_plan_view",
            ) as build_invalid, patch.object(
                Path, "open", side_effect=AssertionError("plan refresh read a file during measurement"),
            ):
                app.configuration.update_plan_summary(replace(config, repeat=7))
                app.configuration.refresh_command_preview(notify=False)
                model = app.query_one("#model", Input)
                with app.prevent(Input.Changed):
                    model.value = ""
                try:
                    assert not app.configuration.refresh_command_preview(notify=False)
                finally:
                    with app.prevent(Input.Changed):
                        model.value = config.model
                project.assert_not_called()
                build_view.assert_not_called()
                build_invalid.assert_not_called()
            assert panel.view is before_view
            assert str(scale_label.content) == before_label
            assert scales.placeholder == before_placeholder
            app.execution.session.snapshot = replace(
                app.execution.session.snapshot, measurement_active=False,
            )
            app.execution.session.kind = ""
            app.set_busy(False)
        execute.assert_not_called()
