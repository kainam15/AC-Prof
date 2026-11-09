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
from acprof.tui.ui.structured import Callout, FactGrid, FactValue
from tests.support.tui_fixtures import AcprofTui


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
        assert not grid.query_one(".fact-section", Collapsible).collapsed
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
        headline = panel.query_one(".status-headline", Static)
        assert app.tr(panel.view.summary.headline) in str(headline.content)
        assert str(headline.content).startswith("! ")
        for label, row in zip(grid.query(".fact-label"), (
            *panel.view.section.rows, *panel.view.section.detail_rows,
        ), strict=True):
            assert str(label.content) == app.tr(row.label)

        def assert_actions_visible():
            actions = app.query_one("#run-actions")
            assert panel.region.bottom <= actions.region.y
            for selector in ("#open-run-settings", "#probe-largest", "#start-run"):
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
        assert await pilot.click("#open-run-settings")
        await pilot.pause()
        assert app.query_one("#experiment-pages", ContentSwitcher).current == "advanced-form"
        assert_actions_visible()


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
        assert panel.view.summary.level == "error"
        assert panel.view.section.rows == ()
        assert panel.view.section.detail_rows == ()
        grid = panel.query_one("#run-plan-facts", FactGrid)
        assert not grid.display
        callout = panel.query_one("#run-plan-callout", Callout)
        assert callout.display
        assert callout.content == panel.view.callout
        assert callout.content.problem == invalid.value.issues[0].reason
        for issue in invalid.value.issues:
            assert str(issue.reason) in callout.content.detail
        assert callout.query_one(".callout-details", Collapsible).collapsed
        assert str(scale_label.content) == app.tr(message_id("run.inputs.scales"))
        assert scales.placeholder == app.tr(message_id("common.state.pending"))

        field.value = original_value
        assert app.configuration.refresh_command_preview(notify=False)
        await pilot.pause()
        assert panel.view is not None
        assert panel.view.section.rows
        assert panel.view.callout is None
        assert grid.display
        assert not callout.display
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
