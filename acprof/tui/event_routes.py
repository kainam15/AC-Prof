"""Declarative Textual event routes forwarded from AcprofTui to controllers."""
from __future__ import annotations

from typing import TYPE_CHECKING

from textual import events, on
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    DataTable,
    Input as TextualInput,
    Select,
    TabbedContent,
    Tree,
)

from acprof.tui.logging.widget import SelectableLog
from acprof.tui.ui.input import BarCursorApp, BarCursorInput as Input
from acprof.tui.ui.navigation import PageNavigation
from acprof.tui.ui.structured import ActionRequested

if TYPE_CHECKING:
    from acprof.tui.experiments.controller import CatalogController
    from acprof.tui.images.controller import ImageController
    from acprof.tui.logging.controller import LogController
    from acprof.tui.models.controller import ModelController
    from acprof.tui.results.controller import ResultController
    from acprof.tui.results.profile_controller import ProfileController
    from acprof.tui.run.execution import ExecutionCoordinator
    from acprof.tui.run.experiment_controller import ExperimentController
    from acprof.tui.run.preflight_controller import PreflightController
    from acprof.tui.settings.controller import ConfigurationController
    from acprof.tui.ui.slash_controller import SlashController
    from acprof.tui.updates import UpdateController


class AppEventRoutes(BarCursorApp):
    """Widget event forwarding mixin isolated from App lifecycle and execution state."""

    if TYPE_CHECKING:
        form_ready: bool
        ui_closing: bool
        models: ModelController
        catalog: CatalogController
        images: ImageController
        configuration: ConfigurationController
        logs: LogController
        results: ResultController
        profiles: ProfileController
        preflight: PreflightController
        slash: SlashController
        experiments: ExperimentController
        execution: ExecutionCoordinator
        updates: UpdateController

        def action_request_stop(self) -> None: ...

    @on(Button.Pressed, '#model-candidates')
    def on_models_open_model_candidates(self) -> None:
        self.models.open_model_candidates()

    @on(Input.Changed, '#model')
    @on(Select.Changed, '#model-source')
    def on_models_candidate_identity_changed(self, event: TextualInput.Changed | Select.Changed) -> None:
        self.models.candidate_identity_changed(event)

    @on(Button.Pressed, '.experiment-picker')
    def on_catalog_select_experiment(self, event: Button.Pressed) -> None:
        self.catalog.select_experiment(event)

    @on(TabbedContent.TabActivated, '#main-tabs')
    def on_images_image_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        if not self.is_running or not self.form_ready or self.ui_closing:
            return
        self.query_one(PageNavigation).sync(event.tabbed_content)
        self.images.image_tab_activated(event)

    @on(Button.Pressed, '#image-view-tree, #image-view-list, #image-view-layers')
    def on_images_image_view_changed(self, event: Button.Pressed) -> None:
        self.images.image_view_changed(event)

    @on(DataTable.HeaderSelected, '#image-table')
    def on_images_sort_image_table(self, event: DataTable.HeaderSelected) -> None:
        self.images.sort_image_table(event)

    @on(Input.Changed, '#image-search')
    @on(Select.Changed, '#image-scope')
    def on_images_image_filter_changed(self) -> None:
        self.images.image_filter_changed()

    @on(DataTable.RowHighlighted, '#image-table')
    @on(DataTable.RowHighlighted, '#image-layer-table')
    @on(Tree.NodeHighlighted, '#image-tree')
    def on_images_show_image_detail(self) -> None:
        self.images.show_image_detail()

    @on(Collapsible.Expanded, '#image-dependencies')
    def on_images_expand_image_dependencies(self) -> None:
        self.images.expand_image_dependencies()

    @on(DataTable.RowSelected, '#image-table')
    @on(Tree.NodeSelected, '#image-tree')
    @on(Button.Pressed, '#image-toggle')
    def on_images_toggle_image_selection(self) -> None:
        self.images.toggle_image_selection()

    @on(Button.Pressed, '#image-clear')
    def on_images_clear_image_selection(self) -> None:
        self.images.clear_image_selection()

    @on(Button.Pressed, '#image-model')
    def on_images_select_model_images(self) -> None:
        self.images.select_model_images()

    @on(Button.Pressed, '#image-storage')
    def on_images_open_storage(self) -> None:
        self.images.open_storage()

    @on(Button.Pressed, '#image-delete')
    def on_images_request_delete_images(self) -> None:
        self.images.request_delete_images()

    @on(Select.Changed, '.ui-preference')
    @on(Checkbox.Changed, '.ui-preference')
    def on_configuration_ui_preference_changed(self) -> None:
        self.configuration.ui_preference_changed()

    @on(Button.Pressed, '#restore-ui-defaults')
    def on_configuration_restore_ui_defaults(self) -> None:
        self.configuration.restore_ui_defaults()

    @on(Select.Changed, '#update-channel')
    def on_version_channel(self, event: Select.Changed) -> None:
        self.updates.select_channel(str(event.value))

    @on(Button.Pressed, '#version-check, #version-apply, #version-log, #version-restart, #version-manual')
    def on_version_action(self, event: Button.Pressed) -> None:
        if event.button.id == "version-check":
            self.updates.start_check(manual=True)
        elif event.button.id == "version-apply":
            self.updates.request_install()
        elif event.button.id == "version-log":
            self.updates.show_log()
        elif event.button.id == "version-manual":
            self.updates.copy_manual()
        else:
            self.updates.restart()

    @on(Button.Pressed, '#save-ui-settings')
    def on_configuration_save_ui_settings(self) -> None:
        self.configuration.save_ui_settings()

    @on(Button.Pressed, '#save-run-default')
    def on_configuration_save_run_default(self) -> None:
        self.configuration.save_run_default()

    @on(Button.Pressed, '#open-environment-settings')
    def on_configuration_open_environment_settings(self) -> None:
        self.configuration.open_environment_settings()

    @on(Button.Pressed, '#open-run-settings')
    def on_configuration_open_run_settings(self) -> None:
        self.configuration.open_run_settings()

    @on(SelectableLog.Updated)
    def on_log_updated(self) -> None:
        self.logs.refresh_status()

    @on(Input.Changed, '#log-search')
    @on(Select.Changed, '#log-level')
    def on_log_view_changed(self) -> None:
        self.logs.view_changed()

    @on(Input.Submitted, '#log-search')
    def on_log_next(self) -> None:
        self.logs.find()

    @on(Button.Pressed, '#copy-log-all')
    def on_log_copy_all(self) -> None:
        self.logs.copy_all()

    @on(Button.Pressed, '#log-tools')
    def on_log_tools(self) -> None:
        self.logs.toggle_tools()

    @on(Button.Pressed, '#export-log')
    def on_log_export(self) -> None:
        self.logs.export()

    @on(Button.Pressed, '#copy-log-export-path')
    def on_log_export_copy_path(self) -> None:
        self.logs.copy_export_path()

    @on(Button.Pressed, '#view-log-export-path')
    def on_log_export_view_path(self) -> None:
        self.logs.view_export_path()

    @on(Button.Pressed, '#follow-log')
    def on_configuration_follow_latest_log(self) -> None:
        self.configuration.follow_latest_log()

    @on(Button.Pressed, '#expand-log')
    @on(Button.Pressed, '#restore-log')
    def on_configuration_toggle_log_view_button(self) -> None:
        self.configuration.toggle_log_view_button()

    @on(Input.Changed, '.config-control')
    @on(Select.Changed, '.config-control')
    @on(Checkbox.Changed, '.config-control')
    def on_configuration_configuration_changed(self) -> None:
        self.configuration.configuration_changed()

    @on(Button.Pressed, '#clear-log')
    def on_results_clear_log_button(self) -> None:
        self.results.clear_log_button()

    @on(Input.Changed, '#experiment-source')
    def on_results_result_source_changed(self) -> None:
        self.results.result_source_changed()

    @on(Input.Changed, '#backfill-source')
    def on_backfill_result_source_changed(self, event: TextualInput.Changed) -> None:
        if not self.is_running or not self.form_ready or self.ui_closing:
            return
        # Read the current value: queued startup events may predate restoration.
        if event.input.value != self.query_one('#experiment-source', Input).value:
            with self.prevent(Input.Changed):
                self.query_one('#experiment-source', Input).value = event.input.value
            self.results.result_source_changed()

    @on(ActionRequested)
    def on_structured_action(self, event: ActionRequested) -> None:
        if event.action_id.startswith("overview-"):
            event.stop()
            self.results.overview_action(event.action_id)
        elif event.action_id.startswith("process-resource-"):
            event.stop()
            if event.action_id == "process-resource-retry":
                self.preflight.quick_check()
            elif event.action_id == "process-resource-details":
                from acprof.tui.run.resource_view import show_details
                show_details(self.preflight)
        elif event.action_id.startswith("backfill-"):
            event.stop()
            self.profiles.backfill_action(event.action_id)

    @on(Button.Pressed, '#refresh-experiment-state')
    def on_results_refresh_experiment(self) -> None:
        self.results.refresh_experiment()

    @on(Button.Pressed, '#refresh-backfill-state')
    def on_backfill_refresh_experiment(self) -> None:
        self.results.refresh_experiment()

    @on(Button.Pressed, '#overview-profiling')
    def on_overview_profiling(self) -> None:
        self.profiles.open_selection_dialog()

    @on(Button.Pressed, '#backfill-cancel')
    @on(Button.Pressed, '#results-cancel')
    def on_backfill_cancel_read(self) -> None:
        if self.execution.session.kind in {"stats", "compare"}:
            self.action_request_stop()
        else:
            self.results.cancel_result_reads()

    @on(Button.Pressed, '#plot-results')
    def on_results_plot_results_button(self) -> None:
        self.results.plot_results_button()

    @on(Input.Changed, '#report-json-path')
    def on_results_report_json_changed(self) -> None:
        self.results.report_json_changed()

    @on(Input.Changed, '#html-report-path')
    def on_results_html_path_changed(self) -> None:
        self.results.update_report_actions()
    @on(Input.Changed, '#html-extra-sources')
    def on_results_html_extra_changed(self) -> None:
        self.results.html.update_extra_button()

    @on(Button.Pressed, '#analysis-export-csv, #analysis-export-json')
    def on_analysis_export(self, event: Button.Pressed) -> None:
        self.results.export_analysis(event.button.name or "csv")

    @on(Button.Pressed, '#report-open')
    def on_results_open_report_button(self) -> None:
        self.results.open_report_button()

    @on(DataTable.RowHighlighted, '#report-table')
    def on_results_report_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self.results.report_row_highlighted(event)

    @on(Button.Pressed, '#report-calculate')
    def on_results_calculate_report_button(self) -> None:
        self.results.calculate_report_button()

    @on(Button.Pressed, "#report-window-stats")
    def on_results_window_statistics(self) -> None:
        self.results.launch_stats()

    @on(Button.Pressed, '#report-compare')
    def on_results_compare_experiments_button(self) -> None:
        self.results.compare_experiments_button()

    @on(Button.Pressed, '#html-report-generate')
    def on_results_generate_html(self) -> None:
        self.results.html.launch()

    @on(Button.Pressed, '#html-report-open')
    def on_results_open_html(self) -> None:
        self.results.html.open()

    @on(Button.Pressed, '#html-report-export')
    def on_results_export_html(self) -> None:
        self.results.html.export()
    @on(Button.Pressed, '#html-extra-toggle')
    def on_results_toggle_html_extra(self) -> None:
        self.results.html.toggle_extra_sources()

    @on(Button.Pressed, '#html-export-toggle')
    def on_results_toggle_html_export(self) -> None:
        self.results.html.toggle_export_fields()

    @on(Button.Pressed, '#html-existing-toggle')
    def on_results_toggle_html_existing(self) -> None:
        self.results.html.toggle_existing_path()

    @on(Button.Pressed, '#html-report-copy')
    def on_results_copy_html_path(self) -> None:
        self.results.html.copy_path()

    @on(Button.Pressed, '#html-report-cancel')
    def on_results_cancel_html(self) -> None:
        if self.execution.session.kind == "html-report":
            self.action_request_stop()

    @on(Button.Pressed, '#open-model-store')
    def on_preflight_open_model_store(self) -> None:
        self.preflight.open_model_store()

    @on(Button.Pressed, '#open-result-profiling')
    def on_profiles_open_selection(self) -> None:
        self.profiles.open_selection_dialog()

    @on(Input.Submitted, '#slash-command')
    def on_slash_slash_command_submitted(self, event: TextualInput.Submitted) -> None:
        self.slash.slash_command_submitted(event)

    @on(Button.Pressed, '#start-run')
    def on_experiments_start_run_button(self) -> None:
        self.experiments.start_run_button()

    @on(Button.Pressed, '#status-return')
    def action_return_from_issue(self) -> None:
        self.preflight.panel.return_to_origin()

    @on(Button.Pressed, '#process-inherited-resources')
    def on_process_inherited_resources(self) -> None:
        self.configuration.use_process_inherited_resources()

    @on(Button.Pressed, '#probe-largest')
    def on_experiments_probe_largest_button(self) -> None:
        self.experiments.probe_largest_button()

    def on_key(self, event: events.Key) -> None:
        if (
            self.is_running and self.form_ready and not self.ui_closing
            and event.key == "shift+enter"
            and isinstance(self.focused, TextualInput)
            and self.focused.id == "log-search"
        ):
            self.logs.find(previous=True)
            event.stop()
            event.prevent_default()
