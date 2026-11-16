"""Present Process permission evidence owned by the existing PreflightController."""
from __future__ import annotations

from typing import TYPE_CHECKING, cast

from textual.widgets import Button

from acprof.experiments.configuration import ConfigIssue
from acprof.host.preparation.preparation_log import redact_text
from acprof.tui.i18n import message_id
from acprof.tui.ui.structured import Callout, FactDetailScreen
from acprof.tui.ui.view_models import CalloutContent, FactRow, ViewAction

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui
    from acprof.tui.run.diagnostics import PreflightCheck
    from acprof.tui.run.preflight_controller import PreflightController


class ProcessResourceCallout(Callout):
    """Keep the focused recovery action visible after responsive layout settles."""

    def on_resize(self) -> None:
        self.call_after_refresh(self.reveal_action)

    def reveal_action(self) -> None:
        app = cast("AcprofTui", self.app)
        focused = app.focused
        if (focused is not None and self in focused.ancestors
                and not app.execution.session.snapshot.measurement_active):
            app.configuration.focus_config_field(focused)


def resource_check(controller: PreflightController) -> PreflightCheck | None:
    return next((check for check in controller.checks if check.check_id == 'process_resources'), None)


def resource_state(controller: PreflightController) -> str:
    if controller.check_running:
        return controller.resource_phase
    if not controller.check_completed or not controller.config_matches():
        return 'stale'
    check = resource_check(controller)
    if check and check.status == 'fail':
        diagnostic = check.resource_diagnostic
        return 'authorization' if diagnostic and diagnostic.code in {'permission_denied', 'delegation_missing'} else 'blocked'
    return 'blocked' if controller.error else 'ready'


def field_issues(controller: PreflightController) -> tuple[ConfigIssue, ...]:
    check = resource_check(controller)
    if not check or not check.resource_diagnostic or not controller.config_matches() or check.status != 'fail':
        return ()
    field = 'cpus' if 'cpu' in check.resource_diagnostic.required else 'mems'
    return (ConfigIssue(field, message_id('run.resources.field_error')),)


def refresh(controller: PreflightController, *, visible: bool) -> None:
    app = controller.app
    app.query_one('#process-resource-help').display = visible
    entry = app.query_one('#process-resource-open', Button)
    entry.display = visible
    if not visible:
        return
    state = resource_state(controller)
    label = message_id('run.resources.state.' + state)
    app.localization.set_text(entry, message_id('run.resources.open', label), 'label')
    entry.disabled = app.is_busy()
    check = resource_check(controller) if controller.config_matches() and not controller.check_running else None
    diagnostic = check.resource_diagnostic if check else None
    problem = check.detail if check and check.status == 'fail' else label
    impact = message_id('run.resources.impact')
    if state == 'ready':
        impact = message_id('run.resources.inherited' if diagnostic and diagnostic.code == 'inherited'
                            else 'run.resources.verified')
    actions_disabled = app.is_busy() or controller.check_running or app.execution.session.snapshot.measurement_active
    app.query_one('#process-resource-callout', Callout).update_content(CalloutContent(
        'warning' if state == 'authorization' else 'error' if state == 'blocked' else
        'success' if state == 'ready' else 'info',
        problem, impact, message_id('run.resources.next'),
        actions=(ViewAction('process-resource-retry', message_id('run.resources.retry'), True, actions_disabled),
                 ViewAction('process-resource-details', message_id('run.resources.details'),
                            disabled=not bool(diagnostic) or actions_disabled)),
    ))
    app.query_one('#process-inherited-resources', Button).disabled = actions_disabled


def open_panel(controller: PreflightController) -> None:
    app = controller.app
    if app.is_busy() or app.execution.session.snapshot.measurement_active:
        return
    app.activate_tab('run-tab')
    app.configuration.show_run_form('advanced-form')
    button = app.query_one('#process-resource-callout Button', Button)
    app.call_after_refresh(app.configuration.focus_config_field, button)


def show_details(controller: PreflightController) -> None:
    app = controller.app
    check = resource_check(controller)
    if (app.is_busy() or controller.check_running or not controller.config_matches()
            or not check or not check.resource_diagnostic or app.execution.session.snapshot.measurement_active):
        return
    diagnostic = check.resource_diagnostic
    lines = [app.tr(diagnostic.summary),
             f'cgroup: {diagnostic.version}', f'current: {diagnostic.current}',
             f'delegated root: {diagnostic.parent}', f'required: {", ".join(diagnostic.required)}',
             f'controllers: {", ".join(diagnostic.controllers)}',
             f'subtree_control: {", ".join(diagnostic.enabled)}', f'writable: {diagnostic.writable}',
             f'repaired: {diagnostic.repaired}', diagnostic.systemd,
             *(app.tr(step) for step in diagnostic.guidance), diagnostic.detail]
    app.push_screen(FactDetailScreen(FactRow(message_id('run.resources.details'),
                                           redact_text('\n\n'.join(line for line in lines if line)), copyable=True)))
