"""Present Process permission evidence owned by the existing PreflightController."""
from __future__ import annotations

from typing import TYPE_CHECKING

from acprof.experiments.configuration import ConfigIssue
from acprof.host.preparation.preparation_log import redact_text
from acprof.tui.i18n import message_id
from acprof.tui.ui.structured import FactDetailScreen
from acprof.tui.ui.view_models import FactRow

if TYPE_CHECKING:
    from acprof.tui.run.diagnostics import PreflightCheck
    from acprof.tui.run.preflight_controller import PreflightController


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
    return 'blocked' if controller.error else 'ready' if check and check.status == 'ok' else 'stale'


def field_issues(controller: PreflightController) -> tuple[ConfigIssue, ...]:
    check = resource_check(controller)
    if (not controller.check_completed or controller.check_running or not check or not check.resource_diagnostic
            or not controller.config_matches() or check.status != 'fail'):
        return ()
    field = 'cpus' if 'cpu' in check.resource_diagnostic.required else 'mems'
    return (ConfigIssue(field, message_id('run.resources.field_error')),)


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
