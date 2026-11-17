"""On-demand diagnostics projected from the same cached GlobalStatus."""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from textual import on
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from acprof.host.preparation.preparation_log import redact_text
from acprof.tui.i18n import message_id
from acprof.tui.run.environment_view import check_label, global_check_view
from acprof.tui.ui.rendering import CjkCompositor
from acprof.tui.ui.structured import ActionButtons, ActionRequested, Callout, FactGrid
from acprof.tui.ui.view_models import CalloutContent, ViewAction, ViewSection

if TYPE_CHECKING:
    from acprof.tui.run.global_status import GlobalStatus
    from acprof.tui.run.preflight_controller import PreflightController


class IssueDiagnosticScreen(ModalScreen[None]):
    BINDINGS = [('escape', 'close', 'Close')]
    FOCUS_SELECTOR = '#issue-close'
    DEFAULT_CSS = """
    IssueDiagnosticScreen { align: center middle; }
    #issue-dialog { width: 92%; max-width: 100; height: 85%; border: round $accent; background: $surface; }
    #issue-title { height: auto; padding: 0 1; text-style: bold; }
    #issue-body { height: 1fr; padding: 0 1; }
    #issue-actions { height: auto; padding: 0 1; }
    #issue-actions Button { width: auto; min-width: 8; }
    #issue-inherit-note, #issue-resolved { height: auto; color: $text-muted; }
    """

    def __init__(self, controller: PreflightController, check_id: str) -> None:
        super().__init__(id={'process_resources': 'process-resource-screen', 'docker': 'docker-repair-screen'}.get(check_id, 'issue-diagnostic-screen'))
        self._compositor = CjkCompositor()
        self.controller = controller
        self.check_id = check_id

    def compose(self):
        tr = self.controller.app.tr
        with Vertical(id='issue-dialog'):
            yield Static('', id='issue-title', markup=False)
            with VerticalScroll(id='issue-body'):
                yield Static(tr(message_id('run.status.resolved')), id='issue-resolved', markup=False)
                yield Callout(CalloutContent('neutral', '', '', ''), id='issue-callout')
                yield FactGrid(ViewSection('', ()), id='issue-facts')
                yield ActionButtons(id='issue-recheck')
                if self.check_id == 'process_resources':
                    yield Button(tr(message_id('run.resources.inherit')), id='process-inherited-resources')
                    yield Static(tr(message_id('run.resources.guide.auto')), id='issue-inherit-note', markup=False)
            with Horizontal(id='issue-actions'):
                yield Button(tr(message_id('common.action.close_dialog')), id='issue-close')

    def on_mount(self) -> None:
        self.update_cached(self.controller.status)
        self.call_after_refresh(self.query_one(self.FOCUS_SELECTOR).focus)

    def on_screen_resume(self) -> None:
        if self.is_mounted:
            self.update_cached(self.controller.status)

    def update_cached(self, snapshot: GlobalStatus) -> None:
        item = next((item for item in snapshot.items if item.check.check_id == self.check_id), None)
        callout = self.query_one('#issue-callout', Callout)
        facts = self.query_one('#issue-facts', FactGrid)
        self.query_one('#issue-resolved').display = item is None
        callout.display = facts.display = False
        allowed = self.controller.app.operation_state().allows('check')
        actions: tuple[ViewAction, ...] = (ViewAction('environment-retry', message_id('run.environment.retry'), disabled=not allowed),)
        if item is not None:
            row = item.check
            self.query_one('#issue-title', Static).update(self.controller.app.tr(check_label(row)))
            detail = redact_text(self.controller.app.tr(row.detail), limit=None)
            cause = redact_text(detail.splitlines()[0], limit=180) if detail else ''
            view = global_check_view(item, technical_detail=detail, cause=cause,
                                     check_allowed=allowed, navigate_allowed=False, environment=self.controller.app.environment)
            if view.callout is not None:
                callout.display = True
                if row.resource_diagnostic is not None:
                    actions += (ViewAction('process-resource-details', message_id('run.resources.details'),
                                           disabled=not allowed or not self.controller.config_matches()),)
                content = replace(view.callout, actions=actions)
                if content != callout.content:
                    callout.update_content(content)
            if view.facts is not None:
                facts.display = True
                if view.facts != facts.section:
                    facts.update_section(view.facts)
        retry = self.query_one('#issue-recheck', ActionButtons)
        retry.update_actions(actions)
        retry.display = not callout.display
        for button in self.query('#process-inherited-resources'):
            button.disabled = not allowed

    @on(ActionRequested)
    def operate(self, event: ActionRequested) -> None:
        if event.action_id == 'environment-retry':
            event.stop()
            self.controller.quick_check()

    @on(Button.Pressed, '#issue-close')
    def action_close(self) -> None:
        self.dismiss(None)
