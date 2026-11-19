"""A repair dialog inside global status; commands are displayed and copied only."""
from __future__ import annotations

from textual import on
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Button, Collapsible, Link, Static

from acprof.host.preparation.preparation_log import redact_text
from acprof.tui.i18n import message_id
from acprof.tui.run.docker_diagnostics import DockerDiagnostic, DockerHost
from acprof.tui.run.docker_guide import DockerStep, repair_steps
from acprof.tui.run.environment_view import check_presentation
from acprof.tui.run.global_status import GlobalStatus
from acprof.tui.run.issue_view import IssueDiagnosticScreen
from acprof.tui.ui.contracts import translated_app
from acprof.tui.ui.presentation import COMPACT_COLLAPSED_SYMBOL, COMPACT_EXPANDED_SYMBOL
from acprof.tui.ui.structured import Callout
from acprof.tui.ui.view_models import CalloutContent


class DockerCommandStep(Vertical):
    def __init__(self, step: DockerStep) -> None:
        super().__init__(classes='docker-step')
        self.step = step

    def compose(self):
        tr = translated_app(self.app).tr
        step = self.step
        yield Static(tr(step.title), classes='docker-step-title', markup=False)
        yield Static(tr(step.note), id='docker-note-' + step.key, markup=False)
        if step.command:
            yield Button(tr(message_id('run.docker_repair.copy')), id='docker-copy-' + step.key, classes='docker-copy-command')
            with Collapsible(title=tr(message_id('run.docker_repair.commands')), collapsed=step.command.count('\n') > 4,
                             collapsed_symbol=COMPACT_COLLAPSED_SYMBOL, expanded_symbol=COMPACT_EXPANDED_SYMBOL):
                yield Static(step.command, classes='docker-command', markup=False)
        if step.url:
            yield Link(tr(message_id('run.docker_repair.docs')), url=step.url, tooltip=step.url)
            yield Button(tr(message_id('run.docker_repair.copy_link')), classes='docker-copy-link')

    @on(Button.Pressed)
    def copy(self, event: Button.Pressed) -> None:
        event.stop()
        value = self.step.url if event.button.has_class('docker-copy-link') else self.step.command
        self.app.copy_to_clipboard(value)
        self.app.notify(translated_app(self.app).tr(message_id('structured.action.copied')))


class DockerSteps(Vertical):
    steps: reactive[tuple[DockerStep, ...]] = reactive((), recompose=True)

    def compose(self):
        for step in self.steps:
            yield DockerCommandStep(step)


class DockerRepairScreen(IssueDiagnosticScreen):
    FOCUS_SELECTOR = '#docker-recheck'
    DEFAULT_CSS = """
    DockerRepairScreen { align: center middle; }
    #docker-dialog { width: 94%; max-width: 104; height: 92%; border: round $accent; background: $surface; }
    #docker-title { height: auto; padding: 0 1; text-style: bold; }
    #docker-status { height: auto; padding: 0 1; color: $accent; }
    #docker-status.status-warning { color: $warning; }
    #docker-status.status-error { color: $error; }
    #docker-status.status-success { color: $success; }
    #docker-body { height: 1fr; padding: 0 1; }
    #docker-body Static { height: auto; }
    #docker-host, #docker-wsl, #docker-admin { margin-bottom: 1; }
    #docker-host { color: $text-muted; }
    #docker-wsl, #docker-admin { color: $warning; }
    DockerSteps, DockerCommandStep { height: auto; }
    DockerCommandStep { border-top: solid $panel; padding: 0 0 1 0; }
    .docker-step-title { text-style: bold; }
    .docker-step Button { width: auto; min-width: 12; height: 3; }
    .docker-command { color: $text; background: $panel; padding: 0 1; }
    #docker-footer { height: auto; padding: 0 1; }
    #docker-footer Button { width: auto; min-width: 10; margin-right: 1; }
    """

    def __init__(self, controller) -> None:
        super().__init__(controller, 'docker')

    def compose(self):
        tr = self.controller.app.tr
        with Vertical(id='docker-dialog'):
            yield Static(tr(message_id('run.docker_repair.title')), id='docker-title', markup=False)
            yield Static('', id='docker-status', markup=False)
            with VerticalScroll(id='docker-body'):
                yield Callout(CalloutContent('neutral', '', '', ''), id='docker-summary',
                              include_actions=False, include_details=False)
                yield Static('', id='docker-host', markup=False)
                yield Static(tr(message_id('run.docker_repair.wsl')), id='docker-wsl', markup=False)
                yield Static(tr(message_id('run.docker_repair.admin')), id='docker-admin', markup=False)
                yield DockerSteps()
                with Collapsible(title=tr(message_id('structured.detail.title')), collapsed=True,
                                 collapsed_symbol=COMPACT_COLLAPSED_SYMBOL, expanded_symbol=COMPACT_EXPANDED_SYMBOL,
                                 id='docker-technical'):
                    yield Static('', id='docker-raw', markup=False)
            with Horizontal(id='docker-footer'):
                yield Button(tr(message_id('run.environment.retry')), id='docker-recheck', variant='primary')
                yield Button(tr(message_id('common.action.close_dialog')), id='docker-close')

    def update_cached(self, snapshot: GlobalStatus) -> None:
        tr = self.controller.app.tr
        item = next((item for item in snapshot.items if item.check.check_id == 'docker'), None)
        diagnostic = item.check.docker_diagnostic if item else None
        error = self.controller.error or self.controller.environment_error
        diagnostic = diagnostic or DockerDiagnostic('unknown', DockerHost(), detail=tr(item.check.detail) if item else error)
        busy = self.controller.check_running
        stale = item is None or item.check.status in {'stale', 'pending'}
        presentation = check_presentation(item.check) if item else None
        status = (tr(message_id('run.docker_repair.checking')) if busy else tr(message_id('run.docker_repair.stale')) if stale
                  else f'{presentation.symbol} {tr(presentation.label)}' if presentation else '')
        status_widget = self.query_one('#docker-status', Static)
        status_widget.update(status)
        status_widget.set_classes('status-' + ('info' if busy or stale else presentation.level if presentation else 'neutral'))
        impact = 'ready' if diagnostic.code == 'ready' and not busy and not stale else 'required' if item and item.blocks else 'optional'
        self.query_one('#docker-summary', Callout).update_content(CalloutContent(
            'info' if busy or stale else presentation.level if presentation else 'neutral',
            message_id('run.docker_repair.recheck_error', redact_text(error, limit=240)) if error and item is None else diagnostic.summary,
            message_id('run.docker_repair.impact.' + impact), diagnostic.next_step,
        ))
        self.query_one('#docker-host', Static).update(tr(message_id('run.docker_repair.host',
            diagnostic.host.name or tr(message_id('common.value.unknown')),
            {'native_linux': 'Native Linux', 'wsl2': 'WSL2 / PARTIAL'}.get(diagnostic.host.environment,
                diagnostic.host.environment or tr(message_id('common.value.unknown'))))))
        self.query_one('#docker-wsl').display = diagnostic.host.environment == 'wsl2'
        self.query_one('#docker-admin').display = diagnostic.code not in {'ready', 'unsupported_environment'}
        self.query_one(DockerSteps).steps = repair_steps(diagnostic)
        self.query_one('#docker-raw', Static).update(redact_text('\n'.join((
            f'code: {diagnostic.code}', f'CLI: {diagnostic.cli}', f'context: {diagnostic.context}',
            f'endpoint: {diagnostic.endpoint}', f'product: {diagnostic.product}',
            f'selection: {diagnostic.selection_source}', '', diagnostic.detail,
        )), limit=None))
        retry = self.query_one('#docker-recheck', Button)
        retry.disabled = not self.controller.app.operation_state().allows('check')
        retry.tooltip = tr(message_id('run.environment.measurement_cached')) if self.controller.app.execution.session.snapshot.measurement_active else None

    @on(Button.Pressed, '#docker-recheck')
    def recheck(self, event: Button.Pressed) -> None:
        event.stop()
        self.controller.quick_check()

    @on(Button.Pressed, '#docker-close')
    def close_button(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_close()
