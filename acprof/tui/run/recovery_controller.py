"""Shared recovery review for experiment starts and history."""
from __future__ import annotations

from dataclasses import replace
from functools import partial
from threading import Event
from typing import TYPE_CHECKING

from textual import on
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Static
from textual.worker import Worker

from acprof.messages import join_messages, message
from acprof.tui.commands import PendingLaunch, format_command
from acprof.tui.experiments.catalog import ExperimentRecord
from acprof.tui.i18n import error_message, message_id
from acprof.tui.run.recovery import RecoveryReview, review_recovery
from acprof.tui.ui.contracts import translated_app
from acprof.tui.ui.dialogs import focus_horizontal_action
from acprof.tui.ui.rendering import CjkCompositor
from acprof.tui.ui.structured import Callout, FactGrid, StatusBanner
from acprof.tui.ui.view_models import CalloutContent, FactRow, StatusSummary, ViewSection

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui


_PHASE_VIEWS = {
    "preparation": (message_id("run.recovery.preparation_incomplete"), message_id("run.recovery.phase.preparation")),
    "measurement": (message_id("run.recovery.existing_evidence"), message_id("run.recovery.phase.measurement")),
    "complete": (message_id("run.recovery.completed_state"), message_id("run.stage.completed")),
    "unknown": (message_id("run.recovery.unknown_state"), message_id("run.recovery.phase.unknown")),
}


def recovery_views(review: RecoveryReview) -> tuple[StatusSummary, CalloutContent, ViewSection]:
    headline, phase_label = _PHASE_VIEWS.get(review.phase, _PHASE_VIEWS["unknown"])
    if review.phase == "complete":
        level = "success"
    elif review.resume is not None:
        level = "neutral" if review.phase == "measurement" else "warning"
    else:
        level = "warning" if review.restart is not None else "error"
    summary = StatusSummary(level=level, headline=headline)
    problem: str = (join_messages("\n", list(review.issues)) if review.issues
                    else message_id("run.recovery.ready_to_resume") if review.resume is not None else review.detail)
    impact = (message_id("run.recovery.original_options_notice") if review.resume is not None
              else message_id("run.recovery.resume_unavailable_impact"))
    next_step = (message_id("run.recovery.new_directory_notice", review.restart.config.output_dir)
                 if review.restart is not None and review.restart.config is not None
                 else message_id("run.recovery.next_adjust_directory"))
    callout = CalloutContent(level=level, problem=problem, impact=impact, next_step=next_step)
    rows = [
        FactRow(message_id("run.recovery.original_directory"), str(review.directory), copyable=True),
        FactRow(message_id("run.recovery.phase_label"), phase_label,
                status="ok" if review.phase == "complete" else "exists" if review.phase == "measurement" else "warning"),
        FactRow(message_id("run.recovery.resume_status"),
                message_id("run.recovery.resume_available") if review.resume is not None else message_id("run.environment.item.fail"),
                status="ok" if review.resume is not None else "warning"),
    ]
    if review.restart is not None and review.restart.config is not None:
        rows.append(FactRow(message_id("run.recovery.new_directory_label"), review.restart.config.output_dir, copyable=True))
    return summary, callout, ViewSection(title="", rows=tuple(rows))


class RecoveryScreen(ModalScreen[str | None]):
    AUTO_FOCUS = ""
    BINDINGS = [("escape", "cancel", "取消")]
    CSS = """
    RecoveryScreen { align: center middle; }
    #recovery-dialog { width: 94%; max-width: 110; height: 85%; border: round $accent; padding: 1; background: $surface; }
    #recovery-title { height: auto; text-style: bold; margin-bottom: 1; }
    #recovery-body { height: 1fr; }
    #recovery-callout, #recovery-facts { margin-top: 1; }
    #recovery-actions { height: auto; margin-top: 1; }
    #recovery-actions Button { width: 1fr; min-width: 8; margin-right: 1; }
    """

    def __init__(self, review: RecoveryReview):
        super().__init__()
        self._compositor = CjkCompositor()
        self.review = review

    def compose(self):
        tr = translated_app(self.app).tr
        summary, callout, facts = recovery_views(self.review)
        with Vertical(id="recovery-dialog"):
            yield Static(tr("恢复 / 重试"), id="recovery-title")
            with VerticalScroll(id="recovery-body"):
                yield StatusBanner(summary, id="recovery-banner", include_actions=False)
                if callout is not None:
                    yield Callout(callout, id="recovery-callout", include_details=False, include_actions=False)
                yield FactGrid(facts, id="recovery-facts", collapsible=False)
                with Collapsible(title=tr("命令详情"), collapsed=True):
                    for label, pending in (("恢复命令", self.review.resume), ("新实验命令", self.review.restart)):
                        if pending:
                            yield Static(tr(message("{0}\n{1}", message(label), format_command(pending.command))), markup=False)
            with Horizontal(id="recovery-actions"):
                yield Button(tr("取消"), id="recovery-cancel")
                yield Button(tr("重试准备" if self.review.phase == "preparation" else "继续实验"),
                             id="recovery-resume", disabled=self.review.resume is None,
                             variant="primary" if self.review.resume else "default")
                yield Button(tr("新建实验"), id="recovery-new", disabled=self.review.restart is None,
                             variant="default" if self.review.resume else "primary")

    def on_key(self, event) -> None:
        if focus_horizontal_action(self, "#recovery-actions", event.key):
            event.prevent_default()
            event.stop()

    @on(Button.Pressed)
    def choose(self, event: Button.Pressed) -> None:
        action = {"recovery-cancel": None, "recovery-resume": "resume", "recovery-new": "restart"}
        if event.button.id in action:
            self.dismiss(action[event.button.id])

    def action_cancel(self) -> None:
        self.dismiss(None)


class RecoveryController:
    """Own recovery state and operations; the App owns lifecycle and scheduling."""

    def __init__(self, app: AcprofTui) -> None:
        self.app = app
        self.recovery_request: Event | None = None
        self.recovery_after_check: tuple[PendingLaunch, ExperimentRecord | None] | None = None

    def review_run_destination(self, pending: PendingLaunch, *, record=None,
                                confirmed: bool = False) -> None:
        if not self.app.allow_operation("run"):
            return
        token = self.recovery_request = Event()
        self.app.read_jobs.add(token)
        self.app.set_busy(True)
        self.execute_recovery_review(pending, record, token, confirmed)

    def execute_recovery_review(self, pending: PendingLaunch, record: ExperimentRecord | None, token: Event, confirmed: bool) -> Worker[None]:
        """Schedule this operation on the owning App worker manager."""
        return self.app.run_worker(
            partial(self._execute_recovery_review_worker, pending, record, token, confirmed),
            name="execute_recovery_review", thread=True, group='recovery', exit_on_error=False,
        )

    def _execute_recovery_review_worker(self, pending: PendingLaunch, record: ExperimentRecord | None,
                                 token: Event, confirmed: bool) -> None:
        try:
            review, error = review_recovery(pending, record=record, project_dir=self.app.project_dir,
                                            python_executable=self.app.python_executable), ""
        except Exception as exc:
            review, error = None, error_message(exc)
        self.app.process.safe_process_callback(self.show_recovery_review, pending, token, confirmed, review, error)

    def show_recovery_review(self, pending: PendingLaunch, token: Event, confirmed: bool,
                              review: RecoveryReview | None, error: str) -> None:
        self.app.read_jobs.discard(token)
        if not self.app.is_running or self.app.ui_closing:
            return
        self.app.set_busy(self.app.is_busy())
        if token is not self.recovery_request or token.is_set():
            return
        self.recovery_request = None
        if error:
            self.app.notify(error, severity="error", timeout=10)
            return
        if pending.config is None:
            self.app.notify(message_id("run.recovery.missing_config"), severity="error", timeout=10)
            return
        if review is None or (confirmed and pending.config.resume and review.resume is not None):
            launch = pending if review is None else review.resume
            if launch is None or launch.config is None:
                self.app.notify(message_id("run.recovery.missing_config"), severity="error", timeout=10)
                return
            self.app.configuration.apply_config(launch.config, preserve_model_edits=review is None)
            if self.app.preflight.preflight_run_reason():
                self.recovery_after_check = (launch, review.record if review else None)
                self.app.preflight.quick_check()
            else:
                self.app.launch(launch)
            return
        self.app.catalog.picker_open = True
        self.app.set_busy(True)
        self.app.push_screen(RecoveryScreen(review), lambda action: self.recovery_selected(review, action))

    def finish_recovery_check(self) -> None:
        pending = self.recovery_after_check
        self.recovery_after_check = None
        if pending is None:
            return
        launch, record = pending
        try:
            # The form uses explicit Docker/Process, while a legacy resume
            # must keep its frozen original CLI runtime (including auto).
            from acprof.tui.run.form import runtime_form_config
            expected = runtime_form_config(replace(launch.config, resume=False)) if launch.config else None
            if expected is None or self.app.configuration.collect_config() != expected:
                self.app.notify("配置已变化；已取消本次恢复启动。", severity="warning")
                return
        except ValueError:
            return
        if self.app.preflight.allow_collection():
            self.review_run_destination(launch, record=record, confirmed=True)

    def recovery_selected(self, review: RecoveryReview, action: str | None) -> None:
        self.app.catalog.picker_closed()
        if action is not None:
            pending = review.resume if action == "resume" else review.restart
            if pending is not None:
                # Re-read immediately before launch: a modal may outlive its snapshot.
                self.review_run_destination(pending, record=review.record if action == "resume" else None,
                                             confirmed=True)
