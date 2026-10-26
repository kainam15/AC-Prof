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

from acprof.messages import message
from acprof.tui.app_contracts import translated_app
from acprof.tui.commands import PendingLaunch, format_command
from acprof.tui.experiment_catalog import ExperimentRecord
from acprof.tui.i18n import error_message
from acprof.tui.recovery import RecoveryReview, review_recovery
from acprof.tui.rendering import CjkCompositor

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui


class RecoveryScreen(ModalScreen[str | None]):
    AUTO_FOCUS = ""
    BINDINGS = [("escape", "cancel", "取消")]
    CSS = """
    RecoveryScreen { align: center middle; }
    #recovery-dialog { width: 94%; max-width: 110; height: 85%; border: round $accent; padding: 1; background: $surface; }
    #recovery-title { height: auto; text-style: bold; margin-bottom: 1; }
    #recovery-body { height: 1fr; }
    #recovery-actions { height: auto; margin-top: 1; }
    #recovery-actions Button { width: 1fr; min-width: 8; margin-right: 1; }
    """

    def __init__(self, review: RecoveryReview):
        super().__init__()
        self._compositor = CjkCompositor()
        self.review = review

    def compose(self):
        tr = translated_app(self.app).tr
        with Vertical(id="recovery-dialog"):
            yield Static(tr("恢复 / 重试"), id="recovery-title")
            with VerticalScroll(id="recovery-body"):
                yield Static(tr(self.review.detail), id="recovery-detail", markup=False)
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
            self.app.notify("恢复实验缺少运行配置", severity="error", timeout=10)
            return
        if review is None or (confirmed and pending.config.resume and review.resume is not None):
            launch = pending if review is None else review.resume
            if launch is None or launch.config is None:
                self.app.notify("恢复实验缺少运行配置", severity="error", timeout=10)
                return
            self.app.configuration.apply_config(launch.config)
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
            # Resume belongs to the confirmed launch, not the editable form.
            if launch.config is None or self.app.configuration.collect_config() != replace(launch.config, resume=False):
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
