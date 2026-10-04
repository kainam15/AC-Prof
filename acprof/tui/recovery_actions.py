"""One recovery review for start, history and the failed-run action."""
from __future__ import annotations

from threading import Event
from typing import TYPE_CHECKING

from textual import on, work
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message_pump import MessagePump
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Static

from acprof.experiment import build_run_command
from acprof.messages import message
from acprof.tui.commands import PendingLaunch, format_command
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
        tr = self.app.tr
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


class RecoveryActions(MessagePump):
    def _review_run_destination(self: AcprofTui, pending: PendingLaunch, *, record=None,
                                confirmed: bool = False) -> None:
        if not self._allow_operation("run"):
            return
        token = self._recovery_request = Event()
        self._read_jobs.add(token)
        self._set_busy(True)
        self._execute_recovery_review(pending, record, token, confirmed)

    @work(thread=True, group="recovery", exit_on_error=False)
    def _execute_recovery_review(self: AcprofTui, pending, record, token, confirmed) -> None:
        from acprof.tui.app import PROJECT_DIR, PYTHON_EXECUTABLE
        try:
            review, error = review_recovery(pending, record=record, project_dir=PROJECT_DIR,
                                            python_executable=PYTHON_EXECUTABLE), ""
        except Exception as exc:
            review, error = None, error_message(exc)
        self._safe_process_callback(self._show_recovery_review, pending, token, confirmed, review, error)

    def _show_recovery_review(self: AcprofTui, pending, token, confirmed, review, error) -> None:
        self._read_jobs.discard(token)
        if not self.is_running or self._ui_closing:
            return
        self._set_busy(self._is_busy())
        if token is not self._recovery_request or token.is_set():
            return
        self._recovery_request = None
        if error:
            self.notify(error, severity="error", timeout=10)
            return
        if review is None or (confirmed and pending.config.resume and review.resume is not None):
            launch = pending if review is None else review.resume
            self._apply_config(launch.config)
            if self._preflight_run_reason():
                self._recovery_after_check = (launch, review.record if review else None)
                self.action_quick_check()
            else:
                self._launch(launch)
            return
        self._picker_open = True
        self._set_busy(True)
        self.push_screen(RecoveryScreen(review), lambda action: self._recovery_selected(review, action))

    def _finish_recovery_check(self: AcprofTui) -> None:
        pending = self._recovery_after_check
        self._recovery_after_check = None
        if pending is None:
            return
        launch, record = pending
        try:
            if self._collect_config() != launch.config:
                self.notify("配置已变化；已取消本次恢复启动。", severity="warning")
                return
        except ValueError:
            return
        if self._allow_collection():
            self._review_run_destination(launch, record=record, confirmed=True)

    def _recovery_selected(self: AcprofTui, review: RecoveryReview, action: str | None) -> None:
        self._picker_closed()
        if action is not None:
            pending = review.resume if action == "resume" else review.restart
            if pending is not None:
                # Re-read immediately before launch: a modal may outlive its snapshot.
                self._review_run_destination(pending, record=review.record if action == "resume" else None,
                                             confirmed=True)

    @on(Button.Pressed, "#review-run-recovery")
    def review_failed_run(self: AcprofTui) -> None:
        from acprof.tui.app import PROJECT_DIR, PYTHON_EXECUTABLE
        if not self._allow_operation("run"):
            return
        config = self._active_run_config
        if config is not None:
            command = build_run_command(config, project_dir=PROJECT_DIR, python_executable=PYTHON_EXECUTABLE)
            self._review_run_destination(PendingLaunch(tuple(command), "run", config))
