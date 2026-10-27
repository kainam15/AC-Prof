"""On-demand posthoc profiler selection for a completed experiment."""
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Static

from acprof.tui.app_contracts import translated_app
from acprof.tui.i18n import message_id
from acprof.tui.profile_status import ProfilerOverview, ProfilerStatus
from acprof.tui.rendering import CjkCompositor
from acprof.tui.views import StatusCheckbox

TOOL_LABELS = {
    "torch": message_id("ui.2265"),
    "ncu": message_id("ui.2266"),
    "nsys": message_id("ui.2267"),
    "massif": message_id("ui.2268"),
}


def tool_status_message(status: ProfilerStatus):
    if status.state == "complete":
        return message_id("ui.2256", status.completed, status.total)
    if status.state == "failed":
        return message_id("ui.2257", status.errors, status.completed, status.total)
    if status.state == "partial":
        return message_id("ui.2258", status.completed, status.total)
    if status.state == "missing":
        return message_id("ui.2259")
    return message_id("ui.0406")


class ProfilerChoiceScreen(ModalScreen[tuple[str, tuple[str, ...]] | None]):
    """Select missing applicable tools; never trigger profiling implicitly."""

    AUTO_FOCUS = "#profile-choice-close"
    BINDINGS = [("escape", "close", "取消")]
    CSS = """
    ProfilerChoiceScreen { align: center middle; }
    #profiler-choice-dialog {
        width: 92%; max-width: 84; height: auto; max-height: 85%;
        border: round $accent; background: $surface; padding: 1 2;
    }
    #profiler-choice-title { height: auto; text-style: bold; margin-bottom: 1; }
    #profiler-choice-experiment { height: auto; max-height: 2; color: $text-muted; }
    #profiler-choice-list { height: auto; max-height: 45vh; margin: 1 0; }
    .profiler-choice-row { grid-size: 2; grid-columns: 1fr 24; height: auto; min-height: 3; }
    .profiler-choice-row Checkbox { width: 1fr; min-width: 0; }
    .profiler-choice-row Static { width: 1fr; height: auto; margin-top: 1; }
    #profiler-choice-note { height: auto; color: $text-muted; }
    #profiler-choice-actions { height: auto; align-horizontal: right; margin-top: 1; }
    #profiler-choice-actions Button { margin-left: 1; }
    """

    def __init__(self, overview: ProfilerOverview):
        super().__init__()
        self._compositor = CjkCompositor()
        self.overview = overview

    def compose(self) -> ComposeResult:
        tr = translated_app(self.app).tr
        with Vertical(id="profiler-choice-dialog"):
            yield Static(tr(message_id("ui.2252")), id="profiler-choice-title")
            yield Static(str(self.overview.result_dir), id="profiler-choice-experiment", markup=False)
            with VerticalScroll(id="profiler-choice-list"):
                for status in self.overview.tools:
                    with Grid(classes="profiler-choice-row"):
                        yield StatusCheckbox(
                            f"{status.tool} · {tr(TOOL_LABELS[status.tool])}",
                            value=status.tool in self.overview.suggested_tools,
                            name=status.tool,
                            id=f"profile-tool-{status.tool}",
                            disabled=not status.needs_collection,
                            classes="option-checkbox profile-tool",
                        )
                        yield Static(tr(tool_status_message(status)), markup=False)
            yield Static(tr(message_id("ui.2261")), id="profiler-choice-note", markup=False)
            with Horizontal(id="profiler-choice-actions"):
                yield Button(tr("取消"), id="profile-choice-close")
                yield Button(tr(message_id("ui.2262")), id="profile-choice-plan")
                yield Button(tr("执行补采"), id="profile-choice-run", variant="warning")

    def on_mount(self) -> None:
        self.update_actions()

    @on(Checkbox.Changed, ".profile-tool")
    def update_actions(self) -> None:
        selected = bool(self.selected_tools())
        self.query_one("#profile-choice-plan", Button).disabled = not selected
        self.query_one("#profile-choice-run", Button).disabled = not selected

    def selected_tools(self) -> tuple[str, ...]:
        return tuple(status.tool for status in self.overview.tools
                     if self.query_one(f"#profile-tool-{status.tool}", Checkbox).value)

    @on(Button.Pressed, "#profile-choice-close")
    def action_close(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#profile-choice-plan")
    def show_plan(self) -> None:
        if self.selected_tools():
            self.dismiss(("dry-run", self.selected_tools()))

    @on(Button.Pressed, "#profile-choice-run")
    def execute(self) -> None:
        if self.selected_tools():
            self.dismiss(("run", self.selected_tools()))
