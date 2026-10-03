"""Small decision/error dialogs belonging to the active collection."""
from __future__ import annotations

from textual import on
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Label, Static

from acprof.host.model_errors import ModelLookupError
from acprof.messages import message
from acprof.tui.rendering import CjkCompositor
from acprof.tui.review_inputs import review_answers, review_input


def phase_summary(snapshot):
    names = {"not_started": "未开始", "running": "进行中", "waiting": "待确认",
             "passed": "通过", "failed": "失败", "cancelled": "已取消"}
    return message("接口解析：{0}　运行验证：{1}　测量：{2}",
                   *(message(names[value]) for value in (snapshot.interface_status, snapshot.runtime_status,
                                                         snapshot.measurement_status)))


class PreparationScreen(ModalScreen):
    BINDINGS = [("escape", "cancel", "取消")]
    DEFAULT_CSS = """
    PreparationScreen { align: center middle; }
    #preparation-dialog { width: 90%; max-width: 90; height: auto; max-height: 90%;
        border: round $accent; background: $surface; padding: 0 1; }
    #preparation-title { height: auto; text-style: bold; }
    #preparation-scroll { height: auto; max-height: 28; }
    #preparation-scroll Static, #preparation-scroll Label { height: auto; }
    #preparation-detail, #preparation-error { color: $error; }
    #preparation-actions { height: 3; }
    #preparation-actions Button { min-width: 12; }
    """

    def __init__(self, event: dict, summary: str):
        super().__init__()
        self._compositor = CjkCompositor()
        self.request = event["request"]
        self.stage = event["stage"]
        self.summary = summary

    def compose(self):
        tr = self.app.tr
        review = self.request["kind"] == "review"
        title = "需要你确认" if review else "准备阶段失败"
        action = "确认并继续" if review else "重新验证" if self.stage == "runtime" else "重试此阶段"
        with Vertical(id="preparation-dialog"):
            yield Label(tr(title), id="preparation-title")
            with VerticalScroll(id="preparation-scroll"):
                yield Static(tr(self.summary), markup=False)
                yield Static(tr("确认后继续当前采集，无需重新开始。" if review else
                                "修复下列问题后重试；已完成的阶段保留。"), markup=False)
                model_error = self.request.get("model_error")
                if model_error:
                    failure = ModelLookupError(**model_error)
                    yield Static(f"{tr(failure.summary)}\n{tr(failure.hint)}", id="preparation-detail", markup=False)
                    with Collapsible(title=tr("错误详情"), collapsed=True):
                        yield Static(failure.detail, markup=False)
                else:
                    yield Static(self.request.get("detail", ""), id="preparation-detail", markup=False)
                if not review and self.stage == "runtime":
                    yield Static(tr("若修复涉及镜像内文件或依赖，请重新准备环境。"), markup=False)
                for index, question in enumerate(self.request.get("questions", [])):
                    yield Label(question["path"])
                    yield Static(question.get("reason", ""), markup=False)
                    if not question.get("read_only"):
                        yield review_input(question, identifier=f"preparation-answer-{index}", translate=tr)
                if self.request.get("summary"):
                    with Collapsible(title=tr("已解析字段与证据"), collapsed=True):
                        yield Static(self.request["summary"], markup=False)
                yield Static("", id="preparation-error", markup=False)
            with Horizontal(id="preparation-actions", classes="action-bar"):
                with Horizontal(classes="action-secondary"):
                    yield Button(tr("终止任务"), id="preparation-cancel")
                    if not review and self.stage == "runtime":
                        yield Button(tr("重新准备环境"), id="preparation-rebuild")
                with Horizontal(classes="action-primary"):
                    yield Button(tr(action), id="preparation-continue", variant="primary")

    @on(Button.Pressed, "#preparation-continue")
    def proceed(self):
        try:
            answers = review_answers(self, self.request.get("questions", []), prefix="preparation-answer")
        except (ValueError, TypeError) as exc:
            self.query_one("#preparation-error", Static).update(str(exc))
            return
        self.dismiss({"action": "answer", "answers": answers} if self.request["kind"] == "review" else {"action": "retry"})

    @on(Button.Pressed, "#preparation-cancel")
    def action_cancel(self):
        self.dismiss({"action": "cancel"})

    @on(Button.Pressed, "#preparation-rebuild")
    def rebuild(self):
        self.dismiss({"action": "rebuild"})
