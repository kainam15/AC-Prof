"""One modal owns model review, preparation progress, errors and cancellation."""
from __future__ import annotations

from textual import on
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Label, Static

from acprof.host.model_errors import ModelLookupError
from acprof.messages import message
from acprof.tui.rendering import CjkCompositor
from acprof.tui.review_inputs import review_answers, review_input

STAGE_LABELS = {
    "resolution": "正在读取模型信息…", "dependencies": "正在读取模型信息…",
    "interface": "正在确认模型接口…", "preflight": "正在准备运行环境…",
    "image": "正在准备运行环境…", "environment": "正在准备运行环境…",
    "model": "正在准备模型…", "input": "正在准备验证输入…", "runtime": "正在验证模型运行…",
}


def phase_summary(snapshot):
    names = {"not_started": "未开始", "running": "进行中", "waiting": "待确认",
             "passed": "通过", "failed": "失败", "cancelled": "已取消"}
    return message("接口解析：{0}　运行验证：{1}　测量：{2}",
                   *(message(names[value]) for value in (snapshot.interface_status, snapshot.runtime_status,
                                                         snapshot.measurement_status)))


def failure_text(request, stage, tr):
    if request.get("model_error"):
        failure = ModelLookupError(**request["model_error"])
        return f"{tr(failure.summary)}\n{tr(failure.hint)}"
    failed = request.get("failed_stage", stage)
    names = {"load": "模型加载", "preprocess": "输入处理", "predict": "模型推理", "completion": "模型推理",
             "postprocess": "输出处理", "validate_output": "输出验证", "interface": "模型接口",
             "resolution": "模型信息", "runtime": "模型运行", "image": "运行环境", "environment": "运行环境",
             "model": "模型准备", "input": "验证输入", "preflight": "运行环境"}
    reasons = {
        "postprocess": ("模型已完成推理，但输出处理失败。", "请检查任务选择或模型声明中的输出处理配置。"),
        "validate_output": ("模型输出与任务要求不匹配。", "请检查任务选择或模型声明中的输出处理配置。"),
        "interface": ("模型接口无法在当前环境中使用。", "请检查模型声明、源码文件和运行依赖。"),
        "load": ("模型未能完成加载。", "请检查模型文件、访问权限和可用内存。"),
    }
    reason, hint = reasons.get(failed, ("此阶段未能完成，尚未开始正式采集。", "请查看详细信息，修复配置或环境后重试。"))
    code = request.get("failure", {}).get("reason_code")
    if code == "resource_limit":
        reason, hint = "可用资源不足，模型检测未通过。", "请增加内存或选择更小的模型后重试。"
    return f"{tr('失败阶段')}：{tr(names.get(failed, '模型运行'))}\n{tr(reason)}\n{tr(hint)}"


class PreparationScreen(ModalScreen):
    BINDINGS = [("escape", "cancel", "取消")]
    DEFAULT_CSS = """
    PreparationScreen { align: center middle; }
    #preparation-dialog { width: 90%; max-width: 100; height: auto; max-height: 90%;
        border: round $accent; background: $surface; padding: 0 1; }
    #preparation-title { height: auto; text-style: bold; }
    #preparation-scroll { height: auto; max-height: 24; }
    #preparation-scroll Static, #preparation-scroll Label { height: auto; }
    #preparation-detail, #preparation-error { color: $error; }
    #preparation-actions { height: 3; }
    #preparation-actions Button { min-width: 10; width: 1fr; }
    """

    def __init__(self, event: dict, *, respond=None):
        super().__init__()
        self._compositor = CjkCompositor()
        self.event = event
        self.respond = respond
        self.sending = False
        self._close_requested = False

    def close(self) -> None:
        """Close when active, preserving any screen currently covering us."""
        self._close_requested = True
        self._close_when_active()

    def _close_when_active(self) -> None:
        # Textual dismiss() pops the active screen, even if called on another.
        if self._close_requested and self.app.screen is self:
            self.dismiss(None)

    def on_screen_resume(self) -> None:
        self._close_when_active()

    def update_event(self, event):
        if self._close_requested:
            return
        self.event = event
        self.sending = False
        self.refresh(recompose=True)

    def compose(self):
        tr = self.app.tr
        request = self.event.get("request", {})
        review = request.get("kind") == "review"
        failed = request.get("kind") == "error"
        ready = request.get("resolved") is True
        with Vertical(id="preparation-dialog"):
            yield Label(tr("模型确认" if review else "模型无法正常运行" if failed else "正在检测模型"), id="preparation-title")
            with VerticalScroll(id="preparation-scroll"):
                if review:
                    fields = "\n".join(f"✓ {name}    {value}" for name, value in request.get("fields", {}).items())
                    yield Static(fields, id="preparation-fields", markup=False)
                    for index, question in enumerate(request.get("questions", [])):
                        yield Label(question["path"])
                        yield Static(tr(question.get("reason", "")), markup=False)
                        if not question.get("read_only"):
                            yield review_input(question, identifier=f"preparation-answer-{index}", translate=tr)
                    yield Static(request.get("detail", ""), id="preparation-detail", markup=False)
                    if request.get("advanced"):
                        with Collapsible(title=tr("高级修改"), collapsed=True):
                            for index, question in enumerate(request["advanced"]):
                                yield Label(question["path"])
                                yield review_input(question, identifier=f"preparation-advanced-{index}", translate=tr)
                            yield Button(tr("重新解析"), id="preparation-revise")
                    with Collapsible(title=tr("高级详情"), collapsed=True):
                        yield Static(request.get("summary", ""), markup=False)
                elif failed:
                    yield Static(failure_text(request, self.event["stage"], tr), id="preparation-detail", markup=False)
                    with Collapsible(title=tr("详细信息"), id="preparation-diagnostics", collapsed=True):
                        yield Static(request.get("detail", ""), id="preparation-traceback", markup=False)
                        if self.event["stage"] == "runtime":
                            yield Button(tr("重新准备环境"), id="preparation-rebuild")
                else:
                    yield Static(tr(STAGE_LABELS.get(self.event["stage"], "正在确认模型能否正常运行…")), id="preparation-stage", markup=False)
                    yield Static(tr("首次使用可能需要准备模型，请稍候。"), markup=False)
                yield Static("", id="preparation-error", markup=False)
            with Horizontal(id="preparation-actions"):
                yield Button(tr("返回配置" if failed else "取消"), id="preparation-cancel")
                if review:
                    if not ready:
                        yield Button(tr("应用字段"), id="preparation-apply", variant="primary")
                    yield Button(tr("确定"), id="preparation-continue", disabled=not ready, variant="primary" if ready else "default")
                elif failed:
                    yield Button(tr("重试"), id="preparation-continue", variant="primary")

    def send(self, result):
        if self.sending or self._close_requested:
            return
        self.sending = True
        if self.respond:
            self.respond(result)
        else:
            self.dismiss(result)

    @on(Button.Pressed, "#preparation-continue")
    def proceed(self):
        request = self.event.get("request", {})
        if request.get("kind") == "review" and not request.get("resolved"):
            return
        self.send({"action": "confirm" if request.get("kind") == "review" else "retry"})

    @on(Button.Pressed, "#preparation-apply")
    def apply_answers(self):
        try:
            answers = review_answers(self, self.event["request"].get("questions", []), prefix="preparation-answer")
        except (ValueError, TypeError) as exc:
            self.query_one("#preparation-error", Static).update(str(exc))
            return
        self.send({"action": "answer", "answers": answers})

    @on(Button.Pressed, "#preparation-revise")
    def revise(self):
        try:
            overrides = review_answers(self, self.event["request"]["advanced"], prefix="preparation-advanced")
        except (ValueError, TypeError) as exc:
            self.query_one("#preparation-error", Static).update(str(exc))
            return
        self.send({"action": "revise", "overrides": overrides})

    @on(Button.Pressed, "#preparation-cancel")
    def action_cancel(self):
        self.send({"action": "cancel"})

    @on(Button.Pressed, "#preparation-rebuild")
    def rebuild(self):
        self.send({"action": "rebuild"})
