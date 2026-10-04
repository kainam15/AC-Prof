"""One modal owns model review, preparation progress, errors and cancellation."""
from __future__ import annotations

from textual import on
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Label, Static

from acprof.host.model_errors import ModelLookupError
from acprof.messages import message
from acprof.preparation_events import encode_reply
from acprof.tui.downloads import download_fields, download_summary
from acprof.tui.input import BarCursorInput as Input
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


def download_failure(request):
    return request.get("download_error") or request.get("model_error", {}).get("download_error")


def failure_text(request, stage, tr):
    if failure := download_failure(request):
        last = failure["attempts"][-1]
        return "\n".join((tr("无法获取 Hugging Face 模型"),
            tr(message("失败阶段：{0} / {1}", last["stage"], last["reason"])),
            tr(message("最终失败 host：{0}", last["host"])),
            tr("请配置自己的系统 VPN/代理后重试，或显式选择 ModelScope 模型。")))
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
    #preparation-scroll { height: 1fr; max-height: 24; }
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
        report = request.get("download_report")
        download = review and report is not None
        title = "模型确认" if review else "模型无法正常运行" if failed else "正在检测模型"
        with Vertical(id="preparation-dialog"):
            yield Label(tr("下载确认" if download else title), id="preparation-title")
            with VerticalScroll(id="preparation-scroll"):
                if review:
                    values = download_fields(report) if download else request.get("fields", {})
                    fields = "\n".join(f"✓ {tr(name)}    {tr('未知') if value is None else tr(value) if download else value}"
                                       for name, value in values.items())
                    yield Static(fields, id="preparation-fields", markup=False)
                    if download:
                        yield Static(tr("仅在准备阶段下载，正式测量离线。"), id="preparation-download-note", markup=False)
                    for index, question in enumerate(request.get("questions", [])):
                        yield Label(tr(question["path"]))
                        yield Static(tr(question.get("reason", "")), markup=False)
                        if not question.get("read_only"):
                            yield review_input(question, identifier=f"preparation-answer-{index}", translate=tr)
                    if request.get("detail"):
                        yield Static(request["detail"], id="preparation-detail", markup=False)
                    if request.get("advanced"):
                        with Collapsible(title=tr("高级修改"), collapsed=True):
                            for index, question in enumerate(request["advanced"]):
                                yield Label(question["path"])
                                yield review_input(question, identifier=f"preparation-advanced-{index}", translate=tr)
                            yield Button(tr("重新解析"), id="preparation-revise")
                    summary = download_summary(report, tr) if download else request.get("summary")
                    if summary and summary.strip():
                        with Collapsible(title=tr("高级详情"), id="preparation-advanced-details", collapsed=True):
                            yield Static(summary, markup=False)
                elif failed:
                    yield Static(failure_text(request, self.event["stage"], tr), id="preparation-detail", markup=False)
                    with Collapsible(title=tr("详细信息"), id="preparation-diagnostics", collapsed=True):
                        if failure := download_failure(request):
                            import json
                            yield Static(json.dumps(failure, ensure_ascii=False, indent=2), markup=False)
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
                    yield Button(tr("确认下载" if download else "确定"), id="preparation-continue",
                                 disabled=not ready, variant="primary" if ready else "default")
                elif failed:
                    yield Button(tr("重试"), id="preparation-continue", variant="primary")
                    yield Button(tr("查看诊断"), id="preparation-show-diagnostics")
                    if download_failure(request):
                        yield Button(tr("改用 ModelScope"), id="preparation-modelscope")

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
            encode_reply(self.event["request"]["id"], "answer", answers=answers)
        except (ValueError, TypeError) as exc:
            self.query_one("#preparation-error", Static).update(str(exc))
            return
        self.send({"action": "answer", "answers": answers})

    @on(Button.Pressed, "#preparation-revise")
    def revise(self):
        try:
            overrides = review_answers(self, self.event["request"]["advanced"], prefix="preparation-advanced")
            encode_reply(self.event["request"]["id"], "revise", overrides=overrides)
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

    @on(Button.Pressed, "#preparation-show-diagnostics")
    def show_diagnostics(self):
        self.query_one("#preparation-diagnostics", Collapsible).collapsed = False

    @on(Button.Pressed, "#preparation-modelscope")
    def switch_modelscope(self):
        self.app.push_screen(ModelSourceScreen(), lambda result: self.send(result) if result else None)


class ModelSourceScreen(ModalScreen):
    """A source change requires a new, user-supplied identity and confirmation."""
    BINDINGS = [("escape", "cancel", "取消")]
    DEFAULT_CSS = """
    ModelSourceScreen { align: center middle; }
    #source-dialog { width: 90%; max-width: 90; height: auto; max-height: 90%;
        border: round $accent; background: $surface; padding: 1; }
    #source-dialog Static, #source-dialog Label { height: auto; }
    #source-actions { height: 3; }
    """

    def compose(self):
        tr = self.app.tr
        with Vertical(id="source-dialog"):
            yield Label(tr("改用 ModelScope"))
            yield Static(tr("HF 与 ModelScope 是不同来源；同名不代表相同权重。请填写 ModelScope 模型 ID，确认后返回配置并创建新实验。"), markup=False)
            yield Input(placeholder="namespace/model", id="source-model-id")
            yield Input(placeholder=tr("revision（留空使用默认分支）"), id="source-revision")
            yield Static("", id="source-error", markup=False)
            with Horizontal(id="source-actions"):
                yield Button(tr("取消"), id="source-cancel")
                yield Button(tr("确认切换来源"), id="source-confirm", variant="primary")

    @on(Button.Pressed, "#source-confirm")
    def confirm(self):
        from huggingface_hub.utils import validate_repo_id
        model_id = self.query_one("#source-model-id", Input).value.strip()
        revision = self.query_one("#source-revision", Input).value.strip()
        try:
            validate_repo_id(model_id)
            if any(char.isspace() for char in revision):
                raise ValueError("invalid revision")
        except ValueError as exc:
            self.query_one("#source-error", Static).update(str(exc))
            return
        self.dismiss({"action": "switch-source", "model_id": model_id, "revision": revision})

    @on(Button.Pressed, "#source-cancel")
    def action_cancel(self):
        self.dismiss(None)
