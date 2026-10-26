"""Decision modal for model review/errors; passive preparation stays in Monitor."""
from __future__ import annotations

from textual import on
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Label, SelectionList, Static

from acprof.host.model_errors import ModelLookupError
from acprof.preparation_events import encode_reply
from acprof.tui.app_contracts import translated_app
from acprof.tui.downloads import download_sections
from acprof.tui.i18n import message_id
from acprof.tui.input import BarCursorInput as Input
from acprof.tui.log import SelectableLog
from acprof.tui.rendering import CjkCompositor
from acprof.tui.review_inputs import review_answers, review_input

STAGE_LABELS = {
    "resolution": "正在读取模型信息…", "dependencies": "正在读取模型信息…",
    "interface": "正在确认模型接口…", "preflight": "正在准备运行环境…",
    "image": "正在准备运行环境…", "environment": "正在准备运行环境…",
    "model": "正在准备模型…", "input": "正在准备验证输入…", "runtime": "正在验证模型运行…",
}


def download_failure(request):
    return request.get("download_error") or request.get("model_error", {}).get("download_error")


def failure_text(request, stage, tr):
    if failure := download_failure(request):
        last = failure["attempts"][-1]
        return "\n".join((tr("无法获取 Hugging Face 模型"),
            tr(message_id('ui.0009', last["stage"], last["reason"])),
            tr(message_id('ui.0010', last["host"])),
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
    BINDINGS = [("escape", "cancel", "终止任务")]
    DEFAULT_CSS = """
    PreparationScreen { align: center middle; }
    /* Native tooltips anchor to the mouse; clear the full three-row button. */
    PreparationScreen > Tooltip { offset-y: -100%; margin: -3 0 0 0; }
    #preparation-dialog { width: 90%; max-width: 100; height: auto; max-height: 90%;
        border: round $accent; background: $surface; padding: 0 1; }
    #preparation-dialog.download-confirmation, #preparation-dialog.preparation-failure { height: 90%; }
    .download-confirmation #preparation-scroll, .preparation-failure #preparation-scroll { max-height: 100%; }
    #preparation-title { height: auto; text-style: bold; }
    #preparation-scroll { height: 1fr; max-height: 24; }
    #preparation-scroll Static, #preparation-scroll Label { height: auto; }
    #preparation-scroll .download-section {
        border-top: solid $panel; border-title-color: $accent; border-title-style: bold;
        padding: 0 1; margin: 1 0 0 0;
    }
    #preparation-download-note { color: $text-muted; margin-top: 1; }
    #preparation-detail, #preparation-error { color: $error; }
    #preparation-diagnostics { height: 1fr; padding: 0; }
    #preparation-diagnostics.-collapsed { height: auto; }
    #preparation-diagnostics > Contents { height: 1fr; padding: 0; }
    #preparation-traceback { height: 1fr; min-height: 3; border: round $panel;
        padding: 0 1; background: transparent; background-tint: transparent; }
    #preparation-traceback:focus { border: round $accent; }
    #preparation-traceback .text-area--selection { background: $accent; color: $background; }
    #preparation-actions { height: 3; }
    #preparation-actions Button { min-width: 10; width: 1fr; }
    #preparation-scroll SelectionList { height: auto; max-height: 10; min-height: 4; }
    .preparation-failure #preparation-actions Button { width: auto; }
    .preparation-failure #preparation-actions #preparation-continue { width: 1fr; }
    #preparation-actions.extra-action { height: 6; layout: grid; grid-size: 2; grid-rows: 3; }
    .preparation-failure #preparation-actions.extra-action Button { width: 1fr; }
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
        tr = translated_app(self.app).tr
        request = self.event.get("request", {})
        review = request.get("kind") == "review"
        failed = request.get("kind") == "error"
        ready = request.get("resolved") is True
        report = request.get("download_report")
        download = review and report is not None
        task_selection = any(item.get("kind") == "multi" for item in request.get("questions", []))
        parameter_review = request.get("purpose") == "workload_parameters"
        title = "模型参数" if parameter_review else "模型确认" if review else "模型无法正常运行" if failed else "正在检测模型"
        with Vertical(id="preparation-dialog", classes=(
            "download-confirmation" if download else "preparation-failure" if failed else ""
        )):
            yield Label(tr("选择采集任务" if task_selection else "下载确认" if download else title), id="preparation-title")
            with VerticalScroll(id="preparation-scroll"):
                if review:
                    values = request.get("fields", {})
                    if download:
                        for section in download_sections(report):
                            widget = Static("\n".join(tr(line) for line in section.lines),
                                            id=f"download-{section.key}", classes="download-section", markup=False)
                            widget.border_title = tr(section.title)
                            yield widget
                        yield Static(tr("仅在准备阶段下载，正式测量离线。"), id="preparation-download-note", markup=False)
                    else:
                        fields = "\n".join(
                            f"✓ {tr(name)}    {tr('未知') if value is None else value}" for name, value in values.items())
                        yield Static(fields, id="preparation-fields", markup=False)
                    for index, question in enumerate(request.get("questions", [])):
                        name = (f"{tr(question['task_name'])} · {tr(question['caption'])}"
                                if question.get("task_name") else tr(question.get("label", question["path"])))
                        yield Label(tr("采集任务") if question.get("kind") == "multi" else name)
                        if question.get("reason"):
                            yield Static(tr(question["reason"]), markup=False)
                        if not question.get("read_only"):
                            yield review_input(question, identifier=f"preparation-answer-{index}", translate=tr)
                            if question.get("kind") == "multi":
                                yield Button(tr("全选任务"), id="preparation-select-all")
                    if parameter_review and request.get("parameter_advanced"):
                        with Collapsible(title=tr("更多生成参数"), collapsed=True):
                            for index, question in enumerate(request["parameter_advanced"]):
                                name = f"{tr(question['task_name'])} · {tr(question['caption'])}"
                                yield Label(name)
                                if question.get("reason"):
                                    yield Static(tr(question["reason"]), markup=False)
                                yield review_input(question, identifier=f"preparation-parameter-{index}", translate=tr)
                    if request.get("detail"):
                        yield Static(request["detail"], id="preparation-detail", markup=False)
                    if request.get("advanced"):
                        with Collapsible(title=tr("高级修改"), collapsed=True):
                            for index, question in enumerate(request["advanced"]):
                                yield Label(question["path"])
                                yield review_input(question, identifier=f"preparation-advanced-{index}", translate=tr)
                            yield Button(tr("重新解析"), id="preparation-revise")
                    summary = None if download else request.get("summary")
                    if summary and summary.strip():
                        if task_selection:
                            with Collapsible(title=tr("任务声明来源"), collapsed=True):
                                yield Static(summary, id="preparation-summary", markup=False)
                        else:
                            yield Static(summary, id="preparation-summary", markup=False)
                elif failed:
                    yield Static(failure_text(request, self.event["stage"], tr), id="preparation-detail", markup=False)
                    with Collapsible(title=tr("详细信息"), id="preparation-diagnostics", collapsed=False):
                        detail = request.get("detail", "")
                        if failure := download_failure(request):
                            import json
                            detail = json.dumps(failure, ensure_ascii=False, indent=2) + "\n\n" + detail
                        log = SelectableLog(id="preparation-traceback", max_lines=max(1, len(detail.splitlines())))
                        log.write(detail)
                        yield log
                else:
                    yield Static(tr(STAGE_LABELS.get(self.event["stage"], "正在确认模型能否正常运行…")), id="preparation-stage", markup=False)
                    yield Static(tr("首次使用可能需要准备模型，请稍候。"), markup=False)
                yield Static("", id="preparation-error", markup=False)
            with Horizontal(id="preparation-actions", classes=(
                "extra-action" if failed and download_failure(request) and self.event["stage"] in {"image", "runtime"} else ""
            )):
                yield Button(tr("返回配置" if failed or task_selection or parameter_review else "终止任务"), id="preparation-cancel",
                             tooltip=tr(message_id("ui.2157")) if failed else None)
                if review:
                    if not ready:
                        if parameter_review:
                            yield Button(tr("使用推荐参数"), id="preparation-defaults")
                        yield Button(tr("应用参数" if parameter_review else "开始所选任务" if task_selection else "应用字段"),
                                     id="preparation-apply", variant="primary")
                    if not task_selection and not parameter_review:
                        yield Button(tr("确认下载" if download else "确定"), id="preparation-continue",
                                     disabled=not ready, variant="primary" if ready else "default")
                elif failed:
                    yield Button(tr("重试"), id="preparation-continue", variant="primary",
                                 tooltip=tr(message_id("ui.2158")))
                    if self.event["stage"] in {"image", "runtime"}:
                        yield Button(tr("重新准备环境"), id="preparation-rebuild",
                                     tooltip=tr(message_id("ui.2159")))
                    if download_failure(request):
                        yield Button(tr("改用 ModelScope"), id="preparation-modelscope",
                                     tooltip=tr(message_id("ui.2160")))

    @on(SelectableLog.Updated)
    def details_updated(self, event: SelectableLog.Updated) -> None:
        # This static failure log does not update the Monitor's log counters.
        event.stop()

    @on(Button.Pressed, "#preparation-select-all")
    def select_all_tasks(self) -> None:
        for selection in self.query(SelectionList):
            selection.select_all()

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

    @on(Button.Pressed, "#preparation-defaults")
    def use_default_parameters(self):
        request = self.event["request"]
        if request.get("purpose") != "workload_parameters":
            return
        fields = request.get("questions", []) + request.get("parameter_advanced", [])
        answers = {field["path"]: field["value"] for field in fields}
        encode_reply(request["id"], "answer", answers=answers)
        self.send({"action": "answer", "answers": answers})

    @on(Button.Pressed, "#preparation-apply")
    def apply_answers(self):
        try:
            request = self.event["request"]
            answers = review_answers(self, request.get("questions", []), prefix="preparation-answer")
            if request.get("purpose") == "workload_parameters":
                answers.update(review_answers(
                    self, request.get("parameter_advanced", []), prefix="preparation-parameter",
                ))
            encode_reply(request["id"], "answer", answers=answers)
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
        request = self.event.get("request", {})
        if (request.get("purpose") == "workload_parameters"
                or any(item.get("kind") == "multi" for item in request.get("questions", []))):
            self.send({"action": "cancel"})
            return
        # Requesting stop opens the app-level confirmation first. Do not mark
        # this decision screen as sent: rejecting the confirmation must leave
        # the pending request fully interactive.
        if self.respond:
            self.respond({"action": "request-stop"})
        else:
            self.dismiss({"action": "request-stop"})

    @on(Button.Pressed, "#preparation-rebuild")
    def rebuild(self):
        self.send({"action": "rebuild"})

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
        tr = translated_app(self.app).tr
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
