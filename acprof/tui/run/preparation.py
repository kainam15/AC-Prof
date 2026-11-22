"""Decision modal for model review/errors; passive preparation stays in Monitor."""
from __future__ import annotations

from rich.text import Text
from textual import on
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Collapsible, Label, Select, Static

from acprof.events.preparation import encode_reply
from acprof.host.preparation.model_errors import ModelLookupError
from acprof.messages import join_messages, message
from acprof.tui.i18n import message_id
from acprof.tui.logging.widget import SelectableLog
from acprof.tui.models.inference_inputs import ParameterRequestInput, update_parameter_dependencies
from acprof.tui.models.review_inputs import review_answers, review_input
from acprof.tui.models.task_choice_view import task_context_view
from acprof.tui.models.task_choices import TaskChoices
from acprof.tui.run.downloads import download_fact_sections
from acprof.tui.ui.contracts import translated_app
from acprof.tui.ui.dialogs import focus_horizontal_action
from acprof.tui.ui.input import BarCursorInput as Input
from acprof.tui.ui.rendering import CjkCompositor
from acprof.tui.ui.structured import Callout, FactDetailScreen, FactGrid, StatusBanner
from acprof.tui.ui.view_models import (
    CalloutContent,
    FactRow,
    StatusSummary,
    ViewSection,
    reason_presentation,
)

STAGE_LABELS = {
    "resolution": "正在读取模型信息…", "dependencies": "正在读取模型信息…",
    "interface": "正在确认模型接口…", "preflight": "正在准备运行环境…",
    "image": "正在准备运行环境…", "environment": "正在准备运行环境…",
    "model": "正在准备模型…", "input": "正在准备验证输入…", "runtime": "正在验证模型运行…",
}


def download_failure(request):
    return request.get("download_error") or request.get("model_error", {}).get("download_error")


_DOCKER_ERROR_CODES = {
    "docker_content_incomplete", "docker_auth_failed", "docker_network_failed", "docker_disk_full",
    "docker_repair_unverified", "docker_repair_forbidden", "docker_frontend_failed",
    "docker_integrity_failed", "docker_tls_failed",
    "docker_manifest_missing", "docker_platform_mismatch", "docker_download_budget",
}
_FAILURE_STAGE_NAMES = {
    "load": "模型加载", "preprocess": "输入处理", "predict": "模型推理", "completion": "模型推理",
    "postprocess": "输出处理", "validate_output": "输出验证", "interface": "模型接口",
    "resolution": "模型信息", "runtime": "模型运行", "image": "运行环境", "environment": "运行环境",
    "model": "模型准备", "input": "验证输入", "preflight": "运行环境",
}
_FAILURE_STAGE_REASONS = {
    "postprocess": ("模型已完成推理，但输出处理失败。", "请检查任务选择或模型声明中的输出处理配置。"),
    "validate_output": ("模型输出与任务要求不匹配。", "请检查任务选择或模型声明中的输出处理配置。"),
    "interface": ("模型接口无法在当前环境中使用。", "请检查模型声明、源码文件和运行依赖。"),
    "load": ("模型未能完成加载。", "请检查模型文件、访问权限和可用内存。"),
}


_FAILURE_CATEGORIES = {
        "backend_unsupported": "architecture", "runtime_task_unsupported": "architecture",
        "runtime_dependency_missing": "dependency", "runtime_dependency_incompatible": "dependency",
        "artifact_incomplete": "artifact", "artifact_layout_unsupported": "artifact",
        "runtime_initialization_failed": "load", "resource_limit": "resource",
        "inference_failed": "inference",
    }


def review_fields_section(values: dict[str, object], *, verified: bool = True) -> ViewSection:
    rows = tuple(
        FactRow(
            str(name),
            message_id("common.value.unknown") if value is None else str(value),
            status=("ok" if verified else None) if value is not None else "unknown",
            copyable=isinstance(value, str) and len(value) > 24,
        )
        for name, value in values.items()
    )
    return ViewSection("", rows)


def failure_text(request, stage, tr):
    """Plain-text projection of the same typed failure view used by the dialog."""
    summary, callout = failure_views(request, stage)
    return "\n".join(tr(value) for value in (
        summary.headline, callout.problem, callout.impact, callout.next_step,
    ) if value)


def _docker_failure_views(error: dict) -> tuple[StatusSummary, CalloutContent]:
    code = error["reason_code"]
    headline = message_id("run.docker.error." + code) if code in _DOCKER_ERROR_CODES else message_id("run.docker.error.generic")
    problems: list[str] = [message_id("run.docker.failed_step", error.get("step") or error.get("image", ""))]
    if dependency_failure := error.get("dependency_failure"):
        problems.insert(0, dependency_failure)
    content = error.get("content")
    if content:
        status_text: str = message_id("run.docker.content." + content["status"])
        if content.get("present") is not None:
            status_text = message("{0} ({1}/{2})", status_text, content["present"], content.get("required") or "?")
        problems.append(status_text)
    if error.get("exit_code") is not None:
        problems.append(message_id("run.docker.exit_code", error["exit_code"]))

    impacts: list[str] = []
    if code == "docker_frontend_failed":
        from acprof.tui.run.downloads import probe_reason
        impacts.append(message_id("run.docker.frontend_source", error.get("registry") or "?",
                                  probe_reason(error.get("network_reason"))))
    if content:
        from acprof.host.images.content import version_needs_upgrade
        if version_needs_upgrade(content.get("engine_version", "")):
            impacts.append(message_id("run.docker.upgrade", content["engine_version"]))
    if fallback := error.get("fallback_error"):
        impacts.append(message_id("run.docker.error." + fallback["reason_code"]))
    if not impacts:
        impacts.append(message_id("run.docker.preparation_failed"))

    next_steps = [message_id("run.docker.frontend_hint" if code == "docker_frontend_failed" else "run.docker.recovery_hint")]
    if error.get("log_path"):
        next_steps.append(message_id("run.docker.log_path", error["log_path"]))
    return StatusSummary(level="error", headline=headline), CalloutContent(
        level="error",
        problem=join_messages(" · ", problems),
        impact=join_messages("\n", impacts),
        next_step=join_messages("\n", next_steps),
    )


def failure_views(request: dict, stage: str) -> tuple[StatusSummary, CalloutContent]:
    """Return structured banner summary and problem/impact/next callout for a failed preparation event."""
    if error := request.get("docker_error"):
        return _docker_failure_views(error)
    if failure := download_failure(request):
        last = failure["attempts"][-1]
        return StatusSummary(level="error", headline="无法获取 Hugging Face 模型"), CalloutContent(
            level="error",
            problem=message_id("downloads.failure.stage_detail", last["stage"], last["reason"]),
            impact=message_id("downloads.failure.host", last["host"]),
            next_step="请配置自己的系统 VPN/代理后重试，或显式选择 ModelScope 模型。",
        )
    if request.get("model_error"):
        failure = ModelLookupError(**request["model_error"])
        return StatusSummary(level="error", headline=failure.summary), CalloutContent(
            level="error",
            problem=failure.summary,
            impact=message_id("run.preparation.model_error_impact"),
            next_step=failure.hint,
        )
    failed = request.get("failed_stage", stage)
    reason, hint = _FAILURE_STAGE_REASONS.get(
        failed, (message_id("run.preparation.stage_failed_problem"), message_id("run.preparation.stage_failed_next")),
    )
    failure = request.get("failure", {})
    code = failure.get("reason_code")
    summary = StatusSummary(level="error", headline=message("{0}：{1}", "失败阶段", _FAILURE_STAGE_NAMES.get(failed, "模型运行")))
    if code in {"metadata_invalid", "metadata_unavailable"} or (
        code in {"access_denied", "artifact_incomplete"} and failure.get("evidence", {}).get("diagnostics")
    ):
        return summary, reason_presentation(code, failure.get("detail", ""))
    if code in _FAILURE_CATEGORIES:
        category = _FAILURE_CATEGORIES[code]
        problem: list[str] = [message_id("run.failure.type", failure.get("exception_type") or code),
                   message_id("run.failure." + category)]
        detail = str(failure.get("detail") or "").splitlines()
        if detail:
            problem.append(message(detail[0][:240] + ("…" if len(detail[0]) > 240 else "")))
        hints = [message_id("run.failure.hint." + category)]
        if path := failure.get("evidence", {}).get("log_path"):
            hints.append(message_id("run.docker.log_path", path))
        if error := failure.get("evidence", {}).get("log_capture_error"):
            hints.append(message_id("run.failure.log_incomplete", error))
        reason, hint = join_messages("\n", problem), join_messages("\n", hints)
    return summary, CalloutContent(
        level="error",
        problem=reason,
        impact=message_id("run.preparation.not_measuring_impact"),
        next_step=hint,
    )


class PreparationScreen(ModalScreen):
    BINDINGS = [("escape", "cancel", "终止任务")]
    DEFAULT_CSS = """
    PreparationScreen { align: center middle; }
    /* Native tooltips anchor to the mouse; clear the full three-row button. */
    PreparationScreen > Tooltip { offset-y: -100%; margin: -3 0 0 0; }
    #preparation-dialog { width: 90%; max-width: 100; height: auto; max-height: 90%;
        border: round $accent; background: $surface; padding: 0 1; }
    #preparation-dialog.download-confirmation, #preparation-dialog.preparation-failure,
    #preparation-dialog.task-selection, #preparation-dialog.inference-configuration { height: 90%; }
    .download-confirmation #preparation-scroll, .preparation-failure #preparation-scroll,
    .task-selection #preparation-scroll, .inference-configuration #preparation-scroll { max-height: 100%; }
    #preparation-title { height: auto; text-style: bold; }
    #preparation-steps { height: 1; color: $text-muted; }
    #preparation-scroll { height: 1fr; max-height: 24; }
    #preparation-scroll Static, #preparation-scroll Label { height: auto; }
    #preparation-scroll .download-section, #preparation-fields { height: auto; }
    #preparation-scroll .download-section {
        border-top: solid $panel; border-title-color: $accent; border-title-style: bold;
        padding: 0 1; margin: 1 0 0 0;
    }
    #preparation-scroll .download-section .fact-grid, #preparation-fields .fact-grid { grid-columns: 20 1fr; }
    #preparation-download-note { color: $text-muted; margin-top: 1; }
    #preparation-banner { margin: 0; padding: 0 1; }
    #preparation-callout { margin: 0; padding: 0 1; }
    #preparation-error { color: $error; }
    #preparation-error { height: auto; display: none; }
    #preparation-diagnostics { height: 1fr; padding: 0; }
    #preparation-diagnostics.-collapsed { height: auto; }
    #preparation-diagnostics > Contents { height: 1fr; padding: 0; }
    #preparation-traceback { height: 1fr; min-height: 5; border: round $panel;
        padding: 0 1; background: transparent; background-tint: transparent; }
    #preparation-traceback:focus { border: round $accent; }
    #preparation-traceback .text-area--selection { background: $accent; color: $background; }
    #preparation-actions { height: 3; }
    #inference-preset-pills { height: auto; margin-bottom: 1; }
    #inference-preset-pills Button {
        height: 1; min-height: 1; min-width: 0; width: auto;
        border: none; padding: 0 1; margin-right: 1;
        background: $boost; color: $foreground;
    }
    #inference-preset-pills Button.-active { background: $accent; color: $background; text-style: bold; }
    #inference-preset-pills Button:hover, #inference-preset-pills Button:focus {
        background: $accent; color: $background; text-style: bold underline;
    }
    #inference-preset-row { height: 3; }
    #inference-preset { width: 1fr; }
    #preparation-preset { width: auto; }
    #preparation-actions Button { min-width: 10; width: 1fr; }
    .task-selection #preparation-actions Button,
    .inference-configuration #preparation-actions Button { min-width: 0; padding: 0; }
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

    @property
    def model_blocked(self) -> bool:
        request = self.event.get("request", {})
        return (self.event["stage"] == "resolution" and request.get("kind") == "error"
                and request.get("failure", {}).get("reason_code") in {"artifact_layout_unsupported", "backend_unsupported"})

    def on_mount(self) -> None:
        if self.model_blocked:
            self.query_one("#preparation-cancel", Button).focus(scroll_visible=False)
        choices = list(self.query("TaskChoices Checkbox").results(Checkbox))
        if choices:
            choices[0].focus(scroll_visible=False)
        self.call_after_refresh(self.refresh_inference_dependencies)

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
        docker_error = request.get("docker_error")
        ready = request.get("resolved") is True
        report = request.get("download_report")
        download = review and report is not None
        task_selection = any(item.get("kind") == "multi" for item in request.get("questions", []))
        parameter_review = request.get("purpose") == "workload_parameters"
        inference_review = request.get("purpose") == "inference_configuration"
        inference_confirmation = request.get("purpose") == "inference_confirmation"
        title = (message_id("models.inference.title") if inference_review else
                 message_id("models.inference.confirmation_title") if inference_confirmation else
                 message_id("models.parameters.title") if parameter_review else
                 "模型确认" if review else "模型无法正常运行" if failed else "正在检测模型")
        if self.model_blocked:
            title = message_id("models.failure.backend_load_blocked")
        elif failed and (docker_error or request.get("failed_stage", self.event["stage"]) in {"image", "environment", "preflight"}):
            title = message_id("run.docker.preparation_failed")
        with Vertical(id="preparation-dialog", classes=(
            "inference-configuration" if inference_review or inference_confirmation else
            "task-selection" if task_selection else "download-confirmation" if download else "preparation-failure" if failed else ""
        )):
            yield Label(tr("选择采集任务" if task_selection else "下载确认" if download else title), id="preparation-title")
            yield Static(
                self._step_indicator(
                    tr,
                    download=download,
                    task_selection=task_selection,
                    inference_step=inference_review or inference_confirmation or parameter_review,
                    failed=failed,
                ),
                id="preparation-steps",
                markup=False,
            )
            with VerticalScroll(id="preparation-scroll"):
                if review:
                    values = request.get("fields", {})
                    if download:
                        for section in download_fact_sections(report):
                            widget = FactGrid(
                                section.section,
                                collapsible=False,
                                id=f"download-{section.key}",
                                classes="download-section",
                            )
                            widget.border_title = tr(section.section.title)
                            yield widget
                        yield Static(tr("仅在准备阶段下载，正式测量离线。"), id="preparation-download-note", markup=False)
                    elif not task_selection:
                        yield FactGrid(
                            review_fields_section(values, verified=not (inference_review or inference_confirmation)),
                            id="preparation-fields", collapsible=False,
                        )
                    if inference_review:
                        yield Static(tr(message_id("models.inference.auto_hint")), markup=False)
                        if presets := request.get("presets"):
                            labels = {
                                "auto": message_id("models.inference.preset_auto"),
                                "benchmark": message_id("models.inference.preset_benchmark"),
                                "author": message_id("models.inference.preset_author"),
                                "custom": message_id("models.inference.preset_custom"),
                            }
                            selected = request.get("selected_preset", "custom")
                            if selected not in {item["id"] for item in presets}:
                                selected = presets[0]["id"]
                            yield Label(tr(message_id("models.inference.preset")))
                            quick_presets = [item for item in presets if item["id"] != "custom"]
                            if quick_presets:
                                with Horizontal(id="inference-preset-pills"):
                                    for item in quick_presets:
                                        pill_id = item["id"]
                                        yield Button(
                                            tr(labels.get(pill_id, item["label"])),
                                            id=f"preparation-quick-preset-{pill_id}",
                                            name=pill_id,
                                            classes="preset-pill" + (" -active" if pill_id == selected else ""),
                                            compact=True,
                                        )
                            with Horizontal(id="inference-preset-row"):
                                yield Select([(tr(labels.get(item["id"], item["label"])), item["id"])
                                              for item in presets], value=selected, allow_blank=False,
                                             id="inference-preset")
                                yield Button(tr(message_id("models.inference.preset_apply")),
                                             id="preparation-preset")
                        yield from self.inference_questions(request.get("questions", []),
                                                            prefix="preparation-answer", translate=tr)
                    else:
                        for index, question in enumerate(request.get("questions", [])):
                            if question.get("kind") != "multi":
                                name = (f"{tr(question['task_name'])} · {tr(question['caption'])}"
                                        if parameter_review else tr(question["path"]))
                                yield Label(name)
                                if question.get("reason"):
                                    yield Static(tr(question["reason"]), markup=False)
                            if not question.get("read_only"):
                                yield review_input(question, identifier=f"preparation-answer-{index}", translate=tr)
                    if task_selection and values:
                        yield FactGrid(task_context_view(values), id="preparation-fields", classes="task-context")
                    if parameter_review and request.get("parameter_advanced"):
                        with Collapsible(title=tr(message_id("models.parameters.more")), collapsed=True):
                            for index, question in enumerate(request["parameter_advanced"]):
                                name = f"{tr(question['task_name'])} · {tr(question['caption'])}"
                                yield Label(name)
                                if question.get("reason"):
                                    yield Static(tr(question["reason"]), markup=False)
                                yield review_input(question, identifier=f"preparation-parameter-{index}", translate=tr)
                    if inference_review and request.get("parameter_advanced"):
                        with Collapsible(title=tr(message_id("models.inference.more")), collapsed=True):
                            yield from self.inference_questions(request["parameter_advanced"],
                                                                prefix="preparation-parameter", translate=tr)
                    if inference_confirmation:
                        yield Static(tr(message_id("models.inference.confirmation_hint")), markup=False)
                    if request.get("detail"):
                        yield Static(request["detail"], id="preparation-detail", markup=False)
                    summary = None if download else request.get("summary")
                    if summary and summary.strip():
                        yield Static(tr(summary), id="preparation-summary", markup=False)
                elif self.model_blocked:
                    failure = request["failure"]
                    evidence = failure.get("evidence", {})
                    fact_labels = (("model_id", "Model"), ("model_revision", "Revision"),
                                   ("declared_task", message_id("models.failure.declared_task")),
                                   ("library_name", message_id("models.failure.library")))
                    rows = tuple(
                        FactRow(label, value, status="ok" if key in {"model_id", "model_revision"} else None,
                                copyable=len(value) > 24)
                        for key, label in fact_labels if (value := evidence.get(key)) and value != "unknown"
                    )
                    if rows:
                        yield FactGrid(ViewSection("", rows), id="preparation-fields", collapsible=False)
                    problem = reason_presentation(failure["reason_code"]).problem
                    if fmt := evidence.get("artifact_format"):
                        problem = message_id("models.failure.incompatible_files", "Core ML" if fmt == "CoreML" else fmt)
                    yield Static(tr(problem), id="preparation-problem", markup=False)
                elif failed:
                    banner, callout = failure_views(request, self.event["stage"])
                    yield StatusBanner(banner, id="preparation-banner", include_actions=False)
                    yield Callout(callout, id="preparation-callout", include_details=False, include_actions=False)
                    with Collapsible(title=tr("详细信息"), id="preparation-diagnostics", collapsed=bool(docker_error or request.get("failure"))):
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
                "extra-action" if failed and (download_failure(request) or docker_error and "repair" in docker_error.get("actions", []))
                and self.event["stage"] in {"image", "environment", "runtime"} else ""
            )):
                if self.model_blocked:
                    yield Button(tr(message_id("models.action.back_to_selection")), id="preparation-cancel", variant="primary")
                    yield Button(tr(message_id("models.action.view_diagnostics")), id="preparation-view-diagnostics")
                    return
                cancel_label = (tr(message_id("common.action.cancel")) if task_selection else
                                tr("返回配置" if failed or parameter_review or inference_review
                                   or inference_confirmation else "终止任务"))
                yield Button(cancel_label, id="preparation-cancel",
                             tooltip=tr(message_id("run.preparation.stop_hint")) if failed else None)
                if task_selection:
                    yield Button(tr(message_id("models.tasks.select_all")), id="preparation-select-all")
                if review:
                    if not ready:
                        if parameter_review:
                            yield Button(tr(message_id("models.parameters.use_recommended")), id="preparation-defaults")
                        yield Button(tr(message_id("models.inference.apply") if inference_review else
                                        message_id("models.parameters.apply") if parameter_review else
                                        message_id("models.tasks.run_selected") if task_selection else "应用字段"),
                                     id="preparation-apply", variant="primary")
                    if inference_confirmation:
                        yield Button(tr(message_id("models.inference.edit")), id="preparation-edit")
                    if not task_selection and not parameter_review and not inference_review:
                        yield Button(tr(message_id("models.inference.confirm") if inference_confirmation else
                                        "确认下载" if download else "确定"), id="preparation-continue",
                                     disabled=not ready, variant="primary" if ready else "default")
                elif failed:
                    retry_label = ("run.docker.frontend_retry" if docker_error and docker_error.get("reason_code") == "docker_frontend_failed"
                                   else "run.environment.retry")
                    yield Button(tr(message_id(retry_label)) if docker_error else tr("重试"), id="preparation-continue", variant="primary",
                                 tooltip=tr(message_id("run.preparation.retry_hint")))
                    if docker_error and "repair" in docker_error.get("actions", []):
                        yield Button(tr(message_id("run.docker.repair")), id="preparation-repair",
                                     tooltip=tr(message_id("run.docker.repair_hint")))
                    if (self.event["stage"] in {"image", "environment", "runtime"}
                            and (not docker_error or "rebuild" in docker_error.get("actions", ["rebuild"]))):
                        yield Button(tr("重新准备环境"), id="preparation-rebuild",
                                     tooltip=tr(message_id("run.preparation.rebuild_hint")))
                    if download_failure(request):
                        yield Button(tr("改用 ModelScope"), id="preparation-modelscope",
                                     tooltip=tr(message_id("models.source.switch_modelscope_hint")))

    def inference_questions(self, questions: list[dict], *, prefix: str, translate):
        import json

        previous_group = None
        for index, question in enumerate(questions):
            scope = question.get("scope", question.get("parameter_spec", {}).get("scope", "inference"))
            group = (scope, question.get("device"), question.get("task_id"))
            if group != previous_group:
                title = translate({
                    "backend": message_id("models.inference.scope_backend"),
                    "inference": message_id("models.inference.scope_inference"),
                    "loading": message_id("models.inference.scope_loading"),
                    "measurement": message_id("models.inference.scope_measurement"),
                    "model": message_id("models.inference.scope_model"),
                    "workload": message_id("models.inference.scope_workload"),
                }.get(scope, message_id("models.inference.scope_inference")))
                suffix = question.get("device") or question.get("task_id")
                yield Label(f"{title} · {suffix}" if suffix else title, classes="section-title")
                previous_group = group
            spec = question.get("parameter_spec", {})
            yield Label(translate(question.get("caption") or spec.get("label") or question["path"]))
            for key in ("reason", "source", "auto_reason", "validation"):
                if key == "auto_reason" and question.get("request", {}).get("mode") != "auto":
                    continue
                value = question.get(key)
                if value:
                    yield Static(translate(value) if isinstance(value, str) else
                                 json.dumps(value, ensure_ascii=False, sort_keys=True), markup=False)
            if question.get("read_only") or spec.get("read_only"):
                yield Static(json.dumps(question.get("resolved_value", question.get("value")),
                                        ensure_ascii=False, sort_keys=True), markup=False)
            else:
                yield review_input(question, identifier=f"{prefix}-{index}", translate=translate)

    @on(Select.Changed)
    @on(Input.Changed)
    def inference_value_changed(self) -> None:
        self.refresh_inference_dependencies()

    def refresh_inference_dependencies(self) -> None:
        if self.is_mounted and self.event.get("request", {}).get("purpose") == "inference_configuration":
            update_parameter_dependencies(list(self.query(ParameterRequestInput)))

    @staticmethod
    def _step_indicator(tr, *, download: bool, task_selection: bool, inference_step: bool, failed: bool) -> Text:
        steps = (
            tr(message_id("models.preparation.step_detect")),
            tr(message_id("models.preparation.step_tasks")),
            tr(message_id("models.preparation.step_inference")),
            tr(message_id("models.preparation.step_download")),
        )
        current = 3 if download else 2 if inference_step else 1 if task_selection else 0
        line = Text()
        for index, label in enumerate(steps):
            if index:
                line.append("  →  ", style="dim")
            if index < current:
                line.append(f"✓ {label}", style="dim")
            elif index == current:
                line.append(label, style="bold red" if failed else "bold")
            else:
                line.append(label, style="dim")
        return line

    @on(Button.Pressed, "#preparation-preset")
    def apply_inference_preset(self) -> None:
        request = self.event.get("request", {})
        if request.get("purpose") != "inference_configuration":
            return
        selected = self.query_one("#inference-preset", Select).value
        if selected in {item["id"] for item in request.get("presets", [])}:
            self.send({"action": "answer", "answers": {"$preset": selected}})

    @on(Button.Pressed, ".preset-pill")
    def apply_quick_preset(self, event: Button.Pressed) -> None:
        request = self.event.get("request", {})
        if request.get("purpose") != "inference_configuration":
            return
        preset_id = event.button.name
        if preset_id in {item["id"] for item in request.get("presets", [])}:
            self.send({"action": "answer", "answers": {"$preset": preset_id}})

    @on(Button.Pressed, "#preparation-edit")
    def edit_configuration(self) -> None:
        if self.event.get("request", {}).get("purpose") == "inference_confirmation":
            self.send({"action": "edit"})

    @on(SelectableLog.Updated)
    def details_updated(self, event: SelectableLog.Updated) -> None:
        # This static failure log does not update the Monitor's log counters.
        event.stop()

    @on(Button.Pressed, "#preparation-select-all")
    def select_all_tasks(self) -> None:
        for selection in self.query(TaskChoices):
            selection.select_all()
        self.query_one("#preparation-error").display = False

    @on(Checkbox.Changed, "TaskChoices Checkbox")
    def clear_choice_error(self) -> None:
        self.query_one("#preparation-error").display = False

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
        reply = {"action": "confirm" if request.get("kind") == "review" else "retry"}
        if request.get("purpose") == "inference_confirmation":
            reply["confirmation_digest"] = request["confirmation_digest"]
            encode_reply(request["id"], reply["action"], confirmation_digest=reply["confirmation_digest"])
        self.send(reply)

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
            if request.get("purpose") in {"workload_parameters", "inference_configuration"}:
                answers.update(review_answers(
                    self, request.get("parameter_advanced", []), prefix="preparation-parameter",
                ))
            encode_reply(request["id"], "answer", answers=answers)
        except (ValueError, TypeError) as exc:
            error = self.query_one("#preparation-error", Static)
            error.update(str(exc))
            error.display = True
            return
        self.send({"action": "answer", "answers": answers})

    @on(Button.Pressed, "#preparation-view-diagnostics")
    def view_model_diagnostics(self) -> None:
        request = self.event["request"]
        failure = request["failure"]
        evidence = failure.get("evidence", {})
        paths = "\n".join(evidence.get("artifact_paths", ()))
        sections = (failure["reason_code"], failure.get("detail"), paths, evidence.get("next_action"),
                    request.get("detail"))
        row = FactRow(message_id("common.diagnostics.title"), "\n\n".join(filter(None, sections)), copyable=True)
        self.app.push_screen(FactDetailScreen(row))

    @on(Button.Pressed, "#preparation-cancel")
    def action_cancel(self):
        if self.model_blocked:
            self.send({"action": "select-model"})
            return
        request = self.event.get("request", {})
        if (request.get("purpose") in {"workload_parameters", "inference_configuration", "inference_confirmation"}
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

    @on(Button.Pressed, "#preparation-repair")
    def repair(self):
        if "repair" in self.event.get("request", {}).get("docker_error", {}).get("actions", []):
            self.send({"action": "repair"})

    @on(Button.Pressed, "#preparation-modelscope")
    def switch_modelscope(self):
        self.app.push_screen(ModelSourceScreen(), lambda result: self.send(result) if result else None)

    def on_key(self, event) -> None:
        if focus_horizontal_action(self, "#preparation-actions", event.key):
            event.prevent_default()
            event.stop()


class ModelSourceScreen(ModalScreen):
    """A source change requires a new, user-supplied identity and confirmation."""
    BINDINGS = [("escape", "cancel", "取消")]
    DEFAULT_CSS = """
    ModelSourceScreen { align: center middle; }
    #source-dialog { width: 90%; max-width: 90; height: auto; max-height: 90%;
        border: round $accent; background: $surface; padding: 1; }
    #source-dialog Static, #source-dialog Label { height: auto; }
    #source-actions { height: 3; align-horizontal: right; }
    #source-actions Button { margin-left: 1; }
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

    def on_key(self, event) -> None:
        if focus_horizontal_action(self, "#source-actions", event.key):
            event.prevent_default()
            event.stop()
