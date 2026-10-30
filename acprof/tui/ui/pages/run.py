"""AC-Prof TUI ui/pages/run.py components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Collapsible, ContentSwitcher, Label, Static, TabPane

from acprof.tui.i18n import message_id
from acprof.tui.models.input import ModelInput
from acprof.tui.run.form import PRESET_OPTIONS
from acprof.tui.settings.field_validation import ConfigField
from acprof.tui.ui.checkbox import StatusCheckbox
from acprof.tui.ui.input import BarCursorInput as Input
from acprof.tui.ui.presentation import format_input_number

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui
from acprof.tui.ui.presentation import COLLAPSED_SYMBOL, EXPANDED_SYMBOL


def compose_number_field(app: AcprofTui, label: str, widget_id: str, value: int | float | str,
                         unit: str, *, placeholder: str = "") -> ComposeResult:
    yield app.localization.localized_widget(Label(label))
    yield ConfigField(Horizontal(
        app.localization.localized_widget(Input(
            value=format_input_number(value), id=widget_id, classes="config-control",
            placeholder=placeholder,
        )), app.localization.localized_widget(Label(unit, classes="field-unit")), classes="number-field"), field_id=widget_id)

def compose_run_tab(app: AcprofTui) -> ComposeResult:
    with TabPane("实验配置", id="run-tab"):
        with Vertical(classes="page-header"):
            yield app.localization.localized_widget(Static("", id="config-summary", classes="page-summary", markup=False))
            yield Static("", id="preflight-run-reason", markup=False)
        with ContentSwitcher(initial="run-form", id="experiment-pages"):
            with VerticalScroll(id="run-form", classes="pane-scroll"):
                with Grid(classes="form-grid"):
                    yield app.localization.localized_widget(Label("模型 ID"))
                    with Horizontal(classes="model-picker-row"):
                        yield ConfigField(app.localization.localized_widget(ModelInput(
                            value=app.configuration.initial_config.model,
                            placeholder="google-bert/bert-base-uncased",
                            id="model", classes="config-control",
                            tooltip="自由输入模型 ID；F4 搜索本地候选与验证记录。",
                        )), field_id="model")
                        yield app.localization.localized_widget(Button("模型候选", id="model-candidates"))
                    yield app.localization.localized_widget(Label("模型来源"))
                    yield ConfigField(app.localization.localized_select((("Hugging Face", "huggingface"),
                        ("ModelScope", "modelscope")), value=app.configuration.initial_config.model_source, allow_blank=False,
                        id="model-source", classes="config-control"), field_id="model-source")
                    yield app.localization.localized_widget(Label("输出目录"))
                    yield ConfigField(app.localization.localized_widget(Input(
                        value=app.configuration.initial_config.output_dir,
                        id="output-dir",
                        classes="config-control",
                    )), field_id="output-dir")

                    yield app.localization.localized_widget(Label("运行预设"))
                    yield app.localization.localized_select(
                        PRESET_OPTIONS,
                        value=app.configuration.initial_preset,
                        allow_blank=False,
                        id="run-preset",
                    )

                    yield app.localization.localized_widget(Label("CPU 列表"))
                    yield ConfigField(app.localization.localized_widget(Input(
                        value=app.configuration.initial_config.cpus,
                        placeholder="1,2,4,8",
                        id="cpus",
                        classes="config-control",
                    )), field_id="cpus")
                    yield app.localization.localized_widget(Label("内存 GB"))
                    yield ConfigField(app.localization.localized_widget(Input(
                        value=app.configuration.initial_config.mems,
                        placeholder="2,4,8,16",
                        id="mems",
                        classes="config-control",
                    )), field_id="mems")

                    yield app.localization.localized_widget(Label("GPU 模式"))
                    yield ConfigField(app.localization.localized_select(
                        app.configuration.gpu_options(),
                        value=app.configuration.initial_config.gpus,
                        allow_blank=False,
                        id="gpus",
                        classes="config-control",
                    ), field_id="gpus")
                    yield app.localization.localized_widget(Label("输入规模", id="input-scales-label"))
                    yield ConfigField(app.localization.localized_widget(Input(
                        value=app.configuration.initial_config.input_scales,
                        placeholder="留空自动规划；如 64,128,256",
                        id="input-scales",
                        classes="config-control",
                    )), field_id="input-scales")

                with app.localization.localized_widget(Collapsible(
                    title="完整命令（自动更新）",
                    collapsed=True,
                    collapsed_symbol=COLLAPSED_SYMBOL,
                    expanded_symbol=EXPANDED_SYMBOL,
                    id="command-details",
                )):
                    yield app.localization.localized_widget(Static("", id="command-preview", markup=False))
                    yield app.localization.localized_widget(Static(
                        "测量期间运行时间持续更新，日志在测量结束后刷新。",
                        id="science-note",
                        markup=False,
                    ))

            with VerticalScroll(id="advanced-form", classes="pane-scroll"):
                with Grid(classes="form-grid"):
                    for label, field, unit in (
                        ("Warmup", "warmup", "次"), ("Repeat", "repeat", "次"),
                        ("采样频率", "sample_hz", "Hz"), ("Idle 基线", "idle_seconds", "s"),
                        ("冷却时间", "idle_cooldown_seconds", "s"), ("连接/读取超时", "request_timeout_seconds", "s"),
                        ("Batch size", "batch_size", ""), ("窗口请求数", "repeat_in_window", "次"),
                        ("自动窗口", "repeat_window_seconds", "s"),
                    ):
                        yield from compose_number_field(
                            app, label, field.replace("_", "-"), getattr(app.configuration.initial_config, field), unit,
                            placeholder="0 表示自动校准" if field == "repeat_in_window" else "",
                        )
                with Grid(classes="form-grid"):
                    yield app.localization.localized_widget(Label("输入规划"))
                    yield ConfigField(app.localization.localized_select(
                        (("自动范围", "auto"), ("最小单一尺度", "minimal")),
                        value=app.configuration.initial_config.input_scale_policy, allow_blank=False,
                        id="input-scale-policy", classes="config-control",
                    ), field_id="input-scale-policy")
                    yield app.localization.localized_widget(Label("运行方式"))
                    yield ConfigField(app.localization.localized_select(
                        (("自动选择 Runtime", "auto"), ("Docker", "docker"), ("Process（无需 Docker）", "process")),
                        value=app.configuration.initial_config.runtime, allow_blank=False,
                        id="runtime", classes="config-control",
                    ), field_id="runtime")
                    with Vertical(id="process-resource-help"):
                        yield app.localization.localized_widget(Static(
                            "Process 的 CPU / 内存填 auto 表示继承限制；数值配额需要可写的 cgroup。",
                            classes="page-hint", markup=False,
                        ))
                        yield app.localization.localized_widget(Button(
                            "使用 Process 继承限制", id="process-inherited-resources",
                        ))
                    yield app.localization.localized_widget(Label("画像模式"))
                    yield ConfigField(app.localization.localized_select(
                        (("完整（RAPL / perf / 抓包）", "full"), ("基础（延迟 / CPU / 内存）", "basic")),
                        value=app.configuration.initial_config.profiling_mode,
                        allow_blank=False, id="profiling-mode", classes="config-control",
                    ), field_id="profiling-mode")
                    yield app.localization.localized_widget(Label("固定 CPU 集合"))
                    yield ConfigField(app.localization.localized_widget(Input(
                        value=app.configuration.initial_config.cpuset_cpus, id="cpuset-cpus",
                        placeholder="可选；如 0-3,8；用于正式采集", classes="config-control",
                    )), field_id="cpuset-cpus")
                    yield app.localization.localized_widget(Label("抓包网卡"))
                    yield ConfigField(app.localization.localized_widget(Input(
                        value=app.configuration.initial_config.sniff_iface,
                        id="sniff-iface",
                        classes="config-control",
                    )), field_id="sniff-iface")
                    yield app.localization.localized_widget(Label("通知"))
                    yield ConfigField(app.localization.localized_select(
                        (
                            ("自动", "auto"),
                            (message_id('common.option.off'), "none"),
                            ("企业微信", "wecom"),
                        ),
                        value=app.configuration.initial_config.notify,
                        allow_blank=False,
                        id="notify",
                        classes="config-control",
                    ), field_id="notify")

                yield app.localization.localized_widget(Static("深度性能分析", classes="section-title"))
                yield app.localization.localized_widget(Static(
                    "按需选择；分析器分别运行，可能增加准备时间。首次采集建议关闭。",
                    classes="page-hint", markup=False,
                ))
                with Grid(classes="profiler-grid"):
                    for widget_id, label, enabled in (
                        ("torch-profiler", "PyTorch Profiler — 算子与框架计算分析",
                         app.configuration.initial_config.compute_profile_tool in {"torch", "both"}),
                        ("ncu-profiler", "NVIDIA NCU — GPU 内核性能分析",
                         app.configuration.initial_config.compute_profile_tool in {"ncu", "both"}),
                        ("massif-profiler", "Valgrind Massif — CPU 内存分析",
                         app.configuration.initial_config.execution_profile_tool in {"massif", "both"}),
                        ("nsys-profiler", "NVIDIA Nsys — CPU/GPU 执行时序分析",
                         app.configuration.initial_config.execution_profile_tool in {"nsys", "both"}),
                    ):
                        yield app.localization.localized_widget(StatusCheckbox(
                            label, value=enabled, id=widget_id,
                            classes="config-control option-checkbox profiler-checkbox",
                        ))

                yield app.localization.localized_widget(Static("识别覆盖（通常留空）", classes="section-title"))
                with Grid(classes="form-grid"):
                    yield app.localization.localized_widget(Label("模型 revision"))
                    yield ConfigField(app.localization.localized_widget(Input(
                        value=app.configuration.initial_config.revision, id="revision", classes="config-control",
                        placeholder="留空解析默认 revision；候选选择会固定 commit SHA",
                    )), field_id="revision")
                    yield app.localization.localized_widget(Label("Task"))
                    yield ConfigField(app.localization.localized_widget(Input(
                        value=app.configuration.initial_config.task,
                        placeholder="如 text-generation",
                        id="task",
                        classes="config-control",
                    )), field_id="task")
                    yield app.localization.localized_widget(Label("Task family"))
                    yield ConfigField(app.localization.localized_select(
                        (
                            ("自动识别", ""),
                            ("NLP", "nlp"),
                            ("CV", "cv"),
                            ("Audio", "audio"),
                            ("Time series", "timeseries"),
                            ("Diffusion", "diffusion"),
                            ("Multimodal", "multimodal"),
                            ("结构化数据／策略", "structured"),
                        ),
                        value=app.configuration.initial_config.task_family,
                        allow_blank=False,
                        id="task-family",
                        classes="config-control",
                    ), field_id="task-family")
                    yield app.localization.localized_widget(Label("Backend"))
                    yield ConfigField(app.localization.localized_widget(Input(
                        value=app.configuration.initial_config.backend,
                        placeholder="留空自动识别",
                        id="backend",
                        classes="config-control",
                    )), field_id="backend")
                    yield app.localization.localized_widget(Label("模型接口声明"))
                    yield ConfigField(app.localization.localized_widget(Input(
                        value=app.configuration.initial_config.model_spec,
                        placeholder="模型接口 JSON，可留空",
                        id="model-spec",
                        classes="config-control",
                    )), field_id="model-spec")
                    yield app.localization.localized_widget(Label("Workload manifest"))
                    yield ConfigField(app.localization.localized_widget(Input(
                        value=app.configuration.initial_config.workload_spec,
                        placeholder="输入素材 manifest，可留空",
                        id="workload-spec",
                        classes="config-control",
                    )), field_id="workload-spec")

                with Collapsible(title=app.tr(message_id('downloads.settings.title')), collapsed=True):
                    with Grid(classes="form-grid"):
                        for key, label, placeholder in (
                            ("max-download", "下载预算", "5GB；留空不设上限"),
                            ("model-store", "Model Store 路径", "留空使用用户缓存目录"),
                        ):
                            yield app.localization.localized_widget(Label(label))
                            yield ConfigField(app.localization.localized_widget(Input(value=getattr(app.configuration.initial_config, key.replace("-", "_")),
                                placeholder=placeholder, id=key, classes="config-control")), field_id=key)
                        yield app.localization.localized_widget(Label("预计下载大小"))
                        yield app.localization.localized_widget(Static("待解析", id="download-estimate", markup=False))
                    yield app.localization.localized_widget(Static("下载策略：自动。使用系统网络环境。", markup=False))
                    with Collapsible(title=app.tr(message_id('common.action.advanced_details')), collapsed=True):
                        yield app.localization.localized_widget(Label("Model Store 容量上限"))
                        yield ConfigField(app.localization.localized_widget(Input(value=app.configuration.initial_config.model_store_max,
                            placeholder="100GB；留空不设上限", id="model-store-max", classes="config-control")), field_id="model-store-max")
                        yield Static("", id="network-download-summary", markup=False)
                    yield app.localization.localized_widget(Button("Model Store 占用与清理", id="open-model-store", classes="config-control"))

                with Horizontal(classes="checkbox-row"):
                    yield app.localization.localized_widget(StatusCheckbox(
                        "启动 OOM 剪枝",
                        value=app.configuration.initial_config.prune_startup_oom,
                        id="prune-startup-oom",
                        classes="config-control option-checkbox",
                    ))
                    yield app.localization.localized_widget(StatusCheckbox(
                        "Idle 诊断",
                        value=app.configuration.initial_config.idle_debug,
                        id="idle-debug",
                        classes="config-control option-checkbox",
                    ))
                with Horizontal(classes="button-row"):
                    yield app.localization.localized_widget(Button("记住实验配置", id="save-run-default"))
                with Collapsible(title=app.tr(message_id('run.images.advanced_options')), collapsed=True, id="image-build-options"):
                    yield app.localization.localized_widget(StatusCheckbox(
                        "重新构建镜像（仅本次）",
                        value=app.configuration.initial_config.rebuild_image,
                        id="rebuild-image",
                        classes="config-control option-checkbox",
                        tooltip="仅用于下一次采集或探测，不保存为默认配置；仍可使用 Docker 构建缓存。",
                    ))

        with Vertical(id="run-footer"):
            yield app.localization.localized_widget(Static("", id="run-plan-summary", markup=False))
            with Horizontal(id="run-actions", classes="action-bar"):
                with Horizontal(classes="action-secondary"):
                    yield app.localization.localized_widget(Button("高级参数", id="open-run-settings"))
                    yield app.localization.localized_widget(Button("探测最大输入", id="probe-largest"))
                with Horizontal(classes="action-primary"):
                    yield app.localization.localized_widget(Button("开始采集", id="start-run", variant="primary"))
