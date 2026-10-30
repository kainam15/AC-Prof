"""An actual environment diagnostics modal with long Chinese error text."""
from __future__ import annotations

from acprof.experiments.configuration import RunConfig
from acprof.tui.app import AcprofTui
from acprof.tui.ui.dialogs import EnvironmentPreflightScreen
from tests.visual_png.scenarios.common import freeze_for_screenshot


def initial_config() -> RunConfig:
    return RunConfig.smoke("fixture/model")


def mount(app: AcprofTui) -> None:
    freeze_for_screenshot(app)
    app.push_screen(EnvironmentPreflightScreen(
        "模型接口校验失败：无法确认推理任务与输出字段。\n\n"
        "模型：fixture/model\n"
        "原因：接口返回了不兼容的配置；请检查模型名称和自定义参数。\n"
        "提示：这段中文专门用于检查弹窗边框、长文本换行与按钮重叠。",
        retry_disabled=True,
    ))
