"""Near-match of the regular BAAI/bge-m3 customized configuration page."""
from __future__ import annotations

from dataclasses import replace

from acprof.experiments.configuration import RunConfig
from acprof.tui.app import AcprofTui
from tests.visual_png.scenarios.common import apply_fixture_preflight, freeze_for_screenshot


def initial_config() -> RunConfig:
    return replace(
        RunConfig.smoke("BAAI/bge-m3"),
        cpus="1,2,4",
        mems="2,4,8",
        input_scales="128,256,512",
        output_dir="results/BAAI-bge-m3",
        warmup=1,
        repeat=3,
    )


def mount(app: AcprofTui) -> None:
    apply_fixture_preflight(app)
    freeze_for_screenshot(app)
