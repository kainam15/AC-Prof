"""Result input errors are translated without rewriting paths."""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import pytest

from acprof.experiments.configuration import RunConfigError
from acprof.tui.commands import prepare_comparison, prepare_plot, prepare_stats
from acprof.tui.i18n import error_message, translate


@pytest.mark.parametrize("operation", ["plot", "stats", "compare"])
def test_invalid_result_inputs_have_english_messages(operation, tmp_path):
    source = "等待{raw}.csv"
    if operation == "plot":
        prepare = partial(prepare_plot, source)
    elif operation == "stats":
        prepare = partial(prepare_stats, source)
    else:
        prepare = partial(
            prepare_comparison, source, source, baseline="left", purpose="same-hardware",
        )
    with pytest.raises(RunConfigError) as caught:
        prepare(project_dir=tmp_path, python_executable=Path(sys.executable))

    message = error_message(caught.value)
    if operation == "plot":
        assert translate(message, "zh") == f"结果文件不存在：{tmp_path / source}"
        assert translate(message, "en") == f"Result file does not exist: {tmp_path / source}"
    else:
        assert translate(message, "zh") == "请选择有效的分层结果目录、清单或独立 CSV。"
        assert translate(message, "en") == (
            "Select a valid layered result directory, manifest, or standalone CSV."
        )
