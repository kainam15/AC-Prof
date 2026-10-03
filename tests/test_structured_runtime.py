"""在 structured 镜像中验证真实 TorchScript 和 skops 加载。"""
import importlib.util
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.runtime


@pytest.mark.skipif(not (all(importlib.util.find_spec(name) for name in (
    "torch", "skops", "sklearn",
))), reason="requires the structured container")
@pytest.mark.parametrize('task', ('tabular-classification', 'tabular-regression', 'reinforcement-learning', 'robotics', 'graph-ml'))
def test_torchscript_and_skops_offline(task):
    from examples.structured.export_models import export_examples
    from examples.structured.smoke import check_skops, check_task
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        export_examples(root)
        check_task(root / task, task, "torchscript")
        check_skops(root)
