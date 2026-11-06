"""Read real load artifacts without importing execution or presentation layers."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from tests.analysis.reporting.test_load_dataset_report import (
    make_dataset_sweep,
    saved_dataset_report,
)

__all__ = ["saved_dataset_report"]

_READ_OFFLINE = """
import importlib.abc
import json
import sys
from pathlib import Path

blocked = (
    "acprof.host", "acprof.container", "acprof.monitors",
    "acprof.cli", "acprof.tui", "acprof.plotting",
)

class RejectExecutionImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in blocked):
            raise ImportError("offline analysis imported " + fullname)
        return None

sys.meta_path.insert(0, RejectExecutionImports())
source = Path(sys.argv[2])
if sys.argv[1] == "point":
    from acprof.analysis.load_report import read_load_report
    result = read_load_report(source)
    print(json.dumps({"version": result["version"], "offered": result["capacity"]["counts"]["offered"]}))
else:
    from acprof.analysis.load_sweep import read_load_sweep
    result = read_load_sweep(source).payload
    print(json.dumps({"valid_points": result["valid_points"], "conditions": len(result["conditions"])}))
"""


@pytest.mark.parametrize("kind", ["point", "sweep"])
def test_offline_load_analysis_never_imports_execution(dataset_report, tmp_path, kind):
    if kind == "point":
        source = tmp_path / "load.json"
        source.write_text(json.dumps(dataset_report), encoding="utf-8")
        expected = {"version": 3, "offered": 2}
    else:
        source = make_dataset_sweep(tmp_path / "sweep", dataset_report, "source")
        expected = {"valid_points": 3, "conditions": 1}
    completed = subprocess.run(
        [sys.executable, "-I", "-c", _READ_OFFLINE, kind, str(source)],
        cwd=tmp_path, capture_output=True, text=True, timeout=20, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == expected
