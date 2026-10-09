"""Tests for advisory module-size reporting, independent of current repo sizes."""

import json
from pathlib import Path

import pytest

from scripts.check_module_sizes import (
    REVIEW_THRESHOLD,
    SOFT_TARGET,
    inspect_module,
    main,
    scan_modules,
)


def _write_module(root: Path, name: str, contents: str) -> Path:
    path = root / "acprof" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return path


def test_physical_line_thresholds_are_advisory(tmp_path, capsys):
    normal = _write_module(tmp_path, "normal.py", "# comment\n" * SOFT_TARGET)
    target = _write_module(tmp_path, "target.py", "# comment\n" * (SOFT_TARGET + 1))
    review = _write_module(tmp_path, "review.py", "# comment\n" * (REVIEW_THRESHOLD + 1))
    assert inspect_module(normal, tmp_path).status == "ok"
    assert inspect_module(target, tmp_path).status == "target"
    assert inspect_module(review, tmp_path).status == "review"
    assert main(["--root", str(tmp_path), "--top", "0"]) == 0
    output = capsys.readouterr().out
    assert "review.py" in output
    assert "target.py" in output
    assert "normal.py" not in output


def test_literal_catalog_is_still_reported_but_not_prioritized(tmp_path):
    contents = "WORDS = {\n" + "".join(f"    {i}: 'word',\n" for i in range(900)) + "}\n"
    path = _write_module(tmp_path, "catalog.py", contents)
    summary = inspect_module(path, tmp_path)
    assert summary.status == "data-review"
    assert summary.data_catalog
    assert summary.functions == 0


def test_large_catalog_with_complex_function_still_needs_review(tmp_path):
    contents = (
        "WORDS = {\n" + "".join(f"    {i}: 'word',\n" for i in range(900)) + "}\n"
        + "def long_function():\n" + "    x = 1\n" * 100
    )
    summary = inspect_module(_write_module(tmp_path, "mixed.py", contents), tmp_path)
    assert summary.status == "review"
    assert not summary.data_catalog
    assert summary.long_callables == 1


def test_longest_callable_and_decision_hints(tmp_path):
    path = _write_module(
        tmp_path, "logic.py",
        "from acprof.host import client\n"
        "class C:\n"
        "    def complex(self, yes, no):\n"
        "        if yes and no:\n"
        "            return 1\n"
        "        return 0\n"
        "    def short(self): return 0\n"
        "def outer():\n"
        "    def inner():\n"
        "        if True: return 1\n"
        "    return inner()\n",
    )
    summary = inspect_module(path, tmp_path)
    assert summary.classes == 1
    assert summary.functions == 4
    assert summary.longest_callable == "outer"
    assert summary.longest_callable_lines == 4
    assert summary.longest_callable_branches == 0
    assert summary.internal_imports == 1
    assert summary.long_callables == 0


def test_error_paths_are_not_silently_ignored(tmp_path, capsys):
    with pytest.raises(FileNotFoundError):
        scan_modules(tmp_path)
    _write_module(tmp_path, "broken.py", "def broken(:\n")
    with pytest.raises(SystemExit) as error:
        main(["--root", str(tmp_path)])
    assert error.value.code == 1
    assert "Module review failed" in capsys.readouterr().err
    with pytest.raises(SystemExit) as error:
        main(["--top", "-1"])
    assert error.value.code == 2


def test_json_reports_every_module(tmp_path, capsys):
    _write_module(tmp_path, "tiny.py", "def f():\n    return 1\n")
    assert main(["--root", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["soft_target"] == SOFT_TARGET
    assert payload["review_threshold"] == REVIEW_THRESHOLD
    assert payload["modules"][0]["path"] == "acprof/tiny.py"
