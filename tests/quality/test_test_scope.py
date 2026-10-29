"""TUI domain marker keeps the quick unit selection separate from UI regressions."""

import subprocess
import sys
from pathlib import Path

from acprof.testing.plugin import TUI_TEST_ROOT, is_tui_suite


def test_tui_marker_uses_real_suite_directory_only():
    assert is_tui_suite(TUI_TEST_ROOT / "widgets" / "test_tui_layout.py")
    assert is_tui_suite(TUI_TEST_ROOT / "test_pure_tui_logic.py")
    assert not is_tui_suite(TUI_TEST_ROOT.parent / "analysis" / "test_math.py")
    assert not is_tui_suite(Path("/tmp/tests/tui/test_external.py"))


def test_path_classification_does_not_depend_on_file_name():
    assert is_tui_suite(TUI_TEST_ROOT / "catalog" / "test_scenario.py")
    assert not is_tui_suite(TUI_TEST_ROOT.parent / "results" / "test_tui_report.py")


def test_pytest_collection_selects_tui_and_excludes_it_from_unit():
    node = (
        "tests/tui/widgets/test_tui_page_chrome.py::TestTuiPageChrome::"
        "test_confirmation_keeps_the_action_color_and_cancel_does_not_launch"
    )
    for marker, expected_count in (("tui", 1), ("unit", 0)):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", marker, node],
            cwd=TUI_TEST_ROOT.parents[1], capture_output=True, text=True, timeout=30,
        )
        if expected_count:
            assert result.returncode == 0, result.stdout + result.stderr
            assert node in result.stdout
            assert "1 test selected" in result.stdout or "1 test collected" in result.stdout
        else:
            assert "1 deselected" in result.stdout, result.stdout + result.stderr
