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
    tui_nodes = {
        "tests/tui/widgets/test_tui_page_chrome.py::TestTuiPageChrome::"
        "test_confirmation_keeps_the_action_color_and_cancel_does_not_launch",
        "tests/models/storage/test_model_store_view_reuse.py::test_cached_view_can_be_selected_at_80x24_without_source_index",
        "tests/execution/validation/test_interface_progress.py::test_monitor_renders_real_interface_completion",
        "tests/visual_png/test_flow_report.py::test_flow_catalog_matches_visible_tui_pages",
    }
    unit_nodes = {
        "tests/models/storage/test_model_store_view_reuse.py::test_candidate_scan_uses_recorded_entry_id_and_stays_read_only",
        "tests/execution/validation/test_interface_progress.py::test_terminal_interface_events_cannot_arrive_inside_measurement",
        "tests/visual_png/test_flow_report.py::test_flow_catalog_references_existing_public_fixtures_and_pages",
    }
    for marker, expected in (("tui", tui_nodes), ("unit", unit_nodes)):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", marker, *sorted(tui_nodes | unit_nodes)],
            cwd=TUI_TEST_ROOT.parents[1], capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        selected = {line.partition("[")[0] for line in result.stdout.splitlines() if line.startswith("tests/")}
        assert selected == expected
