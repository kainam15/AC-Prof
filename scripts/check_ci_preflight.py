"""Fast, deterministic regression checks before the full host CI matrix.

Run from GitHub Actions and the optional local pre-push hook. No Docker,
network, real models, or production measurement are started by this script.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Repeated failures: saved runtime/configuration, recovery, Textual controls,
# missing cgroup I/O counters, private dependencies, and CI evidence gating.
LAYOUT_TESTS = "tests/tui/widgets/test_tui_layout_settings.py"
TEST_TARGETS = (
    f"{LAYOUT_TESTS}::TestTuiLayoutSettings::test_empty_model_can_be_remembered_without_allowing_collection",
    f"{LAYOUT_TESTS}::TestTuiLayoutSettings::test_custom_gpu_order_survives_loading_and_form_collection",
    f"{LAYOUT_TESTS}::TestTuiLayoutSettings::test_saving_ui_preserves_remembered_experiment_and_saving_experiment_preserves_ui",
    f"{LAYOUT_TESTS}::TestTuiMainSettings::test_cli_explicit_arguments_override_saved_defaults_without_rewriting_them",
    f"{LAYOUT_TESTS}::TestTuiModelMemory::test_explicit_settings_saves_preserve_last_model",
    f"{LAYOUT_TESTS}::TestTuiModelMemory::test_first_launch_remembers_model_but_not_unproduced_results",
    "tests/tui/workflows/test_tui_recovery_flow.py::test_start_existing_experiment_reviews_preparation_retry",
    "tests/tui/workflows/test_tui_recovery_flow.py::test_recovery_launch_requires_successful_preflight_and_unchanged_form",
    "tests/tui/workflows/test_tui_runtime_selection.py::test_legacy_saved_auto_uses_docker_without_rewriting_settings",
    "tests/tui/workflows/test_tui_runtime_selection.py::test_preflight_does_not_replace_an_explicit_runtime",
    "tests/runtime/environment/test_cgroup_target_io.py",
    "tests/runtime/environment/test_cgroup_io_monitor.py",
    "tests/quality/test_ci_gate.py",
)


def main() -> int:
    environment = os.environ.copy()
    environment["ACPROF_WECOM_WEBHOOK_URL"] = ""
    environment.pop("PYTEST_ADDOPTS", None)
    commands = (
        [sys.executable, "scripts/check_private_api.py"],
        [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *TEST_TARGETS],
    )
    for command in commands:
        print(f"CI preflight: {' '.join(command)}", flush=True)
        result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
