"""Select locked, offline checks for files changed by a Codex edit."""
from __future__ import annotations

import os
import runpy
import shlex
import subprocess
from collections.abc import Iterator
from pathlib import Path

CI_PREFLIGHT = "scripts/check_ci_preflight.py"
THIS_SCRIPT = "scripts/codex_edit_checks.py"
PROJECT_CONFIG = {"pyproject.toml", "uv.lock"}
COLLECTION_CONFIG = {"tests/conftest.py", "acprof/testing/plugin.py", THIS_SCRIPT, *PROJECT_CONFIG}

# Each node runs one representative regression on edits to its production inputs
# or its own test file. The full size/language matrix still runs in CI.
BEHAVIOR_INPUTS = {
    "tests/tui/results/test_tui_reports.py::TestTuiReports::"
    "test_open_overhead_at_all_sizes_and_language_switch_preserves_data_and_draft[size0]": {
        "acprof/tui/results/controller.py", "acprof/tui/results/views.py",
        "acprof/tui/results/reports.py", "acprof/tui/results/report_files.py",
        "acprof/tui/results/report_actions.py", "acprof/tui/tui.tcss",
    },
    "tests/tui/workflows/test_tui_startup_preflight.py::TestStartupPreflight::"
    "test_success_does_not_reapply_initial_preset_or_notify": {
        "acprof/tui/run/experiment_controller.py", "acprof/tui/settings/controller.py",
        "acprof/experiments/configuration.py", "acprof/experiments/options.py",
    },
    "tests/tui/workflows/test_tui_auto_preflight.py::"
    "test_failed_auto_check_stays_failed_until_manual_retry[failure1]": {
        "acprof/tui/run/preflight_controller.py", "acprof/tui/updates.py",
    },
    "tests/results/storage/test_result_error_diagnostics.py::"
    "test_error_in_mixed_batch_leaves_valid_head_and_writer[absent-insert]": {
        "acprof/results/csv.py", "acprof/results/layers.py", "acprof/result_store/protocol.py",
        "acprof/result_store/writer.py", "acprof/result_store/sources.py",
    },
    "tests/models/discovery/test_experiment_catalog.py::TestExperimentCatalog::"
    "test_duplicate_run_collapses_aliases_and_content_conflict_is_explicit": {
        "acprof/tui/experiments/catalog.py", "acprof/results/layers.py", "tests/support/catalog_fixtures.py",
    },
    "tests/measurement/windows/test_measurement_reliability.py::"
    "test_ready_failure_materializes_full_error_matrix_without_packet_misdiagnosis": {
        "acprof/host/client.py", "acprof/host/orchestrator.py", "acprof/host/preflight.py",
        "acprof/platform.py", "tests/support/platform_fixtures.py",
    },
}


def is_checkable(path: str) -> bool:
    """Accept authored Python files, configuration and mapped non-Python inputs."""
    if path.split("/", 1)[0] in {"internal-testing", "results", "result-past", "evidence", ".venv"}:
        return False
    return (path.endswith((".py", ".pyi")) or path in PROJECT_CONFIG
            or any(path in sources for sources in BEHAVIOR_INPUTS.values()))


def _commands(root: Path, paths: set[str]) -> Iterator[list[str]]:
    python_paths = {path for path in paths if path.endswith((".py", ".pyi"))}
    existing = sorted(path for path in python_paths if (root / path).is_file())
    if existing:
        # Ruff's project-level results glob also matches authored protocol code.
        yield ["ruff", "check", "--output-format", "concise", "--no-force-exclude", "--", *existing]
    if PROJECT_CONFIG & paths or any(path.startswith("acprof/") for path in python_paths):
        yield ["mypy"]

    pytest = ["pytest", "-q", "-x", "--tb=short", "--color=no", "-p", "no:cacheprovider"]
    collect_all = bool(COLLECTION_CONFIG & paths)
    if collect_all or CI_PREFLIGHT in paths or any(path.startswith("tests/") for path in python_paths):
        # A hook can be inspecting a linked worktree. Load *that tree's* CI
        # registry, including edits and deletions, instead of importing a cached copy.
        try:
            targets = runpy.run_path(str(root / CI_PREFLIGHT)).get("TEST_TARGETS")
        except (Exception, SystemExit) as exc:
            # An edit to the registry itself can break its imports or initialization.
            # Preserve that cause in the model-visible diagnostic for this checkout.
            raise ValueError(f"{CI_PREFLIGHT}: {type(exc).__name__}: {exc}") from exc
        if not isinstance(targets, (list, tuple)) or not targets or not all(
            isinstance(target, str) and target.startswith("tests/") for target in targets
        ):
            raise ValueError(f"{CI_PREFLIGHT}: expected a nonempty TEST_TARGETS sequence of test paths")
        selected = [target for target in targets if collect_all or CI_PREFLIGHT in paths
                    or target.partition("::")[0] in paths]
        if collect_all:
            selected.extend(BEHAVIOR_INPUTS)
        if selected:
            yield [*pytest, "--collect-only", "--", *dict.fromkeys(selected)]

    behavior = [node for node, sources in BEHAVIOR_INPUTS.items()
                if paths & sources or node.partition("::")[0] in paths]
    if behavior:
        yield [*pytest, "--require-no-skips", "--", *behavior]


def run_checks(root: Path, paths: set[str]) -> str:
    """Return the first failing check and its diagnostics, or an empty string."""
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    environment.pop("VIRTUAL_ENV", None)
    environment.update(
        UV_PROJECT_ENVIRONMENT=str(root / ".venv"), HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", ACPROF_WECOM_WEBHOOK_URL="",
    )
    for args in _commands(root, {path for path in paths if is_checkable(path)}):
        command = ["uv", "run", "--locked", "--offline", "python", "-m", *args]
        result = subprocess.run(
            command, cwd=root, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", check=False,
        )
        if result.returncode:
            output = result.stdout.strip()
            if len(output) > 6500:
                output = output[:2500] + "\n...\n" + output[-4000:]
            invocation = shlex.join(command)
            if len(invocation) > 700:
                invocation = invocation[:697] + "..."
            return f"{invocation} (exit {result.returncode})\n{output}"
    return ""
