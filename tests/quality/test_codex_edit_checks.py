"""Edit hooks select checks from the affected checkout and propagate actionable failures."""
from types import SimpleNamespace

import pytest

from scripts import codex_edit_checks as edit_checks

OVERVIEW = "test_four_fact_groups_keep_identifiers_times_and_paths_in_details"
STAGED_IMPORT = "test_service_entrypoint_imports_without_checkout[acprof.failures]"
PRESENCE = "test_failed_presence_probe_stays_distinct_from_absence"
DECIMAL_KEY = "test_csv_identity_preserves_all_significant_digits_without_rounding"


@pytest.fixture
def commands(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(edit_checks.subprocess, "run", run)
    return calls


def make_file(root, name, content=""):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_report_patch_includes_explicit_ruff_project_types_and_focused_pilot(tmp_path, commands):
    path = "acprof/tui/results/controller.py"
    make_file(tmp_path, path)
    assert edit_checks.run_checks(tmp_path, {path, "acprof/tui/tui.tcss"}) == ""
    argv = [command for command, _ in commands]
    assert argv[0][-3:] == ["--no-force-exclude", "--", path]
    assert argv[1][-2:] == ["-m", "mypy"]
    assert len(argv) == 3
    targets = argv[2][argv[2].index("--") + 1:]
    assert len(targets) == 1
    assert "test_open_overhead_at_all_sizes" in targets[0]
    assert all(options["cwd"] == tmp_path for _, options in commands)
    assert all(options["env"]["HF_HUB_OFFLINE"] == "1" for _, options in commands)


@pytest.mark.parametrize(("path", "expected"), [
    ("acprof/tui/experiments/catalog.py",
     {OVERVIEW, "test_duplicate_run_collapses_aliases_and_content_conflict_is_explicit"}),
    ("acprof/analysis/compatibility.py", {OVERVIEW}),
    ("acprof/tui/results/overview.py", {OVERVIEW}),
    ("acprof/source_identity.py", {STAGED_IMPORT}),
    ("acprof/failures.py", {STAGED_IMPORT}),
    ("acprof/experiments/constraints.py", {STAGED_IMPORT}),
    ("acprof/artifacts.py", {STAGED_IMPORT, PRESENCE}),
    ("acprof/results/metadata.py", {PRESENCE}),
    ("acprof/results/csv.py", {DECIMAL_KEY, "test_error_in_mixed_batch_leaves_valid_head_and_writer[absent-insert]"}),
    ("acprof/results/measurement_plan.py", {STAGED_IMPORT, DECIMAL_KEY}),
    ("tests/tui/results/test_structured_overview.py", {OVERVIEW}),
    ("tests/runtime/containers/test_service_source_imports.py", {STAGED_IMPORT}),
    ("tests/analysis/statistics/test_statistics_metadata_observations.py", {PRESENCE}),
    ("tests/results/storage/test_result_merge.py", {DECIMAL_KEY}),
])
def test_changed_contract_files_run_only_their_affected_regressions(tmp_path, commands, path, expected):
    make_file(tmp_path, "scripts/check_ci_preflight.py", "TEST_TARGETS = ('tests/smoke.py',)\n")
    make_file(tmp_path, path)
    assert edit_checks.run_checks(tmp_path, {path}) == ""
    behavior = [argv for argv, _kwargs in commands if "--require-no-skips" in argv]
    assert len(behavior) == 1
    nodes = behavior[0][behavior[0].index("--") + 1:]
    assert {node.rsplit("::", 1)[-1] for node in nodes} == expected


def test_deleted_registered_test_still_checks_its_node_from_this_checkout(tmp_path, commands):
    deleted = "tests/old_test.py"
    make_file(tmp_path, "scripts/check_ci_preflight.py", f"TEST_TARGETS = ('{deleted}::test_old', 'tests/other.py')\n")
    assert edit_checks.run_checks(tmp_path, {deleted}) == ""
    assert len(commands) == 1
    argv, _ = commands[0]
    assert "--collect-only" in argv
    assert argv[-1] == f"{deleted}::test_old"
    assert "tests/other.py" not in argv


def test_changed_registry_collects_every_declared_target(tmp_path, commands):
    registry = "scripts/check_ci_preflight.py"
    make_file(tmp_path, registry, "TEST_TARGETS = ('tests/test_sample.py::test_old', 'tests/test_other.py')\n")
    assert edit_checks.run_checks(tmp_path, {registry}) == ""
    assert commands[-1][0][-2:] == ["tests/test_sample.py::test_old", "tests/test_other.py"]


def test_deleted_production_module_runs_mypy_and_ignores_generated_python(tmp_path, commands):
    generated = "results/example.py"
    make_file(tmp_path, generated)
    assert edit_checks.run_checks(tmp_path, {"acprof/removed.py", generated}) == ""
    assert len(commands) == 1
    assert commands[0][0][-1] == "mypy"


def test_no_tools_for_unrelated_files(tmp_path, commands):
    assert edit_checks.run_checks(tmp_path, {"docs/usage/tui.md", "results/import.csv"}) == ""
    assert commands == []


def test_empty_registry_does_not_launch_unbounded_pytest(tmp_path, commands):
    registry = "scripts/check_ci_preflight.py"
    make_file(tmp_path, registry, "TEST_TARGETS = ()\n")
    with pytest.raises(ValueError, match="TEST_TARGETS"):
        edit_checks.run_checks(tmp_path, {registry})
    assert not any("pytest" in argv for argv, _ in commands)


def test_broken_registry_reports_its_cause_without_executing_its_nodes(tmp_path, commands):
    make_file(tmp_path, "scripts/check_ci_preflight.py", "raise RuntimeError('broken target list')\n")
    with pytest.raises(ValueError, match=r"check_ci_preflight.py: RuntimeError: broken target list"):
        edit_checks.run_checks(tmp_path, {"tests/renamed.py"})
    assert commands == []


def test_failure_preserves_diagnostic_and_stops_slow_checks(tmp_path, monkeypatch):
    path = "acprof/results/csv.py"
    make_file(tmp_path, path)
    commands = []

    def fail(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1, stdout=f"{path}:3:8: F821 Undefined name 'oops'\n")

    monkeypatch.setattr(edit_checks.subprocess, "run", fail)
    message = edit_checks.run_checks(tmp_path, {path})
    assert f"{path}:3:8: F821" in message
    assert len(commands) == 1
