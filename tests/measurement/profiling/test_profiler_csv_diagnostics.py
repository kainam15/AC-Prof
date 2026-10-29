"""CSV missing diagnostics must not become profiler failures or trigger recollection."""
import csv
from types import SimpleNamespace

import pytest

from acprof.analysis.audit import MissingReasons
from acprof.host.posthoc import backfill, plans, service
from acprof.host.posthoc.context import (
    SUPPORTED_TOOLS,
    TOOL_ERROR_FIELD,
    TOOL_GPU_MODES,
    TOOL_METRIC_FIELDS,
    PosthocError,
    load_result_context,
)
from acprof.result_layers import publish_result_layers
from acprof.tui import profile_status
from tests.measurement.profiling.test_posthoc import TestPosthocProfile as _PosthocFixture
from tests.runtime.containers.test_layer_read_cancellation import source_hashes
from tests.tui.results.test_profile_overview_snapshot import profile_store, run_summary_worker


def complete_row(tool, error="nan"):
    return {"gpu_mode": TOOL_GPU_MODES[tool][0], "status": "ok",
            **dict.fromkeys(TOOL_METRIC_FIELDS[tool], "0"), TOOL_ERROR_FIELD[tool]: error}


@pytest.mark.parametrize("version", [1, 2, 3])
def test_uncollected_layers_are_missing_in_the_complete_summary_worker(tmp_path, version):
    profile_store(tmp_path, version)
    before = source_hashes(tmp_path)

    _, _, _, summary, error, _, overview, profile_error = run_summary_worker(tmp_path / "result_layers.json")

    assert summary is not None, error
    assert overview is not None, profile_error
    assert [(s.tool, s.state, s.completed, s.total, s.errors) for s in overview.tools] == [
        ("torch", "complete", 2, 2, 0), ("ncu", "missing", 0, 1, 0),
        ("nsys", "missing", 0, 1, 0), ("massif", "missing", 0, 1, 0),
    ]
    assert overview.suggested_tools == ("ncu",)
    assert source_hashes(tmp_path) == before


@pytest.mark.parametrize("tool", SUPPORTED_TOOLS)
@pytest.mark.parametrize("error", [None, "", "nan", " NaN ", "none", "NULL", " n/a "])
def test_csv_missing_diagnostics_accept_complete_values_without_rewriting_rows(tool, error):
    row = complete_row(tool, error)
    before = dict(row)

    assert backfill.row_tool_complete(row, tool)
    assert row == before
    row.update(dict.fromkeys(TOOL_METRIC_FIELDS[tool], "nan"))
    reason = MissingReasons({"compute_profile_tools": [], "execution_profile_tools": []},
                            environment_class="native_linux").for_row(row)
    assert reason(TOOL_METRIC_FIELDS[tool][0]) == "tool_not_enabled"


@pytest.mark.parametrize("tool", SUPPORTED_TOOLS)
@pytest.mark.parametrize("invalid", [None, "nan", "inf", "-inf", "1e999", "invalid"])
def test_every_required_metric_must_be_finite_even_without_a_diagnostic(tool, invalid):
    for field in TOOL_METRIC_FIELDS[tool]:
        row = complete_row(tool)
        if invalid is None:
            row.pop(field)
        else:
            row[field] = invalid
        assert not backfill.row_tool_complete(row, tool), field


@pytest.mark.parametrize("tool", SUPPORTED_TOOLS)
@pytest.mark.parametrize("error", [
    "permission denied", "nan in counter output", "NoneType exception", "N/A (unsupported)",
    "compute_profile_disabled", "unknown", "unavailable", "0", 0, False,
])
def test_real_diagnostics_still_block_completion_and_remain_visible_to_audit(tool, error):
    row = complete_row(tool, error)

    assert not backfill.row_tool_complete(row, tool)
    field = TOOL_METRIC_FIELDS[tool][0]
    row[field] = "nan"
    reason = MissingReasons({}, environment_class="native_linux").for_row(row)
    assert reason(field) == "profiler_reported_error"


@pytest.mark.parametrize("tool", SUPPORTED_TOOLS)
def test_overview_distinguishes_partial_missing_and_real_failure(tmp_path, monkeypatch, tool):
    first, second = complete_row(tool), complete_row(tool, " N/A ")
    second[TOOL_METRIC_FIELDS[tool][0]] = "nan"
    context = SimpleNamespace(result_dir=tmp_path, rows=[first, second])
    monkeypatch.setattr(profile_status, "load_result_context", lambda *args, **kwargs: context)

    def status():
        return next(s for s in profile_status.inspect_profile_result(tmp_path).tools if s.tool == tool)

    assert status() == profile_status.ProfilerStatus(tool, "partial", 1, 2, 0)
    first[TOOL_METRIC_FIELDS[tool][0]] = "nan"
    assert status() == profile_status.ProfilerStatus(tool, "missing", 0, 2, 0)
    second[TOOL_ERROR_FIELD[tool]] = "nan in counter output"
    assert status() == profile_status.ProfilerStatus(tool, "failed", 0, 2, 1)


def test_successful_csv_with_missing_diagnostics_skips_all_work_without_writing(tmp_path, monkeypatch):
    raw = _PosthocFixture()._write_fixture(tmp_path)
    context = load_result_context(tmp_path)
    for row in context.rows:
        for tool in SUPPORTED_TOOLS:
            if row["gpu_mode"] in TOOL_GPU_MODES[tool]:
                row.update(dict.fromkeys(TOOL_METRIC_FIELDS[tool], "0"))
                row[TOOL_ERROR_FIELD[tool]] = "nan"
    with raw.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, context.fieldnames)
        writer.writeheader()
        writer.writerows(context.rows)
    publish_result_layers(raw)
    before = source_hashes(tmp_path)
    monkeypatch.setattr(service, "find_active_processes", lambda *args, **kwargs: [])

    def unexpected_work(*args, **kwargs):
        raise AssertionError("complete CSV must not validate a runtime, reuse a plan, or start a profiler")

    for name in ("_validate_profiler_runtime", "_find_reusable_compute_plan", "_find_reusable_execution_plan",
                 "_collect_compute_plan", "_collect_execution_plan"):
        monkeypatch.setattr(service, name, unexpected_work)

    summary = service.run_posthoc(tmp_path)
    assert summary.skipped_tools == SUPPORTED_TOOLS
    assert summary.collected_tools == summary.reused_tools == ()
    assert summary.updated_rows_by_tool == {}
    assert summary.backup_dir is None
    assert source_hashes(tmp_path) == before

    forced = service.run_posthoc(tmp_path, dry_run=True, force_reprofile=True)
    assert forced.skipped_tools == ()
    assert forced.updated_rows_by_tool == dict.fromkeys(SUPPORTED_TOOLS, 0)
    assert source_hashes(tmp_path) == before


@pytest.mark.parametrize("tool", SUPPORTED_TOOLS)
def test_backfill_preserves_complete_csv_until_forced_but_missing_metrics_still_need_a_plan(tmp_path, tool):
    _PosthocFixture()._write_fixture(tmp_path)
    context = load_result_context(tmp_path)
    for row in context.rows:
        if row["gpu_mode"] in TOOL_GPU_MODES[tool]:
            row.update(dict.fromkeys(TOOL_METRIC_FIELDS[tool], "7"))
            row[TOOL_ERROR_FIELD[tool]] = "nan"
    before = [dict(row) for row in context.rows]

    _, rows, updated = backfill.backfill_rows(context, tools=(tool,))
    assert rows == before
    assert context.rows == before
    assert updated == {tool: 0}
    with pytest.raises(PosthocError, match="plan is required"):
        backfill.backfill_rows(context, tools=(tool,), force=True)
    applicable = next(row for row in context.rows if row["gpu_mode"] in TOOL_GPU_MODES[tool])
    applicable[TOOL_METRIC_FIELDS[tool][0]] = "nan"
    with pytest.raises(PosthocError, match="plan is required"):
        backfill.backfill_rows(context, tools=(tool,))


@pytest.mark.parametrize("tool", SUPPORTED_TOOLS)
@pytest.mark.parametrize("error", ["nan", "null", "N/A"])
def test_json_plan_error_strings_are_not_csv_missing_markers(tmp_path, tool, error):
    fixture = _PosthocFixture()
    fixture._write_fixture(tmp_path)
    context = load_result_context(tmp_path)
    if tool in {"torch", "ncu"}:
        plan = fixture._torch_plan() if tool == "torch" else fixture._compute_plan()
        covers = plans.compute_plan_covers_tool
        profile = plan["profiles"]["cpu"]["torch_profiler_eager"] if tool == "torch" else plan["profiles"]["gpu"]["ncu"]
    else:
        plan = fixture._execution_plan()
        covers = plans.execution_plan_covers_tool
        profile = next(p["tools"][tool] for p in plan["profiles"] if tool in p["tools"])
    assert covers(plan, context, tool)

    profile["entries"][0]["error"] = error

    assert not covers(plan, context, tool)
