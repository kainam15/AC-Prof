"""Exercise the public audit command on recorded data, without running a workload."""
import csv
import json

import pytest

from acprof.cli import audit
from tests.support.comparison_fixtures import ComparisonFixture


def test_audit_lists_partial_values_without_counting_warmup_or_rejecting_the_run(tmp_path, capsys):
    fixture = ComparisonFixture()
    fixture.build(tmp_path)
    path = fixture.left / "result_all.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, row = reader.fieldnames, next(reader)
    assert fields is not None
    row.update(dram_window_energy_j="0", resource_metric_quality=json.dumps({"schema_version": 1, "metrics": {},
        "collectors": {"dram_energy": {"status": "partial", "reason": "reader gap"}}}))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows([{**row, "warmup": 1}, row, {**row, "repeat_idx": 1, "status": "warn"}])
    fixture.change_json(fixture.left, "run_state.json", lambda state: state["options"].update(warmup=1, repeat=2))
    assert audit.main([str(path), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["accepted"]
    assert report["partial_metrics"] == {"dram_window_energy_j": {"reader gap": 2}}
    assert "dram_window_energy_j" not in report["missing_metrics"]
    assert audit.main([str(path)]) == 0
    assert "dram_window_energy_j (2)" in capsys.readouterr().out


def test_text_distinguishes_completed_run_from_incomplete_plan(tmp_path, capsys):
    fixture = ComparisonFixture()
    fixture.build(tmp_path)

    def change_plan(state):
        state["options"]["repeat"] = 2
        state["runtime"]["planned"]["scales"] = [5]

    fixture.change_json(fixture.left, "run_state.json", change_plan)

    assert audit.main([str(fixture.left)]) == 1
    out = capsys.readouterr().out
    assert "运行完成状态=complete" in out
    assert "测量状态=incomplete" in out
    assert "expected=2，actual=1，missing=2，unexpected=1" in out
    assert "--require-" not in out


@pytest.mark.parametrize(("status", "has_plan"), [("complete", False), ("interrupted", True), ("interrupted", False)])
def test_require_complete_explains_each_unmet_requirement(tmp_path, capsys, status, has_plan):
    fixture = ComparisonFixture()
    fixture.build(tmp_path)

    def change_state(state):
        state["status"] = status
        if not has_plan:
            state.pop("runtime")

    fixture.change_json(fixture.left, "run_state.json", change_state)

    assert audit.main([str(fixture.left), "--require-complete"]) == 1
    out = capsys.readouterr().out
    assert ("--require-complete 需要 run_status=complete，当前为 interrupted" in out) == (status != "complete")
    assert ("--require-complete 缺少可核对的计划覆盖证据" in out) == (not has_plan)
    if not has_plan:
        assert "计划覆盖=unknown" in out
        assert "expected=" not in out


def test_unresolved_timeout_explains_failed_row_and_strict_formal_gate(tmp_path, capsys):
    fixture = ComparisonFixture()
    fixture.build(tmp_path)
    path = fixture.left / "result_all.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, row = reader.fieldnames, next(reader)
    assert fields is not None
    row.update(status="error", error="client_request_timeout: planned_request_attempted=unknown")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerow(row)

    assert audit.main([str(path), "--require-complete"]) == 0
    out = capsys.readouterr().out
    assert "终态失败行中有 1 行超时诊断无法确认计划请求是否已发出" in out
    assert "测量状态=incomplete" in out
    assert "--require-ok" not in out
    assert audit.main([str(path), "--require-ok"]) == 1
    out = capsys.readouterr().out
    assert "--require-ok 要求至少一个正式窗口且全部实测成功" in out
    assert "正式窗口实测成功 0/1" in out
