"""Exercise the public audit command on recorded data, without running a workload."""
import csv
import json

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
