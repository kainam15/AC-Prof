"""Compare production input plans without fixture-only quality declarations."""
import json

import pytest

from acprof.analysis.comparison import compare_results
from acprof.host.detect import TaskInfo
from acprof.host.input_plan.materialize import materialize_scale_plan
from tests.support.comparison_fixtures import ComparisonFixture


@pytest.mark.parametrize("legacy", [False, True], ids=["new-plan", "legacy-plan"])
def test_materialized_plan_records_sanity_constraints_without_upgrading_legacy(tmp_path, legacy):
    fixture = ComparisonFixture()
    fixture.build(tmp_path / "results")
    spec = tmp_path / "workload.json"
    spec.write_text(json.dumps({
        "schema_version": 1, "task": "tabular-regression", "feature_dim": 2,
        "input_scales": [2], "seed": 42,
    }))
    task = TaskInfo("example/source", "tabular-regression", "structured",
                    "onnxruntime", "onnxruntime", "fixed", "test")
    planned = materialize_scale_plan(
        task_info=task, scales=[2], batch_size=1, output_dir=str(fixture.left),
        source="manual", workload_spec_path=str(spec),
    )
    path = fixture.left / "input_scale_plan.json"
    plan = json.loads(path.read_text())
    if legacy:
        plan.pop("quality_constraints", None)
        path.write_text(json.dumps(plan))
    else:
        assert plan["quality_constraints"] == {
            "schema_version": 1,
            "scope": "sanity_only_not_accuracy_benchmark",
            "required_status": {"protocol": "verified", "task": "verified"},
        }
        assert planned.plan_sha256

    before = path.read_bytes()
    report = compare_results(fixture.left, fixture.left)

    assert path.read_bytes() == before
    assert report["status"] == ("unknown" if legacy else "compatible")
    assert report["conditions"]["quality_constraints"]["status"] == (
        "unknown" if legacy else "compatible"
    )
