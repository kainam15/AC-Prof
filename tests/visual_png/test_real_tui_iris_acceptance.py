"""Real acceptance evidence cannot pass with an active or unrelated attempt."""
import csv
from copy import deepcopy

import pytest

from acprof.config import CSV_FIELDS
from acprof.result_store.initialization import initialize_results
from acprof.results.layers import publish_result_layers
from acprof.results.layout import ArtifactLayout
from acprof.tui.run.stages import Stage
from scripts.accept_real_tui_iris import (
    check_cancelled_attempt,
    check_closed_attempt,
    check_result_owner,
    execute,
)


@pytest.fixture
def cancelled_attempt(tmp_path):
    return {
        "number": 1, "returncode": 1, "child_returncode": 1,
        "lifecycle_released": True, "coordinator_idle": True, "fresh_token": True,
        "stop_requested": True, "snapshot": {"stage": Stage.STOPPED},
        "result_root": str(tmp_path),
        "run_result": {
            "complete": False, "result_csv": "", "completed_cases": 0,
            "belongs_to_attempt": False,
        },
    }


@pytest.mark.parametrize(("field", "value"), [
    ("child_returncode", None), ("child_returncode", 0),
    ("lifecycle_released", False), ("coordinator_idle", False), ("fresh_token", False),
])
def test_live_or_unreleased_attempt_cannot_pass(cancelled_attempt, field, value):
    cancelled_attempt[field] = value
    with pytest.raises(RuntimeError, match="exit and release"):
        check_closed_attempt(cancelled_attempt)


def test_preparation_cancel_rejects_measurement_from_its_own_attempt(cancelled_attempt):
    later_measurement = [{"attempt": 2, "measurement_active": True}]
    check_cancelled_attempt(cancelled_attempt, later_measurement)
    stages = [*later_measurement, {"attempt": 1, "measurement_active": True}]
    with pytest.raises(RuntimeError, match="before formal measurement"):
        check_cancelled_attempt(cancelled_attempt, stages)


def test_preparation_cancel_cannot_be_reported_as_completed(cancelled_attempt):
    cancelled_attempt["run_result"]["complete"] = True
    with pytest.raises(RuntimeError, match="before formal measurement"):
        check_cancelled_attempt(cancelled_attempt, [])


@pytest.mark.parametrize("with_measurement", [False, True])
def test_retained_preparation_result_must_have_no_measurement_rows(
    cancelled_attempt, tmp_path, with_measurement,
):
    source = tmp_path / "source.csv"
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        if with_measurement:
            row = dict.fromkeys(CSV_FIELDS, "nan")
            row.update(cpu_cores="1", mem_cap_gb="1", gpu_mode="off", input_scale="1",
                       warmup="0", repeat_idx="0", status="ok")
            writer.writerow(row)
    if with_measurement:
        publish_result_layers(source)
    else:
        initialize_results(ArtifactLayout.for_new_run(tmp_path))
    cancelled_attempt["run_result"].update(
        result_csv=str(tmp_path / "result_layers.json"), belongs_to_attempt=True,
    )
    if with_measurement:
        with pytest.raises(RuntimeError, match="persisted measurement rows"):
            check_cancelled_attempt(cancelled_attempt, [])
    else:
        check_cancelled_attempt(cancelled_attempt, [])
        assert cancelled_attempt["cancelled_measurement_rows"] == 0


def test_selected_result_requires_successful_attempt_and_its_directory(cancelled_attempt, tmp_path):
    current = str(tmp_path / "tasks" / "iris" / "results.csv")
    attempt = deepcopy(cancelled_attempt)
    attempt["run_result"].update(complete=True, belongs_to_attempt=True, result_csv=current)
    check_result_owner(current, attempt)
    with pytest.raises(RuntimeError, match="successful attempt"):
        check_result_owner(str(tmp_path / "previous.csv"), attempt)
    foreign = str(tmp_path.parent / "foreign.csv")
    attempt["run_result"]["result_csv"] = foreign
    with pytest.raises(RuntimeError, match="successful attempt"):
        check_result_owner(foreign, attempt)
    attempt["run_result"].update(result_csv=current, belongs_to_attempt=False)
    with pytest.raises(RuntimeError, match="successful attempt"):
        check_result_owner(current, attempt)


async def test_invalid_scenario_is_rejected_before_external_work(tmp_path):
    with pytest.raises(ValueError, match="unknown acceptance scenario"):
        await execute(tmp_path / "evidence", scenario="unknown")
    assert not (tmp_path / "evidence").exists()
