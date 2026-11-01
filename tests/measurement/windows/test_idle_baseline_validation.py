"""Formal energy quality excludes warmup without treating missing evidence as stable."""
import csv
import json

import pytest

from acprof.host.measurement.energy_validation import (
    EnergyProfilingError,
    check_case_cpu_idle_power_stable,
    check_case_gpu_idle_power_stable,
)
from acprof.quality import summarize_quality
from acprof.results.layout import case_sidecar


@pytest.fixture(params=["cpu", "gpu"])
def device_check(request):
    device = request.param
    check = (check_case_cpu_idle_power_stable if device == "cpu"
             else check_case_gpu_idle_power_stable)
    return device, check


def write_case(tmp_path, values):
    path = tmp_path / "case.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "warmup", "status", "gpu_mode", "cpu_idle_power_w", "gpu_idle_power_w",
        ])
        writer.writeheader()
        for value, changes in values:
            writer.writerow({"warmup": 0, "status": "ok", "gpu_mode": "on",
                             "cpu_idle_power_w": value, "gpu_idle_power_w": value,
                             **changes})
    return path


def quality_checks(path):
    return json.loads(case_sidecar(path, "quality_checks").read_text())["checks"]


def test_warmup_outlier_does_not_change_formal_quality(tmp_path, device_check, capsys):
    _, check = device_check
    path = write_case(tmp_path, [(100, {"warmup": 1}), (20, {}), (20.1, {})])
    original = path.read_bytes()

    check(str(path))

    output = capsys.readouterr()
    assert "case check passed" in output.out
    assert "formal_rows=2, excluded_warmup_rows=1" in output.out
    assert output.err == ""
    assert not case_sidecar(path, "quality_checks").exists()
    assert path.read_bytes() == original


def test_formal_drift_keeps_threshold_and_scoped_raw_evidence(tmp_path, device_check):
    device, check = device_check
    path = write_case(tmp_path, [(22, {"warmup": 1}), (20, {}), (24, {})])
    original = path.read_bytes()

    check(str(path))

    record, = quality_checks(path)
    assert record["code"] == f"{device}_idle_baseline_unstable"
    assert record["observed"] == pytest.approx(4 / 22)
    assert record["threshold"] == 0.05
    assert record["evidence"] == {
        "source": str(path), "metric": f"{device}_idle_power_w", "scope": "formal",
        "values_w": [20, 24], "sample_count": 2, "excluded_warmup_rows": 1,
        "formula": "(max-min)/mean",
    }
    assert summarize_quality([record])["quality_status"] == "unknown"
    assert path.read_bytes() == original


@pytest.mark.parametrize("formal_count", [0, 1])
def test_warmup_cannot_fill_missing_formal_samples(
    tmp_path, device_check, capsys, formal_count,
):
    device, check = device_check
    path = write_case(tmp_path, [(20, {"warmup": 1})] + [(20, {})] * formal_count)

    check(str(path))

    record, = quality_checks(path)
    assert record["code"] == f"{device}_idle_baseline_insufficient_samples"
    assert (record["observed"], record["threshold"]) == (formal_count, 2)
    assert record["evidence"]["values_w"] == [20] * formal_count
    assert record["evidence"]["excluded_warmup_rows"] == 1
    quality = summarize_quality([record])
    assert quality["quality_status"] == "unknown"
    assert quality["auto_selection_eligible"] is False
    output = capsys.readouterr()
    assert "passed" not in output.out
    assert "stability is unknown" in output.err


@pytest.mark.parametrize(("value", "changes"), [
    ("nan", {}), (0, {}), (-1, {}), ("inf", {}),
    (20, {"warmup": ""}), (20, {"warmup": "invalid"}), (20, {"warmup": 2}),
])
def test_invalid_formal_rows_still_fail(tmp_path, device_check, value, changes):
    _, check = device_check
    path = write_case(tmp_path, [(20, {"warmup": 1}), (value, changes)])
    with pytest.raises(EnergyProfilingError, match="invalid_rows=1"):
        check(str(path))
    assert not case_sidecar(path, "quality_checks").exists()


def test_incomplete_case_skips_error_rows_without_using_warmup(tmp_path, device_check):
    device, check = device_check
    path = write_case(tmp_path, [
        (20, {"warmup": 1}), ("nan", {"status": "error"}), (20, {}),
    ])
    check(str(path), ignore_error_rows=True)
    record, = quality_checks(path)
    assert record["code"] == f"{device}_idle_baseline_insufficient_samples"
    assert record["observed"] == 1


def test_gpu_off_has_no_gpu_stability_requirement(tmp_path, capsys):
    path = write_case(tmp_path, [("nan", {"gpu_mode": "off", "warmup": 1})])
    check_case_gpu_idle_power_stable(str(path))
    assert capsys.readouterr().out == ""
    assert not case_sidecar(path, "quality_checks").exists()


def test_cpu_and_gpu_quality_append_without_overwriting_evidence(tmp_path):
    path = write_case(tmp_path, [(20, {}), (24, {})])
    check_case_cpu_idle_power_stable(str(path))
    check_case_gpu_idle_power_stable(str(path))
    records = quality_checks(path)
    assert {record["code"] for record in records} == {
        "cpu_idle_baseline_unstable", "gpu_idle_baseline_unstable",
    }
    assert summarize_quality(records)["auto_selection_eligible"] is False


def test_exact_threshold_still_warns(tmp_path):
    path = write_case(tmp_path, [(19.5, {}), (20.5, {})])
    check_case_cpu_idle_power_stable(str(path))
    record, = quality_checks(path)
    assert record["observed"] == record["threshold"] == 0.05
