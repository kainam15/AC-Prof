"""HTML statistics reuse validated evidence and preserve within-run scope."""
import csv
import json
import os
from pathlib import Path

import pytest

from acprof.analysis.model import load_analysis
from acprof.analysis.statistics_report import build_statistics_report, save_unique_statistics_report
from acprof.plotting.report import report_payload, write_report
from acprof.results.layout import ArtifactLayout


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "result_all.csv"
    write_rows(path, [row(index, value) for index, value in enumerate((1, 2, 3))])
    return path


def row(index, value, **extra):
    return {"cpu_cores": "2", "mem_cap_gb": "4", "gpu_mode": "off", "input_scale": "32",
            "repeat_idx": str(index), "warmup": "0", "status": "ok", "latency_app_s": str(value), **extra}


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def statistics(source):
    return report_payload(load_analysis([source]))["window_statistics"][0]


def save_cache(source, **options):
    payload = build_statistics_report(source, **options)
    path, _ = save_unique_statistics_report(ArtifactLayout.from_csv(source).path("analysis"), payload)
    return path


def test_html_computes_missing_statistics_without_changing_source(source):
    original = source.read_bytes()
    result = statistics(source)
    group = next(group for group in result["groups"] if group["metric"] == "latency_app_s")
    assert result["origin"] == "computed"
    assert group["mean"] == 2
    assert group["std"] == 1
    assert (group["ci_low"], group["ci_high"]) == (1, 3)
    assert group["n_windows"] == 3
    assert result["parameters"]["confidence"] == .95
    assert result["parameters"]["resampling_unit"] == "csv_request_window"
    assert source.read_bytes() == original
    assert not source.parent.joinpath("analysis").exists()


def test_fresh_json_reuses_intervals_and_records_actual_bootstrap_options(source, monkeypatch):
    cached = save_cache(source, metrics=["latency_app_s"], resamples=137, seed=19)
    expected = json.loads(cached.read_text())
    monkeypatch.setattr("acprof.analysis.uncertainty.bootstrap_mean_interval",
                        lambda *_args, **_kwargs: pytest.fail("fresh JSON must avoid Bootstrap"))
    result = statistics(source)
    assert result["origin"] == "cached"
    assert result["report_path"] == str(cached)
    assert result["parameters"]["resamples"] == 137
    assert result["parameters"]["seed"] == 19
    assert result["groups"][0]["ci_low"] == expected["groups"][0]["ci_low"]
    assert result["groups"][0]["config_id"] == load_analysis([source]).configs[0]["config_id"]


def test_equal_size_and_mtime_csv_change_invalidates_json(source):
    save_cache(source)
    before = source.stat()
    source.write_text(source.read_text().replace(",ok,3", ",ok,9"))
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    result = statistics(source)
    assert result["origin"] == "computed"
    assert result["groups"][0]["mean"] == 4
    assert result["cache_diagnostics"][0]["reason"] == "source_changed"


@pytest.mark.parametrize("mutation", [
    lambda data: data.update(confidence=.9),
    lambda data: data.update(schema_version=999),
    lambda data: data.update(method="BCa"),
    lambda data: data["groups"][0].update(ci_low=10, ci_high=-10),
    lambda data: data["groups"][0].update(ci_low=10, ci_high=20),
    lambda data: data["groups"][0].update(ci_low=.25, ci_high=.75),
    lambda data: data["groups"][0].update(ci_low=0, ci_high=3),
    lambda data: data["groups"][0].update(ci_low=1, ci_high=4),
    lambda data: data["groups"][0].update(mean=999),
    lambda data: data["groups"][0].pop("ci_low"),
    lambda data: data["groups"][0].update(metric=["latency_app_s"]),
    lambda data: data["groups"].clear(),
], ids=["confidence", "schema", "method", "interval", "above-observations", "below-observations",
        "lower-outside", "upper-outside", "summary", "missing-bound", "invalid-metric", "missing-groups"])
def test_incompatible_or_corrupt_json_is_recomputed_with_a_visible_reason(source, mutation):
    cached = save_cache(source)
    payload = json.loads(cached.read_text())
    mutation(payload)
    cached.write_text(json.dumps(payload))
    result = statistics(source)
    assert result["origin"] == "computed"
    assert result["groups"][0]["mean"] == 2
    assert result["cache_diagnostics"]


@pytest.mark.parametrize("excluded", [
    {"status": "error", "error": "fixture failure"}, {"warmup": "1"},
    {"result_origin": "inferred_not_measured"},
], ids=["failed", "warmup", "inferred"])
def test_cached_interval_bounds_exclude_non_measured_rows(source, excluded):
    write_rows(source, [row(index, value) for index, value in enumerate((1, 2, 3))]
               + [row(3, 100, **excluded)])
    cached = save_cache(source, metrics=["latency_app_s"])
    payload = json.loads(cached.read_text())
    payload["groups"][0].update(ci_low=10, ci_high=20)
    cached.write_text(json.dumps(payload))
    original_csv, original_cache = source.read_bytes(), cached.read_bytes()

    result = statistics(source)

    assert result["origin"] == "computed"
    assert result["cache_diagnostics"][0]["reason"] == "interval_outside_observed_range"
    group = next(item for item in result["groups"] if item["metric"] == "latency_app_s")
    assert (group["ci_low"], group["ci_high"]) == (1, 3)
    assert source.read_bytes() == original_csv
    assert cached.read_bytes() == original_cache


@pytest.mark.parametrize("separate", ["configuration", "metric"])
def test_cached_interval_bounds_do_not_mix_configurations_or_metrics(source, separate):
    metric = "container_attributed_energy_eff_j"
    rows = [row(index, value, **{metric: str(value * 100)})
            for index, value in enumerate((1, 2, 3))]
    if separate == "configuration":
        rows += [row(index, value * 100, cpu_cores="4") for index, value in enumerate((1, 2, 3))]
    write_rows(source, rows)
    cached = save_cache(source, metrics=["latency_app_s", metric])
    payload = json.loads(cached.read_text())
    target = next(group for group in payload["groups"]
                  if group["metric"] == "latency_app_s" and group["cpu_cores"] == 2)
    target.update(ci_low=10, ci_high=20)
    cached.write_text(json.dumps(payload))

    result = statistics(source)

    assert result["origin"] == "computed"
    assert result["cache_diagnostics"][0]["reason"] == "interval_outside_observed_range"
    group = next(item for item in result["groups"]
                 if item["metric"] == "latency_app_s" and item["cpu_cores"] == 2)
    assert (group["ci_low"], group["ci_high"]) == (1, 3)


@pytest.mark.parametrize("value", [.1, -.1, 1e-200])
def test_constant_interval_roundoff_still_reuses_cache(source, value, monkeypatch):
    metric = "container_attributed_energy_eff_j"
    write_rows(source, [row(index, 1, **{metric: str(value)}) for index in range(3)])
    cached = save_cache(source, metrics=[metric], resamples=17)
    expected = json.loads(cached.read_text())["groups"][0]
    monkeypatch.setattr("acprof.analysis.uncertainty.bootstrap_mean_interval",
                        lambda *_args, **_kwargs: pytest.fail("valid interval must retain cache reuse"))

    result = statistics(source)

    assert result["origin"] == "cached"
    assert result["cache_diagnostics"] == []
    group = result["groups"][0]
    assert (group["ci_low"], group["ci_high"]) == (expected["ci_low"], expected["ci_high"])
    assert group["mean"] == pytest.approx(value, rel=1e-12, abs=0)


def test_tiny_observations_do_not_admit_large_absolute_tolerance(source):
    write_rows(source, [row(index, value * 1e-200) for index, value in enumerate((1, 2, 3))])
    cached = save_cache(source, metrics=["latency_app_s"])
    payload = json.loads(cached.read_text())
    payload["groups"][0].update(ci_low=0, ci_high=1e-15)
    cached.write_text(json.dumps(payload))

    result = statistics(source)

    assert result["origin"] == "computed"
    assert result["cache_diagnostics"][0]["reason"] == "interval_outside_observed_range"


def test_quality_change_invalidates_cache_and_keeps_current_evidence(source):
    save_cache(source)
    (source.parent / "quality_checks.json").write_text(json.dumps({"schema_version": 1, "checks": []}))
    result = statistics(source)
    assert result["origin"] == "computed"
    assert result["audit"]["quality_status"] == "passed"
    assert result["audit"]["measurement_status"] == "unknown"
    assert result["cache_diagnostics"][0]["reason"] == "audit_changed"


def test_missing_failed_warmup_and_inferred_windows_remain_distinct(source):
    write_rows(source, [row(0, 1), row(1, "nan"), row(2, 7, status="error"),
                        row(3, 8, status="warn"), row(4, 9, result_origin="inferred_not_measured"),
                        row(0, 100, warmup="1")])
    result = statistics(source)
    group = result["groups"][0]
    assert (group["mean"], group["std"], group["ci_low"], group["ci_high"]) == (1, None, None, None)
    assert (group["n_windows"], group["missing_windows"]) == (1, 1)
    assert group["reason"] == "insufficient_windows"
    assert sum(group["missing_reasons"].values()) == 1
    assert group["excluded_windows"] == {"warmup": 1, "error": 1, "warn": 1, "inferred_not_measured": 1}


def test_failed_only_configuration_has_visible_unavailable_statistics(source):
    write_rows(source, [row(0, "nan", status="error")])
    result = statistics(source)
    assert result["groups"]
    assert all(group["mean"] is None and group["n_windows"] == 0 for group in result["groups"])
    assert result["groups"][0]["reason"] == "no_successful_windows"
    assert result["groups"][0]["excluded_windows"] == {"error": 1}


def test_statistics_do_not_merge_distinct_concurrency_configurations(source):
    write_rows(source, [row(i, i + 1, concurrency="1") for i in range(3)] +
               [row(i, 10, concurrency="2") for i in range(3, 6)])
    save_cache(source)
    result = statistics(source)
    groups = [group for group in result["groups"] if group["metric"] == "latency_app_s"]
    assert {group["mean"] for group in groups} == {2, 10}
    assert len({group["config_id"] for group in groups}) == 2
    assert result["origin"] == "computed"


def test_cache_permission_error_is_not_silently_recomputed(source, monkeypatch):
    cached = save_cache(source)
    original = Path.open

    def denied(path, *args, **kwargs):
        if path == cached:
            raise PermissionError("statistics cache denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(PermissionError, match="statistics cache denied"):
        statistics(source)


@pytest.mark.parametrize("artifact", ["result_all.csv", "quality_checks.json"])
def test_sources_changed_during_statistics_never_publish_html(source, monkeypatch, artifact):
    from acprof.analysis import uncertainty
    original = uncertainty.bootstrap_mean_interval

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        target = source.parent / artifact
        if target == source:
            target.write_text(target.read_text() + "\n")
        else:
            target.write_text(json.dumps({"schema_version": 1, "checks": []}))
        return result

    model = load_analysis([source])
    monkeypatch.setattr(uncertainty, "bootstrap_mean_interval", mutate)
    output = source.parent / "report.html"
    with pytest.raises(ValueError, match="changed|变化"):
        write_report(model, output)
    assert not output.exists()


@pytest.mark.parametrize("directory", ["plots/analysis", "analysis", "."])
def test_layered_results_reuse_json_without_consuming_uncommitted_tail(tmp_path, monkeypatch, directory):
    from acprof.result_store.writer import ResultWriter
    from tests.results.storage.test_incremental_results import initialized, receipt

    layout = initialized(tmp_path)
    with ResultWriter(tmp_path) as writer:
        for index in range(3):
            writer.append_window(receipt(tmp_path, index))
    cached, _ = save_unique_statistics_report(tmp_path / directory, build_statistics_report(layout.result_csv))
    # A committed prefix, rather than all bytes currently in a layer, is the source.
    layer = layout.layer_csv("performance")
    with layer.open("ab") as stream:
        stream.write(b"uncommitted incomplete window\xff")
    original = layer.read_bytes()
    monkeypatch.setattr("acprof.analysis.uncertainty.bootstrap_mean_interval",
                        lambda *_args, **_kwargs: pytest.fail("committed data has not changed"))
    result = statistics(layout.result_csv)
    assert result["origin"] == "cached"
    assert result["report_path"] == str(cached)
    assert result["groups"][0]["mean"] == pytest.approx(1.1)
    assert result["groups"][0]["n_windows"] == 3
    assert layer.read_bytes() == original


def test_broken_newer_json_does_not_hide_an_older_valid_report(source):
    cached = save_cache(source)
    corrupt = cached.with_name("window-statistics-newer.json")
    corrupt.write_text("{broken")
    result = statistics(source)
    assert result["origin"] == "cached"
    assert result["report_path"] == str(cached)
    assert result["cache_diagnostics"][0]["path"] == str(corrupt)


def test_block_bootstrap_settings_and_unavailable_reason_are_preserved(source):
    write_rows(source, [row(index, index) for index in (0, 1, 2, 4, 5, 6)])
    cached = save_cache(source, block_size=2, resamples=101, seed=7)
    result = statistics(source)
    assert result["origin"] == "cached"
    assert result["report_path"] == str(cached)
    assert result["parameters"]["block_size"] == 2
    assert result["groups"][0]["reason"] == "nonconsecutive_windows"
    assert result["groups"][0]["ci_low"] is None


def test_each_source_uses_its_own_statistics_and_quality(source, tmp_path):
    save_cache(source)
    other = tmp_path / "other" / "result_all.csv"
    other.parent.mkdir()
    write_rows(other, [row(index, value) for index, value in enumerate((10, 20, 30))])
    reports = report_payload(load_analysis([source, other]))["window_statistics"]
    assert [report["origin"] for report in reports] == ["cached", "computed"]
    assert [report["groups"][0]["mean"] for report in reports] == [2, 20]
    assert reports[0]["groups"][0]["config_id"] != reports[1]["groups"][0]["config_id"]


def test_successful_observed_windows_do_not_imply_complete_experiment(source):
    (source.parent / "run_state.json").write_text(json.dumps({
        "status": "complete", "options": {"cpus": "2", "mems": "4", "gpus": "off", "warmup": 0, "repeat": 4},
        "runtime": {"planned": {"scales": [32]}},
    }))
    result = statistics(source)
    assert result["groups"][0]["n_windows"] == 3
    assert result["audit"]["coverage"] == {"expected": 4, "actual": 3, "missing": 1, "unexpected": 0}
    assert result["audit"]["run_status"] == "complete"
    assert result["audit"]["measurement_status"] == "incomplete"


def test_metadata_replaced_after_analysis_is_rejected_before_json_reuse(source):
    save_cache(source)
    model = load_analysis([source])
    (source.parent / "quality_checks.json").write_text(json.dumps({"schema_version": 1, "checks": []}))
    with pytest.raises(ValueError, match="source changed"):
        report_payload(model)
