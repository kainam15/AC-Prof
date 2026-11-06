"""Independent server histogram arithmetic and offline report validation."""
from __future__ import annotations

import copy
import hashlib
import json
from contextlib import nullcontext

import pytest

from acprof.analysis.load_report import analyze_load_report
from acprof.analysis.server_metrics import parse_server_metrics, summarize_server_metrics
from acprof.cli.main import main
from acprof.plotting.load_report import write_load_report
from tests.execution.operations.test_server_metrics import command, exposition, metrics_server


def snapshot(text):
    return {"status": "ok", **parse_server_metrics(text, "fixture")}


def test_multiple_engines_are_differenced_before_aggregation():
    before = snapshot(exposition(10, 20) + exposition(20, 40, engine="1").replace("process_start_time_seconds 100\n", ""))
    after = snapshot(exposition(13, 26) + exposition(21, 42, engine="1").replace("process_start_time_seconds 100\n", ""))
    result = summarize_server_metrics(before, after, "fixture")
    assert result["status"] == "ok"
    assert result["phases"]["decode"] == {
        "status": "ok", "reason": None, "count": 4, "sum_s": 8.0, "mean_s": 2.0,
    }


@pytest.mark.parametrize(("change", "reason"), [
    (lambda text: text.replace("_count{", '_count{worker="new",'), "series_changed"),
    (lambda text: text.replace(" 13\n", " 9\n"), "counter_reset_or_creation_changed"),
    (lambda text: text.replace(" 26.0\n", " 19.0\n"), "counter_reset_or_creation_changed"),
    (lambda text: text.replace("process_start_time_seconds 100", "process_start_time_seconds 200"), "process_start_changed"),
    (lambda text: text.replace(" 13\n", " 10\n"), "inconsistent_count_sum_delta"),
])
def test_inconsistent_or_reset_snapshots_do_not_produce_means(change, reason):
    before = snapshot(exposition())
    raw = change(exposition(13, 26.0))
    if reason == "series_changed":
        raw = raw.replace("_sum{", '_sum{worker="new",')
    result = summarize_server_metrics(before, snapshot(raw), "fixture")
    assert result["status"] == "unavailable"
    assert result["phases"]["prefill"] == {
        "status": "unavailable", "reason": reason, "count": None, "sum_s": None, "mean_s": None,
    }


def test_unchanged_histogram_is_no_new_observations_not_zero_latency():
    result = summarize_server_metrics(snapshot(exposition()), snapshot(exposition()), "fixture")
    assert result["phases"]["prefill"]["reason"] == "no_new_observations"
    assert result["phases"]["prefill"]["count"] == 0
    assert result["phases"]["prefill"]["mean_s"] is None


def test_missing_family_and_other_models_are_not_filled_as_zero():
    text = "\n".join(line for line in exposition().splitlines() if "decode" not in line)
    before = snapshot(text + "\n" + exposition(model="other").replace("process_start_time_seconds 100\n", ""))
    after = snapshot(text.replace(" 10", " 11").replace(" 20.0", " 22.0"))
    after["process_start_time_s"] = before["process_start_time_s"]
    result = summarize_server_metrics(before, after, "fixture")
    assert result["status"] == "partial"
    assert result["phases"]["decode"]["reason"] == "metric_not_exposed"
    assert result["phases"]["prefill"]["count"] == 1


@pytest.mark.parametrize("raw", [
    "", exposition(model="other"), exposition().replace(" 10\n", " -1\n"),
    exposition().replace(" 10\n", " 1.5\n"), exposition().replace(" 20.0\n", " NaN\n"),
    exposition().replace(" 20.0\n", " +Inf\n"), exposition().replace(" 10\n", " 9007199254740993\n"),
    exposition().replace(" histogram", " gauge"),
    "\n".join(line for line in exposition().splitlines() if "_sum" not in line),
    exposition() + 'vllm:request_prefill_time_seconds_count{model_name="fixture",engine="0"} 10\n',
    exposition().replace('engine="0"', 'engine="0",engine="1"'),
    exposition().replace(" 20.0\n", " 20.0 123\n"),
], ids=["empty", "model", "negative", "fractional", "nan", "infinity", "inexact",
        "type", "incomplete", "duplicate", "duplicate-label", "timestamp"])
def test_invalid_exposition_is_not_accepted(raw):
    with pytest.raises(ValueError, match="server|Prometheus"):
        parse_server_metrics(raw, "fixture")


def test_escaped_labels_use_prometheus_grammar():
    model = 'quoted"slash\\name\nnext'
    result = parse_server_metrics(exposition(model=model), model)
    assert all(row["labels"]["model_name"] == model for row in result["series"])


@pytest.fixture(scope="module", name="metrics_report")
def saved_metrics_report(tmp_path_factory):
    output = tmp_path_factory.mktemp("server-metrics") / "load"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("acprof.cli.streaming_load.MeasurementLock", nullcontext)
        patch.setattr("acprof.cli.streaming_load.host_identity", lambda _root: {"hostname": "fixture"})
        with metrics_server() as (origin, _events, _auth):
            assert main(command(origin, output, "--server-metrics", "vllm")) == 0
    return json.loads((output / "load.json").read_text())


@pytest.mark.parametrize(("path", "value"), [
    (("server_metrics", "schema_version"), True),
    (("server_metrics", "scope"), "per-request GPU time"),
    (("server_metrics", "before", "sha256"), "invalid"),
    (("server_metrics", "before", "series", 0, "count"), True),
    (("server_metrics", "before", "series", 0, "labels", "model_name"), "other"),
    (("server_metrics", "before", "finished_s"), -1),
    (("server_metrics", "after", "started_s"), 0),
    (("server_metrics", "summary", "phases", "prefill", "mean_s"), 99),
    (("server_metrics", "summary", "phases", "prefill", "count"), True),
    (("identity", "server_metrics", "path"), "/other"),
    (("identity", "server_metrics", "model"), "other"),
    (("identity", "server_metrics", "schema_version"), True),
], ids=["version", "scope", "digest", "count", "labels", "clock", "order",
        "mean", "boolean-summary", "path", "model", "declaration-version"])
def test_tampered_metrics_are_not_published(metrics_report, tmp_path, path, value):
    report = copy.deepcopy(metrics_report)
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    report["identity_sha256"] = hashlib.sha256(json.dumps(report["identity"], sort_keys=True).encode()).hexdigest()
    source, destination = tmp_path / "load.json", tmp_path / "invalid.html"
    source.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="server"):
        write_load_report(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize("part", ["declaration", "observation"])
def test_unpaired_metrics_are_rejected(metrics_report, part):
    report = copy.deepcopy(metrics_report)
    (report["identity"] if part == "declaration" else report).pop("server_metrics")
    report["identity_sha256"] = hashlib.sha256(json.dumps(report["identity"], sort_keys=True).encode()).hexdigest()
    with pytest.raises(ValueError, match="server metrics"):
        analyze_load_report(report)


def test_one_engine_reset_is_not_hidden_by_another_engines_increase():
    before = snapshot(exposition(10, 20) + exposition(20, 40, engine="1").replace("process_start_time_seconds 100\n", ""))
    after = snapshot(exposition(1, 2) + exposition(40, 80, engine="1").replace("process_start_time_seconds 100\n", ""))
    result = summarize_server_metrics(before, after, "fixture")
    assert result["phases"]["prefill"]["mean_s"] is None
    assert result["phases"]["prefill"]["reason"] == "counter_reset_or_creation_changed"


def test_histogram_creation_change_is_a_reset_even_when_counters_increase():
    extra = 'vllm:request_prefill_time_seconds_created{model_name="fixture",engine="0"} '
    before = snapshot(exposition() + extra + "100\n")
    after = snapshot(exposition(13, 26.0) + extra + "200\n")
    result = summarize_server_metrics(before, after, "fixture")
    assert result["status"] == "partial"
    assert result["phases"]["prefill"]["reason"] == "counter_reset_or_creation_changed"
    assert result["phases"]["decode"]["mean_s"] == 2.0


def test_missing_process_start_evidence_does_not_invent_a_restart():
    before = snapshot(exposition().replace("process_start_time_seconds 100\n", ""))
    after = snapshot(exposition(13, 26.0).replace("process_start_time_seconds 100\n", ""))
    assert summarize_server_metrics(before, after, "fixture")["phases"]["prefill"]["mean_s"] == 2.0


def test_zero_duration_phase_is_a_valid_observation():
    result = summarize_server_metrics(snapshot(exposition(10, 0.0)), snapshot(exposition(13, 0.0)), "fixture")
    assert result["phases"]["prefill"]["mean_s"] == 0
    assert result["phases"]["prefill"]["status"] == "ok"


def test_excess_series_are_rejected(monkeypatch):
    monkeypatch.setattr("acprof.analysis.server_metrics.MAX_METRICS_SERIES", 2)
    with pytest.raises(ValueError, match="Prometheus"):
        parse_server_metrics(exposition(), "fixture")


def test_overflowing_cross_engine_sum_is_unavailable():
    before = snapshot(exposition(0, 0.0) + exposition(0, 0.0, engine="1").replace("process_start_time_seconds 100\n", ""))
    after = snapshot(exposition(1, 1e308) + exposition(1, 1e308, engine="1").replace("process_start_time_seconds 100\n", ""))
    assert summarize_server_metrics(before, after, "fixture")["phases"]["prefill"]["reason"] == "delta_overflow"


def test_scrape_interval_must_enclose_the_load_duration(metrics_report):
    report = copy.deepcopy(metrics_report)
    report["server_metrics"]["after"]["started_s"] = (
        report["server_metrics"]["before"]["finished_s"] + report["result"]["duration_s"] / 2
    )
    with pytest.raises(ValueError, match="duration"):
        analyze_load_report(report)


def make_metrics_sweep(root, metrics_report):
    from tests.support.load_sweep_fixtures import change_point, make_sweep

    root = make_sweep(root)
    def add_metrics(point):
        point["identity"]["server_metrics"] = copy.deepcopy(metrics_report["identity"]["server_metrics"])
        point["server_metrics"] = copy.deepcopy(metrics_report["server_metrics"])
        point["server_metrics"]["before"].update(started_s=0, finished_s=0.1)
        point["server_metrics"]["after"].update(started_s=10, finished_s=10.1)
    for index in (1, 2, 3):
        change_point(root, index, add_metrics)
    return root


def test_sweep_conditions_separate_observed_and_unobserved_points(metrics_report, tmp_path):
    from acprof.analysis.load_sweep import read_load_sweep
    from tests.support.load_sweep_fixtures import change_point

    root = make_metrics_sweep(tmp_path / "sweep", metrics_report)
    def remove_metrics(point):
        del point["identity"]["server_metrics"]
        del point["server_metrics"]
    change_point(root, 2, remove_metrics)
    report = read_load_sweep(root).payload
    assert len(report["conditions"]) == 2
    assert report["valid_points"] == 3
    assert "server_metrics" not in report["points"][1]
