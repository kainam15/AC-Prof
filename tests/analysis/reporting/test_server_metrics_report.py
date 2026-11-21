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
from tests.execution.operations.test_server_metrics import (
    command,
    exposition,
    metrics_server,
    sglang_exposition,
)


def snapshot(text):
    return {"status": "ok", **parse_server_metrics(text, "fixture")}


def test_sglang_stage_population_changes_do_not_discard_other_timings():
    def observed(count, exclude):
        text = "\n".join(line for line in sglang_exposition(count).splitlines() if exclude not in line)
        return {"status": "ok", **parse_server_metrics(text, "fixture", kind="sglang")}

    before, after = observed(2, 'stage="foreign"'), observed(5, 'stage="prefill_forward"')
    summary = summarize_server_metrics(before, after, "fixture", kind="sglang")
    assert summary["status"] == "partial"
    assert summary["phases"]["stage:prefill_forward"]["reason"] == "series_changed"
    assert summary["phases"]["stage:decode_loop"]["mean_s"] == 0.75
    assert summary["phases"]["itl"]["count"] == 12
    without_stages = summarize_server_metrics(observed(2, "per_stage_req"), observed(5, "per_stage_req"),
                                              "fixture", kind="sglang")
    assert without_stages["phases"]["stage"]["reason"] == "metric_not_exposed"
    assert without_stages["phases"]["stage"]["mean_s"] is None


@pytest.mark.parametrize(("raw", "kind"), [(exposition(), "sglang"), (sglang_exposition(3), "vllm")])
def test_requested_exporter_never_falls_back_to_another_namespace(raw, kind):
    with pytest.raises(ValueError, match="model_name"):
        parse_server_metrics(raw, "fixture", kind=kind)


@pytest.mark.parametrize("bound", ["0.5", "+Inf"])
def test_bucket_only_worker_cannot_be_omitted_from_the_mean(bound):
    orphan = f'vllm:request_queue_time_seconds_bucket{{model_name="fixture",engine="1",le="{bound}"}} 9\n'
    with pytest.raises(ValueError, match="count/sum"):
        parse_server_metrics(exposition() + orphan, "fixture")


def test_bucket_only_sglang_stage_cannot_disappear_behind_another_stage():
    text = "\n".join(line for line in sglang_exposition(2).splitlines()
                     if 'stage="decode_loop"' not in line or "_bucket{" in line)
    with pytest.raises(ValueError, match="count/sum"):
        parse_server_metrics(text, "fixture", kind="sglang")


def test_infinite_bucket_must_agree_with_count_for_its_worker():
    family = "vllm:request_queue_time_seconds"
    raw = exposition() + f'{family}_bucket{{engine="0",le="+Inf",model_name="fixture"}} 11\n'
    with pytest.raises(ValueError, match=r"\+Inf"):
        parse_server_metrics(raw, "fixture")


@pytest.mark.parametrize("bounds", [[], [("0.5", 2)], [("0.5", 2), ("1", 4), ("+Inf", 10)]])
def test_buckets_check_population_without_consuming_series_budget_or_changing_evidence(monkeypatch, bounds):
    monkeypatch.setattr("acprof.analysis.server_metrics.MAX_METRICS_SERIES", 1)
    family = "vllm:request_queue_time_seconds"
    tags = 'model_name="fixture",engine="0"'
    text = "\n".join([
        f"# TYPE {family} histogram",
        *(f'{family}_bucket{{le="{bound}",{tags}}} {value}' for bound, value in bounds),
        f"{family}_sum{{{tags}}} 20", f"{family}_count{{{tags}}} 10",
        f'{family}_bucket{{model_name="other",le="+Inf"}} 99',
        'sglang:queue_time_seconds_bucket{model_name="fixture",le="+Inf"} 100',
    ])
    assert parse_server_metrics(text, "fixture")["series"] == [{
        "phase": "queue", "labels": {"engine": "0", "model_name": "fixture"},
        "count": 10, "sum_s": 20.0, "created_s": None,
    }]


def test_sglang_per_stage_histogram_requires_a_named_stage():
    with pytest.raises(ValueError, match="stage"):
        parse_server_metrics(sglang_exposition(3).replace(',stage="decode_loop"', ''), "fixture", kind="sglang")


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


def collect_report(output, kind):
    body = (lambda state: sglang_exposition(state["requests"]).encode()) if kind == "sglang" else None
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("acprof.cli.streaming_load.MeasurementLock", nullcontext)
        patch.setattr("acprof.cli.streaming_load.host_identity", lambda _root: {"hostname": "fixture"})
        with metrics_server(metrics_body=body) as (origin, _events, _auth):
            assert main(command(origin, output, "--server-metrics", kind)) == 0
    return json.loads((output / "load.json").read_text())


@pytest.fixture(scope="module", name="metrics_report")
def saved_metrics_report(tmp_path_factory, request):
    return collect_report(tmp_path_factory.mktemp("server-metrics") / "load", getattr(request, "param", "vllm"))


@pytest.mark.parametrize(("path", "value"), [
    (("server_metrics", "kind"), "sglang"),
    (("identity", "server_metrics", "kind"), "sglang"),
    (("identity", "server_metrics", "kind"), ["vllm"]),
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
], ids=["observation-exporter", "declaration-exporter", "malformed-exporter",
        "version", "scope", "digest", "count", "labels", "clock", "order",
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


def test_relabeling_both_exporters_cannot_reuse_backend_specific_snapshots(metrics_report):
    report = copy.deepcopy(metrics_report)
    report["identity"]["server_metrics"]["kind"] = report["server_metrics"]["kind"] = "sglang"
    report["identity_sha256"] = hashlib.sha256(json.dumps(report["identity"], sort_keys=True).encode()).hexdigest()
    with pytest.raises(ValueError, match="server metric phase"):
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


def make_metrics_sweep(root, metrics_report, *, second_report=None):
    from tests.support.load_sweep_fixtures import change_point, make_sweep

    root = make_sweep(root)
    def add_metrics(point, source):
        point["identity"]["server_metrics"] = copy.deepcopy(source["identity"]["server_metrics"])
        point["server_metrics"] = copy.deepcopy(source["server_metrics"])
        point["server_metrics"]["before"].update(started_s=0, finished_s=0.1)
        point["server_metrics"]["after"].update(started_s=10, finished_s=10.1)
    for index in (1, 2, 3):
        source = second_report if second_report is not None and index == 2 else metrics_report
        change_point(root, index, lambda point, source=source: add_metrics(point, source))
    return root


def test_sweep_does_not_pool_metrics_from_different_exporters(metrics_report, tmp_path):
    from acprof.analysis.load_sweep import read_load_sweep

    sglang_report = collect_report(tmp_path / "sglang", "sglang")
    root = make_metrics_sweep(tmp_path / "sweep", metrics_report, second_report=sglang_report)
    sweep = read_load_sweep(root).payload
    assert sweep["valid_points"] == 3
    assert {group["identity"]["server_metrics"]["kind"] for group in sweep["conditions"]} == {"vllm", "sglang"}
    first, second, third = sweep["points"]
    assert first["condition_id"] == third["condition_id"] != second["condition_id"]
    assert second["server_metrics"]["phases"]["stage:decode_loop"]["count"] == 6


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
