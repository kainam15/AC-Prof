"""Browser inputs preserve Python's window selection, identity and publication guard."""
import gc
import hashlib
import json
import tracemalloc
from collections import defaultdict
from unittest.mock import patch

import pytest

from acprof.analysis.model import load_analysis
from acprof.analysis.report_statistics import window_input_chunks, write_window_inputs
from acprof.analysis.statistics_report import build_statistics_report
from acprof.analysis.uncertainty import bootstrap_mean_interval
from acprof.plotting.report import report_payload, write_report
from acprof.results.measurement_plan import decimal_coordinate, decimal_key_text
from acprof.results.metrics import METRICS
from tests.analysis.reporting.test_report_statistics import row, save_cache, write_rows


def test_window_input_stream_does_not_retain_arrays_for_all_metrics(tmp_path):
    metrics = [name for name, metric in METRICS.items() if metric.kind == 'number'
               and metric.window in {'request_window', 'matched_control'}][:32]
    source = tmp_path / 'result_all.csv'
    write_rows(source, [row(index, 1, **{name: str(index % 17 + 1) for name in metrics})
                        for index in range(1024)])
    model = load_analysis([source])
    configs = defaultdict(list)
    for entry in model.raw_rows:
        configs[entry['config_id']].append(entry)
    peaks, sizes = [], []
    for selected in (metrics[:1], metrics):
        # Keep the source width fixed and discard encoded chunks. This measures
        # stream preparation, not loading the model or retaining its full output.
        gc.collect()
        tracemalloc.start()
        try:
            size = sum(len(chunk) for chunk in window_input_chunks(configs, model.sources[0], selected))
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
        sizes.append(size)
    assert sizes[1] > 20 * sizes[0]
    assert peaks[1] < 1.5 * peaks[0], peaks


@pytest.mark.parametrize('metric_count', [1, 16])
def test_report_classifies_window_identity_independently_of_metric_count(tmp_path, metric_count):
    from acprof.analysis import report_statistics

    metrics = [name for name, metric in METRICS.items() if metric.kind == 'number'
               and metric.window in {'request_window', 'matched_control'}][:metric_count]
    source = tmp_path / 'result_all.csv'
    rows = [row(index, .125, cpu_cores='37.25', **{name: '.125' for name in metrics}) for index in range(8)]
    write_rows(source, rows)
    model = load_analysis([source])
    parse = report_statistics.number
    identity_reads = []

    def observe(value):
        if value == '37.25':
            identity_reads.append(value)
        return parse(value)

    with patch('acprof.analysis.report_statistics.number', observe):
        payload = report_payload(model, statistics_options={'metrics': metrics, 'resamples': 1})

    # Initial estimates and browser inputs may each classify the windows.
    # Adding metrics must not repeat that classification for every metric.
    assert len(identity_reads) <= 4 * len(rows)
    report = payload['window_statistics'][0]
    assert all(group['cpu_cores'] == 37.25 and group['mean'] == .125 for group in report['groups'])
    assert all(group['repeat_indices'] == list(range(8)) for group in report['recalculation']['groups'])


def test_invalid_window_identity_stays_unavailable_without_sorting_bad_repeat(tmp_path):
    source = tmp_path / 'result_all.csv'
    write_rows(source, [row('invalid', 2), row(0, 1)])

    report = report_payload(load_analysis([source]))['window_statistics'][0]

    assert all(group['reason'] == 'invalid_window_identity' for group in report['groups'])
    assert all(group['values'] == [] and group['repeat_indices'] == []
               for group in report['recalculation']['groups'])


@pytest.mark.parametrize(("start", "step", "reason"), [
    (2**53, 1, ""), (2**60, 2, "nonconsecutive_windows"), (10**69 - 3, 1, ""),
])
def test_large_repeat_ordinals_preserve_sorted_windows_and_block_gaps(tmp_path, start, step, reason):
    source = tmp_path / "exact.csv"
    indices = [start + step * index for index in range(6)]
    write_rows(source, [row(i, value) for value, i in reversed(list(enumerate(indices, 1)))])
    metric = "latency_app_s"
    options = {"metrics": [metric], "block_size": 2, "resamples": 20}
    cli = build_statistics_report(source, **options)["groups"][0]
    report = report_payload(load_analysis([source]), statistics_options=options)["window_statistics"][0]
    summary = report["groups"][0]
    assert cli["reason"] == summary["reason"] == reason
    assert (cli["ci_low"] is None) == bool(reason)
    inputs = next(group for group in report["recalculation"]["groups"] if group["metric"] == metric)
    assert inputs["values"] == [1, 2, 3, 4, 5, 6]
    assert inputs["repeat_indices"] == [decimal_key_text(decimal_coordinate(str(i))) for i in indices]


def test_window_inputs_encode_large_powers_without_expansion(tmp_path):
    source = tmp_path / "huge.csv"
    write_rows(source, [row("2e100000000", 3), row("1e100000000", 1), row("0.0", 2)])
    report = report_payload(load_analysis([source]), statistics_options={"block_size": 2})["window_statistics"][0]
    inputs = next(group for group in report["recalculation"]["groups"] if group["metric"] == "latency_app_s")
    assert inputs["repeat_indices"] == [0.0, "1E+100000000", "2E+100000000"]
    assert type(inputs["repeat_indices"][0]) is float
    assert inputs["values"] == [2, 1, 3]


@pytest.mark.parametrize("old_order", [None, "binary64"])
def test_report_rejects_cached_intervals_from_rounded_repeat_order(tmp_path, old_order):
    source = tmp_path / "result_all.csv"
    values = [160, 50, 20, 200, 3, 1]
    write_rows(source, [row(2**60 + i, value) for i, value in zip(reversed(range(6)), values, strict=True)])
    metric = "latency_app_s"
    options = {"metrics": [metric], "resamples": 17, "seed": 3}
    cache = save_cache(source, **options)
    saved = json.loads(cache.read_text())
    group = saved["groups"][0]
    correct = group["ci_low"], group["ci_high"]
    # All six old binary64 indices tied; stable float sorting kept file order.
    stale = bootstrap_mean_interval(values, resamples=17, seed=3)
    assert stale != correct
    group["ci_low"], group["ci_high"] = stale
    if old_order is None:
        group.pop("repeat_index_order", None)
    else:
        group["repeat_index_order"] = old_order
    cache.write_text(json.dumps(saved))
    report = report_payload(load_analysis([source]), statistics_options=options)["window_statistics"][0]
    assert report["origin"] == "computed"
    assert report["cache_diagnostics"][0]["reason"] == "repeat_index_order_changed"
    assert report["groups"][0]["repeat_index_order"] == "decimal_v1"
    assert (report["groups"][0]["ci_low"], report["groups"][0]["ci_high"]) == correct


@pytest.mark.parametrize("axis", ["input_scale", "mem_cap_gb", "cpu_cores"])
def test_colliding_display_axes_keep_cache_bounds_and_inputs_distinct(tmp_path, axis):
    source = tmp_path / "exact-axes.csv"
    higher = "2.0000000000000001"
    write_rows(source, [row(i, first + i, **{axis: value})
                        for first, value in ((2, "2"), (20, higher)) for i in range(3)])
    options = {"metrics": ["latency_app_s"], "resamples": 20}
    cache = save_cache(source, **options)
    groups = json.loads(cache.read_text())["groups"]
    assert not groups[0].get("exact_coordinates")
    assert groups[1]["exact_coordinates"] == {axis: higher}
    report = report_payload(load_analysis([source]), statistics_options=options)["window_statistics"][0]
    assert report["origin"] == "cached"
    inputs = [g for g in report["recalculation"]["groups"] if g["metric"] == "latency_app_s"]
    assert [group["values"] for group in inputs] == [[2, 3, 4], [20, 21, 22]]
    assert inputs[1]["exact_coordinates"] == {axis: higher}
    old_cache = json.loads(cache.read_text())
    other_interval = dict(old_cache["groups"][0], ci_low=20, ci_high=21)
    cache.write_text(json.dumps({**old_cache, "groups": [other_interval, old_cache["groups"][1]]}))
    report = report_payload(load_analysis([source]), statistics_options=options)["window_statistics"][0]
    assert report["cache_diagnostics"][0]["reason"] == "interval_outside_observed_range"
    del old_cache["groups"][1]["exact_coordinates"]
    cache.write_text(json.dumps(old_cache))
    report = report_payload(load_analysis([source]), statistics_options=options)["window_statistics"][0]
    assert report["origin"] == "computed"
    assert report["cache_diagnostics"][0]["reason"] == "configuration_groups_changed"


@pytest.mark.parametrize('runtime', ['docker', 'process'])
def test_window_inputs_keep_numeric_identity_and_inherited_resource_axes(tmp_path, runtime):
    source = tmp_path / 'result_all.csv'
    rows = [row(2, 3, cpu_cores='2e0', mem_cap_gb='4.0', input_scale='3.2e1'),
            row(0, 1, cpu_cores='2.0', mem_cap_gb='4e0'), row(1, 'nan'),
            row(3, 99, status='error'), row(4, 99, result_origin='inferred_not_measured'),
            row(0, 99, warmup='1')]
    if runtime == 'process':
        for record in rows:
            record.update(cpu_cores='', mem_cap_gb='', runtime_kind='process',
                          runtime_id='fixture-env', resource_scope='observed_process_tree')
    write_rows(source, rows)
    original = source.read_bytes()

    report = report_payload(load_analysis([source]))['window_statistics'][0]

    groups = [group for group in report['recalculation']['groups'] if group['metric'] == 'latency_app_s']
    assert len(groups) == 1
    group = groups[0]
    assert group['runtime_kind'] == runtime
    assert (group['cpu_cores'], group['mem_cap_gb']) == ((None, None) if runtime == 'process' else (2., 4.))
    assert group['input_scale'] == 32.
    assert group['values'] == [1., None, 3.]
    assert group['repeat_indices'] == [0., 1., 2.]
    assert group['missing_reasons'] == {'unavailable' if runtime == 'process' else 'unavailable_unspecified': 1}
    assert group['excluded_windows'] == {'error': 1, 'inferred_not_measured': 1, 'warmup': 1}
    assert source.read_bytes() == original


def test_inputs_keep_missing_values_and_exclude_non_measurements(tmp_path):
    source = tmp_path / "result_all.csv"
    write_rows(source, [row(0, 1), row(1, "nan"), row(3, 3), row(4, 99, status="error"),
                        row(5, 99, result_origin="inferred_not_measured"), row(0, 99, warmup="1")])
    original = source.read_bytes()
    report = report_payload(load_analysis([source]))["window_statistics"][0]
    inputs = report.get("recalculation")
    assert inputs is not None, "HTML needs Python-selected windows for offline recalculation"
    group = next(group for group in inputs["groups"] if group["metric"] == "latency_app_s")
    assert group["values"] == [1, None, 3]
    assert group["repeat_indices"] == [0, 1, 3]
    assert group["excluded_windows"] == {"error": 1, "inferred_not_measured": 1, "warmup": 1}
    assert sum(group["missing_reasons"].values()) == 1
    assert inputs["status"] == "embedded"
    assert report["engine"] == "python"
    assert source.read_bytes() == original


def test_inputs_preserve_configuration_groups_and_offer_recorded_metrics(tmp_path):
    source = tmp_path / "result_all.csv"
    write_rows(source, [row(i, i + 1, concurrency="1", throughput_samples_per_s="4") for i in range(3)] +
               [row(i, 10, concurrency="2", throughput_samples_per_s="8") for i in range(3, 6)])
    report = report_payload(load_analysis([source]))["window_statistics"][0]
    inputs = report.get("recalculation")
    assert inputs is not None
    groups = [group for group in inputs["groups"] if group["metric"] == "latency_app_s"]
    assert len({group["config_id"] for group in groups}) == 2
    assert [group["values"] for group in groups] == [[1, 2, 3], [10, 10, 10]]
    assert "throughput_samples_per_s" in inputs["metrics"]
    assert "cold_start_s" not in inputs["metrics"]


def test_large_inputs_publish_hash_bound_sidecar_without_overwriting(tmp_path, monkeypatch):
    from acprof.analysis import report_statistics
    # The limit applies to raw windows only; initial Python estimates stay embedded.
    monkeypatch.setattr(report_statistics, "MAX_EMBEDDED_WINDOW_BYTES", 1, raising=False)
    source = tmp_path / "result_all.csv"
    write_rows(source, [row(i, i + 1) for i in range(3)])
    output = write_report(load_analysis([source]), tmp_path / "report.html")
    html = output.read_text()
    raw = html.split('<script id="report-data" type="application/json">')[1].split("</script>")[0]
    report = json.loads(raw)["window_statistics"][0]
    inputs = report.get("recalculation")
    assert inputs is not None
    assert inputs["status"] == "external_required"
    assert "groups" not in inputs
    sidecar = output.parent / inputs["file"]
    assert hashlib.sha256(sidecar.read_bytes()).hexdigest() == inputs["sha256"]
    decoded = json.loads(sidecar.read_bytes())
    assert decoded["result_sha256"] == report["result_sha256"]
    assert next(g for g in decoded["groups"] if g["metric"] == "latency_app_s")["values"] == [1, 2, 3]
    assert report["groups"][0]["mean"] == 2
    output.unlink()
    before = sidecar.read_bytes()
    with pytest.raises(ValueError, match="new file"):
        write_report(load_analysis([source]), output)
    assert sidecar.read_bytes() == before
    assert not output.exists()


def test_publication_error_preserves_inputs_of_an_already_visible_html(tmp_path, monkeypatch):
    from acprof.plotting import report as rendering

    monkeypatch.setattr("acprof.analysis.report_statistics.MAX_EMBEDDED_WINDOW_BYTES", 1)
    source, output = tmp_path / "result_all.csv", tmp_path / "report.html"
    write_rows(source, [row(i, i + 1) for i in range(3)])
    original = rendering.atomic_write

    def publish_then_fail(path, write, **options):
        original(path, write, **options)
        if path == output:
            raise OSError("directory sync failed after publication")

    with patch('acprof.plotting.report.atomic_write', publish_then_fail), pytest.raises(OSError, match="directory sync"):
        rendering.write_report(load_analysis([source]), output)
    assert output.is_file()
    assert (tmp_path / "report.windows-1.json").is_file()


@pytest.mark.parametrize('cancel', [False, True])
def test_late_window_stream_failure_preserves_existing_reports_and_source(tmp_path, monkeypatch, cancel):
    monkeypatch.setattr('acprof.analysis.report_statistics.MAX_EMBEDDED_WINDOW_BYTES', 1)
    source = tmp_path / 'result_all.csv'
    write_rows(source, [row(index, index + 1) for index in range(3)])
    model = load_analysis([source])
    write_report(model, tmp_path / 'previous.html')
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    failure = KeyboardInterrupt('cancelled window stream') if cancel else OSError('window stream failed')
    emitted = []

    def interrupted_chunks(*args):
        chunks = window_input_chunks(*args)
        try:
            yield next(chunks)
            first_group = next(chunks)
            emitted.append(json.loads(first_group))
            yield first_group
            raise failure
        finally:
            chunks.close()

    def interrupted_write(*args):
        with patch('acprof.analysis.report_statistics.window_input_chunks', interrupted_chunks):
            write_window_inputs(*args)

    with patch('acprof.plotting.report.write_window_inputs', interrupted_write), \
            pytest.raises(type(failure), match=str(failure)) as caught:
        write_report(model, tmp_path / 'new.html')

    assert caught.value is failure
    assert len(emitted) == 1
    assert emitted[0]['repeat_indices'] == [0, 1, 2]
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
