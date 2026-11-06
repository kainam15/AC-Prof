"""Browser inputs preserve Python's window selection, identity and publication guard."""
import hashlib
import json

import pytest

from acprof.analysis.model import load_analysis
from acprof.plotting.report import report_payload, write_report
from acprof.results.metrics import METRICS
from tests.analysis.reporting.test_report_statistics import row, write_rows


@pytest.mark.parametrize('metric_count', [1, 16])
def test_report_classifies_window_identity_independently_of_metric_count(tmp_path, monkeypatch, metric_count):
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

    monkeypatch.setattr(report_statistics, 'number', observe)
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

    monkeypatch.setattr(rendering, "atomic_write", publish_then_fail)
    with pytest.raises(OSError, match="directory sync"):
        rendering.write_report(load_analysis([source]), output)
    assert output.is_file()
    assert (tmp_path / "report.windows-1.json").is_file()
