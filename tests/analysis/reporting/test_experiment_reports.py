"""HTML embeds the existing strict independent-run comparison unchanged."""
import json

import pytest

from acprof.analysis.model import load_analysis
from acprof.plotting.report import report_payload
from tests.support.independent_comparison_fixtures import IndependentComparisonFixture


@pytest.fixture
def runs(tmp_path, request):
    fixture = IndependentComparisonFixture()
    fixture.build(request, tmp_path)
    return fixture


def test_two_independent_runs_keep_differences_and_no_window_based_ci(runs):
    left = runs.replicate("left", 0, [1, 2, 3])
    right = runs.replicate("right", 0, [2, 4, 6])
    payload = report_payload(load_analysis([left, right]), statistics_options={"resamples": 99})
    comparison = payload["experiment_comparisons"][0]["report"]
    group = next(group for group in comparison["groups"] if group["metric"] == "latency_app_s")
    assert (group["left"]["mean"], group["right"]["mean"]) == (2, 4)
    assert (group["difference"], group["ratio"], group["percent"]) == (2, 2, 100)
    assert (group["left"]["n_runs"], group["right"]["n_runs"]) == (1, 1)
    assert group["difference_ci"] is group["ratio_ci"] is None
    assert group["reason"] == "insufficient_independent_runs"


def test_independent_run_means_exclude_inferred_values_and_keep_their_count(runs):
    import csv

    left = runs.replicate("left", 0, [2, 999])
    right = runs.replicate("right", 0, [4, 4])
    source = left / "result_all.csv"
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    rows[1]["result_origin"] = "inferred_not_measured"
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(dict.fromkeys([*fields, "result_origin"])))
        writer.writeheader()
        writer.writerows(rows)
    result = runs.compare([left], [right])["groups"][0]
    assert result["left"]["mean"] == 2
    assert result["left"]["inferred_windows"] == 1
    assert result["left"]["failed_windows"] == 0
    assert (result["difference"], result["ratio"]) == (2, 2)


def test_explicit_groups_reuse_strict_bootstrap_and_preserve_source_evidence(runs):
    left = [runs.replicate("left", i, [2, 2, 2]) for i in range(3)]
    right = [runs.replicate("right", i, [4, 4, 4]) for i in range(3)]
    payload = report_payload(load_analysis([*left, *right]), comparison_groups=(left, right),
                             statistics_options={"metrics": ["latency_app_s"], "resamples": 99, "seed": 3})
    reports = payload["experiment_comparisons"]
    assert len(reports) == 1
    actual = reports[0]["report"]
    assert actual["resampling_unit"] == "independent_run_mean"
    assert actual["groups"][0]["difference_ci"] == [2, 2]
    assert actual["groups"][0]["ratio_ci"] == [2, 2]
    assert actual["groups"][0]["left"]["n_runs"] == 3
    assert actual["source_sha256"]


def test_legacy_sources_keep_descriptions_and_explain_unavailable_strict_comparison(tmp_path):
    sources = []
    for name in ("whisper", "diffusion"):
        root = tmp_path / name
        root.mkdir()
        path = root / "result_all.csv"
        path.write_text("cpu_cores,mem_cap_gb,gpu_mode,input_scale,repeat_idx,warmup,status,latency_app_s\n"
                        "2,4,off,32,0,0,ok,2\n")
        (root / "static_meta.json").write_text(json.dumps({"model_name": name, "pipeline_tag": name}))
        sources.append(path)
    payload = report_payload(load_analysis(sources))
    assert len(payload["window_statistics"]) == 2
    comparison = payload["experiment_comparisons"][0]
    assert comparison["report"] is None
    assert comparison["reason"] == "missing_recorded_run_identity"
    assert comparison["display_only"] is True


def test_incompatible_measurement_conditions_retain_exact_reasons(runs):
    left = runs.replicate("left", 0, [1, 2, 3])
    right = runs.replicate("right", 0, [2, 4, 6])
    runs.fixture.change_json(right, "run_state.json", lambda state: state["options"].update(sample_hz=999))
    payload = report_payload(load_analysis([left, right]), statistics_options={"resamples": 99})
    comparison = payload["experiment_comparisons"][0]
    assert comparison["report"]["status"] == "incompatible"
    group = comparison["report"]["groups"][0]
    assert group["difference"] is group["ratio"] is group["difference_ci"] is None
    assert any("measurement_protocol" in check.get("comparison_reasons", [])
               for check in comparison["report"]["condition_checks"])


def test_reusing_comparison_after_source_change_refuses_html_publication(runs, tmp_path):
    from acprof.plotting.report import write_report
    left = runs.replicate("left", 0, [1, 2, 3])
    right = runs.replicate("right", 0, [2, 4, 6])
    comparison = runs.compare([left], [right])
    runs.fixture.change_json(right, "run_state.json", lambda state: state["options"].update(sample_hz=999))
    model = load_analysis([left, right])
    output = tmp_path / "report.html"
    with pytest.raises(ValueError, match="changed|变化"):
        write_report(model, output, independent_report=comparison)
    assert not output.exists()


def test_comparison_cli_exports_matching_html_json_and_csv(runs, tmp_path, capsys):
    import csv
    import re

    from acprof.cli.compare import main
    left = runs.replicate("left", 0, [1, 2, 3])
    right = runs.replicate("right", 0, [2, 4, 6])
    html, csv_path = tmp_path / "comparison.html", tmp_path / "comparison.csv"
    assert main(["--left", str(left), "--right", str(right), "--resamples", "99", "--metric", "latency_app_s",
                 "--html-output", str(html), "--csv-output", str(csv_path)]) == 0
    exported = json.loads(capsys.readouterr().out)
    payload = json.loads(re.search(r'<script id="report-data" type="application/json">(.*?)</script>',
                                  html.read_text(), re.S).group(1))
    assert payload["experiment_comparisons"][0]["report"]["groups"] == exported["groups"]
    with csv_path.open() as stream:
        row = next(csv.DictReader(stream))
    assert (row["difference"], row["ratio"], row["percent"]) == ("2.0", "2.0", "100.0")
    assert row["left.n_runs"] == row["right.n_runs"] == "1"


def test_report_cli_accepts_explicit_replicate_groups(runs, tmp_path, capsys):
    import re

    from acprof.cli.report import main
    left = [runs.replicate("left", i, [2, 2, 2]) for i in range(3)]
    right = [runs.replicate("right", i, [4, 4, 4]) for i in range(3)]
    output = tmp_path / "report.html"
    command = ["--output", str(output), "--metric", "latency_app_s", "--resamples", "99"]
    for side, sources in (("left", left), ("right", right)):
        for source in sources:
            command.extend(["--" + side, str(source)])
    assert main(command) == 0
    assert "ACPROF_REPORT" in capsys.readouterr().out
    payload = json.loads(re.search(r'<script id="report-data" type="application/json">(.*?)</script>',
                                  output.read_text(), re.S).group(1))
    assert payload["experiment_comparisons"][0]["report"]["groups"][0]["difference_ci"] == [2, 2]
