"""A comparison's derived files are retriable as a set, including HTML window inputs."""
import csv
import json
import os
from pathlib import Path

import pytest

from acprof.cli.main import main
from tests.analysis.reporting.test_report_sidecar_publication import assert_complete_bundle
from tests.support.independent_comparison_fixtures import IndependentComparisonFixture


@pytest.fixture
def experiments(tmp_path, monkeypatch):
    fixture = IndependentComparisonFixture()
    fixture.build(None, tmp_path / "experiments")
    monkeypatch.setattr("plotly.offline.get_plotlyjs", lambda: "/* offline fixture */")
    return fixture.replicate("left", 0, [1, 2]), fixture.replicate("right", 0, [2, 3])


@pytest.fixture
def outputs(tmp_path):
    folder = tmp_path / "exports"
    folder.mkdir()
    return {"--output": folder / "comparison.json", "--csv-output": folder / "comparison.csv",
            "--html-output": folder / "comparison.html"}


def compare(experiments, outputs):
    left, right = experiments
    args = ["compare", "--left", str(left), "--right", str(right), "--metric", "latency_app_s", "--resamples", "20"]
    for option, path in outputs.items():
        args.extend([option, str(path)])
    return main(args)


@pytest.mark.parametrize("error", ["suffix", "html_below_json", "csv_below_html", "html_sidecar"])
def test_conflicting_destinations_leave_nothing_to_clear_for_retry(experiments, outputs, error, monkeypatch):
    monkeypatch.setattr("acprof.analysis.report_statistics.MAX_EMBEDDED_WINDOW_BYTES", 1)
    folder = outputs["--output"].parent
    invalid = outputs.copy()
    if error == "suffix":
        invalid["--csv-output"] = folder / "comparison.txt"
    elif error == "html_below_json":
        invalid["--html-output"] = outputs["--output"] / "nested.html"
    elif error == "csv_below_html":
        invalid["--csv-output"] = outputs["--html-output"] / "nested.csv"
    else:
        invalid["--output"] = folder / "comparison.windows-1.json"
    with pytest.raises(SystemExit) as failure:
        compare(experiments, invalid)
    assert failure.value.code != 0
    assert list(folder.iterdir()) == []
    assert compare(experiments, outputs) == 0


def test_late_claim_keeps_competitor_and_cleans_own_files(experiments, outputs, monkeypatch, capsys):
    monkeypatch.setattr("acprof.analysis.report_statistics.MAX_EMBEDDED_WINDOW_BYTES", 1)
    final = outputs["--output"]
    link = os.link

    def claim_before_link(src, dst, *args, **kwargs):
        if Path(dst) == final:
            with final.open("xb") as stream:
                stream.write(b"owned by another writer")
        return link(src, dst, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "link", claim_before_link)
        with pytest.raises(SystemExit) as failure:
            compare(experiments, outputs)
    assert failure.value.code != 0
    assert not capsys.readouterr().out
    assert list(final.parent.iterdir()) == [final]
    assert final.read_bytes() == b"owned by another writer"
    final.unlink()
    assert compare(experiments, outputs) == 0
    sidecars = [final.with_name(f"comparison.windows-{index}.json") for index in (1, 2)]
    assert_complete_bundle(outputs["--html-output"], sidecars)
    assert set(final.parent.iterdir()) == {*outputs.values(), *sidecars}


def test_incompatible_comparison_publishes_all_formats_with_final_paths(experiments, outputs, monkeypatch, capsys):
    monkeypatch.setattr("acprof.analysis.report_statistics.MAX_EMBEDDED_WINDOW_BYTES", 1)
    _, right = experiments
    path = right / "hardware_conditions.json"
    hardware = json.loads(path.read_bytes())
    hardware["cases"]["1c_4g_off"]["host_id"] = "different-host"
    path.write_text(json.dumps(hardware))
    assert compare(experiments, outputs) == 1
    assert not capsys.readouterr().err
    result = json.loads(outputs["--output"].read_bytes())
    html = outputs["--html-output"]
    assert result["status"] == "incompatible"
    assert result["groups"][0]["difference"] is None
    assert result["html_report_path"] == str(html.resolve())
    embedded = html.read_text().split('<script id="report-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    comparisons = json.loads(embedded)["experiment_comparisons"]
    assert {entry["report"]["status"] for entry in comparisons} == {"incompatible"}
    with outputs["--csv-output"].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["analysis.status"] for row in rows] == ["incompatible"]
    sidecars = [html.with_name(f"comparison.windows-{index}.json") for index in (1, 2)]
    assert_complete_bundle(html, sidecars)
    assert set(html.parent.iterdir()) == {*outputs.values(), *sidecars}
