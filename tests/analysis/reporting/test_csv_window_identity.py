"""CSV measurement coordinates keep decimal identity independent of plotting values."""
import csv
import io
import json

import pytest

from acprof.analysis.model import load_analysis
from acprof.analysis.resource_comparison import resource_qualification
from tests.analysis.reporting.test_report_import import (
    browser as browser,
    page as page,
    source as source,
)

CASES = [
    ("large-repeats", [{"repeat_idx": "9007199254740992"}, {"repeat_idx": "9007199254740993"}], [5]),
    ("scaled-indices", [{"repeat_idx": "10e9999"}, {"repeat_idx": "2e10000"}], [5]),
    ("same-index", [{"repeat_idx": "90071992547409930e-1"}, {"repeat_idx": "9007199254740993.00"}], "duplicate"),
    ("same-zero", [{"warmup": "-0.0", "repeat_idx": "+0e5"}, {}], "duplicate"),
    ("near-integer", [{"repeat_idx": "1.0000000000000001"}], "invalid"),
    ("negative-underflow", [{"repeat_idx": "-1e-324"}], "invalid"),
    ("underflow-warmup", [{"warmup": "1e-324"}], "invalid"),
    ("near-one-warmup", [{"warmup": ".99999999999999999"}], "invalid"),
    ("phase", [{"warmup": "1.00"}, {"warmup": "-0E5"}], [7]),
    *[(field, [{field: "2.00"}, {field: "2.0000000000000001"}], [3, 7])
      for field in ("input_scale", "cpu_cores", "mem_cap_gb")],
    ("scale-alias", [{"input_scale": "2.00000000000000010"}, {"input_scale": "20000000000000001e-16"}], "duplicate"),
]


def csv_content(changes, recorded):
    base = {"cpu_cores": "2", "mem_cap_gb": "4", "gpu_mode": "off", "input_scale": "2",
            "warmup": "0", "repeat_idx": "0", "status": "ok", "latency_app_s": "3"}
    if recorded:
        base["run_id"] = "recorded"
    output = io.StringIO()
    writer = csv.DictWriter(output, base)
    writer.writeheader()
    for row, latency in zip(changes, ("3", "7"), strict=False):
        writer.writerow({**base, **row, "latency_app_s": latency})
    return output.getvalue()


@pytest.mark.parametrize("recorded", [False, True], ids=["anonymous", "recorded"])
@pytest.mark.parametrize(("label", "changes", "expected"), CASES, ids=[case[0] for case in CASES])
def test_python_window_identity(tmp_path, recorded, label, changes, expected):
    path = tmp_path / f"{label}.csv"
    path.write_text(csv_content(changes, recorded))
    if isinstance(expected, str):
        with pytest.raises(ValueError, match="duplicate|conflicting" if expected == "duplicate" else "invalid"):
            load_analysis([path])
    else:
        configs = load_analysis([path]).configs
        assert sorted(config["metrics"]["latency_app_s"]["value"] for config in configs) == expected


@pytest.mark.parametrize("recorded", [False, True], ids=["anonymous", "recorded"])
def test_browser_window_identity(page, recorded):
    csvs = [csv_content(changes, recorded) for _, changes, _ in CASES]
    results = page.evaluate("""async csvs => {
      const {registry} = JSON.parse(document.getElementById("report-data").textContent);
      return await Promise.all(csvs.map(async content => {
        try {
          const bundle = await ACProfImport.readCsv(new File([content], "windows.csv"), registry);
          return bundle.configs.map(c => c.metrics.latency_app_s.value).sort((a,b) => a-b);
        } catch (error) {
          if (error.message.includes("Duplicate or conflicting window")) return "duplicate";
          if (error.message.includes("Invalid configuration or window")) return "invalid";
          throw error;
        }
      }));
    }""", csvs)
    assert dict(zip((case[0] for case in CASES), results, strict=True)) == {case[0]: case[2] for case in CASES}


@pytest.mark.parametrize(("field", "label"), [("cpu_cores", "cpu"), ("mem_cap_gb", "memory"), ("input_scale", "input_case")])
def test_exact_axes_survive_export_and_qualify_comparisons(page, tmp_path, field, label):
    text = csv_content([{field: "2"}, {field: "20000000000000001e-16"}], False)
    path = tmp_path / "exact.csv"
    path.write_text(text)
    before, after = load_analysis([path]).configs
    if label == "input_case":
        assert json.loads(after["input_case"])["scale"] == "2.0000000000000001"
        assert resource_qualification(before, after)["status"] == "incompatible"
    else:
        assert after["exact_resources"] == {label: "2.0000000000000001"}
        assert resource_qualification(before, after)["changed_resources"] == [label]
    observed = page.evaluate("""async text => {
      const {registry} = JSON.parse(document.getElementById("report-data").textContent);
      const bundle = await ACProfImport.readCsv(new File([text], "exact.csv"), registry);
      const [a, b] = ACProfImport.validate(bundle, registry).configs;
      return {input:JSON.parse(b.input_case), exact:b.exact_resources || {},
        reason:ACProfViews.descriptiveQualification(a,b,"same-hardware").reasons,
        scaling:ACProfViews.descriptiveQualification(a,b,"resource-scaling").reasons};
    }""", text)
    assert observed["input"] == json.loads(after["input_case"])
    assert observed["exact"] == after.get("exact_resources", {})
    assert label in observed["reason"]
    assert (label in observed["scaling"]) == (label == "input_case")


def test_imported_exact_resources_require_a_canonical_coordinate_matching_the_axis(page):
    results = page.evaluate("""() => {
      const bundle = JSON.parse(document.getElementById("report-data").textContent);
      return [undefined, {cpu:"2.0000000000000001"}, {cpu:2}, {cpu:"2.00000000000000010"},
        {cpu:"20"}, {gpu:"2"}, {cpu:"2"}].map(evidence => {
        const sample = structuredClone(bundle);
        if (evidence !== undefined) sample.configs[0].exact_resources = evidence;
        try { ACProfImport.validate(sample, bundle.registry); return "ok"; }
        catch (error) { return error.message; }
      });
    }""")
    assert results[:2] == ["ok", "ok"]
    assert all("Invalid exact resource coordinate" in message for message in results[2:])


def test_statistics_validate_and_recalculate_exact_repeat_sequences(page):
    from decimal import Decimal

    from acprof.results.measurement_plan import decimal_key_text

    def window(value):
        return value if value < 2**53 else decimal_key_text(Decimal(value))

    boundary = list(range(2**53 - 2, 2**53 + 4))
    carry = list(range(10**20 - 2, 10**20 + 4))
    cases = [
        ([window(i) for i in boundary], "ok"),
        ([window(i) for i in carry], "ok"),
        ([window(i) for i in carry[:-1] + [carry[-1] + 1]], "gap"),
        ([f"{i}E+999999999999999999" for i in range(1, 7)], "gap"),
        *[([0, 1, 2, 3, 4, invalid], "invalid") for invalid in
          [2**53, -1, 5.5, "5", "09007199254740992", "9007199254740992.1", None]],
        ([window(i) for i in reversed(carry)], "invalid"),
        ([window(i) for i in carry[:-1] + [carry[-2]]], "invalid"),
    ]
    outcomes = page.evaluate("""async cases => {
      const data = JSON.parse(document.getElementById("report-data").textContent);
      const group = data.window_statistics[0].recalculation.groups[0];
      group.values = [1, 2, 3, 4, 5, 6];
      const options = {confidence:.95, resamples:10, seed:2, block_size:2};
      const outcomes = [];
      for (const indices of cases) {
        group.repeat_indices = indices;
        let valid = true, result;
        try { ACProfImport.validate(data, data.registry); } catch { valid = false; }
        try {
          const summary = await ACProfStatistics.summarize(group, options);
          result = summary.reason === "nonconsecutive_windows" && summary.ci_low === null ? "gap" :
            summary.reason === "" && Number.isFinite(summary.ci_low) ? "ok" : summary.reason;
        } catch { result = "invalid"; }
        outcomes.push([valid, result]);
      }
      return outcomes;
    }""", [indices for indices, _ in cases])
    assert outcomes == [[expected != "invalid", expected] for _, expected in cases]


def test_python_report_with_large_repeats_recalculates_and_reopens(page, tmp_path):
    from acprof.plotting.report import write_report
    from tests.analysis.reporting.test_report_import import download_json, settled

    path = tmp_path / "large-repeats.csv"
    start = 2**53 - 2
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", "warmup", "repeat_idx", "status", "latency_app_s"])
        for position in reversed(range(6)):
            writer.writerow([2, 4, "off", 1, 0, start + position, "ok", position + 1])
    generated = write_report(load_analysis([path]), tmp_path / "large-repeats.html")
    page.goto(generated.as_uri())
    settled(page)
    indices = page.evaluate("""async html => {
      const {registry} = JSON.parse(document.getElementById("report-data").textContent);
      const [bundle] = await ACProfImport.read(new File([html], "large-repeats.html"), registry, {});
      return bundle.window_statistics[0].recalculation.groups.find(group => group.metric === "latency_app_s").repeat_indices;
    }""", generated.read_text())
    assert indices == [start, start + 1, *[str(start + i) for i in range(2, 6)]]
    page.click("[data-view=statistics]")
    settled(page)
    page.click("#statistics-parameters > summary")
    page.select_option("#statistics-metrics", ["latency_app_s"])
    page.select_option("#statistics-resamples", "1000")
    page.click("#statistics-advanced > summary")
    page.fill("#statistics-block-size", "2")
    page.click("#statistics-recalculate")
    page.wait_for_function("() => document.body.dataset.calculating === 'false'")
    settled(page)
    result = download_json(page)
    assert result["engine"] == "browser"
    group, = result["groups"]
    assert (group["mean"], group["n_windows"], group["reason"]) == (3.5, 6, "")
    assert group["ci_low"] <= 3.5 <= group["ci_high"]
    with page.expect_download() as saved:
        page.click("#export-html")
    restored = tmp_path / "restored.html"
    saved.value.save_as(restored)
    page.goto(restored.as_uri())
    settled(page)
    assert download_json(page) == result
