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
