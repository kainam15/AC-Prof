"""Portable report contract and offline browser experiment management."""
import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

from acprof.analysis.model import load_analysis
from acprof.plotting.report import report_payload, write_report


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "result_all.csv"
    path.write_text(
        "cpu_cores,mem_cap_gb,gpu_mode,input_scale,repeat_idx,warmup,status,latency_app_s,"
        "latency_app_p95_s,throughput_samples_per_s,cpu_energy_total_j,repeat_in_window,model,run_id\n"
        "2,4,off,32,0,0,ok,2,2,5,2,1,fixture/model,original\n"
        "2,4,off,32,1,0,ok,4,4,7,4,3,fixture/model,original\n", encoding="utf-8")
    return path


def test_report_payload_declares_portable_contract_and_provenance(source):
    payload = report_payload(load_analysis([source]))
    assert payload.get("kind") == "acprof_analysis_bundle"
    assert payload["analysis_contract"]["version"] == 1
    assert payload["analysis_contract"]["statistics_engine"] == "python"
    assert payload["sources"][0]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert payload["sources"][0]["display_name"] == source.parent.name


@pytest.fixture(scope="module")
def browser():
    if os.environ.get("ACPROF_BROWSER_TESTS") != "1":
        pytest.skip("opt-in headless browser test")
    from playwright.sync_api import sync_playwright
    with sync_playwright() as playwright:
        instance = playwright.chromium.launch(
            executable_path=os.environ.get("ACPROF_BROWSER_EXECUTABLE"),
            args=["--no-sandbox", "--disable-gpu"], headless=True)
        yield instance
        instance.close()


@pytest.fixture
def page(browser, source, tmp_path):
    report = write_report(load_analysis([source]), tmp_path / "report.html")
    context = browser.new_context(offline=True, viewport={"width": 1280, "height": 900})
    page = context.new_page()
    errors, network = [], []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("request", lambda request: network.append(request.url)
            if request.url.startswith(("http:", "https:")) else None)
    page.goto(report.as_uri())
    settled(page)
    yield page
    context.close()
    assert errors == []
    assert network == []


def settled(page):
    page.wait_for_function("() => document.body.dataset.ready === 'true' && document.body.dataset.busy === 'false'")
    assert page.locator("#error").is_hidden()


def import_file(page, name, content):
    page.locator("#experiment-files").set_input_files({
        "name": name, "mimeType": "application/octet-stream", "buffer": content.encode()})
    page.wait_for_function("() => document.body.dataset.importing === 'false'")
    settled(page)


def download_json(page):
    with page.expect_download() as result:
        page.click("#export-json")
    return json.loads(Path(result.value.path()).read_text())


@pytest.mark.parametrize(("field", "value"), [
    ("input_scale", "64"), ("batch_size", "2"),
    ("input_units_per_request", "64"), ("task_param", "other"),
])
def test_named_csv_import_preserves_python_input_parameters(page, tmp_path, field, value):
    rows = [dict(cpu_cores="2", mem_cap_gb="4", gpu_mode="off", input_scale="32", repeat_idx=str(index),
                 warmup="0", status="ok", latency_app_s=latency, input_case="shared <input>",
                 batch_size="1", input_units_per_request="32", task_param="fixture")
            for index, latency in enumerate(("2", "20"))]
    rows[1][field] = value
    path = tmp_path / "named.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    before = path.read_bytes()
    expected = load_analysis([path]).configs
    import_file(page, path.name, path.read_text())
    assert page.locator("#matrix tbody tr").count() == 3
    actual = [config for config in download_json(page)["analysis_bundle"]["configs"]
              if config.get("statistics_status") == "not_computed"]
    assert len(expected) == len(actual) == 2
    for python_config, browser_config in zip(expected, actual, strict=True):
        assert json.loads(browser_config["input_case"]) == json.loads(python_config["input_case"])
        assert browser_config["metrics"]["latency_app_s"]["value"] == python_config["metrics"]["latency_app_s"]["value"]
    assert path.read_bytes() == before


@pytest.mark.parametrize("width", [1280, 390])
def test_named_input_labels_filters_and_scaling_stay_separate(page, tmp_path, width):
    path = tmp_path / "named.csv"
    path.write_text(
        "cpu_cores,mem_cap_gb,gpu_mode,input_scale,repeat_idx,warmup,status,latency_app_s,input_case\n"
        "2,4,off,32,0,0,ok,4,named <input>\n4,4,off,32,0,0,ok,2,named <input>\n"
        "2,4,off,64,1,0,ok,40,named <input>\n4,4,off,64,1,0,ok,20,named <input>\n")
    report = write_report(load_analysis([path]), tmp_path / "named.html")
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(report.as_uri())
    settled(page)
    assert page.locator("#matrix tbody tr").count() == 4
    labels = page.locator("#filter-input_case option").all_text_contents()[1:]
    assert len(labels) == 2
    assert all("named <input>" in label for label in labels)
    assert any("scale=32" in label for label in labels)
    assert any("scale=64" in label for label in labels)
    page.click("[data-view=scaling]")
    settled(page)
    page.select_option("#scaling-metric", "latency_app_s")
    settled(page)
    assert page.locator("#scaling-charts .scaling-chart").count() == 2
    assert page.locator("#scaling-charts .js-plotly-plot").evaluate_all(
        "charts => charts.map(chart => chart.data[0].y)") == [[4, 2], [40, 20]]
    page.click("[data-view=experiments]")
    settled(page)
    titles = page.locator("#cross-model-charts h3").all_text_contents()
    assert len(titles) == 2
    assert all("named <input>" in title and "scale=" in title and "{" not in title for title in titles)
    assert page.locator("#cross-model-charts .js-plotly-plot").evaluate_all(
        "charts => charts.map(chart => chart.data[0].y)") == [[4, 2], [40, 20]]
    if artifacts := os.environ.get("ACPROF_BROWSER_ARTIFACT_DIR"):
        directory = Path(artifacts)
        directory.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(directory / f"named-input-{width}.png"), full_page=True)
    page.select_option("#filter-input_case", index=1)
    page.click("[data-view=matrix]")
    settled(page)
    assert page.locator("#matrix tbody tr").count() == 2


def test_csv_import_compare_export_remove_preserves_embedded_data(page, source):
    assert page.locator("#add-experiment").count() == 1
    embedded = page.locator("#report-data").text_content()
    import_file(page, "second.csv", source.read_text().replace("original", "second").replace(",2,2,5,", ",1,1,5,"))
    assert page.locator("#experiment-list tbody tr").count() == 2
    assert page.locator("#matrix tbody tr").count() == 2
    page.select_option("#baseline", index=1)
    page.select_option("#comparison-purpose", "same-hardware")
    settled(page)
    payload = download_json(page)
    rows = [row for row in payload["groups"] if row["run_id"] == "second"]
    latency = next(row for row in rows if row["metric"] == "latency_app_p95_s")
    assert latency["value"] == 2.5
    assert latency["delta"] == -.5
    assert latency["comparison"]["status"] == "unknown"
    assert latency["statistics_status"] == "not_computed"
    assert payload["analysis_bundle"]["kind"] == "acprof_analysis_bundle"
    page.click("[data-view=experiments]")
    settled(page)
    assert page.locator("#experiments tbody tr").count() > 0
    assert "未计算" in page.locator("#experiments").inner_text()
    page.locator("#experiment-list button[data-remove]").click()
    settled(page)
    assert page.locator("#experiment-list tbody tr").count() == 1
    assert page.locator("#report-data").text_content() == embedded


@pytest.mark.parametrize(("content", "expected"), [
    ("cpu_cores,status\n2,ok\n", "缺失"),
    ("cpu_cores,cpu_cores\n2,4\n", "重复"),
    ('cpu_cores,mem_cap_gb,gpu_mode,input_scale,repeat_idx,warmup,status\n2,4,off,32,0,0,"ok', "CSV"),
    ('{"kind":"acprof_analysis_bundle","schema_version":999}', "版本"),
])
def test_rejects_invalid_inputs_atomically(page, content, expected):
    import_file(page, "bad.json" if content.startswith("{") else "bad.csv", content)
    assert expected in page.locator("#import-status").inner_text()
    assert page.locator("#experiment-list tbody tr").count() == 1


def test_duplicate_fingerprints_and_run_identity_are_rejected(page, source):
    import_file(page, "renamed.csv", source.read_text())
    assert "重复" in page.locator("#import-status").inner_text()
    import_file(page, "changed.csv", source.read_text().replace(",2,2,5,", ",9,9,5,"))
    assert "run" in page.locator("#import-status").inner_text()
    assert page.locator("#experiment-list tbody tr").count() == 1


@pytest.mark.parametrize("chunk_size", [7, 257, 262144])
def test_chunked_csv_summaries_match_python_including_failures_and_lifecycle(page, tmp_path, chunk_size):
    path = tmp_path / "comparison.csv"
    rows = []
    for index, status, warmup, origin, energy, weight, cold in [
        (0, "ok", 0, "measured", 2, 1, 5), (1, "ok", 0, "measured", 4, 3, 5),
        (2, "error", 0, "measured", 999, 1, 90), (3, "ok", 1, "measured", 888, 1, 80),
        (4, "ok", 0, "inferred_not_measured", 777, 1, 70),
    ]:
        rows.append(dict(cpu_cores=2, mem_cap_gb=4, gpu_mode="off", input_scale=32, repeat_idx=index,
                         warmup=warmup, status=status, result_origin=origin, cpu_energy_total_j=energy,
                         repeat_in_window=weight, latency_app_s=index + 1, cold_start_s=cold,
                         container_mem_usage_peak_bytes=(index + 1) * 100, model='中文😀,"quoted"\nmodel'))
    rows += [{**rows[0], "cpu_cores": 4, "cpu_energy_total_j": "", "cold_start_s": 5},
             {**rows[1], "cpu_cores": 4, "cold_start_s": 6},
             {**rows[2], "cpu_cores": 8}]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    expected = load_analysis([path]).configs
    actual = page.evaluate("""async ({bytes, chunkSize}) => {
      const file = new File([new Uint8Array(bytes)], 'comparison.csv');
      const registry = JSON.parse(document.getElementById('report-data').textContent).registry;
      return await ACProfImport.readCsv(file, registry, {chunkSize});
    }""", {"bytes": list(path.read_bytes()), "chunkSize": chunk_size})
    assert actual["sources"][0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    for python_config, browser_config in zip(expected, actual["configs"], strict=True):
        for key in ("status", "valid_windows", "excluded_rows", "model"):
            assert browser_config[key] == python_config[key]
        for metric, summary in python_config["metrics"].items():
            assert browser_config["metrics"][metric] == pytest.approx(summary)
    assert actual["configs"][0]["metrics"]["observed_energy_per_request_j"]["value"] == 3.5
    assert actual["configs"][0]["metrics"]["observed_energy_j"]["value"] == 14
    assert actual["configs"][1]["metrics"]["cold_start_s"]["value"] is None
    assert actual["window_statistics"] == []


def test_html_and_json_restore_filtered_comparison_offline(page, source, tmp_path):
    import_file(page, "second.csv", source.read_text().replace("original", "second"))
    page.select_option("#baseline", index=1)
    page.select_option("#comparison-purpose", "cross-hardware")
    page.select_option("#filter-cpu", "2")
    page.click("[data-view=pareto]")
    settled(page)
    original = page.locator("#report-data").text_content()
    with page.expect_download() as result:
        page.click("#export-html")
    path = tmp_path / "saved.html"
    result.value.save_as(path)
    assert '<script src=' not in path.read_text()
    page.goto(path.as_uri())
    settled(page)
    assert page.locator("#report-data").text_content() == original
    assert page.locator("#experiment-list tbody tr").count() == 2
    assert page.locator("#pareto-panel").is_visible()
    assert page.locator("#filter-cpu").input_value() == "2"
    assert page.locator("#comparison-purpose").input_value() == "cross-hardware"
    exported = download_json(page)
    assert len(exported["analysis_bundle"]["configs"]) == 2
    # A clean report can import that JSON, preserving unknown CSV comparability.
    source.write_text(source.read_text().replace("original", "third"))
    target = write_report(load_analysis([source]), tmp_path / "third.html")
    page.goto(target.as_uri())
    settled(page)
    import_file(page, "bundle.json", json.dumps(exported))
    assert page.locator("#experiment-list tbody tr").count() == 3
    assert "unknown" in page.locator("#experiment-list").inner_text()
    page.goto(target.as_uri())
    settled(page)
    import_file(page, "saved.html", path.read_text())
    assert page.locator("#experiment-list tbody tr").count() == 3


def test_import_existing_html_retains_python_statistics_and_ignores_scripts(page, source, tmp_path):
    source.write_text(source.read_text().replace("original", "second"))
    report = write_report(load_analysis([source]), tmp_path / "second.html")
    content = report.read_text().replace('</body>', '<script>throw new Error("IMPORTED SCRIPT EXECUTED")</script></body>')
    import_file(page, "second.html", content)
    assert page.locator("#experiment-list tbody tr").count() == 2
    page.click("[data-view=statistics]")
    settled(page)
    payload = download_json(page)
    assert len(payload["reports"]) == 2
    assert payload["reports"][1]["groups"][0]["n_windows"] == 2
    assert payload["reports"][1]["groups"][0]["ci_low"] is None


def test_drag_drop_cancel_and_repeat_import_delete_release_chart_nodes(page, source):
    content = source.read_text().replace("original", "second")
    cdp = page.context.new_cdp_session(page)
    cdp.send("HeapProfiler.collectGarbage")
    initial_heap = cdp.send("Runtime.getHeapUsage")["usedSize"]
    for _ in range(3):
        page.locator("#experiment-manager").evaluate("""(element, content) => {
          const transfer = new DataTransfer(); transfer.items.add(new File([content], 'drop.csv'));
          element.dispatchEvent(new DragEvent('drop', {dataTransfer:transfer, bubbles:true}));
        }""", content)
        page.wait_for_function("() => document.body.dataset.importing === 'false'")
        for view in ("pareto", "scaling", "experiments"):
            page.click(f"[data-view={view}]")
            settled(page)
        assert page.locator("#experiment-list tbody tr").count() == 2
        page.locator("#experiment-list button[data-remove]").click()
        settled(page)
        assert page.locator("#experiment-list tbody tr").count() == 1
        assert page.locator("#cross-model-charts .js-plotly-plot").count() == 1
    cdp.send("HeapProfiler.collectGarbage")
    assert cdp.send("Runtime.getHeapUsage")["usedSize"] - initial_heap < 12 * 1024 * 1024
    cdp.detach()
    # A CSV spanning many chunks remains cancellable and never partially publishes.
    header = content.splitlines()[0]
    large = header + "\n" + "\n".join(
        f"2,4,off,32,{index},0,ok,2,2,5,2,1,fixture/model,large" for index in range(100000))
    page.locator("#experiment-files").set_input_files({"name": "large.csv", "mimeType": "text/csv", "buffer": large.encode()})
    page.click("#cancel-import")
    page.wait_for_function("() => document.body.dataset.importing === 'false'")
    assert "Cancelled" in page.locator("#import-status").inner_text()
    assert page.locator("#experiment-list tbody tr").count() == 1


def test_bilingual_narrow_layout_reset_and_filtered_exports(page, source):
    import_file(page, "second.csv", source.read_text().replace("original", "second"))
    page.set_viewport_size({"width": 390, "height": 844})
    for language in ("en", "zh"):
        page.select_option("#language", language)
        settled(page)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        assert page.locator("html").get_attribute("lang") == ("en" if language == "en" else "zh-CN")
    page.select_option("#filter-experiment_name", "second.csv")
    settled(page)
    assert page.locator("#matrix tbody tr").count() == 1
    payload = download_json(page)
    assert len(payload["analysis_bundle"]["configs"]) == 1
    assert all(row["run_id"] == "second" for row in payload["groups"])
    page.click("#reset")
    settled(page)
    assert page.locator("#matrix tbody tr").count() == 2
    page.select_option("#baseline", index=1)
    page.click("#reset-baseline")
    settled(page)
    assert page.locator("#baseline").input_value() == ""


def test_imported_profiles_use_python_rules_and_never_invent_independent_ci(page, tmp_path):
    from acprof.analysis.conditions import compare_profiles
    from tests.support.comparison_fixtures import ComparisonFixture
    fixture = ComparisonFixture()
    fixture.build(tmp_path / "profiles")
    # Distinct CSV bytes permit independent artifacts without implying independence.
    fixture.change_json(fixture.right, "hardware_conditions.json", lambda hardware:
        hardware["cases"]["1c_4g_off"].update(cpu_model=["different CPU"]))
    csv_path = fixture.right / "result_all.csv"
    csv_path.write_text(csv_path.read_text().replace("0.25", "0.5"))
    fixture.refresh(fixture.right)
    models = [load_analysis([directory]) for directory in (fixture.left, fixture.right)]
    bundles = [report_payload(model, statistics_options={"resamples": 19}) for model in models]
    for index, bundle in enumerate(bundles):
        import_file(page, f"profile-{index}.json", json.dumps(bundle))
    assert page.locator("#experiment-list tbody tr").count() == 3, page.locator("#import-status").inner_text()
    for purpose in ("same-hardware", "cross-hardware", "resource-scaling"):
        expected = compare_profiles(*(model.configs[0]["comparison_profiles"][purpose] for model in models))["status"]
        result = page.evaluate("""({bundles,purpose}) => {
          const workspace = ACProfWorkspace(JSON.parse(document.getElementById('report-data').textContent));
          workspace.add(bundles, 'profiles');
          const rows = workspace.data.configs.slice(1);
          const context = ACProfViews.comparisonContext(rows[0], purpose, workspace.data.baseline_comparisons, workspace.data.registry);
          return {qualification:context.qualification(rows[1]), change:context.delta(rows[1], 'latency_app_s')};
        }""", {"bundles": bundles, "purpose": purpose})
        assert result["qualification"]["status"] == expected
        if expected == "incompatible":
            assert result["change"]["delta"] is None
        else:
            assert result["change"]["statistics_status"] == "not_computed"
            assert result["change"]["delta"] == .25


def test_precomputed_independent_report_survives_import_and_small_samples_reject_ci(page, tmp_path):
    from tests.support.independent_comparison_fixtures import IndependentComparisonFixture
    fixture = IndependentComparisonFixture()
    fixture.build(None, tmp_path / "independent")
    left = fixture.replicate("left", 0, [1, 2, 3])
    right = fixture.replicate("right", 0, [2, 4, 6])
    bundle = report_payload(load_analysis([left, right]), statistics_options={"metrics": ["latency_app_s"], "resamples": 19})
    bad = json.loads(json.dumps(bundle))
    bad["experiment_comparisons"][0]["report"]["groups"][0]["difference_ci"] = [1, 3]
    import_file(page, "bad-ci.json", json.dumps(bad))
    assert "区间" in page.locator("#import-status").inner_text()
    assert page.locator("#experiment-list tbody tr").count() == 1
    import_file(page, "independent.json", json.dumps(bundle))
    assert page.locator("#experiment-list tbody tr").count() == 3
    page.click("[data-view=experiments]")
    settled(page)
    page.select_option("#experiment-left", page.locator("#experiment-left option").filter(has_text="left-0").first.get_attribute("value"))
    page.select_option("#experiment-right", page.locator("#experiment-right option").filter(has_text="right-0").first.get_attribute("value"))
    page.click("[data-view=experiments]")
    settled(page)
    assert "独立运行不足" in page.locator("#experiments tbody").inner_text()
    assert "— / —" in page.locator("#experiments tbody").inner_text()
    exported = download_json(page)
    assert exported["groups"][0]["difference"] == 2
    assert exported["groups"][0]["difference_ci"] is None


def test_large_csv_streams_with_progress_and_releases_rows(page):
    result = page.evaluate("""async () => {
      const header = 'cpu_cores,mem_cap_gb,gpu_mode,input_scale,repeat_idx,warmup,status,latency_app_s\\n';
      const parts = [header];
      for (let i=0; i<30000; i++) parts.push(`2,4,off,32,${i},0,ok,2\\n`);
      const file = new File(parts, 'large.csv');
      let callbacks=0, ticks=0;
      const timer=setInterval(() => ticks++, 0);
      try {
        const bundle=await ACProfImport.readCsv(file, JSON.parse(document.getElementById('report-data').textContent).registry,
          {chunkSize:32768, onProgress:() => callbacks++});
        return {configs:bundle.configs.length, windows:bundle.configs[0].valid_windows,
          mean:bundle.configs[0].metrics.latency_app_s.value, callbacks, ticks,
          bytes:JSON.stringify(bundle).length, sourceBytes:file.size};
      } finally {clearInterval(timer);}
    }""")
    assert result["configs"] == 1
    assert result["windows"] == 30000
    assert result["mean"] == 2
    assert result["callbacks"] > 10
    assert result["ticks"] > 5
    assert result["bytes"] < result["sourceBytes"] / 5


def test_equal_csv_values_with_distinct_python_run_evidence_are_separate_experiments(page, tmp_path):
    from tests.support.independent_comparison_fixtures import IndependentComparisonFixture
    fixture = IndependentComparisonFixture()
    fixture.build(None, tmp_path / "equal")
    left = fixture.replicate("left", 0, [1, 2, 3])
    right = fixture.replicate("right", 0, [1, 2, 3])
    bundle = report_payload(load_analysis([left, right]), statistics_options={"resamples": 19})
    import_file(page, "equal.json", json.dumps(bundle))
    assert page.locator("#experiment-list tbody tr").count() == 3, page.locator("#import-status").inner_text()


def test_formal_group_ci_is_retained_without_browser_estimation(page, tmp_path):
    from tests.support.independent_comparison_fixtures import IndependentComparisonFixture
    fixture = IndependentComparisonFixture()
    fixture.build(None, tmp_path / "groups")
    left = [fixture.replicate("left", i, [i + 1, i + 2, i + 3]) for i in range(3)]
    right = [fixture.replicate("right", i, [i + 4, i + 5, i + 6]) for i in range(3)]
    bundle = report_payload(load_analysis([*left, *right]), comparison_groups=(left, right),
                            statistics_options={"metrics": ["latency_app_s"], "resamples": 19})
    expected = bundle["experiment_comparisons"][0]["report"]["groups"]
    assert expected[0]["difference_ci"] is not None
    import_file(page, "formal.json", json.dumps(bundle))
    assert page.locator("#experiment-list tbody tr").count() == 7, page.locator("#import-status").inner_text()
    page.click("[data-view=experiments]")
    settled(page)
    choices = page.locator("#experiment-left option").evaluate_all("nodes => nodes.map(node => node.value)")
    page.select_option("#experiment-left", next(value for value in choices if len(json.loads(value)) == 3))
    choices = page.locator("#experiment-right option").evaluate_all("nodes => nodes.map(node => node.value)")
    page.select_option("#experiment-right", next(value for value in choices if len(json.loads(value)) == 3))
    page.click("[data-view=experiments]")
    settled(page)
    assert download_json(page)["groups"] == expected


def test_descriptive_changes_match_python_and_conflicting_csv_inputs_withhold_deltas(page):
    from acprof.analysis.changes import metric_change
    pairs = [(0, 4), (-2, 4), (2, 0), (2, 4), (None, 4), (1e-6, 1e2)]
    for direction in ("lower", "higher", "neutral"):
        actual = page.evaluate("({pairs,direction}) => pairs.map(([a,b]) => ACProfViews.descriptiveChange(a,b,direction))",
                               {"pairs": pairs, "direction": direction})
        assert actual == [metric_change(a, b, direction=direction) for a, b in pairs]
    result = page.evaluate("""() => {
      const a={experiment_id:'a',config_id:'a',comparison_profiles:{},cpu:2,memory:4,device:'CPU',input_scale:32,
        input_case:JSON.stringify({task_param:'classification',batch_size:1}),metrics:{x:{value:2}}};
      const b={...a,experiment_id:'b',config_id:'b',input_case:JSON.stringify({task_param:'generation',batch_size:1}),metrics:{x:{value:4}}};
      return ACProfViews.comparisonContext(a,'same-hardware').delta(b,'x');
    }""")
    assert result["reason"] == "incompatible"
    assert "input_case" in result["reasons"]
    assert result["delta"] is None


def test_process_inherited_resources_remain_unknown(page, source):
    content = "cpu_cores,mem_cap_gb,gpu_mode,input_scale,repeat_idx,warmup,status,latency_app_s,runtime_kind\nnan,nan,off,32,0,0,ok,2,process\n"
    import_file(page, "process.csv", content)
    assert page.locator("#experiment-list tbody tr").count() == 2
    configs = download_json(page)["analysis_bundle"]["configs"]
    config = next(config for config in configs if config["runtime_kind"] == "process")
    assert config["cpu"] is None
    assert config["memory"] is None
    assert config["metrics"]["latency_app_s"]["value"] == 2


def test_legacy_unknown_runs_keep_distinct_models(page, source):
    content = source.read_text().replace('original', 'unknown')
    lines = content.splitlines()
    content = '\n'.join(lines + [line.replace('fixture/model', 'fixture/other') for line in lines[1:]]) + '\n'
    actual = page.evaluate("""async text => {
      const registry=JSON.parse(document.getElementById('report-data').textContent).registry;
      return (await ACProfImport.readCsv(new File([text],'legacy.csv'),registry)).configs;
    }""", content)
    assert len(actual) == 2
    assert [config['valid_windows'] for config in actual] == [2, 2]
    assert {config['run_id'] for config in actual} == {'unknown'}


def test_experiment_selectors_share_one_visible_comparison(page, source):
    for name in ("second", "third"):
        import_file(page, name + ".csv", source.read_text().replace("original", name))
    page.select_option("#active-experiment-left", index=1)
    third = page.locator("#active-experiment-right option").filter(has_text="third.csv").get_attribute("value")
    page.select_option("#active-experiment-right", third)
    page.click("[data-view=experiments]")
    settled(page)
    assert "third.csv" in page.locator("#experiment-right").input_value()
    second = page.locator("#experiment-right option").filter(has_text="second.csv").get_attribute("value")
    page.select_option("#experiment-right", second)
    settled(page)
    assert "second.csv" in page.locator("#active-experiment-right option:checked").inner_text()
    rows = download_json(page)["groups"]
    assert rows
    assert {row["run_id"] for row in rows} == {"second"}
