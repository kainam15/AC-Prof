"""Portable report contract and offline browser experiment management."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path

import pytest

from acprof.analysis.model import load_analysis
from acprof.plotting.report import report_payload, write_report
from tests.analysis.statistics.test_observed_energy_quality import energy_csv as energy_csv


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


@pytest.mark.parametrize('width', [390, 1280])
def test_cached_precision_survives_export_and_is_not_reused_after_browser_recalculation(page, source, tmp_path, width):
    from acprof.analysis.statistics_report import (
        build_statistics_report,
        save_unique_statistics_report,
    )
    source.write_text(
        'cpu_cores,mem_cap_gb,gpu_mode,input_scale,repeat_idx,warmup,status,latency_app_s\n'
        '2,4,off,32,0,0,ok,1\n2,4,off,32,1,0,ok,2\n2,4,off,32,2,0,ok,3\n')
    saved = build_statistics_report(source, metrics=['latency_app_s'], precision_target=.05, resamples=100)
    save_unique_statistics_report(tmp_path / 'analysis', saved)
    report = write_report(load_analysis([source]), tmp_path / 'precision.html')
    page.goto(report.as_uri())
    page.set_viewport_size({'width': width, 'height': 900})
    settled(page)
    page.click('[data-view=statistics]')
    settled(page)
    original = download_json(page)
    assert original['precision'] == saved['precision']
    assert original['groups'][0]['precision_status'] == 'not_met'
    assert 'not_met' in page.locator('#statistics tbody').inner_text()
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')

    page.locator('#statistics-parameters').evaluate('(element) => element.open = true')
    page.locator('#statistics-resamples').select_option('1000')
    page.click('#statistics-recalculate')
    page.wait_for_function("() => document.body.dataset.calculating === 'false'")
    settled(page)
    recalculated = download_json(page)
    assert recalculated['engine'] == 'browser'
    assert 'precision' not in recalculated
    assert 'precision_status' not in recalculated['groups'][0]
    page.click('#statistics-restore')
    settled(page)
    assert download_json(page)['precision'] == original['precision']


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
    page.click("#report-filters > summary")
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
    page.click("#report-filters > summary")
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


@pytest.fixture
def linked_bundle(tmp_path):
    paths = []
    for name, latency in (("left", 2), ("right", 20)):
        path = tmp_path / f"{name}.csv"
        path.write_text(
            "cpu_cores,mem_cap_gb,gpu_mode,input_scale,repeat_idx,warmup,status,latency_app_s,run_id\n"
            f"2,4,off,32,0,0,ok,{latency},{name}\n"
            f"4,4,off,32,0,0,ok,{latency * 2},{name}\n", encoding="utf-8")
        paths.append(path)
    return report_payload(load_analysis(paths), statistics_options={"metrics": ["latency_app_s"], "resamples": 19})


@pytest.mark.parametrize("extension", ["json", "html"])
@pytest.mark.parametrize("defect", ["unused_source", "foreign_config", "foreign_report"])
def test_bundle_source_links_reject_before_changing_workspace(page, linked_bundle, defect, extension):
    damaged = json.loads(json.dumps(linked_bundle))
    if defect == "unused_source":
        damaged["sources"].append({**damaged["sources"][0], "path": "unused.csv", "sha256": "a" * 64})
    elif defect == "foreign_config":
        damaged["window_statistics"][0]["groups"][0]["config_id"] = damaged["configs"][2]["config_id"]
    else:
        report = damaged["window_statistics"][0]
        report["source"] = damaged["sources"][1]["path"]
        report["result_sha256"] = damaged["sources"][1]["sha256"]
    content = json.dumps(damaged)
    if extension == "html":
        content = f'<script id="report-data" type="application/json">{content}</script>'
    before = download_json(page)["analysis_bundle"]
    import_file(page, f"damaged.{extension}", content)
    assert "来源" in page.locator("#import-status").inner_text()
    assert page.locator("#experiment-list tbody tr").count() == 1
    assert download_json(page)["analysis_bundle"] == before
    # Rejection must not reserve these sources/run IDs or poison a later valid import.
    import_file(page, "valid.json", json.dumps(linked_bundle))
    assert page.locator("#experiment-list tbody tr").count() == 3
    page.click("[data-view=statistics]")
    settled(page)
    payload = download_json(page)
    assert sorted(group["mean"] for report in payload["reports"][1:] for group in report["groups"]) == [2, 4, 20, 40]


def test_bundle_source_links_survive_filter_export_remove_and_html_restore(page, linked_bundle, tmp_path):
    import_file(page, "valid.json", json.dumps(linked_bundle))
    page.click("#report-filters > summary")
    page.select_option("#filter-cpu", "2")
    settled(page)
    exported = download_json(page)["analysis_bundle"]
    assert len(exported["configs"]) == 3
    assert page.evaluate("bundle => ACProfImport.validate(bundle, bundle.registry)", exported) == exported
    page.click("#reset")
    settled(page)
    page.locator("#experiment-list button[data-remove]").first.click()
    settled(page)
    exported = download_json(page)["analysis_bundle"]
    assert len(exported["sources"]) == 2
    assert all(config["run_id"] != "left" for config in exported["configs"])
    assert page.evaluate("bundle => ACProfImport.validate(bundle, bundle.registry)", exported) == exported
    with page.expect_download() as result:
        page.click("#export-html")
    saved = tmp_path / "retained.html"
    result.value.save_as(saved)
    page.goto(saved.as_uri())
    settled(page)
    assert download_json(page)["analysis_bundle"] == exported


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


@pytest.mark.parametrize("chunk_size", [pytest.param(None, id="default-256k-crlf"), pytest.param(1, id="bytewise-crlf")])
def test_read_csv_preserves_quoted_records_at_chunk_edges(page, tmp_path, chunk_size):
    header = "cpu_cores,mem_cap_gb,gpu_mode,input_scale,repeat_idx,warmup,status,latency_app_s,notes,model\r\n"
    first = header + "2,4,off,32,0,0,ok,2,"
    quoted_model = '"雪😀,""quoted""\r\nsecond line"'
    end = "," + quoted_model + "\r"
    # In the default-size case, CR is the final byte of the first 256 KiB read.
    padding = 256 * 1024 - len((first + end).encode()) if chunk_size is None else 0
    content = first + "x" * padding + end + "\n\r\n2,4,off,32,1,0,ok,4,," + quoted_model
    path = tmp_path / "quoted.csv"
    path.write_bytes(content.encode())
    python_config, = load_analysis([path]).configs
    imported = page.evaluate("""async ({text, size}) => {
      const file = new File([text], "quoted.csv");
      const {registry} = JSON.parse(document.getElementById("report-data").textContent);
      const bundle = size == null ? await ACProfImport.readCsv(file, registry)
        : await ACProfImport.readCsv(file, registry, {chunkSize:size});
      return {sha256:bundle.sources[0].sha256, configs:bundle.configs};
    }""", {"text": content, "size": chunk_size})
    assert imported["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    config, = imported["configs"]
    assert config["model"] == python_config["model"] == '雪😀,"quoted"\r\nsecond line'
    assert config["valid_windows"] == python_config["valid_windows"] == 2
    assert config["metrics"]["latency_app_s"]["value"] == python_config["metrics"]["latency_app_s"]["value"] == 3


@pytest.mark.parametrize(("scenario", "expected"), [
    ("large-first-block", "2 windows"), ("unfinished-header", "header exceeds"), ("unfinished-row", "row exceeds"),
])
def test_csv_buffer_limit_applies_to_unfinished_records(page, scenario, expected):
    outcome = page.evaluate(r"""async scenario => {
      const limit = 8 * 1024 * 1024;
      const header = "cpu_cores,mem_cap_gb,gpu_mode,input_scale,repeat_idx,warmup,status,latency_app_s,notes\r\n";
      const field = "x".repeat(limit / 2);
      const row = "2,4,off,32,0,0,ok,2,";
      const parts = scenario === "large-first-block" ? [header, row, field, "\r\n2,4,off,32,1,0,ok,4,", field]
        : scenario === "unfinished-header" ? [field, field, "x".repeat(limit / 4), "\r\n"]
        : [header, row, '"', field, field, "x".repeat(limit / 4), '"\r\n'];
      const file = new File(parts, "sized.csv");
      const {registry} = JSON.parse(document.getElementById("report-data").textContent);
      try {
        const options = {chunkSize:scenario === "large-first-block" ? file.size : limit / 4};
        const bundle = await ACProfImport.readCsv(file, registry, options);
        return bundle.configs[0].valid_windows + " windows";
      } catch (err) { return err.message; }
    }""", scenario)
    if scenario == "large-first-block":
        assert outcome == expected
    else:
        assert expected in outcome


def test_long_quoted_field_does_not_rescan_each_chunk_prefix(page):
    result = page.evaluate(r"""async () => {
      const header = "cpu_cores,mem_cap_gb,gpu_mode,input_scale,repeat_idx,warmup,status,latency_app_s,notes\r\n";
      const text = header + '2,4,off,32,0,0,ok,2,"' + "line\r\n".repeat(16384) +
        'end"\r\n2,4,off,32,1,0,ok,4,done';
      const file = new File([text], "long-field.csv");
      const {registry} = JSON.parse(document.getElementById("report-data").textContent);
      const original = Papa.Parser;
      let parsedChars = 0;
      Papa.Parser = function(options) {
        const parser = new original(options);
        const parse = parser.parse.bind(parser);
        parser.parse = (input, ...args) => { parsedChars += input.length; return parse(input, ...args); };
        return parser;
      };
      try {
        const {configs} = await ACProfImport.readCsv(file, registry, {chunkSize:1024});
        return {sourceChars:text.length, parsedChars, windows:configs[0].valid_windows, mean:configs[0].metrics.latency_app_s.value};
      } finally { Papa.Parser = original; }
    }""")
    assert (result["windows"], result["mean"]) == (2, 3)
    assert result["sourceChars"] <= result["parsedChars"] <= 4 * result["sourceChars"]


@pytest.mark.parametrize("broken", ['"closed"x\r\n', '"open'], ids=["trailing-junk", "unterminated"])
def test_malformed_quoted_record_never_publishes_partial_import(page, broken):
    header = "cpu_cores,mem_cap_gb,gpu_mode,input_scale,repeat_idx,warmup,status,latency_app_s,model\r\n"
    text = header + '2,4,off,32,0,0,ok,2,"good"\r\n2,4,off,32,1,0,ok,4,' + broken
    import_file(page, "broken.csv", text)
    assert "Malformed CSV" in page.locator("#import-status").inner_text()
    assert page.locator("#experiment-list tbody tr").count() == 1


def test_html_and_json_restore_filtered_comparison_offline(page, source, tmp_path):
    import_file(page, "second.csv", source.read_text().replace("original", "second"))
    page.click("#report-filters > summary")
    page.select_option("#baseline", index=1)
    page.select_option("#comparison-purpose", "cross-hardware")
    page.click("#fixed-conditions > summary")
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
    page.click("#report-filters > summary")
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


@pytest.fixture
def same_path_bundles(tmp_path):
    directory = tmp_path / "recorded"
    directory.mkdir()
    path = directory / "result_all.csv"
    path.write_text(
        "cpu_cores,mem_cap_gb,gpu_mode,input_scale,repeat_idx,warmup,status,latency_app_s\n"
        "2,4,off,32,0,0,ok,2\n4,4,off,32,0,0,ok,4\n", encoding="utf-8")
    bundles = []
    for run_id in ("left", "right"):
        (directory / "run_state.json").write_text(json.dumps({"run_id": run_id}))
        bundles.append(report_payload(load_analysis([path]), statistics_options={"metrics": ["latency_app_s"], "resamples": 19}))
    assert bundles[0]["sources"][0]["path"] == bundles[1]["sources"][0]["path"]
    assert bundles[0]["sources"][0]["sha256"] == bundles[1]["sources"][0]["sha256"]
    return bundles


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("local_ids", [False, True])
def test_same_path_runs_have_distinct_import_identities(page, same_path_bundles, reverse, local_ids):
    bundles = same_path_bundles[::-1] if reverse else same_path_bundles
    if local_ids:
        # Portable IDs need only be unique within their own bundle.
        for bundle in bundles:
            ids = {config["config_id"]: f"config-{index}" for index, config in enumerate(bundle["configs"])}
            for config in bundle["configs"]:
                config["config_id"] = ids[config["config_id"]]
            for report in bundle["window_statistics"]:
                for group in [*report["groups"], *report.get("recalculation", {}).get("groups", [])]:
                    group["config_id"] = ids[group["config_id"]]
            bundle["baseline_comparisons"]["pairs"] = {
                ids[key]: {ids[other]: pair for other, pair in pairs.items()}
                for key, pairs in bundle["baseline_comparisons"]["pairs"].items()}
    actual = page.evaluate("""bundles => {
      const original = JSON.parse(document.getElementById('report-data').textContent);
      const before = JSON.stringify(bundles), workspace = ACProfWorkspace(original);
      workspace.add(bundles, 'same path');
      const exported = workspace.bundle(workspace.data.configs);
      return {exported:ACProfImport.validate(exported, original.registry), unchanged:before === JSON.stringify(bundles)};
    }""", bundles)
    assert actual["unchanged"]
    exported = actual["exported"]
    assert len({source["path"] for source in exported["sources"]}) == 3
    assert len({config["config_id"] for config in exported["configs"]}) == 5
    configs = {config["config_id"]: config for config in exported["configs"]}
    for report in exported["window_statistics"][1:]:
        for group in report["groups"]:
            assert configs[group["config_id"]]["source"] == report["source"]
            assert group["mean"] == configs[group["config_id"]]["metrics"]["latency_app_s"]["value"]
    for source in exported["sources"][1:]:
        assert source["original_path"] == bundles[0]["sources"][0]["path"]
    for config in exported["configs"][1:]:
        assert set(exported["baseline_comparisons"]["pairs"][config["config_id"]]).issubset(configs)


@pytest.mark.parametrize("width", [1280, 390])
def test_same_path_runs_compare_export_restore_and_remove(page, same_path_bundles, tmp_path, width):
    original_url = page.url
    page.set_viewport_size({"width": width, "height": 900})
    for index, bundle in enumerate(same_path_bundles):
        content = json.dumps(bundle)
        if index:
            content = f'<script id="report-data" type="application/json">{content}</script>'
        import_file(page, "run.html" if index else "run.json", content)
    assert page.locator("#experiment-list tbody tr").count() == 3
    assert page.locator("#experiment-left option").count() == 3
    exported = download_json(page)["analysis_bundle"]
    left, right = [next(config for config in exported["configs"] if config["run_id"] == run and config["cpu"] == 2)
                   for run in ("left", "right")]
    page.select_option("#active-experiment-left", left["experiment_id"])
    page.select_option("#active-experiment-right", right["experiment_id"])
    page.click("#report-filters > summary")
    page.select_option("#filter-cpu", "2")
    page.select_option("#comparison-purpose", "same-hardware")
    page.click("[data-view=experiments]")
    settled(page)
    assert json.loads(page.locator("#experiment-left").input_value()) == [left["source"]]
    assert json.loads(page.locator("#experiment-right").input_value()) == [right["source"]]
    page.select_option("#experiment-metric", "latency_app_s")
    settled(page)
    comparison = download_json(page)
    assert len(comparison["groups"]) == 1
    assert comparison["groups"][0]["baseline_value"] == comparison["groups"][0]["value"] == 2
    assert comparison["groups"][0]["delta"] is None
    assert comparison["groups"][0]["comparison"]["status"] == "incompatible"
    assert "invalid_artifacts" in comparison["groups"][0]["comparison"]["reasons"]
    assert comparison["groups"][0]["statistics_status"] == "not_computed"
    page.click("[data-view=matrix]")
    settled(page)
    for config in (left, right):
        identity = page.locator(f'#matrix tr[data-config-id="{config["config_id"]}"] td').first
        assert f"run_id={config['run_id']}" in identity.inner_text()
    page.locator(f'#matrix tr[data-config-id="{right["config_id"]}"]').click()
    settled(page)
    filtered = download_json(page)["analysis_bundle"]
    assert len(filtered["configs"]) == len(filtered["sources"]) == 2
    with page.expect_download() as result:
        page.click("#export-html")
    saved = tmp_path / "same-path.html"
    result.value.save_as(saved)
    page.goto(saved.as_uri())
    settled(page)
    assert page.locator("#baseline").input_value() == left["config_id"]
    assert page.locator("#filter-cpu").input_value() == "2"
    assert page.locator("#matrix tr.selected").get_attribute("data-config-id") == right["config_id"]
    assert download_json(page)["analysis_bundle"] == filtered
    page.click("#experiment-manager > summary")
    page.locator(f'#experiment-list button[data-remove="{left["experiment_id"]}"]').click()
    settled(page)
    remaining = download_json(page)["analysis_bundle"]
    assert [config["run_id"] for config in remaining["configs"]] == ["right"]
    assert page.evaluate("bundle => ACProfImport.validate(bundle, bundle.registry)", remaining) == remaining
    page.goto(original_url)
    settled(page)
    import_file(page, "filtered.json", json.dumps(filtered))
    assert page.locator("#experiment-list tbody tr").count() == 3
    imported = download_json(page)["analysis_bundle"]
    assert {config["config_id"] for config in imported["configs"] if config["run_id"] != "original"} == {left["config_id"], right["config_id"]}


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



def test_cross_runtime_report_only_exposes_qualified_metric_deltas(page):
    result = page.evaluate("""() => {
        const baseline={config_id:"docker",experiment_id:"a"}, target={config_id:"process",experiment_id:"b"};
        const pairs={pairs:{docker:{process:{policies:{"same-hardware":0},cross_runtime:true,
          metric_reasons:{latency_app_p95_s:"incomplete_cross_runtime_metric_evidence"},
          metrics:{latency_app_s:{delta:0.1}}}}},qualifications:[{status:"incompatible",matched_cross_runtime:true}]};
        const context=ACProfViews.comparisonContext(baseline,"same-hardware",pairs);
        return [context.delta(target,"latency_app_s"), context.delta(target,"latency_s"), context.automaticQualification(target),
          context.delta(target,"latency_app_p95_s")];
    }""")
    assert result[0] == {"delta": 0.1, "scope": "matched_cross_runtime_metric", "statistics_status": "not_computed"}
    assert result[1]["delta"] is None
    assert result[2]["status"] == "incompatible"
    assert result[3]["delta"] is None
    assert result[3]["reasons"] == ["incomplete_cross_runtime_metric_evidence"]


def test_csv_import_checks_sampling_only_for_contributing_readings(page, tmp_path):
    metric = "throughput_samples_per_s"
    path = tmp_path / "sampling.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", "repeat_idx", "warmup",
                         "status", "result_origin", metric, "resource_metric_quality"])
        for cpu in (1, 2):
            samples = [
                ("ok", "", 0, "complete", "application"),
                ("ok", "", 6, "complete", "application"),
                ("ok", "", "", "complete", "packet"),
                ("warn", "", 100, "complete", "packet"),
                ("ok", "inferred_not_measured", 200, "complete", "packet"),
                ("ok", "", 300, "partial", "packet"),
            ]
            if cpu == 2:
                samples.append(("ok", "", 4, "complete", "packet"))
            for repeat, (status, origin, value, quality, denominator) in enumerate(samples):
                evidence = {metric: {"quality": quality, "sampling": {"latency_denominator": denominator}}}
                writer.writerow([cpu, 4, "off", 32, repeat, 0, status, origin, value, json.dumps(evidence)])

    expected = load_analysis([path]).configs
    import_file(page, path.name, path.read_text())
    imported = [config for config in download_json(page)["analysis_bundle"]["configs"]
                if config.get("statistics_status") == "not_computed"]
    for configs in (expected, imported):
        compatible, mixed = sorted(configs, key=lambda config: config["cpu"])
        assert compatible["metrics"][metric] == {"value": 3, "n": 2, "missing": 2, "aggregation": "mean"}
        assert mixed["metrics"][metric] == {"value": None, "n": 3, "missing": 2, "aggregation": "mean"}
        observations = compatible["metric_observations"][metric]
        assert {item["quality"] for item in observations} == {"complete", "partial", "unavailable"}
        assert any(item.get("reason") == "missing_window_value" for item in observations)
        assert any(item["quality"] == "complete" and item["sampling"]["latency_denominator"] == "packet"
                   for item in observations)


def test_nested_sampling_survives_csv_aggregation_and_json_baseline_comparison(page, tmp_path):
    metric = "latency_app_p95_s"
    sampling = {"policy": "periodic", "windows": [{"phase": "active", "interval_s": 1}, {"phase": "idle"}]}
    reordered = {"windows": [{"interval_s": 1, "phase": "active"}, {"phase": "idle"}], "policy": "periodic"}
    reversed_windows = {**sampling, "windows": sampling["windows"][::-1]}

    def write_samples(run, rows):
        path = tmp_path / f"{run}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", "repeat_idx", "warmup",
                             "status", "run_id", metric, "resource_metric_quality"])
            for cpu, repeat, value, contract in rows:
                evidence = {metric: {"quality": "complete", "sampling": contract}}
                writer.writerow([cpu, 4, "off", 32, repeat, 0, "ok", run, value, json.dumps(evidence)])
        return path

    reference = write_samples("reference", [(1, 0, 1, sampling)])
    candidate = write_samples("candidate", [(1, 0, 2, sampling), (1, 1, 6, reordered),
                                           (2, 0, 2, sampling), (2, 1, 6, reversed_windows)])
    import_file(page, "reference.json", json.dumps(report_payload(load_analysis([reference]))))
    import_file(page, candidate.name, candidate.read_text())

    configs = download_json(page)["analysis_bundle"]["configs"]
    imported = [config for config in configs if config["run_id"] == "candidate"]
    for group in (load_analysis([candidate]).configs, imported):
        matching, mixed = sorted(group, key=lambda c: c["cpu"])
        assert matching["metrics"][metric]["value"] == 4
        assert matching["metrics"][metric]["n"] == 2
        assert [obs["sampling"] for obs in matching["metric_observations"][metric]] == [sampling]
        assert mixed["metrics"][metric]["value"] is None
        assert len(mixed["metric_observations"][metric]) == 2

    baseline = next(config for config in configs if config["run_id"] == "reference")
    page.click("#report-filters > summary")
    page.select_option("#baseline", baseline["config_id"])
    settled(page)
    row = next(row for row in download_json(page)["groups"]
               if row["run_id"] == "candidate" and row["cpu"] == 1 and row["metric"] == metric)
    assert (row["delta"], row["scope"], row["statistics_status"]) == (3, "browser_descriptive", "not_computed")


def test_imported_metric_delta_requires_symmetric_observation_evidence(page):
    from acprof.results.profile import observation_comparison_reason, observations_comparable

    known = {"quality": "complete"}
    cases = [
        ([], [], ""),
        ([], [known], "metric_contract_missing_on_one_side"),
        ([known], [], "metric_contract_missing_on_one_side"),
        ([known], [{**known, "sampling": None}], "different_source_scope_window_or_sampling"),
        ([known], [{"quality": "unknown", "reason": "legacy"}], ""),
        ([known], [{"quality": "partial"}], "incomplete_metric_quality"),
        ([{**known, "sampling": [{"a": 1, "b": 2}]}], [{**known, "sampling": [{"b": 2, "a": 1}]}], ""),
        ([{**known, "sampling": {"interval_s": 1}}], [{**known, "sampling": {"interval_s": 2}}],
         "different_source_scope_window_or_sampling"),
    ]
    observed = page.evaluate("""cases => cases.map(([left, right]) => {
      const config = (id, evidence, value) => ({experiment_id:id, config_id:id, metrics:{x:{value}},
        ...(evidence.length ? {metric_observations:{x:evidence}} : {})});
      return ACProfViews.comparisonContext(config('a', left, 2)).delta(config('b', right, 6), 'x');
    })""", cases)
    for (left, right, detail), change in zip(cases, observed, strict=True):
        assert observation_comparison_reason(left, right) == detail
        assert observations_comparable(left, right) is (not detail)
        assert change.get("reasons", []) == ([detail] if detail else [])
        assert (change["delta"], change["reason"]) == (
            (None, "different_or_incomplete_metric_contract") if detail else (4, ""))


def test_withheld_metric_reason_visible_in_matrix_experiments_and_exports(page, source):
    metric = "latency_app_p95_s"
    with source.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["run_id"] = "candidate"
        row["resource_metric_quality"] = json.dumps({metric: {"quality": "complete", "sampling": {"interval_s": 2}}})
    content = io.StringIO()
    writer = csv.DictWriter(content, list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    import_file(page, "candidate.csv", content.getvalue())
    configs = download_json(page)["analysis_bundle"]["configs"]
    left, right = (next(config for config in configs if config["run_id"] == run) for run in ("original", "candidate"))
    page.select_option("#active-experiment-left", left["experiment_id"])
    page.select_option("#active-experiment-right", right["experiment_id"])
    page.click("#report-filters > summary")
    page.select_option("#baseline", left["config_id"])
    for view in ("matrix", "experiments"):
        page.click(f"[data-view={view}]")
        if view == "experiments":
            page.select_option("#experiment-metric", metric)
        for lang, text in (("zh", "指标的观测口径不同"), ("en", "Metric observation contracts differ")):
            page.select_option("#language", lang)
            settled(page)
            row = page.locator(f'#{view} tr[data-config-id="{right["config_id"]}"]')
            assert text in row.inner_text()
        exported = next(row for row in download_json(page)["groups"]
                        if row["config_id"] == right["config_id"] and row["metric"] == metric)
        assert exported["value"] == 3
        assert exported["delta"] is None
        assert exported["reason"] == "different_or_incomplete_metric_contract"
        assert exported["reasons"] == ["different_source_scope_window_or_sampling"]
        with page.expect_download() as result:
            page.click("#export-csv")
        with Path(result.value.path()).open(newline="") as stream:
            entry = next(row for row in csv.DictReader(stream)
                         if row["config_id"] == right["config_id"] and row["metric"] == metric)
        assert entry["delta"] == ""
        assert json.loads(entry["reasons"]) == exported["reasons"]


def test_precomputed_metric_reason_survives_import_and_rejects_conflicting_delta(page, linked_bundle):
    metric = "latency_app_s"
    left, right = sorted((c for c in linked_bundle["configs"] if c["run_id"] == "left"), key=lambda c: c["cpu"])
    pair = linked_bundle["baseline_comparisons"]["pairs"][left["config_id"]][right["config_id"]]
    pair["metric_reasons"] = {metric: "incomplete_metric_quality"}
    import_file(page, "conflict.json", json.dumps(linked_bundle))
    assert "Invalid metric comparison reason" in page.locator("#import-status").inner_text()
    assert page.locator("#experiment-list tbody tr").count() == 1

    del pair["metrics"][metric]
    extra_metric = "additional_latency_s"
    linked_bundle["registry"][extra_metric] = {**linked_bundle["registry"][metric], "name": extra_metric}
    for config in linked_bundle["configs"]:
        config["metrics"][extra_metric] = config["metrics"][metric].copy()
    expected_reasons = {metric: "incomplete_metric_quality", extra_metric: "additional_quality_detail"}
    pair["metric_reasons"] = expected_reasons
    import_file(page, "corrected.json", json.dumps(linked_bundle))
    saved = download_json(page)["analysis_bundle"]
    left, right = sorted((c for c in saved["configs"] if c["run_id"] == "left"), key=lambda c: c["cpu"])
    pair = saved["baseline_comparisons"]["pairs"][left["config_id"]][right["config_id"]]
    assert pair["metric_reasons"] == expected_reasons
    page.click("#report-filters > summary")
    page.click("#custom-metrics > summary")
    page.select_option("#metrics", metric)
    page.select_option("#comparison-purpose", "within-experiment")
    page.select_option("#baseline", left["config_id"])
    settled(page)
    assert "指标观测质量不完整" in page.locator(f'#matrix tr[data-config-id="{right["config_id"]}"]').inner_text()


def test_csv_import_distinguishes_historical_and_partial_token_coverage(page, tmp_path):
    path = tmp_path / "token-coverage.csv"
    token_rate = "output_tokens_per_s_app"
    columns = ["cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", "repeat_idx", "warmup",
               "status", token_rate, "resource_metric_quality"]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for cpu, token_metrics in [(1, {}), (2, {token_rate: {
            "quality": "partial", "reason": "missing_output_token_count",
            "observed_requests": 1, "completed_requests": 2,
        }})]:
            writer.writerow([cpu, 4, "off", 64, 0, 0, "ok", 10, json.dumps({
                "schema_version": 1, "metrics": token_metrics, "collectors": {"client_http": {"status": "available"}}
            })])
    python = load_analysis([path]).configs
    imported = page.evaluate("""async text => {
      const registry = JSON.parse(document.getElementById('report-data').textContent).registry;
      return (await ACProfImport.readCsv(new File([text], 'token-coverage.csv'), registry)).configs;
    }""", path.read_text())
    for configs in (python, imported):
        legacy, partial = sorted(configs, key=lambda config: config["cpu"])
        assert legacy["metric_observations"][token_rate][0]["quality"] == "unknown"
        assert legacy["metrics"][token_rate]["value"] == 10
        assert partial["metric_observations"][token_rate][0]["quality"] == "partial"
        assert partial["metrics"][token_rate]["value"] is None


def test_csv_import_validates_collector_evidence_per_metric(page, tmp_path):
    latency, energy, sibling = "latency_app_s", "cpu_energy_total_j", "process_io_read_bytes"
    available = {"status": "available"}

    def evidence(groups, metrics=None):
        return {"schema_version": 1, "metrics": metrics or {}, "collectors": {"process_io": available, **groups}}

    cases = [
        ("partial", latency, evidence({"client_http": {"status": "partial"}}), 4, "partial"),
        ("null-state", latency, evidence({"client_http": None}), 4, "error"),
        ("bad-state-before-direct", latency, evidence({"client_http": []}, {latency: {"quality": "complete"}}), 4, "error"),
        ("missing-state", latency, evidence({}), 4, "unknown"),
        ("invalid-sampling", energy, evidence({"cpu_energy": {**available, "sampling": []}}), 4, "error"),
        ("invalid-reader", latency, evidence({"client_http": {**available, "reader_errors": {latency: None}}}), 4, "error"),
        ("invalid-reason", latency, evidence({"client_http": {"status": "not_applicable", "reason": {}}}), 4, "error"),
        ("invalid-direct-reason", latency, evidence({}, {latency: {"quality": "complete", "reason": False}}), 4, "error"),
        ("invalid-missing-hint", latency, evidence({"client_http": {**available, "preflight_missing": {latency: {}}}}), None, "error"),
        ("unrelated-reader", latency, evidence({"client_http": {**available, "reader_errors": {"other": []}}}), 4, "complete"),
        ("unused-hint", latency, evidence({"client_http": {**available, "preflight_missing": None}}), 4, "complete"),
        ("direct", latency, evidence({"client_http": {**available, "reader_errors": None}}, {latency: {"quality": "complete"}}), 4, "complete"),
        ("unsupported-runtime", "container_mem_usage_peak_bytes", evidence({"cgroup": None}), 4, "not_applicable"),
        ("legacy", latency, {latency: {"quality": "complete"}}, 4, "complete"),
        ("invalid-envelope", latency, {"schema_version": 1, "metrics": {}, "collectors": None}, 4, "error"),
    ]
    path = tmp_path / "collector-evidence.csv"
    fields = ["cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", "repeat_idx", "warmup", "status",
              "runtime_kind", "model", latency, energy, sibling, "container_mem_usage_peak_bytes", "resource_metric_quality"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for scale, (label, name, detail, value, _) in enumerate(cases, start=1):
            writer.writerow({"cpu_cores": 1, "mem_cap_gb": 4, "gpu_mode": "off", "input_scale": scale, "repeat_idx": 0,
                             "warmup": 0, "status": "ok", "runtime_kind": "process", "model": label,
                             name: value, sibling: 17, "resource_metric_quality": "   " + json.dumps(detail)})
    imported = page.evaluate("""async text => {
      const {registry} = JSON.parse(document.getElementById("report-data").textContent);
      return (await ACProfImport.readCsv(new File([text], "collector-evidence.csv"), registry)).configs;
    }""", path.read_text())
    for configs in (imported, load_analysis([path]).configs):
        for config, (label, name, _, value, quality) in zip(
                sorted(configs, key=lambda item: item["input_scale"]), cases, strict=True):
            observations, metrics = config["metric_observations"], config["metrics"]
            assert observations[name][0]["quality"] == quality, label
            if label in {"invalid-reason", "invalid-direct-reason"}:
                assert observations[name][0]["reason"] == "metric reason must be text", label
            assert metrics[name]["value"] == (value if quality in {"complete", "unknown"} else None), label
            sibling_quality = {"legacy": "unknown", "invalid-envelope": "error"}.get(label, "complete")
            assert observations[sibling][0]["quality"] == sibling_quality, label
            assert metrics[sibling]["value"] == (None if sibling_quality == "error" else 17), label


def test_import_derived_energy_matches_python_windows_and_comparisons(page, energy_csv):
    model = load_analysis([energy_csv])
    expected = {c["cpu"]: c for c in model.configs}
    result = page.evaluate("""async ({text, baseline}) => {
      const {registry} = JSON.parse(document.getElementById("report-data").textContent);
      const bundle = await ACProfImport.readCsv(new File([text], "energy.csv"), registry);
      const byCpu = Object.fromEntries(bundle.configs.map(c => [c.cpu, {...c, experiment_id:"csv"}]));
      const names = ["observed_energy_per_request_j", "observed_energy_j"];
      const context = ACProfViews.comparisonContext({...baseline, experiment_id:"python"}, "within-experiment", {}, registry);
      const changes = [11, 12].map(cpu => names.map(name => context.delta(byCpu[cpu], name)));
      const legacy = ACProfImport.clone(bundle);
      for (const c of legacy.configs) for (const name of names) delete c.metric_observations[name];
      const old = ACProfImport.validate(legacy, registry).configs.find(c => c.cpu === 12);
      const legacyDeltas = names.map(name => context.delta({...old, experiment_id:"legacy"}, name));
      return {configs:bundle.configs, changes, legacyDeltas};
    }""", {"text": energy_csv.read_text(), "baseline": expected[10]})
    for actual in result["configs"]:
        config = expected[actual["cpu"]]
        for name in ("observed_energy_per_request_j", "observed_energy_j", "cpu_energy_total_j", "gpu_energy_total_j"):
            assert actual["metrics"][name] == config["metrics"][name], (actual["cpu"], name)
            assert actual["metric_observations"][name] == config["metric_observations"][name], (actual["cpu"], name)
    changed_sampling, equivalent_sampling = result["changes"]
    assert all(delta["delta"] is None and delta["reasons"] == ["different_source_scope_window_or_sampling"]
               for delta in changed_sampling)
    assert [delta["delta"] for delta in equivalent_sampling] == [-2, -8]
    assert all(delta["delta"] is None and delta["reasons"] == ["metric_contract_missing_on_one_side"]
               for delta in result["legacyDeltas"])


def test_import_derived_energy_keeps_evidence_for_missing_gpu_column(page, tmp_path):
    cpu, gpu = "cpu_energy_total_j", "gpu_energy_total_j"
    path = tmp_path / "missing-component.csv"
    fields = ["cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", "repeat_idx", "warmup", "status",
              cpu, "repeat_in_window", "resource_metric_quality"]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        denied = {"schema_version": 1, "metrics": {
            cpu: {"quality": "complete"}, gpu: {"quality": "permission_denied", "reason": "energy counter denied"}}}
        sampled = {"schema_version": 1, "metrics": {cpu: {"quality": "complete"}},
                   "collectors": {"gpu_power": {"status": "available", "sampling": {"sample_hz": 5}}}}
        for case, evidence in enumerate([denied, sampled], start=1):
            writer.writerow([case, 4, "on", 1, 0, 0, "ok", 3.5, 2, json.dumps(evidence)])
    expected = {c["cpu"]: c for c in load_analysis([path]).configs}
    imported = page.evaluate("""async text => {
      const {registry} = JSON.parse(document.getElementById("report-data").textContent);
      return (await ACProfImport.readCsv(new File([text], "missing-component.csv"), registry)).configs;
    }""", path.read_text())
    assert {c["cpu"] for c in imported} == {1, 2}
    for actual in imported:
        case = actual["cpu"]
        assert gpu not in actual["metric_observations"]
        for name in ("observed_energy_per_request_j", "observed_energy_j"):
            observation, = actual["metric_observations"][name]
            assert observation == expected[case]["metric_observations"][name][0]
            if case == 1:
                assert observation["quality"] == "permission_denied"
                assert observation["reason"] == f"{gpu}: energy counter denied"
            else:
                assert observation["quality"] == "unavailable"
                assert observation["components"][gpu]["sampling"] == {"sample_hz": 5, "energy_method": None}
            assert actual["metrics"][name]["value"] is None
