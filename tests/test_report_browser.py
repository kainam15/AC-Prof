"""Opt-in headless browser evidence; run with ACPROF_BROWSER_TESTS=1 and Playwright."""
import csv
import json
import os
import shutil
import tempfile
from functools import partial
from importlib.metadata import version
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not (os.environ.get("ACPROF_BROWSER_TESTS") == "1"), reason="opt-in headless browser test")
class TestReportBrowser:
    @pytest.fixture(scope="class", autouse=True)
    def _class_setup(self, request, tmp_path_factory):
        cls = request.cls
        from playwright.sync_api import sync_playwright

        from acprof.analysis.model import load_analysis
        from acprof.plotting.report import write_report
        cls.temporary = tempfile.TemporaryDirectory()
        request.addfinalizer(partial(cls.temporary.cleanup))
        cls.root = Path(cls.temporary.name)
        from comparison_fixtures import ComparisonFixture
        fixture = ComparisonFixture()
        fixture.build(tmp_path_factory.mktemp("comparison"))
        shutil.copytree(fixture.left, cls.root, dirs_exist_ok=True)
        fixture.change_json(cls.root, "run_state.json", lambda state: (state.pop("runtime"),
            state["options"].update(cpus="2,4,8", mems="4,8")))
        fixture.change_json(cls.root, "input_scale_plan.json", lambda plan:
            plan["entries"][0].update(input_scale=32))
        hardware = json.loads((cls.root / "hardware_conditions.json").read_text())["cases"]["1c_4g_off"]
        contract = fixture.contract
        contract["input"].update(planned_scale=32, actual_scale=32)
        rows = []
        for cpu, mem, latency, energy in [(2, 4, .10, 4), (4, 4, .04, 5), (8, 4, .08, 6),
                                          (2, 8, .09, 4), (4, 8, .03, 5)]:
            rows.append({"cpu_cores": cpu, "mem_cap_gb": mem, "gpu_mode": "off",
                         "input_scale": 32, "repeat_idx": 0, "warmup": 0, "status": "ok",
                         "latency_app_p95_s": latency, "throughput_samples_per_s": 1 / latency,
                         "cpu_energy_total_j": energy, "repeat_in_window": 2,
                         "container_mem_usage_peak_bytes": mem * 1024, "cpu_ipc": 2,
                         "cold_start_s": 1, "concurrency": 1})
            rows[-1].update(environment_class="native_linux", workload_contract=json.dumps({
                "schema_version": 1, "request_count": 2, "variants": [{"count": 2, "contract": contract}]}))
        csv_path = cls.root / "result_all.csv"
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        fixture.change_json(cls.root, "static_meta.json", lambda meta: meta.update(
            model_name="fixture/model", runtime_backend="torch", batch_size=1, pipeline_tag="tabular-regression"))
        fixture.write_json(cls.root, "hardware_conditions.json", {"schema_version": 1, "cases": {
            f"{row['cpu_cores']}c_{row['mem_cap_gb']}g_off": hardware for row in rows}})
        (cls.root / "quality_checks.json").write_text(json.dumps({"schema_version": 1, "checks": []}))
        cls.model = load_analysis([csv_path])
        cls.report = write_report(cls.model, cls.root / "report.html", comparison_purpose="resource-scaling")
        cls.playwright = sync_playwright().start()
        request.addfinalizer(partial(cls.playwright.stop))
        cls.artifacts = Path(os.environ["ACPROF_BROWSER_ARTIFACT_DIR"]).resolve() if os.environ.get("ACPROF_BROWSER_ARTIFACT_DIR") else None
        if cls.artifacts:
            cls.artifacts.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cls.report, cls.artifacts / "report.html")
        executable = os.environ.get("ACPROF_BROWSER_EXECUTABLE")
        cls.browser = cls.playwright.chromium.launch(executable_path=executable, headless=True,
                                                     args=["--no-sandbox", "--disable-gpu"])
        request.addfinalizer(partial(cls.browser.close))
        if cls.artifacts:
            (cls.artifacts / "browser-info.json").write_text(json.dumps({
                "playwright": version("playwright"), "chromium": cls.browser.version,
                "explicit_executable_override": executable, "network_mode": "offline",
            }, indent=2), encoding="utf-8")
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.fixture_root = tmp_path
        self.context = self.browser.new_context(offline=True, viewport={"width": 1440, "height": 1000})
        self._request.addfinalizer(partial(self.context.close))
        if self.artifacts:
            self.context.tracing.start(screenshots=True, snapshots=True, sources=True)
        self.page = self.context.new_page()
        self.errors, self.remote_requests, self.console = [], [], []
        self._request.addfinalizer(partial(self.save_failure_evidence))
        self.page.on("console", lambda msg: self.console.append({"type": msg.type, "text": msg.text}))
        self.page.on("request", lambda request: self.remote_requests.append(request.url)
                     if request.url.startswith(("http://", "https://")) else None)
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.goto(self.report.as_uri())
        self.settled()

    def save_failure_evidence(self):
        if not self.artifacts:
            return
        from acprof.testing.plugin import PHASE_REPORTS
        reports = self._request.node.stash.get(PHASE_REPORTS, {})
        failed = any(report.failed for report in reports.values())
        if failed:
            target = self.artifacts / self._request.node.name
            target.mkdir(parents=True, exist_ok=True)
            self.page.screenshot(path=str(target / "page.png"), full_page=True)
            (target / "page.html").write_text(self.page.content(), encoding="utf-8")
            (target / "browser.json").write_text(json.dumps({"page_errors": self.errors,
                "remote_requests": self.remote_requests, "console": self.console}, ensure_ascii=False, indent=2))
            current = urlsplit(self.page.url)
            if current.scheme == "file":
                shutil.copyfile(unquote(current.path), target / "report.html")
            self.context.tracing.stop(path=str(target / "trace.zip"))
        else:
            self.context.tracing.stop()

    def settled(self):
        self.page.wait_for_function("document.body.dataset.ready === 'true' && document.body.dataset.busy === 'false'")
        assert (self.errors) == ([])
        assert (self.remote_requests) == ([])
        assert (self.page.locator("#error").is_hidden())

    def test_baseline_sort_filter_and_shared_selection(self):
        assert (self.page.locator("#matrix tbody tr").count()) == (5)
        row = self.page.locator("#matrix tbody tr").first
        config = row.get_attribute("data-config-id") or ""
        assert (config)
        self.page.select_option("#baseline", config)
        self.settled()
        assert ("Δ 0 s · 0%") in (row.inner_text())
        row.locator("td").first.click()
        self.settled()
        assert (config[-8:]) in (self.page.locator("#selection").inner_text())
        self.page.locator("#matrix th button").first.click()
        self.settled()
        assert (self.page.locator("#matrix tbody tr td:nth-child(2)").first.inner_text().splitlines()[0]) == ("0.1")
        self.page.select_option("#filter-memory", "4")
        self.settled()
        assert (self.page.locator("#matrix tbody tr").count()) == (3)
        assert ("当前筛选隐藏") in (self.page.locator("#selection").inner_text())
        self.page.click("[data-view=pareto]")
        self.settled()
        assert ("3 个有效点") in (self.page.locator("#pareto-note").inner_text())

    def test_pareto_direction_ties_missing_and_neutral(self):
        result = self.page.evaluate("""() => {
          const profiles = JSON.parse(document.getElementById('report-data').textContent).configs[0].comparison_profiles;
          const make = (id, x, y, extra={}) => ({config_id:id, run_id:'run', environment_class:'unknown',
            task:'task', input_case:'case', status:'ok', auto_selection_eligible:true, comparison_profiles:profiles,
            metrics:{x:{value:x}, y:{value:y}}, ...extra});
          const rows = [make('a', 10, 4), make('b', 20, 5), make('c', 15, 6), make('tie', 20, 5),
            make('missing', null, 0), make('failure', 100, 0, {status:'failed'}),
            make('bad-output',100,0,{auto_selection_eligible:false}),
            make('legacy',100,0,{auto_selection_eligible:undefined}),
            make('other', 100, 0, {comparison_profiles:{}})];
          const registry = {x:{direction:'higher'}, y:{direction:'lower'}};
          return {front: ACProfViews.pareto(rows,'x','y',registry),
            neutral: ACProfViews.pareto(rows,'x','y',{...registry,y:{direction:'neutral'}}),
            lower: ACProfViews.score(1,ACProfViews.range([1,9],{scale:'linear'}),{direction:'lower',scale:'linear'}),
            higher: ACProfViews.score(9,ACProfViews.range([1,9],{scale:'linear'}),{direction:'higher',scale:'linear'}),
            logZero: ACProfViews.score(0,ACProfViews.range([0,9],{scale:'log'}),{direction:'lower',scale:'log'})};
        }""")
        assert (set(result["front"])) == ({"a", "b", "tie"})
        assert (result["neutral"]) == ([])
        assert (result["lower"]) == (1)
        assert (result["higher"]) == (1)
        assert (result["logZero"]) is None

    def test_zero_baseline_and_environment_isolation(self):
        result = self.page.evaluate("""() => {
          const profiles = JSON.parse(document.getElementById('report-data').textContent).configs[0].comparison_profiles;
          const a = {run_id:'a', environment_class:'native_linux', comparison_profiles:profiles, metrics:{x:{value:0}}};
          const b = {...a,metrics:{x:{value:4}}};
          const changed = structuredClone(profiles);
          changed['same-hardware'].checks.comparability_class.value = 'other-environment';
          return {zero:ACProfViews.delta(b,a,'x'),
            mixed:ACProfViews.delta({...b,comparison_profiles:changed},a,'x'),
            unknown:ACProfViews.delta({...b,comparison_profiles:{}},a,'x'),
            sorted:ACProfViews.sort([{...a,config_id:'zero'}, {...b,config_id:'four'},
              {...a,config_id:'missing',metrics:{x:{value:null}}}], 'x', true).map(c=>c.config_id)};
        }""")
        assert (result["zero"]["delta"]) == (4)
        assert (result["zero"]["percent"]) is None
        assert (result["mixed"]["reason"]) == ("incompatible")
        assert (result["unknown"]["reason"]) == ("unknown")
        assert (result["sorted"]) == (["four", "zero", "missing"])

    def test_scaling_keeps_other_resources_fixed_and_clicks_link_back(self):
        self.page.click("[data-view=scaling]")
        self.settled()
        series = self.page.evaluate("""() => {
          const chart = document.querySelector('.scaling-chart');
          return chart.data.map(trace => ({x:trace.x, y:trace.y, id:trace.customdata[0]}));
        }""")
        assert (len(series)) == (2)
        assert (series[0]["x"]) == ([2, 4, 8])
        assert (series[0]["y"]) == ([.1, .04, .08])
        assert (series[1]["x"]) == ([2, 4])
        self.page.evaluate("""id => document.querySelector('.scaling-chart').emit('plotly_click',
          {points:[{customdata:id}]})""", series[0]["id"])
        self.settled()
        self.page.click("[data-view=matrix]")
        self.settled()
        assert (self.page.locator("tr.selected").get_attribute("data-config-id")) == (series[0]["id"])

    def test_axis_selection_size_empty_filters_and_narrow_layout(self):
        self.page.click("[data-view=pareto]")
        self.page.select_option("#tradeoff-preset", "throughput_samples_per_s,observed_energy_per_request_j")
        self.page.select_option("#size-metric", "container_mem_usage_peak_bytes")
        self.settled()
        assert ("吞吐量") in (self.page.evaluate("document.getElementById('pareto-chart').layout.xaxis.title.text"))
        sizes: list[float] = self.page.evaluate("document.getElementById('pareto-chart').data[0].marker.size")
        assert (max(sizes)) > (min(sizes))
        self.page.select_option("#filter-memory", "8")
        self.page.select_option("#filter-cpu", "8")
        self.settled()
        assert ("0 个有效点") in (self.page.locator("#pareto-note").inner_text())
        self.page.set_viewport_size({"width": 640, "height": 900})
        self.page.click("[data-view=matrix]")
        self.settled()
        assert (self.page.locator("#matrix-empty").is_visible())
        assert (self.page.evaluate("document.documentElement.scrollWidth <= innerWidth"))

    def test_quality_filter_preserves_observations_and_displays_loader_evidence(self):
        from acprof.analysis.model import load_analysis
        from acprof.plotting.report import write_report
        from acprof.quality import loading_quality
        root = self.root / "quality"
        root.mkdir()
        with (self.root / "result_all.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, row = reader.fieldnames, next(reader)
        from acprof.result_layers import publish_result_rows
        publish_result_rows(fields, [row], root)
        checks = loading_quality({"missing_keys": ["head.weight"]}, source="fixture-loader")
        (root / "quality_checks.json").write_text(json.dumps({"schema_version": 1, "checks": checks}))
        output = write_report(load_analysis([self.root / "result_all.csv", root]), root / "report.html")
        self.page.goto(output.as_uri())
        self.settled()
        assert (self.page.locator("#matrix tbody tr").count()) == (6)
        self.page.locator("#matrix tbody tr").filter(has_text="weights_reinitialized").locator("td").first.click()
        self.settled()
        self.page.locator("#quality-evidence summary").click()
        evidence = self.page.locator("#quality-detail").inner_text()
        assert ("fixture-loader") in (evidence)
        assert ('"quality_status": "blocked"') in (evidence)
        self.page.select_option("#quality-filter", "eligible")
        self.settled()
        assert (self.page.locator("#matrix tbody tr").count()) == (5)
        assert ("当前筛选隐藏") in (self.page.locator("#selection").inner_text())
        self.page.select_option("#quality-filter", "all")
        self.settled()
        assert (self.page.locator("#matrix tbody tr").count()) == (6)

    def test_input_order_mismatch_blocks_delta_and_baseline_frontier_but_retains_points(self):
        from comparison_fixtures import ComparisonFixture

        from acprof.analysis.comparison import compare_results
        from acprof.analysis.model import load_analysis
        from acprof.plotting.report import write_report
        fixture = ComparisonFixture()
        fixture.build(self.fixture_root / "comparison" )
        for directory, latency, energy in ((fixture.left, .1, 4), (fixture.right, .05, 2)):
            path = directory / "result_all.csv"
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                fields, row = reader.fieldnames, next(reader)
            row.update(latency_app_p95_s=latency, cpu_energy_total_j=energy, repeat_in_window=3)
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fields)
                writer.writeheader()
                writer.writerow(row)
            fixture.write_json(directory, "quality_checks.json", {"schema_version": 1, "checks": []})
        fixture.change_json(fixture.right, "input_scale_plan.json", lambda plan:
                            plan["entries"][0]["payload"]["features"].reverse())
        fixture.refresh(fixture.left)
        fixture.refresh(fixture.right)
        assert (compare_results(fixture.left, fixture.right)["status"]) == ("incompatible")
        model = load_analysis([fixture.left, fixture.right])
        output = write_report(model, fixture.root / "incompatible.html", baseline=model.configs[0]["config_id"])
        self.page.goto(output.as_uri())
        self.settled()
        row = self.page.locator(f'tr[data-config-id="{model.configs[1]["config_id"]}"]')
        assert ("planned_inputs") in (row.inner_text())
        assert ("-50%") not in (row.inner_text())
        assert ("Δ") not in (row.inner_text())
        assert ("incompatible: 1") in (self.page.locator("#comparison-note").inner_text())
        self.page.click("[data-view=pareto]")
        self.settled()
        assert ("2 个有效点") in (self.page.locator("#pareto-note").inner_text())
        assert ("1 个非支配点") in (self.page.locator("#pareto-note").inner_text())

    def test_browser_and_python_agree_on_exact_workload_distributions(self):
        from copy import deepcopy

        from comparison_fixtures import ComparisonFixture

        from acprof.analysis.conditions import compare_profiles, workload_case_profile
        base = ComparisonFixture()
        base.build(self.fixture_root / "comparison")
        contract = base.contract
        other = deepcopy(contract)
        other["input"]["feature_dim"] = 9

        def profile(contracts, counts):
            evidence = workload_case_profile({"request_count": sum(counts), "variants": [
                {"contract": item, "count": count} for item, count in zip(contracts, counts)]})
            return {"purpose": "same-hardware", "valid": True, "checks": {}, "workload": {"case": evidence}}

        generation = deepcopy(contract)
        generation["task"] = "text-generation"
        changed_output = deepcopy(generation)
        changed_output["output"]["count"] = 99
        partial = deepcopy(contract)
        partial["output"]["shape"] = None
        changed_partial = deepcopy(partial)
        changed_partial["input"]["feature_dim"] = 9
        pairs = [(profile([contract, other], [99, 1]), profile([contract, other], [1, 99]), "incompatible"),
                 (profile([contract, other], [99, 1]), profile([contract, other], [990, 10]), "compatible"),
                 (profile([partial], [1]), profile([partial], [2]), "unknown"),
                 (profile([partial], [1]), profile([changed_partial], [2]), "incompatible"),
                 (profile([generation], [1]), profile([changed_output], [2]), "unknown")]
        for a, b, expected in pairs:
            assert (compare_profiles(a, b)["status"]) == (expected)
        actual = self.page.evaluate("pairs => pairs.map(([a,b]) => ACProfViews.compareProfiles(a,b).status)", pairs)
        assert (actual) == ([expected for _, _, expected in pairs])

    def test_color_ranges_read_each_group_metric_once_and_preserve_normalization(self):
        result = self.page.evaluate("""() => {
          let reads = 0;
          const registry = {x:{direction:'lower',scale:'linear'}, y:{direction:'higher',scale:'log'}};
          const rows = Array.from({length:200}, (_,i) => ({config_id:String(i), auto_selection_eligible:true,
            comparison_profiles:{'same-hardware':{status:'compatible',cohort:String(i % 2)}},
            metrics:{get x(){reads++; return {value:i+1}},get y(){reads++; return {value:10 ** (i % 3)}}}}));
          const ranges = ACProfViews.colorRanges(rows, ['x','y'], registry);
          return {reads,groups:ranges.size,
            first:ACProfViews.score(1,ranges.get('0').x,registry.x),
            last:ACProfViews.score(199,ranges.get('0').x,registry.x),
            log:ACProfViews.score(10,ranges.get('0').y,registry.y),
            zero:ACProfViews.score(0,ranges.get('0').y,registry.y),
            neutral:ACProfViews.score(1,ranges.get('0').x,{direction:'neutral',scale:'linear'})};
        }""")
        assert (result) == ({"reads": 400, "groups": 2, "first": 1, "last": 0,
                                  "log": .5, "zero": None, "neutral": None})

    def test_synthetic_matrix_growth_keeps_sort_colors_and_filter_values(self):
        from copy import deepcopy

        from acprof.analysis.model import DEFAULT_MATRIX, AnalysisModel
        from acprof.plotting.report import write_report
        measurements = []
        for count in (100, 400, 1000):
            configs = []
            for index in range(count):
                config = deepcopy(self.model.configs[0])
                config.update(config_id=f"synthetic-{index}", run_id=f"synthetic-run-{index}",
                              experiment_batch="even" if index % 2 == 0 else "odd")
                for metric in DEFAULT_MATRIX:
                    config["metrics"][metric]["value"] = (index + 1) / 100
                configs.append(config)
            model = AnalysisModel(self.model.sources, [], configs)
            output = write_report(model, self.root / f"synthetic-{count}.html")
            self.page.goto(output.as_uri())
            self.settled()
            ready_ms = self.page.evaluate("performance.now()")
            assert (self.page.locator("#matrix tbody tr").count()) == (count)
            row = self.page.locator('tr[data-config-id="synthetic-2"] td:nth-child(2)')
            original_color = row.evaluate("cell => getComputedStyle(cell).backgroundColor")
            started = self.page.evaluate("performance.now()")
            self.page.locator("#matrix th button").first.click()
            self.settled()
            sorted_ms = self.page.evaluate("performance.now()") - started
            assert (row.evaluate("cell => getComputedStyle(cell).backgroundColor")) == (original_color)
            assert (row.inner_text()) == ("0.03")
            started = self.page.evaluate("performance.now()")
            self.page.select_option("#filter-experiment_batch", "even")
            self.settled()
            filtered_ms = self.page.evaluate("performance.now()") - started
            assert (self.page.locator("#matrix tbody tr").count()) == (count // 2)
            assert (row.inner_text()) == ("0.03")
            measurements.append({"configs": count, "metrics": len(DEFAULT_MATRIX), "ready_ms": ready_ms,
                                 "sort_ms": sorted_ms, "filter_ms": filtered_ms})
        if self.artifacts:
            (self.artifacts / "matrix-scaling.json").write_text(json.dumps({"browser": self.browser.version,
                "scope": "synthetic offline report rendering; not inference or formal measurement",
                "timing": "navigation to readiness, then interactions to readiness; shared development host",
                "measurements": measurements}, indent=2))
