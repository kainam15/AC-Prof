"""Opt-in headless browser evidence; run with ACPROF_BROWSER_TESTS=1 and Playwright."""
import csv
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.environ.get("ACPROF_BROWSER_TESTS") == "1", "opt-in headless browser test")
class ReportBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright

        from acprof.analysis.model import load_analysis
        from acprof.plotting.report import write_report
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        rows = []
        for cpu, mem, latency, energy in [(2, 4, .10, 4), (4, 4, .04, 5), (8, 4, .08, 6),
                                          (2, 8, .09, 4), (4, 8, .03, 5)]:
            rows.append({"cpu_cores": cpu, "mem_cap_gb": mem, "gpu_mode": "off",
                         "input_scale": 32, "repeat_idx": 0, "warmup": 0, "status": "ok",
                         "latency_app_p95_s": latency, "throughput_samples_per_s": 1 / latency,
                         "cpu_energy_total_j": energy, "repeat_in_window": 2,
                         "container_mem_usage_peak_bytes": mem * 1024, "cpu_ipc": 2,
                         "cold_start_s": 1, "concurrency": 1})
        csv_path = cls.root / "result_all.csv"
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (cls.root / "static_meta.json").write_text(json.dumps({
            "model_name": "fixture/model", "runtime_backend": "torch", "batch_size": 1}))
        cls.report = write_report(load_analysis([csv_path]), cls.root / "report.html")
        cls.playwright = sync_playwright().start()
        cls.addClassCleanup(cls.playwright.stop)
        executable = os.environ.get("ACPROF_BROWSER_EXECUTABLE") or shutil.which("google-chrome") or shutil.which("chromium")
        cls.browser = cls.playwright.chromium.launch(executable_path=executable, headless=True,
                                                     args=["--no-sandbox", "--disable-gpu"])
        cls.addClassCleanup(cls.browser.close)

    def setUp(self):
        self.context = self.browser.new_context(offline=True, viewport={"width": 1440, "height": 1000})
        self.addCleanup(self.context.close)
        self.page = self.context.new_page()
        self.errors = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.goto(self.report.as_uri())
        self.settled()

    def settled(self):
        self.page.wait_for_function("document.body.dataset.ready === 'true' && document.body.dataset.busy === 'false'")
        self.assertEqual(self.errors, [])
        self.assertTrue(self.page.locator("#error").is_hidden())

    def test_baseline_sort_filter_and_shared_selection(self):
        self.assertEqual(self.page.locator("#matrix tbody tr").count(), 5)
        row = self.page.locator("#matrix tbody tr").first
        config = row.get_attribute("data-config-id") or ""
        self.assertTrue(config)
        self.page.select_option("#baseline", config)
        self.settled()
        self.assertIn("Δ 0 s · 0%", row.inner_text())
        row.click()
        self.settled()
        self.assertIn(config[-8:], self.page.locator("#selection").inner_text())
        self.page.locator("#matrix th button").first.click()
        self.settled()
        self.assertEqual(self.page.locator("#matrix tbody tr td:nth-child(2)").first.inner_text().splitlines()[0], "0.1")
        self.page.select_option("#filter-memory", "4")
        self.settled()
        self.assertEqual(self.page.locator("#matrix tbody tr").count(), 3)
        self.assertIn("当前筛选隐藏", self.page.locator("#selection").inner_text())
        self.page.click("[data-view=pareto]")
        self.settled()
        self.assertIn("3 个有效点", self.page.locator("#pareto-note").inner_text())

    def test_pareto_direction_ties_missing_and_neutral(self):
        result = self.page.evaluate("""() => {
          const make = (id, x, y, extra={}) => ({config_id:id, run_id:'run', environment_class:'unknown',
            task:'task', input_case:'case', status:'ok', metrics:{x:{value:x}, y:{value:y}}, ...extra});
          const rows = [make('a', 10, 4), make('b', 20, 5), make('c', 15, 6), make('tie', 20, 5),
            make('missing', null, 0), make('failure', 100, 0, {status:'failed'}),
            make('other', 100, 0, {environment_class:'native_linux'})];
          const registry = {x:{direction:'higher'}, y:{direction:'lower'}};
          return {front: ACProfViews.pareto(rows,'x','y',registry),
            neutral: ACProfViews.pareto(rows,'x','y',{...registry,y:{direction:'neutral'}}),
            lower: ACProfViews.score(1,[1,9],{direction:'lower',scale:'linear'}),
            higher: ACProfViews.score(9,[1,9],{direction:'higher',scale:'linear'}),
            logZero: ACProfViews.score(0,[0,9],{direction:'lower',scale:'log'})};
        }""")
        self.assertEqual(set(result["front"]), {"a", "b", "tie", "other"})
        self.assertEqual(result["neutral"], [])
        self.assertEqual(result["lower"], 1)
        self.assertEqual(result["higher"], 1)
        self.assertIsNone(result["logZero"])

    def test_zero_baseline_and_environment_isolation(self):
        result = self.page.evaluate("""() => {
          const a = {run_id:'a', environment_class:'unknown', task:'x', input_case:'x', metrics:{x:{value:0}}};
          const b = {...a,metrics:{x:{value:4}}};
          return {zero:ACProfViews.delta(b,a,'x'),
            mixed:ACProfViews.delta({...b,environment_class:'native_linux'},a,'x'),
            unknown:ACProfViews.delta({...b,run_id:'b'},a,'x'),
            sorted:ACProfViews.sort([{...a,config_id:'zero'}, {...b,config_id:'four'},
              {...a,config_id:'missing',metrics:{x:{value:null}}}], 'x', true).map(c=>c.config_id)};
        }""")
        self.assertEqual(result["zero"]["delta"], 4)
        self.assertIsNone(result["zero"]["percent"])
        self.assertEqual(result["mixed"]["reason"], "different_conditions")
        self.assertEqual(result["unknown"]["reason"], "different_conditions")
        self.assertEqual(result["sorted"], ["four", "zero", "missing"])

    def test_scaling_keeps_other_resources_fixed_and_clicks_link_back(self):
        self.page.click("[data-view=scaling]")
        self.settled()
        series = self.page.evaluate("""() => {
          const chart = document.querySelector('.scaling-chart');
          return chart.data.map(trace => ({x:trace.x, y:trace.y, id:trace.customdata[0]}));
        }""")
        self.assertEqual(len(series), 2)
        self.assertEqual(series[0]["x"], [2, 4, 8])
        self.assertEqual(series[0]["y"], [.1, .04, .08])
        self.assertEqual(series[1]["x"], [2, 4])
        self.page.evaluate("""id => document.querySelector('.scaling-chart').emit('plotly_click',
          {points:[{customdata:id}]})""", series[0]["id"])
        self.settled()
        self.page.click("[data-view=matrix]")
        self.settled()
        self.assertEqual(self.page.locator("tr.selected").get_attribute("data-config-id"), series[0]["id"])

    def test_axis_selection_size_empty_filters_and_narrow_layout(self):
        self.page.click("[data-view=pareto]")
        self.page.select_option("#tradeoff-preset", "throughput_samples_per_s,observed_energy_per_request_j")
        self.page.select_option("#size-metric", "container_mem_usage_peak_bytes")
        self.settled()
        self.assertIn("吞吐量", self.page.evaluate("document.getElementById('pareto-chart').layout.xaxis.title.text"))
        sizes: list[float] = self.page.evaluate("document.getElementById('pareto-chart').data[0].marker.size")
        self.assertGreater(max(sizes), min(sizes))
        self.page.select_option("#filter-memory", "8")
        self.page.select_option("#filter-cpu", "8")
        self.settled()
        self.assertIn("0 个有效点", self.page.locator("#pareto-note").inner_text())
        self.page.set_viewport_size({"width": 640, "height": 900})
        self.page.click("[data-view=matrix]")
        self.settled()
        self.assertTrue(self.page.locator("#matrix-empty").is_visible())
        self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
