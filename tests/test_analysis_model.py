"""Historical CSV analysis, provenance and the offline visualization entry point."""
import csv
import json
import tempfile
import unittest
from pathlib import Path


class AnalysisModelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def source(self, rows, *, meta=None, directory="run"):
        root = self.root / directory
        root.mkdir()
        path = root / "result_all.csv"
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        if meta is not None:
            (root / "static_meta.json").write_text(json.dumps(meta))
        return path

    @staticmethod
    def row(**changes):
        return {"cpu_cores": "2", "mem_cap_gb": "4", "gpu_mode": "off", "input_scale": "32",
                "repeat_idx": "0", "warmup": "0", "status": "ok", "repeat_in_window": "2",
                "latency_app_p95_s": ".04", "throughput_samples_per_s": "50",
                "container_mem_usage_peak_bytes": "1024", "cpu_energy_total_j": "3", **changes}

    def test_history_stays_unchanged_and_window_statistics_keep_their_scope(self):
        from acprof.analysis.model import load_analysis
        path = self.source([
            self.row(extra=" preserve ", energy_total_j="999"),
            self.row(repeat_idx="1", repeat_in_window="8", latency_app_p95_s=".08",
                     container_mem_usage_peak_bytes="2048", cpu_energy_total_j="5"),
            self.row(warmup="1", latency_app_p95_s="9"),
            self.row(repeat_idx="2", status="error", latency_app_p95_s="10"),
        ], meta={"model_name": "example/model", "runtime_backend": "torch", "batch_size": 2})
        before = path.read_bytes()
        data = load_analysis([path])
        config = data.configs[0]
        self.assertEqual(config["status"], "partial")
        self.assertEqual(config["environment_class"], "unknown")
        self.assertEqual(config["model"], "example/model")
        self.assertEqual(config["metrics"]["latency_app_p95_s"]["value"], .06)
        self.assertEqual(config["metrics"]["latency_app_p95_s"]["n"], 2)
        self.assertEqual(config["metrics"]["container_mem_usage_peak_bytes"]["value"], 2048)
        self.assertEqual(config["metrics"]["observed_energy_j"]["value"], 46)
        self.assertEqual(config["metrics"]["observed_energy_per_request_j"]["value"], 4.6)
        self.assertIsNone(config["metrics"]["cpu_ipc"]["value"])
        self.assertIsNone(config["metrics"]["qps"]["value"])
        self.assertEqual(data.raw_rows[0]["row"]["extra"], " preserve ")
        self.assertEqual(data.raw_rows[0]["row"]["energy_total_j"], "999")
        record = next(r for r in data.records if r["metric"] == "latency_app_p95_s")
        self.assertTrue({"run_id", "model", "runtime", "device", "cpu", "memory",
                         "concurrency", "metric", "value", "unit"} <= record.keys())
        self.assertEqual(record["source_row"], 2)
        self.assertEqual(path.read_bytes(), before)

    def test_cases_and_sources_never_collapse_across_workload_or_environment(self):
        from acprof.analysis.model import load_analysis
        a = self.source([self.row(), self.row(input_scale="64"),
                         self.row(cpu_cores="4"), self.row(concurrency="2"),
                         self.row(environment_class="wsl2")])
        b = self.source([self.row()], directory="second", meta={"model_name": "other/model"})
        data = load_analysis([a, b])
        self.assertEqual(len(data.configs), 6)
        self.assertEqual(len({c["config_id"] for c in data.configs}), 6)
        self.assertEqual({c["environment_class"] for c in data.configs}, {"unknown", "wsl2"})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_analysis([a, a.parent])

    def test_inferred_failures_and_nonfinite_values_cannot_become_measurements(self):
        from acprof.analysis.model import load_analysis
        path = self.source([self.row(result_origin="inferred_not_measured"),
                            self.row(cpu_cores="4", latency_app_p95_s="inf", cpu_ipc="nan",
                                     gpu_mode="on", gpu_energy_total_j="nan")])
        data = load_analysis([path])
        a, b = data.configs
        self.assertEqual(a["status"], "inferred_not_measured")
        self.assertIsNone(a["metrics"]["latency_app_p95_s"]["value"])
        self.assertIsNone(b["metrics"]["latency_app_p95_s"]["value"])
        self.assertIsNone(b["metrics"]["observed_energy_j"]["value"])
        json.dumps(data.to_dict(), allow_nan=False)

    def test_lifecycle_values_are_deduplicated_and_estimates_never_fill_pmu(self):
        from acprof.analysis.model import load_analysis
        data = load_analysis([self.source([
            self.row(cold_start_s="2", cold_start_started_at="start-a", cpu_cycles_est_app="100"),
            self.row(repeat_idx="1", cold_start_s="2", cold_start_started_at="start-a"),
            self.row(repeat_idx="2", cold_start_s="4", cold_start_started_at="start-b"),
        ])])
        metrics = data.configs[0]["metrics"]
        self.assertEqual(metrics["cold_start_s"]["value"], 3)
        self.assertEqual(metrics["cold_start_s"]["n"], 2)
        self.assertIsNone(metrics["cpu_cycles_per_request"]["value"])

    def test_csv_and_metadata_errors_are_visible(self):
        from acprof.analysis.model import load_analysis
        path = self.source([self.row(), self.row()])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_analysis([path])
        path.write_text("cpu_cores,cpu_cores\n1,2\n")
        with self.assertRaisesRegex(ValueError, "columns"):
            load_analysis([path])
        path.write_text("cpu_cores,status,warmup\n2,ok,0\n")
        (path.parent / "static_meta.json").write_text("{broken")
        with self.assertRaises(ValueError):
            load_analysis([path])

    def test_missing_energy_is_not_a_partial_total_and_zero_remains_a_value(self):
        from acprof.analysis.model import load_analysis
        path = self.source([self.row(cpu_energy_total_j="0", qps="2", cpu_ipc="0"),
                            self.row(repeat_idx="1", cpu_energy_total_j="nan")])
        data = load_analysis([path])
        metrics = data.configs[0]["metrics"]
        self.assertIsNone(metrics["observed_energy_j"]["value"])
        self.assertIsNone(metrics["observed_energy_per_request_j"]["value"])
        self.assertEqual(metrics["qps"]["value"], 2)
        self.assertEqual(metrics["cpu_ipc"]["value"], 0)
        self.assertEqual(len(data.summary[0]), 19)

    def test_v2_metadata_and_blank_csv_handling(self):
        from acprof.analysis.model import load_analysis
        from acprof.artifact_layout import ArtifactLayout
        source = self.source([self.row()])
        layout = ArtifactLayout.for_new_run(self.root / "v2")
        layout.initialize()
        layout.result_csv.write_bytes(source.read_bytes())
        layout.path("run_state.json").parent.mkdir(parents=True, exist_ok=True)
        layout.path("run_state.json").write_text(json.dumps({"run_id": "recorded-run", "status": "complete"}))
        data = load_analysis([layout.root])
        self.assertEqual(data.configs[0]["run_id"], "recorded-run")
        self.assertEqual(data.sources[0]["run_state"], "complete")
        layout.result_csv.write_text("cpu_cores,status,warmup\n\n\n")
        with self.assertRaisesRegex(ValueError, "no measurement rows"):
            load_analysis([layout.root])
