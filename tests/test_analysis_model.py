"""Historical CSV analysis, provenance and the offline visualization entry point."""
import csv
import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest


class TestAnalysisModel:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.temporary = tmp_path
        self.root = Path(str(self.temporary))

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
        assert (config["status"]) == ("partial")
        assert (config["environment_class"]) == ("unknown")
        assert (config["model"]) == ("example/model")
        assert (config["metrics"]["latency_app_p95_s"]["value"]) == (.06)
        assert (config["metrics"]["latency_app_p95_s"]["n"]) == (2)
        assert (config["metrics"]["container_mem_usage_peak_bytes"]["value"]) == (2048)
        assert (config["metrics"]["observed_energy_j"]["value"]) == (46)
        assert (config["metrics"]["observed_energy_per_request_j"]["value"]) == (4.6)
        assert (config["metrics"]["cpu_ipc"]["value"]) is None
        assert (config["metrics"]["qps"]["value"]) is None
        assert (data.raw_rows[0]["row"]["extra"]) == (" preserve ")
        assert (data.raw_rows[0]["row"]["energy_total_j"]) == ("999")
        record = next(r for r in data.records if r["metric"] == "latency_app_p95_s")
        assert ({"run_id", "model", "runtime", "device", "cpu", "memory",
                         "concurrency", "metric", "value", "unit"} <= record.keys())
        assert (record["source_row"]) == (2)
        assert (path.read_bytes()) == (before)

    def test_result_csv_is_hashed_and_parsed_without_whole_file_read(self):
        from acprof.analysis.model import load_analysis

        path = self.source([self.row()])
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        original_read_bytes = Path.read_bytes

        def guarded_read_bytes(candidate):
            if candidate == path:
                raise AssertionError("analysis must stream result CSV")
            return original_read_bytes(candidate)

        with patch.object(Path, "read_bytes", guarded_read_bytes):
            data = load_analysis([path])

        assert data.sources[0]["sha256"] == expected
        assert data.configs[0]["metrics"]["latency_app_p95_s"]["value"] == .04

    def test_result_csv_change_during_streaming_analysis_is_rejected(self):
        from acprof.analysis.model import load_analysis

        path = self.source([self.row()])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with patch("acprof.analysis.model.file_sha256", side_effect=[digest, "0" * 64]):
            with pytest.raises(ValueError, match="changed during analysis"):
                load_analysis([path])

    def test_cases_and_sources_never_collapse_across_workload_or_environment(self):
        from acprof.analysis.model import load_analysis
        a = self.source([self.row(), self.row(input_scale="64"),
                         self.row(cpu_cores="4"), self.row(concurrency="2"),
                         self.row(environment_class="wsl2")])
        b = self.source([self.row()], directory="second", meta={"model_name": "other/model"})
        data = load_analysis([a, b])
        assert (len(data.configs)) == (6)
        assert (len({c["config_id"] for c in data.configs})) == (6)
        assert ({c["environment_class"] for c in data.configs}) == ({"unknown", "wsl2"})
        with pytest.raises(ValueError, match="duplicate"):
            load_analysis([a, a.parent])

    def test_inferred_failures_and_nonfinite_values_cannot_become_measurements(self):
        from acprof.analysis.model import load_analysis
        path = self.source([self.row(result_origin="inferred_not_measured"),
                            self.row(cpu_cores="4", latency_app_p95_s="inf", cpu_ipc="nan",
                                     gpu_mode="on", gpu_energy_total_j="nan")])
        data = load_analysis([path])
        a, b = data.configs
        assert (a["status"]) == ("inferred_not_measured")
        assert (a["metrics"]["latency_app_p95_s"]["value"]) is None
        assert (b["metrics"]["latency_app_p95_s"]["value"]) is None
        assert (b["metrics"]["observed_energy_j"]["value"]) is None
        json.dumps(data.to_dict(), allow_nan=False)

    def test_lifecycle_values_are_deduplicated_and_estimates_never_fill_pmu(self):
        from acprof.analysis.model import load_analysis
        data = load_analysis([self.source([
            self.row(cold_start_s="2", cold_start_started_at="start-a", cpu_cycles_est_app="100"),
            self.row(repeat_idx="1", cold_start_s="2", cold_start_started_at="start-a"),
            self.row(repeat_idx="2", cold_start_s="4", cold_start_started_at="start-b"),
        ])])
        metrics = data.configs[0]["metrics"]
        assert (metrics["cold_start_s"]["value"]) == (3)
        assert (metrics["cold_start_s"]["n"]) == (2)
        assert (metrics["cpu_cycles_per_request"]["value"]) is None

    def test_csv_and_metadata_errors_are_visible(self):
        from acprof.analysis.model import load_analysis
        path = self.source([self.row(), self.row()])
        with pytest.raises(ValueError, match="duplicate"):
            load_analysis([path])
        path.write_text("cpu_cores,cpu_cores\n1,2\n")
        with pytest.raises(ValueError, match="columns"):
            load_analysis([path])
        path.write_text("cpu_cores,status,warmup\n2,ok,0\n")
        (path.parent / "static_meta.json").write_text("{broken")
        with pytest.raises(ValueError):
            load_analysis([path])

    def test_missing_energy_is_not_a_partial_total_and_zero_remains_a_value(self):
        from acprof.analysis.model import load_analysis
        path = self.source([self.row(cpu_energy_total_j="0", qps="2", cpu_ipc="0"),
                            self.row(repeat_idx="1", cpu_energy_total_j="nan")])
        data = load_analysis([path])
        metrics = data.configs[0]["metrics"]
        assert (metrics["observed_energy_j"]["value"]) is None
        assert (metrics["observed_energy_per_request_j"]["value"]) is None
        assert (metrics["qps"]["value"]) == (2)
        assert (metrics["cpu_ipc"]["value"]) == (0)
        assert (data.summary[0]["status"]) == ("ok")
        assert (data.summary[0]["observed_energy_j"]) is (None)

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
        assert (data.configs[0]["run_id"]) == ("recorded-run")
        assert (data.sources[0]["run_state"]) == ("complete")
        layout.result_csv.write_text("cpu_cores,status,warmup\n\n\n")
        with pytest.raises(ValueError, match="no measurement rows"):
            load_analysis([layout.root])

    def test_quality_evidence_survives_summary_without_hiding_observations(self):
        from acprof.analysis.model import load_analysis
        from acprof.quality import loading_quality
        path = self.source([self.row()])
        checks = loading_quality({"missing_keys": ["head.weight"]}, source="loader")
        (path.parent / "quality_checks.json").write_text(json.dumps({"schema_version": 1, "checks": checks}))
        model = load_analysis([path])
        config = model.configs[0]
        assert (config["status"]) == ("ok")
        assert (config["quality_status"]) == ("blocked")
        assert not (config["auto_selection_eligible"])
        assert ("weights_reinitialized") in (config["quality_reasons"])
        assert (config["metrics"]["latency_app_p95_s"]["value"]) == (.04)
        assert (model.summary[0]["quality_checks"][0]["evidence"]["source"]) == ("loader")
        assert (model.sources[0]["quality_status"]) == ("blocked")

    def test_legacy_quality_is_unknown_even_with_successful_rows(self):
        from acprof.analysis.model import load_analysis
        model = load_analysis([self.source([self.row()])])
        assert (model.configs[0]["quality_status"]) == ("unknown")
        assert not (model.configs[0]["auto_selection_eligible"])
        assert ("quality_evidence_missing") in (model.configs[0]["quality_reasons"])
        assert (model.configs[0]["measurement_status"]) == ("unknown")

    @pytest.mark.parametrize('recorded', (True, False))
    def test_moving_recorded_or_legacy_experiment_preserves_configuration_identity(self, recorded):
        from acprof.analysis.model import load_analysis
        source = self.source([self.row()], directory=f"original-{recorded}")
        if recorded:
            (source.parent / "run_state.json").write_text(json.dumps({"run_id": "stable-run"}))
        before = load_analysis([source]).configs[0]["config_id"]
        moved = self.root / f"renamed-{recorded}"
        source.parent.rename(moved)
        after = load_analysis([moved]).configs[0]["config_id"]
        assert (after) == (before)

    def test_backup_is_rejected_as_duplicate_instead_of_another_configuration(self):
        from acprof.analysis.model import load_analysis
        source = self.source([self.row()])
        (source.parent / "run_state.json").write_text(json.dumps({"run_id": "stable-run"}))
        copy = self.root / "backup"
        shutil.copytree(source.parent, copy)
        with pytest.raises(ValueError, match="duplicate measurement.*stable-run"):
            load_analysis([source, copy])

    def test_same_recorded_measurement_with_changed_values_is_a_content_conflict(self):
        from acprof.analysis.model import load_analysis
        source = self.source([self.row()])
        other = self.source([self.row(latency_app_p95_s=".02")], directory="edited-backup")
        for path in (source, other):
            (path.parent / "run_state.json").write_text(json.dumps({"run_id": "stable-run"}))
        with pytest.raises(ValueError, match="conflicting measurement.*stable-run"):
            load_analysis([source, other])

    def test_recorded_csv_run_id_cannot_override_run_state(self):
        from acprof.analysis.model import load_analysis
        source = self.source([self.row(run_id="another-run")])
        (source.parent / "run_state.json").write_text(json.dumps({"run_id": "stable-run"}))
        with pytest.raises(ValueError, match="inconsistent run_id"):
            load_analysis([source])
