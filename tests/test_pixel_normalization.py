import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.pyplot as matplotlib_pyplot
import pandas as pd
import pytest
from client_fixtures import patch_client, patch_client_settings

import acprof.plotting.config as plotting_config
import acprof.plotting.data as plotting_data
import acprof.plotting.metrics as plotting_metrics
from acprof.config import CSV_FIELDS
from acprof.host import client
from acprof.host.client import ClientRunner
from acprof.host.client_config import ClientConfig
from acprof.host.orchestrator import _write_case_error_csv
from acprof.packet import merge_packet_latency
from acprof.pixel_metrics import pixel_counts_from_metadata


class TestPixelNormalization:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        from platform_fixtures import native_policy
        native_policy(self._request)
        self.runner = ClientRunner(ClientConfig())

    def write_result(self, root, *, family="diffusion", scale_type="resolution_px",
                     batch=2, entries=None, rows=None, workload=None):
        root = Path(root)
        if entries is None:
            entries = [
                {"input_scale": side, "input_metadata": {
                    "output_pixel_count_per_image": pixels,
                }}
                for side, pixels in ((128, 16384), (256, 65536))
            ]
        plan = {"schema_version": 2, "task_family": family, "entries": entries}
        plan_bytes = json.dumps(plan).encode()
        (root / "input_scale_plan.json").write_bytes(plan_bytes)
        meta = {"schema_version": 7, "task_family": family, "input_scale_type": scale_type,
                "batch_size": batch, "workload": workload or {},
                "input_scale_plan_sha256": hashlib.sha256(plan_bytes).hexdigest()}
        (root / "static_meta.json").write_text(json.dumps(meta))
        if rows is None:
            rows = [
                {"input_scale": 128, "input_units_per_request": 256,
                 "container_attributed_energy_eff_j": 327.68,
                 "container_attributed_j_per_input_unit": 1.28,
                 "latency_s": 0.032768, "latency_app_s": 0.065536},
                {"input_scale": 256, "input_units_per_request": 512,
                 "container_attributed_energy_eff_j": 1310.72,
                 "container_attributed_j_per_input_unit": 2.56,
                 "latency_s": 0.131072, "latency_app_s": 0.262144},
            ]
        metadata_by_scale = {entry["input_scale"]: entry.get("input_metadata", {}) for entry in entries}
        rows = [{**pixel_counts_from_metadata(metadata_by_scale.get(row["input_scale"]), batch,
                                              task_family=family, workload=workload), **row} for row in rows]
        path = root / "result_all.csv"
        pd.DataFrame([
            {"cpu_cores": 1, "mem_cap_gb": 4, "gpu_mode": "off",
             "status": "ok", "warmup": 0, **row}
            for row in rows
        ]).to_csv(path, index=False)
        return path

    def test_diffusion_uses_recorded_area_and_keeps_scale_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_result(tmp)
            originals = {p: p.read_bytes() for p in Path(tmp).iterdir()}
            df = plotting_data.prepare_df(str(path))
            assert ("output_pixels_per_request") in (df)
            assert (df.output_pixels_per_request.tolist()) == ([32768, 131072])
            assert (df.input_units_per_request.tolist()) == ([256, 512])
            assert (df.container_attributed_j_per_input_unit.tolist()) == ([1.28, 2.56])
            assert (df.container_attributed_j_per_output_megapixel.tolist()) == ([10000, 10000])
            assert (df.latency_s_per_output_megapixel.tolist()) == ([1, 1])
            assert (df.latency_app_s_per_output_megapixel.tolist()) == ([2, 2])
            assert (df.input_pixels_per_request.isna().all())
            for p, before in originals.items():
                assert (p.read_bytes()) == (before)

    def test_cv_video_uses_dimensions_and_frame_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_result(tmp, family="cv", scale_type="resolution_scale", batch=1,
                entries=[{"input_scale": 0.5, "input_metadata": {
                    "image_width": 112, "image_height": 112, "num_frames": 3}}],
                rows=[{"input_scale": 0.5, "input_units_per_request": 0.5,
                       "container_attributed_energy_eff_j": 37.632}])
            df = plotting_data.prepare_df(str(path))
        assert ("input_pixels_per_request") in (df)
        assert (df.input_pixels_per_request.tolist()) == ([37632])
        assert (df.container_attributed_j_per_input_megapixel.tolist()) == ([1000])
        assert (df.output_pixels_per_request.isna().all())

    def test_multimodal_uses_materialized_rectangular_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_result(tmp, family="multimodal", batch=2,
                entries=[{"input_scale": 20, "input_metadata": {
                    "image_width": 20, "image_height": 30}}],
                rows=[{"input_scale": 20, "container_attributed_energy_eff_j": 1.2}])
            df = plotting_data.prepare_df(str(path))
        assert ("input_pixels_per_request") in (df)
        assert (df.input_pixels_per_request.tolist()) == ([1200])
        assert (df.container_attributed_j_per_input_megapixel.tolist()) == ([1000])

    def test_diffusion_video_does_not_multiply_frames_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_result(tmp, entries=[{"input_scale": 128, "input_metadata": {
                "output_pixel_count_per_frame": 16384,
                "output_num_frames": 4, "output_pixel_count_per_video": 65536}}],
                rows=[{"input_scale": 128, "container_attributed_energy_eff_j": 1310.72}])
            df = plotting_data.prepare_df(str(path))
        assert ("output_pixels_per_request") in (df)
        assert (df.output_pixels_per_request.tolist()) == ([131072])
        assert (df.container_attributed_j_per_output_megapixel.tolist()) == ([10000])

    @pytest.mark.parametrize('family,scale_type', (('nlp', 'seq_length'), ('audio', 'duration_s'), ('diffusion', 'denoising_steps')))
    def test_non_image_scales_are_not_squared(self, family, scale_type):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_result(tmp, family=family, scale_type=scale_type,
                entries=[{"input_scale": 128, "input_metadata": {}}])
            df = plotting_data.prepare_df(str(path))
            assert ("output_pixels_per_request") in (df)
            assert (df.output_pixels_per_request.isna().all())
            assert (df.input_pixels_per_request.isna().all())
            assert (df.container_attributed_j_per_input_unit.tolist()) == ([1.28, 2.56])

    def test_explicit_counts_do_not_read_the_input_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_result(tmp)
            (Path(tmp) / "input_scale_plan.json").write_text("not a plan")
            df = plotting_data.prepare_df(str(path))
        assert (df.output_pixels_per_request.tolist()) == ([32768, 131072])

    def test_explicit_pixel_counts_work_without_sidecars_and_refresh_stale_rates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_result(tmp, rows=[{
                "input_scale": 128, "output_pixels_per_request": 32768,
                "container_attributed_energy_eff_j": 327.68,
                "container_attributed_j_per_output_megapixel": 999,
                "latency_s": 0.032768, "latency_s_per_output_megapixel": 999}])
            (Path(tmp) / "static_meta.json").unlink()
            (Path(tmp) / "input_scale_plan.json").unlink()
            df = plotting_data.prepare_df(str(path))
        assert (df.container_attributed_j_per_output_megapixel.tolist()) == ([10000])
        assert (df.latency_s_per_output_megapixel.tolist()) == ([1])

    def test_missing_pixel_counts_are_not_reconstructed_from_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_result(tmp)
            rows = pd.read_csv(path)
            rows["input_pixels_per_request"] = ""
            rows["output_pixels_per_request"] = ""
            rows.to_csv(path, index=False)
            df = plotting_data.prepare_df(str(path))
        assert (df.output_pixels_per_request.isna().all())
        assert (df.container_attributed_j_per_output_megapixel.isna().all())

    def test_invalid_counts_and_energy_remain_nan_but_zero_energy_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_result(tmp, rows=[
                {"input_scale": 128, "output_pixels_per_request": pixels,
                 "container_attributed_energy_eff_j": energy}
                for pixels, energy in ((16384, 0), (0, 1), (-1, 1),
                                       (16384, -1), (16384, float("nan")), (1.5, 1))
            ])
            df = plotting_data.prepare_df(str(path))
        values = df.container_attributed_j_per_output_megapixel.tolist()
        assert (values[0]) == (0)
        assert (all(math.isnan(value) for value in values[1:]))

    def test_resolution_panel_does_not_fall_back_to_edge_when_geometry_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_result(tmp)
            rows = pd.read_csv(path)
            rows["output_pixels_per_request"] = float("nan")
            rows.to_csv(path, index=False)
            df = plotting_data.prepare_df(str(path))
        title, _, rows, columns, panels, shared = next(
            spec for spec in plotting_config.METRIC_OVERVIEW_PLOTS
            if spec[1] == "service_efficiency_overview_vs_scale.png")
        try:
            with patch.object(matplotlib_pyplot, "close"):
                plotting_metrics.plot_metric_overview(df, panels=panels, rows=rows,
                    columns=columns, shared_y_groups=shared, title=title,
                    xlabel="resolution_px", out_png=None)
                axis = matplotlib_pyplot.gcf().axes[3]
            assert (len(axis.lines)) == (0)
            assert ("No data") in ([text.get_text() for text in axis.texts])
            assert ("Output Megapixel") in (axis.get_title())
        finally:
            matplotlib_pyplot.close("all")

    def test_non_image_panel_keeps_the_task_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_result(tmp, family="audio", scale_type="duration_s", entries=[])
            df = plotting_data.prepare_df(str(path))
        try:
            with patch.object(matplotlib_pyplot, "close"):
                plotting_metrics.plot_metric(df, "container_attributed_j_per_input_unit",
                    "Energy per Input Unit", "J/input unit", "duration_s", None)
                axis = matplotlib_pyplot.gca()
            assert (list(axis.lines[0].get_ydata())) == ([1.28, 2.56])
            assert (axis.get_ylabel()) == ("J/input unit")
        finally:
            matplotlib_pyplot.close("all")


    @pytest.mark.parametrize('filename,index,expected,unit', (('service_efficiency_overview_vs_scale.png', 3, [10000, 10000], 'J/Mpixel'), ('latency_overview_vs_scale.png', 2, [1, 1], 's/Mpixel')))
    def test_energy_and_latency_panels_plot_pixels(self, filename, index, expected, unit):
        with tempfile.TemporaryDirectory() as tmp:
            df = plotting_data.prepare_df(str(self.write_result(tmp)))
        title, _, rows, columns, panels, shared = next(
            spec for spec in plotting_config.METRIC_OVERVIEW_PLOTS if spec[1] == filename)
        try:
            with patch.object(matplotlib_pyplot, "close"):
                plotting_metrics.plot_metric_overview(df, panels=panels, rows=rows,
                    columns=columns, shared_y_groups=shared, title=title,
                    xlabel="resolution_px", out_png=None)
                axis = matplotlib_pyplot.gcf().axes[index]
            assert ("Output Megapixel") in (axis.get_title())
            assert (unit) in (axis.get_ylabel())
            assert (list(axis.lines[0].get_ydata())) == (expected)
        finally:
            matplotlib_pyplot.close("all")

    def test_live_client_writes_pixels_and_application_latency(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "result.csv")
            entry = {"input_scale": 128.0, "scale_label": "res128px", "payload": {},
                     "input_metadata": {"output_pixel_count_per_image": 16384}}
            with patch_client_settings(self.runner, OUT_CSV=path, WARMUP=0, REPEAT=1,
                    REPEAT_IN_WINDOW=2, BATCH_SIZE=2, TASK_FAMILY="diffusion",
                    USE_ENERGY=False, USE_MIPS=False, energy_mod=None,
                    cpu_energy_mod=None, resource_usage_mod=None,
                    input_scale_entries=[entry]), patch.object(client.requests, "get",
                    return_value=SimpleNamespace(status_code=200, text="ok")), patch_client(self.runner, "_one_request", return_value={"latency_app_s": 0.065536,
                    "effective_input_scale": 128.0}):
                self.runner.main()
            with open(path) as f:
                row = next(csv.DictReader(f))
        assert ("output_pixels_per_request") in (row)
        assert (row["output_pixels_per_request"]) == ("32768.000000")
        assert (row["input_units_per_request"]) == ("256.000000")
        assert (row["latency_app_s_per_output_megapixel"]) == ("2.000000")
        assert (row["latency_s_per_output_megapixel"]) == ("nan")
        assert (row["container_attributed_j_per_output_megapixel"]) == ("nan")

    def test_packet_merge_recomputes_pixel_latency_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, destination = root / "result.csv", root / "merged.csv"
            pd.DataFrame([{"sniff_group_id": "case", "input_scale": 128,
                "output_pixels_per_request": 32768,
                "latency_s": 99, "latency_s_per_output_megapixel": 99,
                "latency_app_s_per_output_megapixel": 2,
                "container_attributed_j_per_output_megapixel": 10000}]).to_csv(source, index=False)
            latencies = root / "latencies.json"
            latencies.write_text(json.dumps({"schema_version": 2, "requests": {"case:1": {"latency_s": 0.032768}}}))
            merge_packet_latency.main([str(source), str(latencies), str(destination)])
            with destination.open() as f:
                row = next(csv.DictReader(f))
        assert (row["latency_s_per_output_megapixel"]) == ("1.000000")
        assert (row["latency_app_s_per_output_megapixel"]) == ("2")
        assert (row["container_attributed_j_per_output_megapixel"]) == ("10000")

    def test_partial_legacy_case_preserves_measurements_without_optional_pixel_columns(self):
        fields = [field for field in CSV_FIELDS
                  if not field.endswith(("_pixels_per_request", "_megapixel"))]
        original = dict.fromkeys(fields, "nan")
        original.update(input_scale="128", repeat_idx="0", warmup="0", status="ok",
                        error="", container_attributed_energy_eff_j="327.680000")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partial.csv"
            with path.open("w") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerow(original)
            preserved, added = _write_case_error_csv(
                task_info=SimpleNamespace(task_family="diffusion"), out_csv=str(path),
                cpu=1, mem=4, gpu="off", warmup=0, repeat=1, repeat_in_window=1,
                input_scales="128,256", error="request failed", preserve_existing=True)
            with path.open() as f:
                rows = list(csv.DictReader(f))
        assert ((preserved, added)) == ((1, 1))
        assert ({field: rows[0][field] for field in fields}) == (original)
        assert (rows[0]["output_pixels_per_request"]) == ("nan")
        assert (rows[1]["status"]) == ("error")
