import csv
import json
import math
import os
import sys
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest

import acprof.analysis.latency_model as analysis_latency_model
import acprof.analysis.latency_report as analysis_latency_report
import acprof.plotting.config as plotting_config
import acprof.plotting.data as plotting_data
import acprof.plotting.latency as plotting_latency
import acprof.plotting.metrics as plotting_metrics
from acprof.cli import plot
from acprof.platform import Environment


class TestLatencyModelReport:
    def test_report_calculation_returns_artifacts_without_writing_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        frame = pd.DataFrame(self._rows(gpu_modes=("off",)))
        result = analysis_latency_report.build_latency_model_report(frame, {})
        assert result.report['report_schema_version'] == 2
        assert result.report['status'] == 'ok'
        assert len(result.residuals) == 48
        assert list(tmp_path.iterdir()) == []
        skipped = analysis_latency_report.build_latency_model_report(pd.DataFrame(), {})
        assert skipped.report['status'] == 'skipped'
        assert skipped.residuals == ()

    def test_export_keeps_environment_identity_and_rejects_mixed_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame = pd.DataFrame(self._rows(gpu_modes=("off",)))
            frame["environment_class"] = "wsl2"
            analysis_latency_report.write_latency_model_report(frame, Environment("wsl2").metadata(), temporary)
            report, residuals = self._read_artifacts(temporary)
            assert (report["comparability_class"]) == ("wsl2")
            assert (report["collection_tier"]) == ("partial")
            assert (residuals)
            assert ({row["environment_class"] for row in residuals}) == ({"wsl2"})
            frame.loc[0, "environment_class"] = "native_linux"
            with pytest.raises(ValueError, match="mixed or inconsistent environments"):
                analysis_latency_report.write_latency_model_report(frame, Environment("wsl2").metadata(), temporary)

    def test_skipped_report_preserves_unknown_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            analysis_latency_report.write_latency_model_report(pd.DataFrame(), {}, temporary)
            report, _ = self._read_artifacts(temporary)
            assert (report["comparability_class"]) == ("unknown")

    def test_v2_plot_cli_writes_reports_and_figures_below_plots(self):
        from contextlib import ExitStack
        from pathlib import Path

        from acprof.artifact_layout import ArtifactLayout
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ArtifactLayout.for_new_run(root).initialize()
            csv_path = self._write_fixture(temporary, self._rows(gpu_modes=("off",)))
            with ExitStack() as stack:
                # Exercise the real CLI and report writer; raster rendering has its own tests.
                renderers = []
                for module in (plotting_metrics, plot.plotting_diagnostics, plotting_latency):
                    for name in dir(module):
                        if name.startswith("plot_") and callable(getattr(module, name)):
                            renderers.append(stack.enter_context(patch.object(module, name)))
                plot.main([csv_path])
            assert ((root / "plots/latency_model/latency_model_report.json").is_file())
            assert not ((root / "latency_model").exists())
            outputs = [Path(call.kwargs["out_png"]) for renderer in renderers for call in renderer.call_args_list
                       if call.kwargs.get("out_png")]
            assert (outputs)
            assert (all(path.is_relative_to(root / "plots") for path in outputs))

    FIELDNAMES = [
        "cpu_cores",
        "mem_cap_gb",
        "gpu_mode",
        "input_scale",
        "repeat_idx",
        "warmup",
        "status",
        "latency_s",
    ]

    @staticmethod
    def _base_latency(
        cpu: int,
        mem: int,
        gpu_mode: str,
        input_scale: int,
    ) -> float:
        log_scale = math.log(input_scale)
        log_mem = math.log(mem)
        if gpu_mode == "off":
            log_cpu = math.log(cpu)
            log_latency = (
                -4.0
                + 0.90 * log_scale
                - 0.65 * log_cpu
                + 0.02 * log_mem
                - 0.04 * log_scale * log_cpu
            )
        else:
            inverse_cpu = 1.0 / cpu
            log_latency = (
                -7.0
                + 0.55 * log_scale
                + 1.20 * inverse_cpu
                + 0.01 * log_mem
                - 0.06 * log_scale * inverse_cpu
            )
        return math.exp(log_latency)

    def _rows(
        self,
        gpu_modes: tuple[str, ...] = ("off", "on"),
        break_one_gpu_max_scale_case: bool = False,
        break_gpu_resource_config: bool = False,
    ) -> list[dict[str, str]]:
        rows = []
        for cpu in (1, 2, 4, 8):
            for mem in (4, 8, 16):
                for gpu_mode in gpu_modes:
                    for input_scale in (64, 128, 256, 512):
                        latency = self._base_latency(
                            cpu,
                            mem,
                            gpu_mode,
                            input_scale,
                        )
                        if (
                            break_one_gpu_max_scale_case
                            and gpu_mode == "on"
                            and cpu == 8
                            and mem == 16
                            and input_scale == 512
                        ):
                            latency *= 1.5
                        if (
                            break_gpu_resource_config
                            and gpu_mode == "on"
                            and cpu == 8
                            and mem == 16
                        ):
                            latency *= 0.5
                        for repeat_idx, repeat_factor in enumerate((0.99, 1.0, 1.01)):
                            rows.append({
                                "cpu_cores": str(cpu),
                                "mem_cap_gb": str(mem),
                                "gpu_mode": gpu_mode,
                                "input_scale": str(input_scale),
                                "repeat_idx": str(repeat_idx),
                                "warmup": "0",
                                "status": "ok",
                                "latency_s": f"{latency * repeat_factor:.12f}",
                            })
        return rows

    def _write_fixture(
        self,
        output_dir: str,
        rows: list[dict[str, str]],
    ) -> str:
        csv_path = os.path.join(output_dir, "result_all.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

        with open(
            os.path.join(output_dir, "static_meta.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump({
                "schema_version": 7,
                "model_name": "google-bert/bert-base-uncased",
                "task_family": "nlp",
                "input_scale_type": "seq_length",
            }, f)
        return csv_path

    @staticmethod
    def _read_artifacts(output_dir: str) -> tuple[dict, list[dict[str, str]]]:
        model_output_dir = os.path.join(
            output_dir,
            analysis_latency_report.LATENCY_MODEL_DIR,
        )
        with open(
            os.path.join(model_output_dir, analysis_latency_report.LATENCY_MODEL_REPORT),
            "r",
            encoding="utf-8",
        ) as f:
            report = json.load(f)
        with open(
            os.path.join(model_output_dir, analysis_latency_report.LATENCY_MODEL_RESIDUALS),
            "r",
            encoding="utf-8",
            newline="",
        ) as f:
            residual_rows = list(csv.DictReader(f))
        return report, residual_rows

    def test_plot_main_writes_group_validated_positive_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_fixture(tmp, self._rows())

            with patch.object(sys, "argv", ["acprof plot", csv_path]), patch.object(
                plotting_metrics,
                'plot_metric',
            ), patch.object(plotting_metrics, 'plot_cold_start_bar'):
                plot.main()

            model_output_dir = os.path.join(tmp, analysis_latency_report.LATENCY_MODEL_DIR)
            assert (os.path.isdir(model_output_dir))
            for artifact_name in (
                analysis_latency_report.LATENCY_MODEL_REPORT,
                analysis_latency_report.LATENCY_MODEL_RESIDUALS,
                plotting_config.LATENCY_MODEL_RESIDUAL_PLOT,
                plotting_config.LATENCY_MODEL_FIT_CURVES_PLOT,
            ):
                assert (os.path.isfile(
                        os.path.join(model_output_dir, artifact_name)
                    ))
                assert not (os.path.exists(os.path.join(tmp, artifact_name)))

            report, residual_rows = self._read_artifacts(tmp)

        assert (report["report_schema_version"]) == (2)
        assert (report["status"]) == ("ok")
        assert (report["prediction_ready"])
        assert (report["positive_prediction_form"])
        assert (report["target_metric"]) == ("latency_s")
        assert (report["model_name"]) == ("google-bert/bert-base-uncased")
        assert (report["task_family"]) == ("nlp")
        assert (report["raw_rows"]) == (288)
        assert (report["case_rows"]) == (96)
        assert (report["aggregation"]["target_statistic"]) == ("median")
        assert not (report["aggregation"]["repetitions_split_across_train_and_test"])
        assert (set(report["models"])) == ({"cpu", "gpu"})

        for hardware_model in ("cpu", "gpu"):
            model_report = report["models"][hardware_model]
            assert (model_report["status"]) == ("ok")
            assert (model_report["prediction_ready"])
            assert ("log_input_scale_x_") in (" ".join(model_report["feature_columns"]))
            config_validation = model_report["validation"][
                "resource_configuration_holdout"
            ]
            scale_validation = model_report["validation"]["input_scale_holdout"]
            assert (config_validation["available"])
            assert (config_validation["train_test_group_overlap_count"]) == (0)
            assert (scale_validation["available"])
            assert (scale_validation["strict_extrapolation"])
            assert (scale_validation["train_input_scale_max"]) < (scale_validation["test_input_scale_min"])
            assert (model_report["metrics"]["resource_configuration_holdout"]["r2"]) > (0.99)
            assert (model_report["metrics"]["input_scale_holdout"]["r2"]) > (0.99)
            assert (model_report["metrics"]["resource_configuration_holdout"][
                    "nonpositive_prediction_count"
                ]) == (0)

        assert (len(residual_rows)) == (96)
        assert (all(row["split"] == "out_of_fold_test" for row in residual_rows))
        assert (all(int(row["repeat_count"]) == 3 for row in residual_rows))
        assert (all(float(row["resource_config_oof_predicted_latency_s"]) > 0.0 for row in residual_rows))
        assert (all(row["report_schema_version"] == "2" for row in residual_rows))
        for row in residual_rows:
            assert ("predicted_latency_s") not in (row)
            assert ("residual_s") not in (row)
            assert (float(row["resource_config_oof_residual_s"])) == (float(row["latency_s"]) - float(row["resource_config_oof_predicted_latency_s"])) or round(abs((float(row["resource_config_oof_residual_s"])) - (float(row["latency_s"]) - float(row["resource_config_oof_predicted_latency_s"]))), 8) == 0
            if float(row["input_scale"]) == 512.0:
                assert (row["max_scale_holdout_predicted_latency_s"]) != ("")
            else:
                assert (row["max_scale_holdout_predicted_latency_s"]) == ("")
        unique_cases = {
            (
                row["hardware_model"],
                row["cpu_cores"],
                row["mem_cap_gb"],
                row["input_scale"],
            )
            for row in residual_rows
        }
        assert (len(unique_cases)) == (len(residual_rows))

    def test_gpu_only_matrix_no_longer_fails_as_singular(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = self._rows(gpu_modes=("on",))
            csv_path = self._write_fixture(tmp, rows)
            df = plotting_data.prepare_df(csv_path)
            analysis_latency_report.write_latency_model_report(df, plotting_data.read_static_meta(csv_path), tmp)
            report, residual_rows = self._read_artifacts(tmp)

        assert (report["status"]) == ("ok")
        assert (set(report["models"])) == ({"gpu"})
        gpu_report = report["models"]["gpu"]
        assert (gpu_report["status"]) == ("ok")
        assert (gpu_report["quality_gate"]["failures"]) == ([])
        config_validation = gpu_report["validation"][
            "resource_configuration_holdout"
        ]
        assert (config_validation["folds"]) == (12)
        assert (config_validation["completed_folds"]) == (12)
        assert (config_validation["train_test_group_overlap_count"]) == (0)
        assert (gpu_report["metrics"]["resource_configuration_holdout"][
                "prediction_count"
            ]) == (48)
        assert (gpu_report["metrics"]["input_scale_holdout"]["prediction_count"]) == (12)
        for validation_name in (
            "resource_configuration_holdout",
            "input_scale_holdout",
        ):
            metrics = gpu_report["metrics"][validation_name]
            assert (metrics["nonfinite_prediction_count"]) == (0)
            assert (metrics["nonpositive_prediction_count"]) == (0)
        assert (len(residual_rows)) == (48)
        assert (all(float(row["fitted_predicted_latency_s"]) > 0.0 for row in residual_rows))

    def test_cpu_log_scale_squared_feature_captures_curvature(self) -> None:
        rows = self._rows(gpu_modes=("off",))
        repeat_factors = (0.99, 1.0, 1.01)
        for row in rows:
            cpu = int(row["cpu_cores"])
            mem = int(row["mem_cap_gb"])
            input_scale = int(row["input_scale"])
            log_scale = math.log(input_scale)
            log_cpu = math.log(cpu)
            log_latency = (
                -4.50
                + 0.40 * log_scale
                + 0.08 * log_scale**2
                - 0.70 * log_cpu
                + 0.01 * math.log(mem)
                - 0.04 * log_scale * log_cpu
            )
            repeat_factor = repeat_factors[int(row["repeat_idx"])]
            row["latency_s"] = f"{math.exp(log_latency) * repeat_factor:.12f}"

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_fixture(tmp, rows)
            df = pd.read_csv(csv_path)
            analysis_latency_report.write_latency_model_report(
                df,
                plotting_data.read_static_meta(csv_path),
                tmp,
            )
            report, residual_rows = self._read_artifacts(tmp)

        cpu_report = report["models"]["cpu"]
        assert ("log_input_scale_squared") in (cpu_report["selected_feature_columns"])
        assert (cpu_report["metrics"]["fit"]["relative_mae"]) < (1e-6)
        fitted_relative_errors = [
            abs(
                float(row["fitted_predicted_latency_s"])
                - float(row["latency_s"])
            )
            / float(row["latency_s"])
            for row in residual_rows
        ]
        assert (max(fitted_relative_errors)) < (1e-6)

    def test_cpu_log_response_surface_captures_resource_interactions(self) -> None:
        rows = self._rows(gpu_modes=("off",))
        repeat_factors = (0.99, 1.0, 1.01)
        for row in rows:
            cpu = int(row["cpu_cores"])
            mem = int(row["mem_cap_gb"])
            input_scale = int(row["input_scale"])
            log_scale = math.log(input_scale)
            log_cpu = math.log(cpu)
            log_mem = math.log(mem)
            log_latency = (
                -4.50
                + 0.40 * log_scale
                + 0.08 * log_scale**2
                - 0.70 * log_cpu
                + 0.11 * log_cpu**2
                + 0.02 * log_mem
                - 0.04 * log_scale * log_cpu
                + 0.03 * log_scale * log_mem
                - 0.05 * log_cpu * log_mem
            )
            repeat_factor = repeat_factors[int(row["repeat_idx"])]
            row["latency_s"] = (
                f"{math.exp(log_latency) * repeat_factor:.12f}"
            )

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_fixture(tmp, rows)
            df = pd.read_csv(csv_path)
            analysis_latency_report.write_latency_model_report(
                df,
                plotting_data.read_static_meta(csv_path),
                tmp,
            )
            report, _ = self._read_artifacts(tmp)

        cpu_report = report["models"]["cpu"]
        for feature_name in (
            "log_cpu_cores_squared",
            "log_input_scale_x_log_mem_cap_gb",
            "log_cpu_cores_x_log_mem_cap_gb",
        ):
            assert (feature_name) in (cpu_report["selected_feature_columns"])
        for metric_name in (
            "fit",
            "resource_configuration_holdout",
            "input_scale_holdout",
        ):
            assert (cpu_report["metrics"][metric_name][
                    "mean_absolute_percentage_error"
                ]) < (1e-6)

    def test_gpu_inverse_square_feature_captures_cpu_saturation(self) -> None:
        rows = self._rows(gpu_modes=("on",))
        repeat_factors = (0.99, 1.0, 1.01)
        for row in rows:
            cpu = int(row["cpu_cores"])
            mem = int(row["mem_cap_gb"])
            input_scale = int(row["input_scale"])
            inverse_cpu = 1.0 / cpu
            log_scale = math.log(input_scale)
            log_latency = (
                -6.85
                + 0.39 * log_scale
                - 1.80 * inverse_cpu
                + 2.38 * inverse_cpu**2
                + 0.01 * math.log(mem)
                + 0.24 * log_scale * inverse_cpu
                - 0.16 * log_scale * inverse_cpu**2
            )
            repeat_factor = repeat_factors[int(row["repeat_idx"])]
            row["latency_s"] = f"{math.exp(log_latency) * repeat_factor:.12f}"

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_fixture(tmp, rows)
            df = pd.read_csv(csv_path)
            analysis_latency_report.write_latency_model_report(
                df,
                plotting_data.read_static_meta(csv_path),
                tmp,
            )
            report, residual_rows = self._read_artifacts(tmp)

        gpu_report = report["models"]["gpu"]
        assert ("inverse_cpu_cores_squared") in (gpu_report["selected_feature_columns"])
        assert ("log_input_scale_x_inverse_cpu_cores_squared") in (gpu_report["selected_feature_columns"])
        assert (gpu_report["metrics"]["fit"]["relative_mae"]) < (1e-6)
        fitted_relative_errors = [
            abs(
                float(row["fitted_predicted_latency_s"])
                - float(row["latency_s"])
            )
            / float(row["latency_s"])
            for row in residual_rows
        ]
        assert (max(fitted_relative_errors)) < (1e-6)

    def test_gpu_log_spline_captures_shared_input_scale_regime_change(self) -> None:
        rows = []
        repeat_factors = (0.99, 1.0, 1.01)
        for cpu in (1, 2, 4, 8):
            for mem in (4, 8, 16):
                for input_scale in (80, 160, 240, 320, 400, 480):
                    inverse_cpu = 1.0 / cpu
                    log_scale = math.log(input_scale)
                    # A shared GPU execution-regime change after scale 240.
                    # The post-change slope remains stable, so a forward
                    # maximum-scale holdout can validate the spline boundary.
                    regime_shift = 0.75 if input_scale > 240 else 0.0
                    log_latency = (
                        -7.0
                        + 0.55 * log_scale
                        + regime_shift
                        + 1.20 * inverse_cpu
                        + 0.01 * math.log(mem)
                        - 0.06 * log_scale * inverse_cpu
                    )
                    latency = math.exp(log_latency)
                    for repeat_idx, repeat_factor in enumerate(repeat_factors):
                        rows.append({
                            "cpu_cores": str(cpu),
                            "mem_cap_gb": str(mem),
                            "gpu_mode": "on",
                            "input_scale": str(input_scale),
                            "repeat_idx": str(repeat_idx),
                            "warmup": "0",
                            "status": "ok",
                            "latency_s": (
                                f"{latency * repeat_factor:.12f}"
                            ),
                        })

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_fixture(tmp, rows)
            df = pd.read_csv(csv_path)
            analysis_latency_report.write_latency_model_report(
                df,
                plotting_data.read_static_meta(csv_path),
                tmp,
            )
            report, _ = self._read_artifacts(tmp)

        gpu_report = report["models"]["gpu"]
        assert (gpu_report["input_scale_basis"]["type"]) == ("continuous_piecewise_linear_spline_in_log_space")
        assert (gpu_report["input_scale_basis"]["knots"]) == ([160.0, 240.0, 320.0, 400.0])
        assert ("log_input_scale_hinge_at_240") in (gpu_report["selected_feature_columns"])
        for metric_name in (
            "fit",
            "resource_configuration_holdout",
            "input_scale_holdout",
        ):
            assert (gpu_report["metrics"][metric_name][
                    "mean_absolute_percentage_error"
                ]) < (1e-6)
        assert (gpu_report["status"]) == ("ok")

    def test_gpu_unstable_upper_boundary_uses_continuous_affine_tail(self) -> None:
        rows = []
        scale_latency = {
            1: 0.335,
            2: 0.503,
            5: 0.916,
            10: 1.534,
            20: 2.336,
            30: 3.506,
        }
        for cpu in (1, 2, 4, 8):
            for mem in (4, 8, 16):
                resource_factor = math.exp(
                    0.02 / cpu + 0.001 * math.log(mem)
                )
                for input_scale, base_latency in scale_latency.items():
                    latency = base_latency * resource_factor
                    for repeat_idx, repeat_factor in enumerate((0.99, 1.0, 1.01)):
                        rows.append({
                            "cpu_cores": str(cpu),
                            "mem_cap_gb": str(mem),
                            "gpu_mode": "on",
                            "input_scale": str(input_scale),
                            "repeat_idx": str(repeat_idx),
                            "warmup": "0",
                            "status": "ok",
                            "latency_s": (
                                f"{latency * repeat_factor:.12f}"
                            ),
                        })

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_fixture(tmp, rows)
            df = pd.read_csv(csv_path)
            analysis_latency_report.write_latency_model_report(
                df,
                plotting_data.read_static_meta(csv_path),
                tmp,
            )
            report, _ = self._read_artifacts(tmp)

        gpu_report = report["models"]["gpu"]
        upper_tail = gpu_report["input_scale_basis"]["upper_extrapolation"]
        scale_validation = gpu_report["validation"]["input_scale_holdout"]
        assert (upper_tail["enabled"])
        assert (upper_tail["calibration"]["mean_absolute_percentage_error"]) > (analysis_latency_model.LATENCY_MODEL_GPU_UPPER_TAIL_ACTIVATION_MAPE)
        assert (gpu_report["metrics"]["input_scale_holdout"][
                "mean_absolute_percentage_error"
            ]) < (0.04)
        assert not (scale_validation["r2_quality_gate_applicable"])
        assert (gpu_report["metrics"]["input_scale_holdout"]["r2"]) < (0.0)
        assert (gpu_report["status"]) == ("ok")

    def test_bad_gpu_extrapolation_cannot_hide_behind_cpu_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = self._rows(break_one_gpu_max_scale_case=True)
            csv_path = self._write_fixture(tmp, rows)
            df = pd.read_csv(csv_path)
            analysis_latency_report.write_latency_model_report(df, plotting_data.read_static_meta(csv_path), tmp)
            report, _ = self._read_artifacts(tmp)

        assert (report["models"]["cpu"]["status"]) == ("ok")
        gpu_report = report["models"]["gpu"]
        pooled_metrics = gpu_report["metrics"]["input_scale_holdout"]
        assert (pooled_metrics["r2"]) >= (analysis_latency_model.LATENCY_MODEL_MIN_VALIDATION_R2)
        assert (pooled_metrics["relative_mae"]) <= (analysis_latency_model.LATENCY_MODEL_MAX_VALIDATION_RELATIVE_MAE)
        assert (gpu_report["validation"]["input_scale_holdout"][
                "worst_case_relative_error"
            ]) > (analysis_latency_model.LATENCY_MODEL_MAX_VALIDATION_CASE_RELATIVE_ERROR)
        assert (gpu_report["status"]) == ("poor_fit")
        assert (report["status"]) == ("poor_fit")
        assert not (report["prediction_ready"])
        assert not (report["quality_gate"]["passed"])
        assert (any(
                "gpu: input_scale_holdout" in failure
                for failure in report["quality_gate"]["failures"]
            ))

    def test_bad_held_out_configuration_cannot_hide_in_pooled_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = self._rows(break_gpu_resource_config=True)
            csv_path = self._write_fixture(tmp, rows)
            df = pd.read_csv(csv_path)
            analysis_latency_report.write_latency_model_report(df, plotting_data.read_static_meta(csv_path), tmp)
            report, _ = self._read_artifacts(tmp)

        gpu_report = report["models"]["gpu"]
        pooled_metrics = gpu_report["metrics"]["resource_configuration_holdout"]
        assert (pooled_metrics["r2"]) >= (analysis_latency_model.LATENCY_MODEL_MIN_VALIDATION_R2)
        assert (pooled_metrics["relative_mae"]) <= (analysis_latency_model.LATENCY_MODEL_MAX_VALIDATION_RELATIVE_MAE)
        assert (gpu_report["validation"]["resource_configuration_holdout"][
                "worst_fold_relative_mae"
            ]) > (analysis_latency_model.LATENCY_MODEL_MAX_CONFIGURATION_FOLD_RELATIVE_MAE)
        assert (gpu_report["status"]) == ("poor_fit")
        assert (report["status"]) == ("poor_fit")
        assert not (report["prediction_ready"])
        assert (any(
                "held-out configuration fold" in failure
                for failure in gpu_report["quality_gate"]["failures"]
            ))

    def test_two_scales_are_not_misreported_as_valid_extrapolation(self) -> None:
        rows = [
            row
            for row in self._rows(gpu_modes=("on",))
            if int(row["input_scale"]) in (64, 128)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_fixture(tmp, rows)
            df = pd.read_csv(csv_path)
            analysis_latency_report.write_latency_model_report(df, plotting_data.read_static_meta(csv_path), tmp)
            report, _ = self._read_artifacts(tmp)

        gpu_report = report["models"]["gpu"]
        scale_validation = gpu_report["validation"]["input_scale_holdout"]
        assert not (scale_validation["available"])
        assert ("at least 3 input scales") in (scale_validation["reason"])
        assert (gpu_report["status"]) == ("unvalidated")
        assert (report["status"]) == ("unvalidated")
        assert not (report["prediction_ready"])

    def test_invalid_gpu_mode_skips_and_replaces_stale_residuals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = self._rows(gpu_modes=("on",))
            for row in rows:
                row["gpu_mode"] = "mystery-device"
            csv_path = self._write_fixture(tmp, rows)
            model_output_dir = os.path.join(tmp, analysis_latency_report.LATENCY_MODEL_DIR)
            os.makedirs(model_output_dir, exist_ok=True)
            residuals_path = os.path.join(
                model_output_dir,
                analysis_latency_report.LATENCY_MODEL_RESIDUALS,
            )
            with open(residuals_path, "w", encoding="utf-8") as f:
                f.write("stale-marker\n")

            df = pd.read_csv(csv_path)
            analysis_latency_report.write_latency_model_report(df, plotting_data.read_static_meta(csv_path), tmp)
            report, residual_rows = self._read_artifacts(tmp)
            with open(residuals_path, "r", encoding="utf-8") as f:
                residual_text = f.read()

        assert (report["status"]) == ("skipped")
        assert not (report["prediction_ready"])
        assert ("unsupported gpu_mode") in (report["reason"])
        assert (residual_rows) == ([])
        assert ("stale-marker") not in (residual_text)
        assert ("report_schema_version") in (residual_text.splitlines()[0])

    def test_residual_plot_writes_diagnostic_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_fixture(tmp, self._rows())
            df = pd.read_csv(csv_path)
            analysis_latency_report.write_latency_model_report(
                df,
                plotting_data.read_static_meta(csv_path),
                tmp,
            )
            model_output_dir = os.path.join(tmp, analysis_latency_report.LATENCY_MODEL_DIR)
            residuals_path = os.path.join(
                model_output_dir,
                analysis_latency_report.LATENCY_MODEL_RESIDUALS,
            )
            out_png = os.path.join(
                model_output_dir,
                plotting_config.LATENCY_MODEL_RESIDUAL_PLOT,
            )

            plotted = plotting_latency.plot_latency_model_residuals(
                residuals_path,
                out_png,
            )

            assert (plotted)
            assert (os.path.isfile(out_png))
            assert (os.path.getsize(out_png)) > (0)

    def test_fit_curve_plot_writes_cpu_and_gpu_configuration_curves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_fixture(tmp, self._rows())
            df = pd.read_csv(csv_path)
            analysis_latency_report.write_latency_model_report(
                df,
                plotting_data.read_static_meta(csv_path),
                tmp,
            )
            model_output_dir = os.path.join(tmp, analysis_latency_report.LATENCY_MODEL_DIR)
            residuals_path = os.path.join(
                model_output_dir,
                analysis_latency_report.LATENCY_MODEL_RESIDUALS,
            )
            report_path = os.path.join(
                model_output_dir,
                analysis_latency_report.LATENCY_MODEL_REPORT,
            )
            out_png = os.path.join(
                model_output_dir,
                plotting_config.LATENCY_MODEL_FIT_CURVES_PLOT,
            )

            plotted = plotting_latency.plot_latency_model_fit_curves(
                residuals_path,
                report_path,
                out_png,
            )

            assert (plotted)
            assert (os.path.isfile(out_png))
            assert (os.path.getsize(out_png)) > (0)

    def test_residual_plot_skips_header_only_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            residuals_path = os.path.join(
                tmp,
                analysis_latency_report.LATENCY_MODEL_RESIDUALS,
            )
            with open(
                residuals_path,
                "w",
                encoding="utf-8",
                newline="",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=analysis_latency_report.LATENCY_MODEL_RESIDUAL_FIELDS,
                )
                writer.writeheader()
            out_png = os.path.join(
                tmp,
                plotting_config.LATENCY_MODEL_RESIDUAL_PLOT,
            )

            plotted = plotting_latency.plot_latency_model_residuals(
                residuals_path,
                out_png,
            )

            assert not (plotted)
            assert not (os.path.exists(out_png))
