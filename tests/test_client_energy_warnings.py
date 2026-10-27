import csv
import io
import json
import math
import os
import tempfile
from contextlib import ExitStack, redirect_stderr
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from client_fixtures import patch_client

from acprof.config import CSV_FIELDS
from acprof.host import client
from acprof.host.client import ClientRunner
from acprof.host.measurement import metrics as client_metrics, requests as window_requests
from acprof.host.measurement.config import ClientConfig
from acprof.monitors import energy_cpu


class TestClientWindowEnergy:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        from platform_fixtures import native_policy
        native_policy(self._request)
        temporary = tmp_path
        self.runner = ClientRunner(ClientConfig(out_csv=os.path.join(str(temporary), "result.csv")))
        gpu_uuid = patch_client(self.runner, "GPU_DEVICE_UUID", "GPU-fixture")
        gpu_uuid.start()
        self._request.addfinalizer(partial(gpu_uuid.stop))
        for name in ("IDLE_SECONDS", "IDLE_COOLDOWN_SECONDS"):
            mocked = patch_client(self.runner, name, 0.0)
            mocked.start()
            self._request.addfinalizer(partial(mocked.stop))

    @pytest.mark.parametrize(("variant", "valid"), [
        ("reordered", True),
        ("missing_column", False),
        ("moved_column", False),
    ])
    def test_append_respects_existing_column_order_and_rejects_invalid_headers(self, variant, valid):
        reordered = ["status", "error", *[field for field in CSV_FIELDS
                                          if field not in {"status", "error"}]]
        fields = {
            "reordered": reordered,
            "missing_column": reordered[:-1],
            "moved_column": [*reordered[:-1], reordered[0]],
        }[variant]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "result.csv")
            original = dict.fromkeys(fields, "nan")
            original.update(cpu_cores="1", status="ok", error="")
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(original)
            before = path.read_bytes()
            with ExitStack() as stack:
                for name, value in {
                    "OUT_CSV": str(path), "CPU_CORES": "2", "GPU_MODE": "off",
                    "USE_ENERGY": False, "USE_MIPS": False, "IDLE_DEBUG": False,
                    "PROFILING_MODE": "full", "SNIFF_GROUPS_PATH": "",
                    "input_scale_entries": [{"input_scale": 1.0}],
                }.items():
                    stack.enter_context(patch_client(self.runner, name, value))
                ready = stack.enter_context(patch.object(
                    client.requests, "get", side_effect=client.requests.exceptions.ConnectionError("offline fixture")))
                if valid:
                    self.runner.main()
                else:
                    with pytest.raises(RuntimeError, match="CSV.*columns"):
                        self.runner.main()
                    ready.assert_not_called()
                    assert (path.read_bytes()) == (before)
                    assert (list(Path(tmp).iterdir())) == ([path])
                    return
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                assert (reader.fieldnames) == (fields)
            assert (rows[0]) == (original)
            assert (len(rows)) == (2)
            assert (rows[1]["cpu_cores"]) == ("2")
            assert (rows[1]["status"]) == ("error")
            assert ("offline fixture") in (rows[1]["error"])

    @pytest.mark.parametrize('invalid', ['[16,32]', '[16,32,32]', '[16,32,128]'])
    def test_frozen_scale_order_changes_execution_without_changing_payloads(self, invalid):
        entries = [{"input_scale": scale, "payload": {"text": f"payload-{scale}"}}
                   for scale in (16, 32, 64)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "input_scale_plan.json")
            path.write_text(json.dumps({"schema_version": 2, "entries": entries}))
            original = path.read_bytes()
            with patch_client(self.runner, "INPUT_SCALE_PLAN_FILE", str(path)), patch.object(self.runner.config, "input_scale_order", "[64,16,32]"):
                loaded = self.runner._load_input_scale_entries()
                assert ([e["input_scale"] for e in loaded]) == ([64, 16, 32])
                assert ([e["payload"]["text"] for e in loaded]) == (["payload-64", "payload-16", "payload-32"])
                assert (path.read_bytes()) == (original)
                with patch.object(self.runner.config, "input_scale_order", invalid):
                    with pytest.raises(ValueError, match="frozen matrix input-scale"):
                        self.runner._load_input_scale_entries()

    @pytest.mark.parametrize(("invalid_value", "expected"), [('NaN', 'non-finite'), ('1e999', 'non-finite')])
    def test_input_plan_rejects_nonfinite_numbers_before_measurement(self, invalid_value, expected):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "input_scale_plan.json")
            path.write_text(
                '{"schema_version":2,"entries":[{"input_scale":' + invalid_value
                + ',"payload":{"text":"unchanged"}}]}'
            )
            original = path.read_bytes()
            with patch_client(self.runner, "INPUT_SCALE_PLAN_FILE", str(path)):
                with pytest.raises(ValueError, match=expected):
                    self.runner._load_input_scale_entries()
            assert (path.read_bytes()) == (original)

    @pytest.mark.parametrize(("policy", "available"), [
        ("auto", False),
        ("required", False),
        ("required", True),
    ])
    def test_full_client_dram_policy_and_separate_window_request_units(self, policy, available):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            result = energy_cpu._nan_result(cpu_idle_power_w=1.0)
            result.cpu_energy_total_j = 12.0
            result.vcpu_energy_eff_j = 6.0
            if available:
                result.dram = energy_cpu.DRAMEnergyResult(
                    window_energy_j=10.0, window_duration_s=2.0, avg_power_w=5.0,
                    peak_power_w=7.0, idle_power_w=1.0,
                    window_effective_energy_j=8.0, status="verified")
            monitor = Mock()
            monitor.idle_power_w = 1.0
            monitor.idle_trace = {}
            monitor.stop.return_value = (result, "", [])
            path = Path(tmp, "result.csv")
            settings = {
                "OUT_CSV": str(path), "PROFILING_MODE": "full", "DRAM_ENERGY": policy,
                "WARMUP": 0, "REPEAT": 1, "REPEAT_IN_WINDOW": 2,
                "USE_ENERGY": False, "USE_MIPS": False, "GPU_MODE": "off",
                "energy_mod": None, "resource_usage_mod": None,
                "cpu_energy_mod": SimpleNamespace(CPUEnergyMonitor=Mock(return_value=monitor)),
                "input_scale_entries": [{"input_scale": 1.0, "scale_label": "one", "payload": {}}],
            }
            for name, value in settings.items():
                stack.enter_context(patch_client(self.runner, name, value))
            stack.enter_context(patch.object(client.requests, "get",
                return_value=SimpleNamespace(status_code=200, text="ok")))
            stack.enter_context(patch_client(self.runner, "_one_request",
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0}))
            if policy == "required" and not available:
                with pytest.raises(client.EnergyAbort, match="required DRAM"):
                    self.runner.main()
                return
            self.runner.main()
            with path.open() as stream:
                row = next(csv.DictReader(stream))
            assert (row["status"]) == ("ok"), row["error"]
            assert (float(row["container_attributed_energy_eff_j"])) == (3.0)
            assert (row["result_origin"]) == ("formal_measurement")
            if available:
                assert (float(row["dram_window_energy_j"])) == (10.0)
                assert (float(row["dram_energy_per_request_j"])) == (5.0)
                assert (float(row["dram_effective_energy_per_request_j"])) == (4.0)
            else:
                assert (row["dram_energy_status"]) == ("unavailable")
                assert (row["dram_window_energy_j"]) == ("nan")

    @pytest.mark.parametrize(("threshold", "expected"), [(0.05, 2 / 3), (0.1, 1 / 3)])
    def test_latency_metrics_use_the_explicit_slo_threshold(self, threshold, expected) -> None:
        latencies = [0.01, 0.06, 0.2, float("nan"), float("inf")]
        assert (client_metrics._latency_distribution_metrics(
                "latency_app", latencies, slow_latency_threshold_s=threshold,
            )[
                "latency_app_slow_ratio"
            ]) == (expected) or round(abs((client_metrics._latency_distribution_metrics(
                "latency_app", latencies, slow_latency_threshold_s=threshold,
            )[
                "latency_app_slow_ratio"
            ]) - (expected)), 7) == 0

    def test_default_idle_diag_path_uses_dedicated_debug_directory(self) -> None:
        with patch_client(self.runner, "IDLE_DIAG_PATH", ""):
            assert (self.runner._idle_diag_path("results/model/result_case.csv")) == (os.path.join(
                    "results",
                    "model",
                    "debug_idle_diag",
                    "result_case.csv.idle_diag.jsonl",
                ))

    def test_matched_control_only_starts_energy_monitors_and_applies_baselines(self) -> None:
        events = []

        class FakeEnergyMonitor:
            def __init__(self, name, avg_field):
                self.name = name
                self.avg_field = avg_field
                self.applied = None

            def start(self):
                events.append(f"start:{self.name}")

            def stop(self):
                events.append(f"stop:{self.name}")
                result = SimpleNamespace(**{self.avg_field: 7.0})
                if self.name == "gpu":
                    return result, "GPU", "", [(0.0, 7.0), (1.0, 7.0)]
                return result, "", [SimpleNamespace(timestamp=0.0), SimpleNamespace(timestamp=1.0)]

            def apply_control_baseline(self, result, samples, trace=False):
                events.append(f"apply:{self.name}:{trace}")
                self.applied = (result, samples)

        class FakeResourceMonitor:
            def start(self):
                events.append("start:resource")

            def stop(self):
                events.append("stop:resource")
                return None, "", []

        class FakeMIPSMonitor:
            def start(self):
                events.append("start:mips")

            def stop(self, repeat_in_window, latency_app_s):
                events.append(f"stop:mips:{repeat_in_window}:{latency_app_s}")

        gpu_monitor = FakeEnergyMonitor("gpu", "avg_power_total_w")
        cpu_monitor = FakeEnergyMonitor("cpu", "cpu_avg_power_total_w")

        with patch_client(self.runner, "IDLE_SECONDS", 2.0), patch_client(self.runner, "IDLE_DEBUG", True
        ), patch.object(
            client.time,
            "sleep",
            side_effect=lambda seconds: events.append(f"sleep:{seconds}"),
        ):
            from acprof.host.measurement.monitors import MonitorGroup, run_matched_control_window
            monitors = MonitorGroup(close=False)
            for name, monitor in (("gpu", gpu_monitor), ("cpu", cpu_monitor),
                                  ("resource", FakeResourceMonitor()), ("mips", FakeMIPSMonitor())):
                monitors.add(name, monitor)
            run_matched_control_window(monitors, idle_seconds=2.0, trace=True)

        assert (events) == ([
                "start:gpu",
                "start:cpu",
                "sleep:2.0",
                "stop:gpu",
                "stop:cpu",
                "apply:gpu:True",
                "apply:cpu:True",
            ])
        assert (gpu_monitor.applied) is not None
        assert (cpu_monitor.applied) is not None

    def test_one_request_reports_prepared_body_and_response_counts(self) -> None:
        class FakeResponse:
            status_code = 200
            text = json.dumps({
                "effective_input_scale": 3,
                "output_length": 12,
                "output_token_count": 5,
            })
            reason = "OK"
            request = SimpleNamespace(body='{"text":"\u2713"}')

            @staticmethod
            def json():
                return {
                    "effective_input_scale": 3,
                    "output_length": 12,
                    "output_token_count": 5,
                }

        payload = {"text": "hello", "params": {"z": 1, "a": 2}}
        with patch.object(client.requests, "post", return_value=FakeResponse()), patch.object(
            client.time,
            "perf_counter",
            side_effect=[10.0, 10.25],
        ), patch_client(self.runner, "_FIRST_PREDICT_APP_S", float("nan")):
            result = self.runner._one_request(3.0, "req", payload_override=payload)
            first_predict_app_s = self.runner.first_predict_app_s

        assert (result["request_payload_bytes"]) == (14.0)
        assert (result["output_length"]) == (12.0)
        assert (result["output_token_count"]) == (5.0)
        assert (result["task_param"]) == ('{"a":2,"z":1}')
        assert (first_predict_app_s) == (0.25)

    def test_one_request_rejects_nonfinite_success_json(self) -> None:
        class FakeResponse:
            status_code = 200
            text = '{"effective_input_scale":3,"workload_contract":{"score":NaN}}'
            reason = "OK"
            request = SimpleNamespace(body="{}")

            @staticmethod
            def json():
                return json.loads(FakeResponse.text)

        with patch.object(client.requests, "post", return_value=FakeResponse()), patch.object(
            client.time,
            "perf_counter",
            side_effect=[10.0, 10.25],
        ), pytest.raises(ValueError, match="non-finite"):
            self.runner._one_request(3.0, "req", payload_override={"text": "hello"})

    def test_task_param_includes_top_level_timeseries_prediction_length(self) -> None:
        payload = {
            "context": [[0.1, 0.2]],
            "prediction_length": 64,
        }
        assert (window_requests.canonical_task_param(payload)) == ('{"prediction_length":64}')

    def test_request_shape_metrics_are_averaged_per_window(self) -> None:
        responses = iter([
            {
                "latency_app_s": 0.1,
                "effective_input_scale": 1.0,
                "request_payload_bytes": 100,
                "output_length": 10,
                "output_token_count": 3,
            },
            {
                "latency_app_s": 0.2,
                "effective_input_scale": 1.0,
                "request_payload_bytes": 104,
                "output_length": 14,
                "output_token_count": 5,
            },
        ])
        payload = {
            "audio_base64": "UklGRg==",
            "audio_format": "wav",
            "sample_rate": 16000,
            "params": {"task": "transcribe", "language": "english"},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = os.path.join(tmp_dir, "result.csv")
            with patch_client(self.runner, "OUT_CSV", out_csv), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 1), patch_client(self.runner, "REPEAT_IN_WINDOW", 2
            ), patch_client(self.runner, "BATCH_SIZE", 2), patch_client(self.runner, "CPU_CORES", "2"
            ), patch_client(self.runner, "USE_ENERGY", False), patch_client(self.runner, "USE_MIPS", False
            ), patch_client(self.runner, "energy_mod", None), patch_client(self.runner, "cpu_energy_mod", None
            ), patch_client(self.runner, "resource_usage_mod", None), patch_client(self.runner,
                "input_scale_entries",
                [
                    {
                        "input_scale": 1.0,
                        "scale_label": "dur1s",
                        "payload": payload,
                        "input_metadata": {"input_num_samples": 16000},
                    }
                ],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner, "_one_request", side_effect=lambda *args, **kwargs: next(responses)):
                self.runner.main()

            with open(out_csv, encoding="utf-8", newline="") as f:
                row = next(csv.DictReader(f))

        assert (row["input_units_per_request"]) == ("2.000000")
        assert (row["input_num_samples"]) == ("16000.000000")
        assert (row["request_payload_bytes"]) == ("102.000000")
        assert (row["output_length_avg"]) == ("12.000000")
        assert (row["output_token_count_avg"]) == ("4.000000")
        assert (row["latency_app_s_per_input_unit"]) == ("0.075000")
        assert (row["throughput_samples_per_s_per_cpu_core"]) == ("6.666667")
        assert (row["task_param"]) == ('{"language":"english","task":"transcribe"}')

    def test_auto_repeat_window_prepares_each_scale_with_warmup_only(self) -> None:
        request_ids = []

        def fake_one_request(scale_value, req_id, payload_override=None):
            request_ids.append(req_id)
            return {
                "latency_app_s": 0.5,
                "effective_input_scale": float(scale_value),
            }

        with patch_client(self.runner, "CASE_NAME", "case"
        ), patch_client(self.runner, "REPEAT_IN_WINDOW", 0
        ), patch_client(self.runner, "REPEAT_WINDOW_SECONDS", 0.05, create=True
        ), patch_client(self.runner, "AUTO_WARMUP_REQUESTS", 2, create=True
        ), patch_client(self.runner,
            "_one_request",
            side_effect=fake_one_request,
        ):
            repeat_counts = [
                self.runner._prepare_repeat_window(1.0, "seq1", {}),
                self.runner._prepare_repeat_window(2.0, "seq2", {}),
            ]

        assert (repeat_counts) == ([1, 1])
        assert (request_ids) == ([
                "case_seq1_auto_warmup0",
                "case_seq1_auto_warmup1",
                "case_seq2_auto_warmup0",
                "case_seq2_auto_warmup1",
            ])

    def test_auto_repeat_window_continues_until_target_duration_when_requests_get_faster(self) -> None:
        measurement_req_ids = []

        def fake_one_request(scale_value, req_id, payload_override=None):
            if "_auto_warmup" not in req_id:
                measurement_req_ids.append(req_id)
            latency = 0.2
            return {
                "latency_app_s": latency,
                "effective_input_scale": float(scale_value),
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            with patch_client(self.runner, "OUT_CSV", out_csv
            ), patch_client(self.runner, "CASE_NAME", "case"
            ), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 1
            ), patch_client(self.runner, "REPEAT_IN_WINDOW", 0
            ), patch_client(self.runner, "REPEAT_WINDOW_SECONDS", 1.0, create=True
            ), patch_client(self.runner, "AUTO_WARMUP_REQUESTS", 0, create=True
            ), patch_client(self.runner, "USE_ENERGY", False
            ), patch_client(self.runner, "energy_mod", None
            ), patch_client(self.runner,
                "cpu_energy_mod",
                None,
            ), patch_client(self.runner,
                "resource_usage_mod",
                None,
            ), patch_client(self.runner,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "seq1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner,
                "_one_request",
                side_effect=fake_one_request,
            ):
                self.runner.main()
                with open(out_csv, encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

        assert (rows[0]["repeat_in_window"]) == ("5")
        assert (measurement_req_ids) == ([
                "case_seq1_r0:0",
                "case_seq1_r0:1",
                "case_seq1_r0:2",
                "case_seq1_r0:3",
                "case_seq1_r0:4",
            ])

    def test_latency_app_distribution_fields_are_written_per_window(self) -> None:
        latencies = iter([0.01, 0.02, 0.10, 0.20, 0.30])

        def fake_one_request(scale_value, req_id, payload_override=None):
            return {
                "latency_app_s": next(latencies),
                "effective_input_scale": float(scale_value),
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            Path(tmp_dir, "static_meta.json").write_text(json.dumps({
                "schema_version": 7,
                "latency_slo": {"threshold_s": 0.2, "source": "task:fill-mask"},
            }), encoding="utf-8")
            with patch_client(self.runner, "OUT_CSV", out_csv
            ), patch_client(self.runner, "CASE_NAME", "case"
            ), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 1
            ), patch_client(self.runner, "REPEAT_IN_WINDOW", 5
            ), patch.dict(
                os.environ, {"SLOW_LATENCY_THRESHOLD_S": "0.001"}
            ), patch_client(self.runner, "USE_ENERGY", False
            ), patch_client(self.runner, "energy_mod", None
            ), patch_client(self.runner,
                "cpu_energy_mod",
                None,
            ), patch_client(self.runner,
                "resource_usage_mod",
                None,
            ), patch_client(self.runner,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "seq1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner,
                "_one_request",
                side_effect=fake_one_request,
            ):
                self.runner.main()
                with open(out_csv, encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

            assert (os.path.isfile(f"{out_csv}.requests.jsonl"))
            with open(f"{out_csv}.requests.jsonl", encoding="utf-8") as requests_file:
                request_windows = [json.loads(line) for line in requests_file]

        assert (request_windows[0]["latency_app_s"]) == ([0.01, 0.02, 0.10, 0.20, 0.30])
        assert (request_windows[0]["sniff_group_id"]) == ("case_seq1_r0")
        assert (request_windows[0]["status"]) == ("ok")
        assert (rows[0]["latency_app_s"]) == ("0.126000")
        assert (rows[0]["latency_app_request_count"]) == ("5.000000")
        assert (rows[0]["latency_app_p50_s"]) == ("0.100000")
        assert (rows[0]["latency_app_p90_s"]) == ("0.300000")
        assert (rows[0]["latency_app_p95_s"]) == ("0.300000")
        assert (rows[0]["latency_app_std_s"]) == ("0.110562")
        assert (rows[0]["latency_app_cv"]) == ("0.877478")
        assert (rows[0]["latency_app_iqr_s"]) == ("0.180000")
        assert (rows[0]["latency_app_max_s"]) == ("0.300000")
        assert (rows[0]["latency_app_slow_ratio"]) == ("0.200000")
        assert (rows[0]["latency_app_tail_ratio"]) == ("3.000000")

    def test_efficiency_metrics_are_derived_without_new_measurements(self) -> None:
        metrics = client_metrics._derived_efficiency_metrics(
            gpu_mode="on",
            batch_size=2,
            latency_app_s=0.5,
            output_token_count_avg=10.0,
            gpu_energy_eff_j=0.3,
            vcpu_energy_eff_j=0.2,
            input_units_per_request=8.0,
        )

        assert (metrics["container_attributed_energy_eff_j"]) == (0.5) or round(abs((metrics["container_attributed_energy_eff_j"]) - (0.5)), 7) == 0
        assert (metrics["container_attributed_samples_per_j"]) == (4.0) or round(abs((metrics["container_attributed_samples_per_j"]) - (4.0)), 7) == 0
        assert (metrics["container_attributed_edp_app_js"]) == (0.25) or round(abs((metrics["container_attributed_edp_app_js"]) - (0.25)), 7) == 0
        assert (metrics["output_tokens_per_s_app"]) == (20.0) or round(abs((metrics["output_tokens_per_s_app"]) - (20.0)), 7) == 0
        assert (metrics["container_attributed_j_per_output_token"]) == (0.05) or round(abs((metrics["container_attributed_j_per_output_token"]) - (0.05)), 7) == 0
        assert (metrics["container_attributed_j_per_input_unit"]) == (0.0625) or round(abs((metrics["container_attributed_j_per_input_unit"]) - (0.0625)), 7) == 0

        missing_gpu = client_metrics._derived_efficiency_metrics(
            gpu_mode="on",
            batch_size=1,
            latency_app_s=0.5,
            output_token_count_avg=float("nan"),
            gpu_energy_eff_j=float("nan"),
            vcpu_energy_eff_j=0.2,
        )
        assert (math.isnan(missing_gpu["container_attributed_energy_eff_j"]))

    def test_cold_start_row_metrics_include_phases_and_first_predict(self) -> None:
        with patch_client(self.runner,
            "COLD_START_STARTED_AT",
            "2026-08-23T10:00:00.000+08:00",
        ), patch_client(self.runner,
            "COLD_START_READY_AT",
            "2026-08-23T10:00:01.000+08:00",
        ), patch_client(self.runner,
            "COLD_START_CONTAINER_LAUNCH_S",
            "0.1",
        ), patch_client(self.runner,
            "COLD_START_SERVER_SETUP_S",
            "0.2",
        ), patch_client(self.runner,
            "COLD_START_CUDA_INIT_S",
            "0.05",
        ), patch_client(self.runner,
            "COLD_START_MODEL_LOAD_S",
            "0.55",
        ), patch_client(self.runner,
            "COLD_START_READY_WAIT_S",
            "0.1",
        ), patch_client(self.runner,
            "COLD_START_S",
            "1.0",
        ), patch_client(self.runner,
            "_FIRST_PREDICT_APP_S",
            0.25,
        ):
            metrics = self.runner._cold_start_row_metrics()

        assert (metrics["cold_start_container_launch_s"]) == ("0.1")
        assert (metrics["cold_start_cuda_init_s"]) == ("0.05")
        assert (metrics["cold_start_model_load_s"]) == ("0.55")
        assert (metrics["cold_start_first_predict_app_s"]) == ("0.250000")
        assert (metrics["cold_start_s"]) == ("1.0")

    def test_process_first_predict_is_not_reported_as_docker_cold_start(self) -> None:
        with patch_client(self.runner, "RUNTIME_KIND", "process"), patch_client(
            self.runner, "_FIRST_PREDICT_APP_S", 0.25
        ), patch_client(self.runner, "COLD_START_S", "1.0"):
            metrics = self.runner._cold_start_row_metrics()
        assert set(metrics.values()) == {"nan"}

    def test_manual_repeat_in_window_skips_auto_warmup(self) -> None:
        def fake_one_request(scale_value, req_id, payload_override=None):
            assert ("_auto_warmup") not in (req_id)
            return {
                "latency_app_s": 0.5,
                "effective_input_scale": float(scale_value),
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            with patch_client(self.runner, "OUT_CSV", out_csv
            ), patch_client(self.runner, "CASE_NAME", "case"
            ), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 1
            ), patch_client(self.runner, "REPEAT_IN_WINDOW", 20
            ), patch_client(self.runner, "USE_ENERGY", False
            ), patch_client(self.runner, "energy_mod", None
            ), patch_client(self.runner,
                "cpu_energy_mod",
                None,
            ), patch_client(self.runner,
                "resource_usage_mod",
                None,
            ), patch_client(self.runner,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "seq1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner,
                "_one_request",
                side_effect=fake_one_request,
            ) as one_request:
                self.runner.main()
                with open(out_csv, encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

        assert (rows[0]["repeat_in_window"]) == ("20")
        assert (one_request.call_count) == (20)

    def test_gpu_energy_uses_one_matched_control_baseline_per_workload(self) -> None:
        sleep_calls = []

        class FakeGpuMonitor:
            apply_control_calls = 0

            def __init__(self, *args, **kwargs):
                self.idle_power_w = float("nan")

            def apply_control_baseline(self, result, samples, trace=False):
                type(self).apply_control_calls += 1
                self.idle_power_w = 10.0
                return 10.0

            def start(self):
                pass

            def stop(self):
                return (
                    SimpleNamespace(
                        idle_power_w=self.idle_power_w,
                        energy_iters=10,
                        avg_power_total_w=12.0,
                        peak_power_total_w=13.0,
                        energy_total_j=1.0,
                        avg_power_eff_w=2.0,
                        peak_power_eff_w=3.0,
                        energy_eff_j=0.2,
                    ),
                    "Test GPU",
                    "",
                    [],
                )

            def close(self):
                pass

        def fake_one_request(scale_value, req_id, payload_override=None):
            return {
                "latency_app_s": 0.5,
                "effective_input_scale": float(scale_value),
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            with patch_client(self.runner, "OUT_CSV", out_csv
            ), patch_client(self.runner, "CASE_NAME", "case"
            ), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 2
            ), patch_client(self.runner, "REPEAT_IN_WINDOW", 1
            ), patch_client(self.runner, "USE_ENERGY", True
            ), patch_client(self.runner,
                "energy_mod",
                SimpleNamespace(GPUEnergyMonitor=FakeGpuMonitor),
            ), patch_client(self.runner,
                "IDLE_COOLDOWN_SECONDS",
                2.5,
            ), patch_client(self.runner,
                "cpu_energy_mod",
                None,
            ), patch_client(self.runner,
                "resource_usage_mod",
                None,
            ), patch_client(self.runner,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "seq1", "payload": {}}],
            ), patch.object(
                client.time,
                "sleep",
                side_effect=lambda seconds: sleep_calls.append(seconds),
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner,
                "_one_request",
                side_effect=fake_one_request,
            ):
                self.runner.main()
                with open(out_csv, encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames or []
                    rows = list(reader)

        assert (FakeGpuMonitor.apply_control_calls) == (2)
        assert (sleep_calls) == ([2.5, 2.5])
        assert ("gpu_idle_power_w") in (fieldnames)
        assert ("idle_power_w") not in (fieldnames)
        assert ("gpu_energy_iters") in (fieldnames)
        assert ("energy_iters") not in (fieldnames)
        assert ("gpu_energy_eff_j") in (fieldnames)
        assert ("energy_eff_j") not in (fieldnames)
        assert (rows[0]["gpu_idle_power_w"]) == ("10.000000")
        assert (rows[0]["gpu_energy_eff_j"]) == ("0.200000")
        assert (rows[0]["gpu_avg_power_total_w"]) == ("12.000000")
        assert ("gpu_power_w") not in (rows[0])

    def test_idle_cooldown_applies_before_cpu_idle_without_gpu(self) -> None:
        sleep_calls = []

        class FakeCPUMonitor:
            apply_control_calls = 0

            def __init__(self, **kwargs):
                self.idle_power_w = float("nan")

            def apply_control_baseline(self, result, samples, trace=False):
                type(self).apply_control_calls += 1
                self.idle_power_w = 5.0
                return self.idle_power_w

            def start(self):
                pass

            def stop(self):
                return SimpleNamespace(
                    cpu_energy_iters=2,
                    cpu_idle_power_w=self.idle_power_w,
                    cpu_avg_power_total_w=6.0,
                    cpu_peak_power_total_w=7.0,
                    cpu_energy_total_j=1.0,
                    cpu_avg_power_eff_w=1.0,
                    cpu_peak_power_eff_w=2.0,
                    cpu_energy_eff_j=0.5,
                    vcpu_cpu_share=0.5,
                    vcpu_cpu_time_s=0.1,
                    vcpu_avg_power_total_w=3.0,
                    vcpu_peak_power_total_w=4.0,
                    vcpu_energy_total_j=0.3,
                    vcpu_avg_power_eff_w=0.4,
                    vcpu_peak_power_eff_w=0.5,
                    vcpu_energy_eff_j=0.2,
                ), "", []

            def close(self):
                pass

        def fake_one_request(scale_value, req_id, payload_override=None):
            return {
                "latency_app_s": 0.5,
                "effective_input_scale": float(scale_value),
            }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            with patch_client(self.runner, "OUT_CSV", out_csv
            ), patch_client(self.runner, "CASE_NAME", "case"
            ), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 2
            ), patch_client(self.runner, "REPEAT_IN_WINDOW", 1
            ), patch_client(self.runner, "USE_ENERGY", False
            ), patch_client(self.runner,
                "IDLE_COOLDOWN_SECONDS",
                2.5,
                create=True,
            ), patch_client(self.runner,
                "energy_mod",
                None,
            ), patch_client(self.runner,
                "cpu_energy_mod",
                SimpleNamespace(CPUEnergyMonitor=lambda **kwargs: FakeCPUMonitor(**kwargs)),
            ), patch_client(self.runner,
                "resource_usage_mod",
                None,
            ), patch_client(self.runner,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "seq1", "payload": {}}],
            ), patch.object(
                client.time,
                "sleep",
                side_effect=lambda seconds: sleep_calls.append(seconds),
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner,
                "_one_request",
                side_effect=fake_one_request,
            ):
                self.runner.main()

        assert (FakeCPUMonitor.apply_control_calls) == (2)
        assert (sleep_calls) == ([2.5, 2.5])

    def test_client_entrypoint_prints_friendly_energy_abort_without_traceback(self) -> None:
        stderr = io.StringIO()
        with patch_client(self.runner,
            "main",
            side_effect=client.EnergyAbort("gpu_idle_power_w failed"),
        ), pytest.raises(SystemExit) as raised, redirect_stderr(stderr):
            self.runner.run_cli()

        assert (raised.value.code) == (1)
        assert ("[energy][ERROR] gpu_idle_power_w failed") in (stderr.getvalue())
        assert ("Traceback") not in (stderr.getvalue())

    def test_one_request_converts_http_timeout_to_case_abort(self) -> None:
        with patch_client(self.runner,
            "REQUEST_TIMEOUT_SECONDS",
            0.25,
        ), patch.object(
            client.requests,
            "post",
            side_effect=client.requests.exceptions.ReadTimeout("slow inference"),
        ), pytest.raises(client.RequestTimeoutAbort) as raised:
            self.runner._one_request(
                10.0,
                req_id="case_dur10_auto_warmup0",
                payload_override={},
            )

        message = str(raised.value)
        assert ("inactivity timeout: 0.25s") in (message)
        assert ("input_scale=10") in (message)
        assert ("req_id=case_dur10_auto_warmup0") in (message)
        assert (raised.value.input_scale) == (10.0)
        assert (raised.value.request_id) == ("case_dur10_auto_warmup0")
        assert (raised.value.timeout_s) == (0.25)

    def test_measurement_timeout_escapes_row_level_error_handling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir, patch_client(self.runner,
            "OUT_CSV",
            f"{tmp_dir}/result.csv",
        ), patch_client(self.runner,
            "CASE_NAME",
            "case",
        ), patch_client(self.runner,
            "WARMUP",
            0,
        ), patch_client(self.runner,
            "REPEAT",
            1,
        ), patch_client(self.runner,
            "REPEAT_IN_WINDOW",
            2,
        ), patch_client(self.runner,
            "USE_ENERGY",
            False,
        ), patch_client(self.runner,
            "USE_MIPS",
            False,
        ), patch_client(self.runner,
            "energy_mod",
            None,
        ), patch_client(self.runner,
            "cpu_energy_mod",
            None,
        ), patch_client(self.runner,
            "resource_usage_mod",
            None,
        ), patch_client(self.runner,
            "input_scale_entries",
            [{"input_scale": 10.0, "scale_label": "dur10", "payload": {}}],
        ), patch.object(
            client.requests,
            "get",
            return_value=SimpleNamespace(status_code=200, text="ok"),
        ), patch_client(self.runner,
            "_one_request",
            side_effect=[{"latency_app_s": 0.25, "effective_input_scale": 10.0}, client.RequestTimeoutAbort("slow inference")],
        ):
            with pytest.raises(client.RequestTimeoutAbort):
                self.runner.main()
            with open(f"{tmp_dir}/result.csv.requests.jsonl", encoding="utf-8") as f:
                window = json.loads(f.readline())
            assert (window["status"]) == ("error")
            assert (window["latency_app_s"]) == ([0.25])
            assert (window["failed_request_id"]) == ("case_dur10_r0:1")

    def test_client_entrypoint_uses_dedicated_timeout_exit_code(self) -> None:
        stderr = io.StringIO()
        with patch_client(self.runner,
            "main",
            side_effect=client.RequestTimeoutAbort("slow inference"),
        ), pytest.raises(SystemExit) as raised, redirect_stderr(stderr):
            self.runner.run_cli()

        assert (raised.value.code) == (client.CLIENT_REQUEST_TIMEOUT_EXIT_CODE)
        assert ("[case][ERROR] slow inference") in (stderr.getvalue())
        assert ("Traceback") not in (stderr.getvalue())

    def test_client_entrypoint_persists_structured_timeout_context(self) -> None:
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp_dir:
            sidecar_path = os.path.join(tmp_dir, "client-error.json")
            exc = client.RequestTimeoutAbort(
                "slow inference",
                input_scale=30.0,
                request_id="case_dur30_auto_warmup0",
                timeout_s=300.0,
            )
            with patch_client(self.runner,
                "CLIENT_ERROR_PATH",
                sidecar_path,
            ), patch_client(self.runner,
                "main",
                side_effect=exc,
            ), pytest.raises(SystemExit), redirect_stderr(stderr):
                self.runner.run_cli()

            with open(sidecar_path, encoding="utf-8") as f:
                payload = json.load(f)

        assert (payload["error_type"]) == ("client_request_timeout")
        assert (payload["input_scale"]) == (30.0)
        assert (payload["request_timeout_s"]) == (300.0)
        assert (payload["request_phase"]) == ("auto_repeat_window_warmup")
        assert (payload["request_index_in_window"]) == (0)

    def test_client_entrypoint_cleans_failed_timeout_sidecar_temporary(self) -> None:
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp_dir:
            sidecar_path = os.path.join(tmp_dir, "client-error.json")
            exc = client.RequestTimeoutAbort(
                "slow inference",
                input_scale=30.0,
                request_id="case_dur30_auto_warmup0",
                timeout_s=300.0,
            )
            real_replace = os.replace

            def fail_sidecar_replace(source, destination):
                if os.fspath(destination) == sidecar_path:
                    raise OSError("disk error")
                return real_replace(source, destination)

            with patch_client(self.runner, "CLIENT_ERROR_PATH", sidecar_path), patch_client(
                    self.runner, "main", side_effect=exc), patch(
                    "acprof.artifacts.os.replace", side_effect=fail_sidecar_replace), pytest.raises(
                    SystemExit), redirect_stderr(stderr):
                self.runner.run_cli()

            assert not os.path.exists(sidecar_path)
            assert os.listdir(tmp_dir) == []
        assert "failed to persist structured timeout context" in stderr.getvalue()

    def test_sniff_group_id_is_hidden_from_csv_but_kept_for_packet_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            with patch_client(self.runner, "OUT_CSV", out_csv
            ), patch_client(self.runner, "CASE_NAME", "case"
            ), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 1
            ), patch_client(self.runner, "REPEAT_IN_WINDOW", 1
            ), patch_client(self.runner, "USE_ENERGY", False
            ), patch_client(self.runner, "energy_mod", None
            ), patch_client(self.runner,
                "cpu_energy_mod",
                None,
            ), patch_client(self.runner,
                "resource_usage_mod",
                None,
            ), patch_client(self.runner,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "seq1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner,
                "_one_request",
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0},
            ):
                self.runner.main()
                with open(out_csv, encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    fieldnames = reader.fieldnames or []

            with open(f"{out_csv}.sniff_groups.jsonl", encoding="utf-8") as f:
                sidecar_rows = [json.loads(line) for line in f if line.strip()]

        assert ("sniff_group_id") not in (fieldnames)
        assert ("sniff_group_id") not in (rows[0])
        assert (sidecar_rows) == ([{"sniff_group_id": "case_seq1_r0"}])

    def test_negative_effective_metrics_are_reported_per_field(self) -> None:
        warnings = client_metrics._eff_negative_warnings(
            avg_power_eff_w=-0.1,
            peak_power_eff_w=0.0,
            energy_eff_j=-0.001,
        )

        assert (warnings) == (["gpu_avg_power_eff_w<0", "gpu_energy_eff_j<0"])

    def test_cpu_vcpu_negative_effective_metrics_keep_full_field_names(self) -> None:
        warnings = client_metrics._named_negative_warnings({
            "cpu_energy_eff_j": -0.001,
            "vcpu_avg_power_eff_w": -0.1,
            "vcpu_energy_total_j": 1.0,
        })

        assert (warnings) == (["cpu_energy_eff_j<0", "vcpu_avg_power_eff_w<0"])

    def test_cpu_monitor_unavailable_keeps_successful_row_ok(self) -> None:
        class FakeUnavailableCPUMonitor:
            def apply_control_baseline(self, result, samples, trace=False):
                return float("nan")

            def start(self):
                return None

            def stop(self):
                return energy_cpu._nan_result(), "RAPL unavailable", []

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            with patch_client(self.runner, "OUT_CSV", out_csv
            ), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 1
            ), patch_client(self.runner, "REPEAT_IN_WINDOW", 1
            ), patch_client(self.runner, "USE_ENERGY", False
            ), patch_client(self.runner, "energy_mod", None
            ), patch_client(self.runner,
                "cpu_energy_mod",
                SimpleNamespace(CPUEnergyMonitor=lambda **kwargs: FakeUnavailableCPUMonitor()),
            ), patch_client(self.runner,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner,
                "_one_request",
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0},
            ):
                self.runner.main()
                with open(out_csv, encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

        assert (rows[0]["status"]) == ("ok")
        assert (rows[0]["error"]) == ("")
        assert (rows[0]["cpu_energy_total_j"]) == ("nan")

    def test_idle_debug_writes_csv_fields_and_diagnostic_jsonl(self) -> None:
        class FakeCPUMonitor:
            idle_values = iter([5.0, 5.5])
            trace_intervals = []

            def __init__(self, **kwargs):
                self.idle_power_w = float("nan")
                self.idle_trace = {}

            def apply_control_baseline(self, result, samples, trace=False):
                type(self).trace_intervals.append(trace)
                self.idle_power_w = next(type(self).idle_values)
                self.idle_trace = {
                    "idle_trace_schema": "cpu_rapl_idle_v1",
                    "actual_idle_duration_s": 3.0,
                    "rapl_trace": {
                        "interval_s": 0.1,
                        "power_windows": [{"t0_s": 0.0, "t1_s": 0.1, "power_w": 6.0}],
                    },
                    "idle_proc_cpu_top": [{"pid": 123, "comm": "python", "cpu_time_ms": 10.0}],
                    "idle_container_cpu_delta_s": 0.01,
                }
                return self.idle_power_w

            def start(self):
                return None

            def stop(self):
                return SimpleNamespace(
                    cpu_energy_iters=2,
                    cpu_idle_power_w=self.idle_power_w,
                    cpu_avg_power_total_w=6.0,
                    cpu_peak_power_total_w=7.0,
                    cpu_energy_total_j=1.0,
                    cpu_avg_power_eff_w=1.0,
                    cpu_peak_power_eff_w=2.0,
                    cpu_energy_eff_j=0.5,
                    vcpu_cpu_share=0.5,
                    vcpu_cpu_time_s=0.1,
                    vcpu_avg_power_total_w=3.0,
                    vcpu_peak_power_total_w=4.0,
                    vcpu_energy_total_j=0.3,
                    vcpu_avg_power_eff_w=0.4,
                    vcpu_peak_power_eff_w=0.5,
                    vcpu_energy_eff_j=0.2,
                ), "", []

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            diag_path = os.path.join(
                tmp_dir,
                "debug_idle_diag",
                "result.csv.idle_diag.jsonl",
            )
            with patch_client(self.runner, "OUT_CSV", out_csv
            ), patch_client(self.runner, "CASE_NAME", "case"
            ), patch_client(self.runner, "CPU_CORES", "1"
            ), patch_client(self.runner, "MEM_CAP_GB", "2"
            ), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 2
            ), patch_client(self.runner, "REPEAT_IN_WINDOW", 1
            ), patch_client(self.runner, "USE_ENERGY", False
            ), patch_client(self.runner, "IDLE_DEBUG", True, create=True
            ), patch_client(self.runner, "IDLE_DIAG_PATH", "", create=True
            ), patch_client(self.runner, "energy_mod", None
            ), patch_client(self.runner,
                "cpu_energy_mod",
                SimpleNamespace(CPUEnergyMonitor=lambda **kwargs: FakeCPUMonitor(**kwargs)),
            ), patch_client(self.runner,
                "resource_usage_mod",
                None,
            ), patch_client(self.runner,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "seq1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner,
                "_one_request",
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0},
            ), patch_client(self.runner,
                "collect_idle_debug_snapshot",
                return_value={
                    "snapshot_scope": "after_idle",
                    "loadavg": [0.1, 0.2, 0.3],
                    "top_cpu_processes": [{"pid": 123, "comm": "python", "cpu_pct": 4.5}],
                    "docker_containers": [{"name": "case"}],
                    "docker_stats": [{"name": "case", "cpu_perc": "0.1%"}],
                },
                create=True,
            ), patch_client(self.runner,
                "_now_iso",
                side_effect=[
                    "2026-05-02T10:00:00+08:00",
                    "2026-05-02T10:00:01+08:00",
                ],
                create=True,
            ):
                self.runner.main()
                with open(out_csv, encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))
                with open(diag_path, encoding="utf-8") as f:
                    diag_rows = [json.loads(line) for line in f if line.strip()]

        assert (rows[0]["cpu_idle_measured_at"]) == ("2026-05-02T10:00:00+08:00")
        assert (rows[0]["gpu_idle_measured_at"]) == ("nan")
        assert (rows[0]["gpu_idle_rel_range_so_far"]) == ("nan")
        assert (rows[0]["cpu_idle_rel_range_so_far"]) == ("0.000000")
        assert (rows[1]["cpu_idle_measured_at"]) == ("2026-05-02T10:00:01+08:00")
        assert (rows[1]["gpu_idle_measured_at"]) == ("nan")
        assert (rows[1]["gpu_idle_rel_range_so_far"]) == ("nan")
        assert (rows[1]["cpu_idle_rel_range_so_far"]) == ("0.095238")
        assert (FakeCPUMonitor.trace_intervals) == ([True, True])
        assert (len(diag_rows)) == (2)
        assert (diag_rows[1]["sniff_group_id"]) == ("case_seq1_r1")
        assert (diag_rows[1]["cpu_idle_measured_at"]) == ("2026-05-02T10:00:01+08:00")
        assert (diag_rows[1]["idle_trace_schema"]) == ("cpu_rapl_idle_v1")
        assert (diag_rows[1]["actual_idle_duration_s"]) == (3.0)
        assert (diag_rows[1]["rapl_trace"]["interval_s"]) == (0.1)
        assert (diag_rows[1]["idle_proc_cpu_top"][0]["comm"]) == ("python")
        assert (diag_rows[1]["idle_container_cpu_delta_s"]) == (0.01)
        assert (diag_rows[1]["cpu_idle_valid_count"]) == (2)
        assert (diag_rows[1]["cpu_idle_mean_w"]) == (5.25) or round(abs((diag_rows[1]["cpu_idle_mean_w"]) - (5.25)), 7) == 0
        assert (diag_rows[1]["snapshot_scope"]) == ("after_idle")
        assert (diag_rows[1]["loadavg"]) == ([0.1, 0.2, 0.3])
        assert (diag_rows[1]["top_cpu_processes"][0]["comm"]) == ("python")
        assert (diag_rows[1]["docker_containers"][0]["name"]) == ("case")
        assert (diag_rows[1]["docker_stats"][0]["cpu_perc"]) == ("0.1%")

    def test_idle_debug_writes_gpu_idle_fields_and_diagnostics(self) -> None:
        class FakeGpuMonitor:
            idle_values = iter([10.0, 11.0])
            trace_args = []

            def __init__(self, **kwargs):
                self.idle_power_w = float("nan")
                self.idle_trace = {}

            def apply_control_baseline(self, result, samples, trace=False):
                type(self).trace_args.append(trace)
                self.idle_power_w = next(type(self).idle_values)
                self.idle_trace = {
                    "gpu_idle_trace_schema": "nvml_gpu_idle_v1",
                    "gpu_idle_sample_count": 2,
                    "gpu_idle_power_samples": [{"t_s": 0.0, "power_w": self.idle_power_w}],
                }
                return self.idle_power_w

            def start(self):
                return None

            def stop(self):
                if self.idle_trace:
                    self.idle_trace['gpu_idle_baseline_reselected_due_to_workload'] = True
                return SimpleNamespace(
                    energy_iters=2,
                    idle_power_w=self.idle_power_w,
                    avg_power_total_w=self.idle_power_w + 1.0,
                    peak_power_total_w=self.idle_power_w + 2.0,
                    energy_total_j=1.0,
                    avg_power_eff_w=1.0,
                    peak_power_eff_w=2.0,
                    energy_eff_j=0.5,
                ), "Fake GPU", "", []

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            diag_path = f"{out_csv}.idle_diag.jsonl"
            with patch_client(self.runner, "OUT_CSV", out_csv
            ), patch_client(self.runner, "CASE_NAME", "case"
            ), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 2
            ), patch_client(self.runner, "REPEAT_IN_WINDOW", 1
            ), patch_client(self.runner, "USE_ENERGY", True
            ), patch_client(self.runner, "IDLE_DEBUG", True, create=True
            ), patch_client(self.runner, "IDLE_DIAG_PATH", diag_path, create=True
            ), patch_client(self.runner,
                "energy_mod",
                SimpleNamespace(GPUEnergyMonitor=lambda **kwargs: FakeGpuMonitor(**kwargs)),
            ), patch_client(self.runner,
                "cpu_energy_mod",
                None,
            ), patch_client(self.runner,
                "resource_usage_mod",
                None,
            ), patch_client(self.runner,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "seq1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner,
                "_one_request",
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0},
            ), patch_client(self.runner,
                "collect_gpu_idle_debug_snapshot",
                return_value={
                    "gpu_snapshot_scope": "after_gpu_idle",
                    "nvidia_smi_gpu": {"pstate": "P0", "clocks_sm_mhz": 1200.0},
                    "nvidia_smi_pmon": [{"pid": 123, "type": "G", "command": "Xorg"}],
                },
                create=True,
            ), patch_client(self.runner,
                "_now_iso",
                side_effect=[
                    "2026-05-02T10:00:00+08:00",
                    "2026-05-02T10:00:01+08:00",
                ],
                create=True,
            ):
                self.runner.main()
                with open(out_csv, encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))
                with open(diag_path, encoding="utf-8") as f:
                    diag_rows = [json.loads(line) for line in f if line.strip()]

        assert (FakeGpuMonitor.trace_args) == ([True, True])
        assert (rows[0]["gpu_idle_measured_at"]) == ("2026-05-02T10:00:00+08:00")
        assert (rows[0]["gpu_idle_rel_range_so_far"]) == ("0.000000")
        assert (rows[1]["gpu_idle_measured_at"]) == ("2026-05-02T10:00:01+08:00")
        assert (rows[1]["gpu_idle_rel_range_so_far"]) == ("0.095238")
        assert (rows[1]["cpu_idle_measured_at"]) == ("nan")
        assert (rows[1]["cpu_idle_rel_range_so_far"]) == ("nan")
        assert (diag_rows[1]["gpu_idle_trace_schema"]) == ("nvml_gpu_idle_v1")
        assert diag_rows[1]['gpu_idle_baseline_reselected_due_to_workload'] is True
        assert (diag_rows[1]["cpu_idle_measured_at"]) == ("nan")
        assert (diag_rows[1]["gpu_idle_sample_count"]) == (2)
        assert (diag_rows[1]["gpu_idle_valid_count"]) == (2)
        assert (diag_rows[1]["gpu_idle_mean_w"]) == (10.5)
        assert (diag_rows[1]["gpu_snapshot_scope"]) == ("after_gpu_idle")
        assert (diag_rows[1]["nvidia_smi_gpu"]["pstate"]) == ("P0")
        assert (diag_rows[1]["nvidia_smi_pmon"][0]["command"]) == ("Xorg")

    def test_idle_debug_disabled_fills_nan_and_does_not_write_diag_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_csv = f"{tmp_dir}/result.csv"
            diag_path = f"{out_csv}.idle_diag.jsonl"
            with patch_client(self.runner, "OUT_CSV", out_csv
            ), patch_client(self.runner, "WARMUP", 0
            ), patch_client(self.runner, "REPEAT", 1
            ), patch_client(self.runner, "REPEAT_IN_WINDOW", 1
            ), patch_client(self.runner, "USE_ENERGY", False
            ), patch_client(self.runner, "IDLE_DEBUG", False, create=True
            ), patch_client(self.runner, "IDLE_DIAG_PATH", diag_path, create=True
            ), patch_client(self.runner, "energy_mod", None
            ), patch_client(self.runner,
                "cpu_energy_mod",
                None,
            ), patch_client(self.runner,
                "resource_usage_mod",
                None,
            ), patch_client(self.runner,
                "input_scale_entries",
                [{"input_scale": 1.0, "scale_label": "1", "payload": {}}],
            ), patch.object(
                client.requests,
                "get",
                return_value=SimpleNamespace(status_code=200, text="ok"),
            ), patch_client(self.runner,
                "_one_request",
                return_value={"latency_app_s": 0.5, "effective_input_scale": 1.0},
            ):
                self.runner.main()
                with open(out_csv, encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))

        assert (rows[0]["cpu_idle_measured_at"]) == ("nan")
        assert (rows[0]["gpu_idle_measured_at"]) == ("nan")
        assert (rows[0]["gpu_idle_rel_range_so_far"]) == ("nan")
        assert (rows[0]["cpu_idle_rel_range_so_far"]) == ("nan")
        assert not (os.path.exists(diag_path))
