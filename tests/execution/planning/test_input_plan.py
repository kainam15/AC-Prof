import json
import os
import re
import tempfile
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.host import docker_runtime, input_plan, model_schema
from acprof.host.detect import TaskInfo
from acprof.host.images import runtime as runtime_images
from acprof.host.input_plan import families, materialize, planner, probe


class TestInputPlan:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        from tests.support.platform_fixtures import native_policy
        native_policy(self._request)
        conditions = patch("acprof.host.orchestrator.record_case_conditions")
        conditions.start()
        self._request.addfinalizer(partial(conditions.stop))
        self.output_dir = str(tmp_path)

    def test_manual_nlp_scales_write_effective_scale_plan(self) -> None:
        task_info = TaskInfo(
            model_id="google-bert/bert-base-uncased",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="hub_api",
        )
        session = docker_runtime.RunningContainer(
            name="probe_google-bert--bert-base-uncased_1c_4g_off",
            base_url="http://127.0.0.1:8106",
            host_port=8106,
            cold_start_s=1.0,
        )

        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.host.input_plan.families.start_probe_session",
            return_value=session,
        ), patch(
            "acprof.host.input_plan.families.stop_service_session",
        ), patch(
            "acprof.host.input_plan.families.post_probe_payload",
            return_value={
                "effective_input_scale": 254.0,
                "truncated_by_limit": False,
                "reason": "within_model_limit",
                "payload": {"text": "hello [MASK]", "params": {}},
            },
        ):
            planned = input_plan.plan_input_scales(
                task_info=task_info,
                image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off"],
                batch_size=1,
                output_dir=tmp,
                input_scales="64",
            )

            assert (planned.scales) == ([254.0])
            assert (planned.source) == ("manual")
            assert (planned.plan_file) is not None
            assert planned.plan_file is not None
            assert (os.path.exists(planned.plan_file))
            with open(planned.plan_file, encoding="utf-8") as f:
                plan = json.load(f)

        assert (plan["entries"][0]["input_scale"]) == (254.0)
        assert (plan["entries"][0]["payload"]) == ({"text": "hello [MASK]", "params": {}})

    @pytest.mark.parametrize('task_family', ['cv', 'timeseries'])
    def test_manual_non_nlp_scales_write_reusable_payload_plan(self, task_family) -> None:
        class FakeWorkloadGenerator:
            def generate(self, scale: float) -> dict:
                return {"value": float(scale)}

            def effective_input_scale(
                self,
                scale: float,
                payload: dict | None = None,
            ) -> float:
                return float(scale)

            def scale_label(self, scale: float) -> str:
                return f"scale{scale:g}"

            def max_input_scale(self) -> float:
                return 2048.0

        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.workloads.get_generator",
            return_value=FakeWorkloadGenerator(),
        ), patch.object(families, "start_probe_session", return_value=SimpleNamespace(name="probe")), patch.object(
            families, "stop_service_session"
        ), patch.object(families, "request_scale_meta", return_value={
            "max_effective_input_scale": 512, "input_scale_type": "context_length",
            "reason": "test model context limit",
        }
        ):
            task_info = TaskInfo(
                model_id=f"test/{task_family}",
                pipeline_tag="image-classification" if task_family == "cv" else "time-series-forecasting",
                task_family=task_family,
                runtime_backend="transformers_pipeline" if task_family == "cv" else "chronos",
                library_name="transformers",
                model_revision="main",
                detection_method="unit",
            )

            planned = input_plan.plan_input_scales(
                task_info=task_info,
                image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off"],
                batch_size=1,
                output_dir=tmp,
                input_scales="1,2",
            )

            assert (planned.scales) == ([1.0, 2.0])
            assert (planned.source) == ("manual")
            assert (planned.plan_file) == (os.path.join(tmp, "input_scale_plan.json"))
            assert planned.plan_file is not None
            with open(planned.plan_file, encoding="utf-8") as f:
                plan = json.load(f)

            assert (plan["entries"]) == ([
                    {
                        "input_scale": 1.0,
                        "scale_label": "scale1",
                        "input_metadata": {},
                        "payload": {"value": 1.0},
                    },
                    {
                        "input_scale": 2.0,
                        "scale_label": "scale2",
                        "input_metadata": {},
                        "payload": {"value": 2.0},
                    },
                ])
            assert (plan["schema_version"]) == (2)
            assert re.search(r"^[0-9a-f]{64}$", planned.plan_sha256)

    def test_auto_audio_scales_use_workload_manifest_defaults(self) -> None:
        class FakeWorkloadGenerator:
            def default_input_scales(self) -> list[float]:
                return [1.0, 2.0, 5.0, 10.0, 20.0, 30.0]

        task_info = TaskInfo(
            model_id="test/audio",
            pipeline_tag="automatic-speech-recognition",
            task_family="audio",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="unit",
        )

        expected = input_plan.PlannedInputScales(
            scales=[1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
            source="workload_spec",
            plan_file="/tmp/input_scale_plan.json",
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.workloads.get_generator",
            return_value=FakeWorkloadGenerator(),
        ), patch.object(
            planner,
            "plan_audio_scales",
            return_value=expected,
        ) as plan_audio:
            planned = input_plan.plan_input_scales(
                task_info=task_info,
                image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off"],
                batch_size=1,
                output_dir=tmp,
            )

        assert (planned) is (expected)
        assert (plan_audio.call_args.kwargs["scales"]) == ([1.0, 2.0, 5.0, 10.0, 20.0, 30.0])

    def test_audio_scale_meta_records_fixed_frontend_and_decoder_limit(self) -> None:
        response = {
            "input_scale_type": "duration_s",
            "required_sampling_rate": 16000,
            "max_short_form_duration_s": 30,
            "model_input_num_samples": 480000,
            "model_input_frames": 3000,
            "short_form_fixed_padding": True,
            "fixed_frontend_num_samples": 480000,
            "fixed_frontend_num_frames": 3000,
            "frontend_feature_bins": 128,
            "encoder_positions": 1500,
            "decoder_output_token_limit": 448,
            "model_type": "whisper",
            "reason": "fixed frontend; 448 is an output limit",
        }
        session = docker_runtime.RunningContainer(
            name="probe",
            base_url="http://127.0.0.1:1",
            host_port=1,
            cold_start_s=0.0,
        )
        with patch.object(
            probe,
            "request_scale_meta",
            return_value=response,
        ):
            metadata = probe.request_audio_scale_meta(session, {})

        assert (metadata["short_form_fixed_padding"])
        assert (metadata["fixed_frontend_num_samples"]) == (480000)
        assert (metadata["fixed_frontend_num_frames"]) == (3000)
        assert (metadata["frontend_feature_bins"]) == (128)
        assert (metadata["decoder_output_token_limit"]) == (448)

    def test_audio_output_schema_describes_effective_scale_and_nullable_tokens(self) -> None:
        task_info = TaskInfo(
            model_id="openai/whisper-large-v3",
            pipeline_tag="automatic-speech-recognition",
            task_family="audio",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="unit",
        )

        _, output_format = model_schema._model_io_formats(task_info)
        properties = output_format["json_schema"]["properties"]

        assert (properties["effective_input_scale"]) == ({"type": "number"})
        assert (properties["output_token_count"]) == ({"type": ["integer", "null"]})

    def test_workload_spec_is_rejected_for_unimplemented_families(self) -> None:
        task_info = TaskInfo(
            model_id="test/forecast",
            pipeline_tag="time-series-forecasting",
            task_family="timeseries",
            runtime_backend="chronos",
            library_name="chronos",
            model_revision="main",
            detection_method="unit",
        )
        with tempfile.TemporaryDirectory() as tmp, pytest.raises(ValueError, match="workload-spec is not declared for time-series-forecasting"):
            input_plan.plan_input_scales(
                task_info=task_info,
                image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off"],
                batch_size=1,
                output_dir=tmp,
                workload_spec_path="/tmp/not-used.json",
            )

    def test_audio_manifest_limit_is_checked_before_starting_probe(self) -> None:
        task_info = TaskInfo(
            model_id="openai/whisper-large-v3",
            pipeline_tag="automatic-speech-recognition",
            task_family="audio",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="unit",
        )
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            families,
            "start_probe_session",
        ) as start_probe, pytest.raises(ValueError, match="long-form workload"):
            families.plan_audio_scales(
                task_info=task_info,
                image_info=runtime_images.ImageInfo(tag="acprof-test:latest"),
                cpu_list=[1],
                mem_list=[4],
                gpu_list=["off"],
                scales=[31.0],
                batch_size=1,
                output_dir=tmp,
                source="manual",
                workload_spec_path=None,
            )
        start_probe.assert_not_called()

    def test_materialized_plan_v2_records_workload_and_input_metadata(self) -> None:
        class FakeWorkloadGenerator:
            def generate(self, scale: float) -> dict:
                return {"value": float(scale), "params": {"mode": "fixed"}}

            def effective_input_scale(self, scale: float, payload=None) -> float:
                return float(scale)

            def scale_label(self, scale: float) -> str:
                return f"scale{scale:g}"

            def input_metadata(self, scale: float, payload=None) -> dict:
                return {"input_num_samples": int(scale * 100)}

            def plan_metadata(self) -> dict:
                return {"workload_id": "fixture-v1"}

        task_info = TaskInfo(
            model_id="test/audio",
            pipeline_tag="automatic-speech-recognition",
            task_family="audio",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="unit",
        )
        constraints = {"max_short_form_duration_s": 30}
        with tempfile.TemporaryDirectory() as tmp, patch(
            "acprof.workloads.get_generator",
            return_value=FakeWorkloadGenerator(),
        ):
            planned = materialize.materialize_scale_plan(
                task_info=task_info,
                scales=[1.0, 2.0],
                batch_size=1,
                output_dir=tmp,
                source="unit",
                model_constraints=constraints,
            )
            assert planned.plan_file is not None
            with open(planned.plan_file, encoding="utf-8") as plan_file:
                plan = json.load(plan_file)

        assert (plan["schema_version"]) == (2)
        assert (plan["workload"]) == ({"workload_id": "fixture-v1"})
        assert (plan["model_constraints"]) == (constraints)
        assert (plan["entries"][0]["input_metadata"]["input_num_samples"]) == (100)
        assert (planned.workload["model_constraints"]) == (constraints)
        assert re.search(r"^[0-9a-f]{64}$", planned.plan_sha256)
