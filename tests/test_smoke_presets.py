"""Smoke uses resolved workloads and presets preserve user constraints."""
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from acprof.experiment import RunConfig, RunConfigError, build_run_command
from acprof.host import input_plan
from acprof.host.detect import TaskInfo
from acprof.host.runtime_images import ImageInfo
from acprof.run_args import build_parser


class SmokePresetTests(unittest.TestCase):
    def test_smoke_defers_single_scale_until_task_resolution(self):
        config = RunConfig.smoke("openai/whisper-tiny")
        self.assertEqual(config.input_scales, "")
        command = build_run_command(config, project_dir=Path.cwd())
        self.assertEqual(command[command.index("--input-scale-policy") + 1], "minimal")
        options = build_parser().parse_args(command[command.index("--model"):])
        self.assertEqual(options.input_scale_policy, "minimal")

    def test_task_minimum_uses_workload_tokens_seconds_and_image_scale(self):
        for task, family, expected, size_type in (
            ("text-classification", "nlp", 64.0, "seq_length"),
            ("automatic-speech-recognition", "audio", 1.0, "duration_s"),
            ("image-classification", "cv", 0.1, "resolution_scale"),
        ):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                info = TaskInfo("example/model", task, family,
                    "transformers_pipeline" if family == "nlp" else "transformers_model", "transformers", "fixed", "unit")
                with patch.object(input_plan, "_start_probe_session", return_value=SimpleNamespace(name="probe")), patch.object(
                    input_plan, "stop_container_session"
                ), patch.object(input_plan, "_request_scale_meta", return_value={
                    "input_scale_type": "duration_s", "required_sampling_rate": 16000,
                    "source_sampling_rate": 16000, "short_form_fixed_padding": True,
                    "max_short_form_duration_s": 30.0, "model_type": "whisper", "reason": "Whisper short-form audio",
                }), patch.object(input_plan, "_post_probe_payload", side_effect=lambda session, payload, description: {
                    "effective_input_scale": expected, "truncated_by_limit": False, "reason": "valid fixture input", "payload": payload,
                }):
                    plan = input_plan.plan_input_scales(info, ImageInfo("unused"), [1], [4], ["off"],
                        1, directory, input_scale_policy="minimal")
                self.assertEqual(plan.scales, [expected])
                self.assertEqual(plan.source, "minimal")
                self.assertEqual(input_plan.input_plan_summary(info, plan)["scale_type"], size_type)
                payload = json.loads(Path(plan.plan_file).read_text())
                self.assertEqual([entry["input_scale"] for entry in payload["entries"]], [expected])
                self.assertEqual(hashlib.sha256(Path(plan.plan_file).read_bytes()).hexdigest(), plan.plan_sha256)

    def test_minimal_plan_freeze_and_resume_keep_source_scales_hash_and_static_metadata(self):
        from acprof.host import run_state
        from acprof.host.static_metadata import StaticMeta, enrich_static_meta_from_input_plan
        info = TaskInfo("example/model", "image-classification", "cv", "transformers_model", "transformers", "fixed", "unit")
        image = ImageInfo("sha256:" + "a" * 64)
        with tempfile.TemporaryDirectory() as temporary, patch.object(run_state, "MEASUREMENT_LOCK_ROOT", Path(temporary)), patch.object(
            run_state, "host_identity", return_value={"test": "isolated"}
        ), patch("acprof.host.runtime_images.require_image_identity"):
            directory = Path(temporary) / "result"
            options = {"input_scale_policy": "minimal"}
            state = run_state.RunState(directory, options, resume=False, project_dir=str(Path.cwd()))
            try:
                plan = input_plan.plan_input_scales(info, image, [1], [4], ["off"], 1, str(directory), input_scale_policy="minimal")
                original = Path(plan.plan_file).read_bytes()
                state.bind_runtime(info, image, plan, "", "")
            finally:
                state.close()
            restored = run_state.RunState(directory, options, resume=True, project_dir=str(Path.cwd()))
            try:
                _, _, resumed, _, _ = restored.restore_runtime()
                self.assertEqual(resumed, plan)
                self.assertEqual(restored.data["runtime"]["planned"]["source"], "minimal")
                self.assertEqual(Path(resumed.plan_file).read_bytes(), original)
                self.assertEqual(input_plan.input_plan_summary(info, resumed), {"scales": [0.1], "scale_type": "resolution_scale"})
                meta = StaticMeta(model_name=info.model_id, model_revision="fixed", task_family="cv",
                    pipeline_tag=info.pipeline_tag, runtime_backend=info.runtime_backend, image_tag=image.tag,
                    batch_size=1, input_scale_type="resolution_scale", run_command="fixture", model_download_url="fixture",
                    gpu="none", gpu_mem_total_bytes=None, model_cache_bytes=0, docker_image_bytes=0,
                    environment="test", cpu_power_source="unavailable", vcpu_power_method="unavailable",
                    cpu_governor="unknown", cpu_boost="unknown")
                static = enrich_static_meta_from_input_plan(meta, resumed)
                self.assertEqual(static.input_scale_plan_sha256, hashlib.sha256(original).hexdigest())
                self.assertEqual(static.workload, plan.workload)
            finally:
                restored.close()

    def test_text_minimum_reuses_strict_token_planner(self):
        info = TaskInfo("example/model", "text-classification", "nlp", "transformers_pipeline", "transformers", "fixed", "unit")
        with tempfile.TemporaryDirectory() as directory, patch.object(input_plan, "_plan_manual_nlp_scales",
            return_value=input_plan.PlannedInputScales([64], "manual")) as planner:
            plan = input_plan.plan_input_scales(info, ImageInfo("unused"), [1], [4], ["off"],
                1, directory, input_scale_policy="minimal")
        self.assertEqual(planner.call_args.kwargs["scales"], [64.0])
        self.assertEqual(plan.source, "minimal")

    def test_explicit_scales_are_never_replaced_by_minimal_policy(self):
        info = TaskInfo("example/model", "text-classification", "nlp", "transformers_pipeline", "transformers", "fixed", "unit")
        with tempfile.TemporaryDirectory() as directory, patch.object(input_plan, "_plan_manual_nlp_scales",
            side_effect=RuntimeError("manual scale exceeds model limit")) as planner:
            with self.assertRaisesRegex(RuntimeError, "manual scale"):
                input_plan.plan_input_scales(info, ImageInfo("unused"), [1], [4], ["off"],
                    1, directory, input_scales="8192", input_scale_policy="minimal")
        self.assertEqual(planner.call_args.kwargs["scales"], [8192.0])

    def test_presets_keep_download_budget_storage_and_model_selection(self):
        current = replace(RunConfig(model="demo/model"), max_download="5GB", download_mode="official",
            model_store="/custom/cache", model_store_max="20GB", output_dir="/custom/output",
            task="image-classification", task_family="cv", backend="transformers_model", model_spec="custom.json", notify="wecom")
        for preset in ("smoke", "main"):
            changed = current.with_preset(preset)
            for name in ("max_download", "download_mode", "model_store", "model_store_max", "output_dir",
                         "model", "task", "task_family", "backend", "model_spec", "notify"):
                self.assertEqual(getattr(changed, name), getattr(current, name), name)

    def test_incompatible_preset_during_resume_requires_new_experiment(self):
        current = replace(RunConfig.smoke("demo/model"), resume=True)
        with self.assertRaisesRegex(RunConfigError, "新实验"):
            current.with_preset("main")


if __name__ == "__main__":
    unittest.main()
