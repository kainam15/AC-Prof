import csv
import hashlib
import io
import json
import sys
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest

from acprof.cli import run
from acprof.config import CSV_FIELDS
from acprof.host.detect import TaskInfo
from acprof.host.input_plan import PlannedInputScales
from acprof.host.runtime_images import ImageInfo
from acprof.host.static_metadata import StaticMeta


class RunRecoveryFixture:
    def build(self, request, tmp_path):
        self._request = request
        from platform_fixtures import native_policy
        native_policy(self._request)
        temporary = tmp_path
        self.root = Path(str(temporary))
        self.directory = self.root / "org--model"
        self.task = TaskInfo("org/model", "fill-mask", "nlp", "transformers_pipeline",
                             "transformers", "a" * 40, "manual")
        self.image = ImageInfo(tag="sha256:" + "b" * 64)
        self.calls = []

    def write_case(self, **kwargs):
        self.calls.append(kwargs["cpu"])
        path = self.directory / f'.acprof/work/cases/{kwargs["cpu"]}c_4g_off/result.csv'
        row = dict.fromkeys(CSV_FIELDS, "nan")
        row.update(cpu_cores=str(kwargs["cpu"]), mem_cap_gb="4", gpu_mode="off",
                   input_scale="64", warmup="0", repeat_idx="0", status="ok", error="",
                   latency_app_s="0.1", latency_s="0.09", throughput_samples_per_s="10",
                   container_cpu_util_avg_pct="0", container_mem_usage_avg_bytes="1024",
                   cpu_energy_total_j="1", vcpu_energy_total_j="0", cpu_instructions_per_request="0")
        with path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            if path.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(row)
        return str(path)

    def prepare_plan(self, **kwargs):
        path = Path(kwargs["output_dir"]) / "metadata/input_scale_plan.json"
        path.write_text(json.dumps({"schema_version": 2, "entries": [
            {"input_scale": 64, "payload": {"text": "test input"}}
        ]}))
        return PlannedInputScales([64.0], "manual", str(path), {},
                                  hashlib.sha256(path.read_bytes()).hexdigest())

    def invoke(self, *extra, case=None, validation=None, output=None):
        from acprof.platform import Environment
        metadata = StaticMeta(
            **Environment("native_linux").metadata(),
            model_name="org/model", model_revision="a" * 40, task_family="nlp",
            pipeline_tag="fill-mask", runtime_backend="transformers_pipeline",
            image_tag=self.image.tag, batch_size=1, input_scale_type="seq_length",
            run_command="fixture", model_download_url="https://huggingface.co/org/model",
            gpu="none", gpu_mem_total_bytes=None, model_cache_bytes=1, docker_image_bytes=1,
            environment="linux", cpu_power_source="rapl", vcpu_power_method="rapl_cgroup_cpu_share",
            cpu_governor="performance", cpu_boost="off", cgroup_version="v2",
        )
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", [
                "acprof run", "--model", "org/model", "--cpus", "1,2", "--mems", "4",
                "--gpus", "off", "--input-scales", "64", "--warmup", "0", "--repeat", "1",
                "--repeat-in-window", "1", "--notify", "none", "--no-prune-startup-oom",
                "--output-dir", str(self.root), "--matrix-order", "declared", *extra,
            ]))
            for name in ("bootstrap_project_env", "require_collection_host", "require_native_docker",
                         "require_packet_latency_prerequisites", "require_cpu_energy_prerequisites",
                         "require_mips_prerequisites"):
                stack.enter_context(patch.object(run, name))
            stack.enter_context(patch.object(run, "start_terminal_log", return_value=None))
            stack.enter_context(patch.object(run, "require_cgroup_prerequisites", return_value="v2"))
            stack.enter_context(patch("acprof.host.detect.detect_task", return_value=self.task))
            stack.enter_context(patch("acprof.host.interface_probe.probe_interface", return_value={"status": "ok"}))
            stack.enter_context(patch("acprof.host.runtime_images.configure_runtime_profile"))
            stack.enter_context(patch("acprof.host.runtime_images.prepare_image", return_value=self.image))
            stack.enter_context(patch("acprof.host.runtime_images.require_image_identity"))
            stack.enter_context(patch("acprof.host.runtime_validation.validate_runtime",
                                      side_effect=validation, return_value={"status": "ok"}))
            stack.enter_context(patch("acprof.host.static_metadata.collect_static_meta", return_value=metadata))
            stack.enter_context(patch("acprof.host.input_plan.plan_input_scales", side_effect=self.prepare_plan))
            stack.enter_context(patch("acprof.host.orchestrator.run_single_case", side_effect=case or self.write_case))
            stack.enter_context(redirect_stdout(output if output is not None else io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            run.main()

    def assert_missing_required_measurements_rejected(self, mode):
        def missing_memory(**kwargs):
            path = Path(self.write_case(**kwargs))
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            if kwargs["cpu"] == 2:
                rows[-1]["container_mem_usage_avg_bytes"] = "nan"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            return str(path)
        with pytest.raises(SystemExit) as caught:
            self.invoke("--profiling-mode", mode, case=missing_memory)
        assert (caught.value.code) == (1)
        report = json.loads((self.directory / "capability_report.json").read_text())
        assert (report["measurement"]["container_memory"]["status"]) == ("unavailable")
        assert not (report["full_profile_complete"])
        assert ((self.directory / "result_layers.json").is_file())
        assert (len(list(self.directory.glob(".acprof/work/cases/*/result.csv")))) == (2)
        state = json.loads((self.directory / ".acprof/run_state.json").read_text())
        assert (state["status"]) == ("failed")
        assert (state["artifacts"]["static_meta.json"]) == (hashlib.sha256((self.directory / "static_meta.json").read_bytes()).hexdigest())

    def interrupt_after_first(self):
        def interrupted(**kwargs):
            if kwargs["cpu"] == 2:
                raise KeyboardInterrupt()
            return self.write_case(**kwargs)
        with pytest.raises(KeyboardInterrupt):
            self.invoke(case=interrupted)
        self.calls.clear()
