"""用户入口、结果归属和迟到读取的行为回归；不启动真实采集。"""
import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from acprof.artifact_layout import ArtifactLayout
from acprof.experiment import RunConfig
from acprof.tui.diagnostics import summarize_result_csv
from acprof.tui.run_form import infer_preset, matches_preset
from acprof.tui.run_results import RunArtifacts, inspect_run_result


def write_csv(path, latencies=(0.01, 0.2)):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["cpu_cores", "mem_cap_gb", "gpu_mode",
            "input_scale", "warmup", "repeat_idx", "status", "latency_app_s"])
        writer.writeheader()
        for scale, latency in zip((64, 128), latencies):
            writer.writerow(dict(cpu_cores=1, mem_cap_gb=4, gpu_mode="off", input_scale=scale,
                                 warmup=0, repeat_idx=0, status="ok", latency_app_s=latency))


class SummaryAndPresetTests(unittest.TestCase):
    def test_execution_settings_do_not_change_preset(self):
        config = replace(RunConfig.smoke("demo/model"), output_dir="elsewhere", model_store="cache",
                         download_mode="direct", notify="auto", resume=True, skip_build=True)
        self.assertEqual(infer_preset(config), "smoke")
        self.assertTrue(matches_preset(config, "smoke"))
        self.assertFalse(matches_preset(replace(config, cpus="2"), "smoke"))

    def test_summary_keeps_input_scales_separate_and_single_windows_insufficient(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.csv"
            write_csv(path)
            summary = summarize_result_csv(path)
            groups = getattr(summary, "groups", ())
            self.assertEqual(len(groups), 2, "两个输入规模必须分别显示")
            self.assertEqual([group["mean"] for group in groups], [0.01, 0.2])
            self.assertTrue(all(group["reason"] == "insufficient_windows" for group in groups))
            self.assertTrue(all(group["ci_low"] is None for group in groups))


class RunAttributionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "run"
        self.before = RunArtifacts.read(self.root)
        self.layout = ArtifactLayout.for_new_run(self.root)
        self.layout.initialize()
        self.state: dict = dict(schema_version=1, layout_version=2, run_id="run-1", status="complete", outcome="ok",
            attempts=[dict(pid=123, started_at="2026-10-02T01:00:00Z", ended_at="2026-10-02T01:01:00Z")],
            options=dict(cpus="1", mems="4", gpus="off", warmup=0, repeat=1),
            runtime=dict(planned=dict(scales=[64, 128])),
            cases={"case.csv": dict(status="complete", completed_at="2026-10-02T01:01:00Z")})
        self.save_state()
        write_csv(self.layout.result_csv)
        self.layout.path("capability_report.json").write_text(json.dumps({"requested_measurements_complete": True}))

    def save_state(self):
        path = self.layout.path("run_state.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state))

    def test_complete_requires_matching_attempt_and_audited_coverage(self):
        result = inspect_run_result(self.before, 123)
        self.assertEqual(result.stage(0, False, ""), "已完成")
        self.assertEqual(result.result_csv, str(self.layout.result_csv))
        self.assertEqual((result.completed_cases, result.total_cases, result.new_cases), (1, 1, 1))
        self.assertEqual(inspect_run_result(self.before, 999).stage(0, False, ""), "失败")
        self.state["runtime"]["planned"]["scales"].append(256)
        self.save_state()
        self.assertEqual(inspect_run_result(self.before, 123).stage(0, False, ""), "部分完成")

    def test_zero_exit_with_missing_requested_metrics_is_partial(self):
        self.layout.path("capability_report.json").write_text(json.dumps({"requested_measurements_complete": False}))
        self.assertEqual(inspect_run_result(self.before, 123).stage(0, False, ""), "部分完成")

    def test_preflight_failure_and_unchanged_resume_never_promote_history(self):
        before = RunArtifacts.read(self.root)
        self.assertFalse(inspect_run_result(before, 123).belongs_to_attempt)
        self.state["attempts"].append(dict(pid=124, started_at="new", ended_at="end"))
        self.save_state()
        result = inspect_run_result(before, 124)
        self.assertTrue(result.belongs_to_attempt)
        self.assertEqual(result.result_csv, "")
        self.assertEqual(result.new_cases, 0)

    def test_stopped_case_is_retained_without_merged_csv(self):
        self.layout.result_csv.unlink()
        self.state["status"] = "interrupted"
        self.save_state()
        result = inspect_run_result(self.before, 123)
        self.assertEqual(result.stage(0, True, ""), "已停止")
        self.assertEqual(result.stage(1, False, ""), "部分完成")
        self.assertEqual(result.new_cases, 1)
        self.assertEqual(result.result_csv, "")
