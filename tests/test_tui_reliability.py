"""用户入口、结果归属和迟到读取的行为回归；不启动真实采集。"""
import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from acprof.experiment import RunConfig
from acprof.tui.diagnostics import summarize_result_csv
from acprof.tui.run_form import infer_preset, matches_preset


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
