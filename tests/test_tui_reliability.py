"""用户入口、结果归属和迟到读取的行为回归；不启动真实采集。"""
import unittest
from dataclasses import replace

from acprof.experiment import RunConfig
from acprof.tui.run_form import infer_preset, matches_preset


class SummaryAndPresetTests(unittest.TestCase):
    def test_execution_settings_do_not_change_preset(self):
        config = replace(RunConfig.smoke("demo/model"), output_dir="elsewhere", model_store="cache",
                         download_mode="direct", notify="auto", resume=True, skip_build=True)
        self.assertEqual(infer_preset(config), "smoke")
        self.assertTrue(matches_preset(config, "smoke"))
        self.assertFalse(matches_preset(replace(config, cpus="2"), "smoke"))
