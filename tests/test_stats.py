"""窗口统计的日期命名、内容去重及既有 CLI 输出契约。"""
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from acprof.cli.stats import main


class StatisticsOutputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.csv = self.directory / "结果.csv"
        self.csv.write_text(
            "cpu_cores,mem_cap_gb,gpu_mode,input_scale,warmup,repeat_idx,status,latency_app_s\n"
            "2,8,off,64,0,0,ok,0.01\n2,8,off,64,0,1,ok,0.02\n2,8,off,64,0,2,ok,0.03\n",
            encoding="utf-8",
        )
        self.output = self.directory / "analysis"

    def arguments(self, *extra):
        return [str(self.csv), "--resamples", "30", "--output-dir", str(self.output), *extra]

    def calculate(self, *extra):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main(self.arguments(*extra)), 0)
        line = stdout.getvalue().strip()
        self.assertTrue(line.startswith("ACPROF_STATS "), line)
        return json.loads(line.removeprefix("ACPROF_STATS "))

    def test_timestamp_and_duplicate_reuse_leave_source_and_existing_report_unchanged(self):
        source = self.csv.read_bytes()
        first = self.calculate()
        path = Path(first["report_path"])
        self.assertFalse(first["reused"])
        self.assertRegex(path.name, r"^window-statistics-\d{8}-\d{6}-\d{6}\.json$")
        datetime.strptime(path.stem.removeprefix("window-statistics-"), "%Y%m%d-%H%M%S-%f")
        saved, modified = path.read_bytes(), path.stat().st_mtime_ns
        self.assertEqual(self.calculate(), {"report_path": str(path), "reused": True})
        self.assertEqual(list(self.output.iterdir()), [path])
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), (saved, modified))
        self.assertEqual(self.csv.read_bytes(), source)

    def test_existing_uuid_report_is_reused_despite_json_formatting(self):
        generated = Path(self.calculate()["report_path"])
        existing = generated.with_name("window-statistics-1b1047da59ca46f186c93ac5ad86f8d0.json")
        data = json.loads(generated.read_text(encoding="utf-8"))
        generated.rename(existing)
        existing.write_text(json.dumps(data, sort_keys=True, ensure_ascii=True), encoding="utf-8")
        saved = existing.read_bytes()
        self.assertEqual(self.calculate(), {"report_path": str(existing), "reused": True})
        self.assertEqual(list(self.output.iterdir()), [existing])
        self.assertEqual(existing.read_bytes(), saved)

    def test_statistics_settings_and_source_changes_create_distinct_reports(self):
        first = Path(self.calculate()["report_path"])
        original = first.read_bytes()
        for options in (("--confidence", "0.9"), ("--seed", "1"), ("--resamples", "40"),
                        ("--block-size", "2"), ("--metric", "latency_app_s")):
            with self.subTest(options=options):
                receipt = self.calculate(*options)
                self.assertFalse(receipt["reused"])
                self.assertNotEqual(receipt["report_path"], str(first))
        self.csv.write_text(self.csv.read_text(encoding="utf-8").replace("0.03", "0.04"), encoding="utf-8")
        self.assertFalse(self.calculate()["reused"])
        self.assertEqual(len(list(self.output.glob("*.json"))), 7)
        self.assertEqual(first.read_bytes(), original)

    def test_corrupt_and_modified_reports_do_not_count_as_identical(self):
        original = Path(self.calculate()["report_path"])
        data = json.loads(original.read_text(encoding="utf-8"))
        data["groups"][0]["mean"] = 999
        original.write_text(json.dumps(data), encoding="utf-8")
        corrupt = self.output / "window-statistics-broken.json"
        corrupt.write_bytes(b'{"schema_version":')
        invalid_encoding = self.output / "window-statistics-invalid.json"
        invalid_encoding.write_bytes(b"\xff")
        receipt = self.calculate()
        self.assertFalse(receipt["reused"])
        self.assertNotEqual(receipt["report_path"], str(original))
        self.assertEqual(json.loads(original.read_text())["groups"][0]["mean"], 999)
        self.assertEqual(corrupt.read_bytes(), b'{"schema_version":')

    def test_same_timestamp_never_overwrites_different_report(self):
        with patch("acprof.cli.stats.datetime") as clock:
            clock.now.return_value = datetime(2026, 9, 27, 12, 34, 56, 123456)
            first = Path(self.calculate()["report_path"])
            original = first.read_bytes()
            second = Path(self.calculate("--seed", "1")["report_path"])
        self.assertEqual(first.name, "window-statistics-20260927-123456-123456.json")
        self.assertEqual(second.name, "window-statistics-20260927-123456-123457.json")
        self.assertEqual(first.read_bytes(), original)

    def test_concurrent_calculations_publish_only_one_report(self):
        command = [sys.executable, "-m", "acprof.cli.stats", *self.arguments()]
        processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                     for _ in range(2)]
        try:
            receipts = []
            for process in processes:
                stdout, stderr = process.communicate(timeout=20)
                self.assertEqual(process.returncode, 0, stderr)
                receipts.append(json.loads(stdout.strip().removeprefix("ACPROF_STATS ")))
            self.assertEqual(receipts[0]["report_path"], receipts[1]["report_path"])
            self.assertEqual(sorted(receipt["reused"] for receipt in receipts), [False, True])
            self.assertEqual(len(list(self.output.iterdir())), 1)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.communicate()

    def test_stdout_and_explicit_output_keep_their_existing_contract(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main([str(self.csv), "--resamples", "30"]), 0)
        self.assertEqual(json.loads(stdout.getvalue())["schema_version"], 1)
        target = self.directory / "chosen.json"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.csv), "--resamples", "30", "--output", str(target)]), 0)
        original = target.read_bytes()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main([str(self.csv), "--output", str(target)])
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(target.read_bytes(), original)

    def test_stats_and_tui_summary_preserve_output_quality_evidence(self):
        from acprof.quality import loading_quality
        from acprof.tui.diagnostics import result_summary_text, summarize_result_csv
        from acprof.tui.i18n import translate
        checks = loading_quality({"missing_keys": ["head.weight"]}, source="loader-log")
        (self.directory / "quality_checks.json").write_text(json.dumps({"schema_version": 1, "checks": checks}))
        stream = io.StringIO()
        with redirect_stdout(stream):
            self.assertEqual(main([str(self.csv), "--metric", "latency_app_s", "--resamples", "30"]), 0)
        report = json.loads(stream.getvalue())
        self.assertEqual(report["quality_status"], "blocked")
        self.assertFalse(report["auto_selection_eligible"])
        summary = result_summary_text(summarize_result_csv(self.csv), self.csv)
        self.assertIn("weights_reinitialized", str(summary))
        self.assertIn("loader-log", translate(summary, "en"))


if __name__ == "__main__":
    unittest.main()
