"""Report entry point, offline serialization and artifact protection."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from acprof.cli.main import main


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "result_all.csv"
        self.source.write_text(
            "cpu_cores,mem_cap_gb,gpu_mode,input_scale,repeat_idx,warmup,status,latency_app_p95_s\n"
            "2,4,off,32,0,0,ok,0.04\n", encoding="utf-8")

    def test_report_help_is_public_and_does_not_start_collection(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            with self.assertRaises(SystemExit) as result:
                main(["report", "--help"])
        self.assertEqual(result.exception.code, 0, stream.getvalue())
        self.assertIn("--baseline", stream.getvalue())
        result = subprocess.run([sys.executable, "-c", (
            "import sys; from acprof.cli.main import main\n"
            "try: main(['report', '--help'])\n"
            "except SystemExit: pass\n"
            "assert 'plotly' not in sys.modules\n"
            "assert 'acprof.host.run_state' not in sys.modules\n"
        )], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_offline_report_escapes_data_and_protects_existing_output(self):
        name = "__STYLE__ </script><script>alert(1)</script>"
        (self.root / "static_meta.json").write_text(json.dumps({"model_name": name}))
        output = self.root / "report.html"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["report", str(self.source)]), 0)
        content = output.read_text()
        self.assertNotIn("</script><script>alert(1)</script>", content)
        self.assertIn("__STYLE__ \\u003c/script", content)
        self.assertNotIn('<script src=', content)
        self.assertIn("Plotly", content)
        self.assertIn("Permission is hereby granted", content)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["report", str(self.source), "--output", str(output)])
        self.assertEqual(output.read_text(), content)

    def test_baseline_is_unique_and_errors_do_not_publish_a_report(self):
        from acprof.analysis.model import load_analysis
        from acprof.plotting.report import report_payload
        model = load_analysis([self.source])
        config = model.configs[0]
        self.assertEqual(report_payload(model, config["run_id"])["baseline"], config["config_id"])
        with self.source.open("a") as stream:
            stream.write("4,4,off,32,0,0,ok,0.02\n")
        model = load_analysis([self.source])
        with self.assertRaisesRegex(ValueError, "唯一"):
            report_payload(model, model.configs[0]["run_id"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["report", str(self.source), "--baseline", "does-not-exist"])
        self.assertFalse((self.root / "report.html").exists())

    @unittest.skipUnless(os.name == "posix", "Linux measurement lock")
    def test_report_refuses_to_run_during_a_formal_collection(self):
        from acprof.host.run_state import MeasurementLock
        with MeasurementLock(), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["report", str(self.source)])
        self.assertFalse((self.root / "report.html").exists())

    def test_malformed_csv_reports_an_error_without_a_traceback(self):
        self.source.write_text('cpu_cores,status,warmup\n"2,ok,0\n')
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream), self.assertRaises(SystemExit):
            main(["report", str(self.source)])
        self.assertIn("报告生成失败", stream.getvalue())
        self.assertNotIn("Traceback", stream.getvalue())
