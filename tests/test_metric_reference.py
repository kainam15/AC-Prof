"""指标文档在不同 locale 下保持 UTF-8，检查失败时不改写文件。"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.render_metric_reference import render

PROJECT_DIR = Path(__file__).resolve().parents[1]


class MetricReferenceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "docs").mkdir()
        self.path = self.root / "docs/Metric_Reference.md"

    def run_reference(self, *args):
        # Disable UTF-8 mode and reject implicit locale-dependent file encoding
        # on every platform, including UTF-8 Linux CI runners.
        return subprocess.run(
            [sys.executable, "-X", "utf8=0", "-X", "warn_default_encoding",
             "-W", "error::EncodingWarning", "-c", (
                 "import sys; from pathlib import Path; "
                 "from scripts import render_metric_reference as reference; "
                 "reference.ROOT = Path(sys.argv.pop(1)); "
                 "raise SystemExit(reference.main())"
             ), str(self.root), *args],
            cwd=PROJECT_DIR,
            env={**os.environ, "LC_ALL": "C", "PYTHONCOERCECLOCALE": "0",
                 "PYTHONIOENCODING": "utf-8"},
            capture_output=True, text=True, encoding="utf-8", timeout=30, check=False,
        )

    def test_generation_uses_utf8_and_lf_without_bom(self):
        result = self.run_reference()
        self.assertEqual(result.returncode, 0, result.stderr)
        raw = self.path.read_bytes()
        self.assertTrue(raw.startswith("# 指标登记表速查\n\n".encode("utf-8")))
        self.assertNotIn(b"\r", raw)
        self.assertEqual(raw.decode("utf-8"), render())

    def test_check_accepts_utf8_with_lf_or_crlf_without_rewriting(self):
        for newline in ("\n", "\r\n"):
            with self.subTest(newline=newline):
                raw = render().replace("\n", newline).encode("utf-8")
                self.path.write_bytes(raw)
                result = self.run_reference("--check")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.path.read_bytes(), raw)

    def test_check_rejects_missing_document_without_creating_it(self):
        result = self.run_reference("--check")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("python scripts/render_metric_reference.py", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(self.path.exists())

    def test_check_rejects_stale_document_without_rewriting(self):
        raw = "# 过期的指标登记表\n".encode("utf-8")
        self.path.write_bytes(raw)
        result = self.run_reference("--check")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("python scripts/render_metric_reference.py", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(self.path.read_bytes(), raw)

    def test_check_rejects_gbk_document_without_rewriting(self):
        raw = render().encode("gbk")
        self.path.write_bytes(raw)
        result = self.run_reference("--check")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("UTF-8", result.stderr)
        self.assertIn("python scripts/render_metric_reference.py", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(self.path.read_bytes(), raw)
