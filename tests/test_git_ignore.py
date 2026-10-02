"""Generated CSV stays ignored while reviewable fixtures can be committed."""
import subprocess
import tempfile
import unittest
from pathlib import Path


class GitIgnoreTests(unittest.TestCase):
    def test_csv_locations(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(["git", "init", "--quiet", directory], check=True)
            (Path(directory) / ".gitignore").write_bytes((root / ".gitignore").read_bytes())
            for path, ignored in {
                "results/foo.csv": True,
                "internal-testing/foo.csv": True,
                "result-past/foo.csv": True,
                "ad-hoc.csv": True,
                "tests/fixtures/foo.csv": False,
                "tests/fixtures/nested/foo.csv": False,
                "examples/foo.csv": False,
                "examples/nested/foo.csv": False,
            }.items():
                with self.subTest(path=path):
                    result = subprocess.run(["git", "-c", "core.excludesFile=/dev/null", "check-ignore", "--no-index", "-q", path], cwd=directory)
                    self.assertEqual(result.returncode, 0 if ignored else 1)
