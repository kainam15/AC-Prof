"""验证 CI 的退出状态和跳过证据，避免空测试/缺依赖伪装为通过。"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


class TestValidationRunner:
    def run_fixture(self, body, *options):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_sample.py").write_text("import pytest\n" + body)
            report = root / "report.json"
            process = subprocess.run([
                sys.executable, str(ROOT / "scripts/run_tests.py"),
                "--directory", directory, "--report", str(report), *options,
            ], capture_output=True, text=True)
            return process, json.loads(report.read_text()) if report.exists() else None

    def test_missing_runtime_fails_strict_job_and_records_reason(self):
        process, report = self.run_fixture(
            "class TestSample:\n"
            "    @pytest.mark.skip(reason='no runtime')\n"
            "    def test_runtime(self): pass\n", "--require-no-skips",
        )
        assert (process.returncode) == (1), process.stderr
        assert (report["counts"]["skipped"]) == (1)
        assert not (report["successful"])
        assert (report["tests"][0]["reason"]) == ("no runtime")

    def test_optional_skip_remains_visible(self):
        process, report = self.run_fixture(
            "class TestSample:\n"
            "    @pytest.mark.skip(reason='no runtime')\n"
            "    def test_runtime(self): pass\n",
        )
        assert (process.returncode) == (0), process.stderr
        assert (report) is not None
        assert (report["successful"])
        assert (report["counts"]["passed"]) == (0)

    def test_empty_discovery_is_failure(self):
        process, report = self.run_fixture("")
        assert (process.returncode) == (1), process.stderr
        assert (report["counts"]["run"]) == (0)

    def test_missing_required_pattern_cannot_hide_behind_other_passing_tests(self):
        process, report = self.run_fixture(
            "class TestSample:\n"
            "    def test_runtime(self): pass\n",
            "--require-no-skips", "--pattern", "test_sample.py", "--pattern", "test_missing_runtime.py",
        )
        assert (process.returncode) == (1), process.stderr
        assert not (report['successful'])
        assert (report['discovery_counts']['test_missing_runtime.py']) == (0)

    def test_failed_parameter_is_failure_and_has_evidence(self):
        process, report = self.run_fixture(
            "class TestSample:\n"
            "    @pytest.mark.parametrize('value', [2])\n"
            "    def test_values(self, value):\n"
            "        assert 1 == value\n",
        )
        assert (process.returncode) == (1), process.stderr
        assert (report) is not None
        assert ("1 == 2") in (report["tests"][0]["reason"])

    def test_shards_cover_every_test_once_and_preserve_failures(self):
        body = (
            "class TestSample:\n"
            "    def test_a(self): pass\n"
            "    def test_b(self): pytest.fail('real failure')\n"
            "    def test_c(self): pass\n"
            "    def test_d(self): pass\n"
            "    def test_e(self): pass\n"
        )
        _, whole = self.run_fixture(body)
        expected = {test["id"] for test in whole["tests"]}
        selected = []
        for index in range(3):
            process, report = self.run_fixture(
                body, "--shard-index", str(index), "--shard-count", "3",
            )
            assert (report) is not None, process.stderr
            selected.extend(test["id"] for test in report["tests"])
            assert (report["shard"]["index"]) == (index)
            assert (report["shard"]["count"]) == (3)
            assert (report["shard"]["discovered"]) == (5)
            assert (report["shard"]["selected"]) == (report["counts"]["run"])
            assert (report["shard"]["suite_sha256"]) == (whole["shard"]["suite_sha256"])
            assert (process.returncode) == (1 if index == 1 else 0), process.stderr
        assert (set(selected)) == (expected)
        assert (len(selected)) == (len(expected))

    @pytest.mark.parametrize('index,count', ((-1, 2), (2, 2), (0, 0)))
    def test_invalid_or_empty_shard_cannot_pass(self, index, count):
        body = "class TestSample:\n    def test_one(self): pass\n"
        process, report = self.run_fixture(
            body, "--shard-index", str(index), "--shard-count", str(count),
        )
        assert (process.returncode) == (2)
        assert (report) is None
        process, report = self.run_fixture(body, "--shard-index", "1", "--shard-count", "2")
        assert (report) is not None, process.stderr
        assert (process.returncode) == (1)
        assert (report["counts"]["run"]) == (0)
