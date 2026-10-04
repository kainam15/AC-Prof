"""窗口统计的日期命名、内容去重及既有 CLI 输出契约。"""
import io
import json
import re
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from acprof.cli.stats import main


class TestStatisticsOutput:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        temporary = tmp_path
        self.directory = Path(str(temporary))
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
            assert (main(self.arguments(*extra))) == (0)
        line = stdout.getvalue().strip()
        assert (line.startswith("ACPROF_STATS ")), line
        return json.loads(line.removeprefix("ACPROF_STATS "))

    def test_statistics_reuses_one_bounded_result_snapshot(self, monkeypatch):
        original_open = Path.open
        original_read_bytes = Path.read_bytes
        modes = []

        def tracked_open(target, *args, **kwargs):
            if target == self.csv:
                modes.append(args[0] if args else kwargs.get("mode", "r"))
            return original_open(target, *args, **kwargs)

        def reject_materialized_read(target):
            if target == self.csv:
                raise AssertionError("statistics must not materialize the result CSV")
            return original_read_bytes(target)

        monkeypatch.setattr(Path, "open", tracked_open)
        monkeypatch.setattr(Path, "read_bytes", reject_materialized_read)
        self.calculate()
        assert (modes) == (["rb", "rb"])

    def test_statistics_rejects_source_change_before_publication(self, monkeypatch):
        original_open = Path.open
        binary_opens = 0

        def mutate_before_verification(target, *args, **kwargs):
            nonlocal binary_opens
            mode = args[0] if args else kwargs.get("mode", "r")
            if target == self.csv and mode == "rb":
                binary_opens += 1
                if binary_opens == 2:
                    with original_open(target, "ab") as stream:
                        stream.write(b"\n")
            return original_open(target, *args, **kwargs)

        monkeypatch.setattr(Path, "open", mutate_before_verification)
        stderr = io.StringIO()
        with redirect_stderr(stderr), pytest.raises(SystemExit) as raised:
            main(self.arguments())
        assert (raised.value.code) == (1)
        assert ("统计期间结果 CSV 发生变化") in (stderr.getvalue())
        assert (binary_opens) == (2)
        assert not (self.output.exists())

    def test_timestamp_and_duplicate_reuse_leave_source_and_existing_report_unchanged(self):
        source = self.csv.read_bytes()
        first = self.calculate()
        path = Path(first["report_path"])
        assert not (first["reused"])
        assert re.search(r"^window-statistics-\d{8}-\d{6}-\d{6}\.json$", path.name)
        datetime.strptime(path.stem.removeprefix("window-statistics-"), "%Y%m%d-%H%M%S-%f")
        saved, modified = path.read_bytes(), path.stat().st_mtime_ns
        assert (self.calculate()) == ({"report_path": str(path), "reused": True})
        assert (list(self.output.iterdir())) == ([path])
        assert ((path.read_bytes(), path.stat().st_mtime_ns)) == ((saved, modified))
        assert (self.csv.read_bytes()) == (source)

    def test_existing_uuid_report_is_reused_despite_json_formatting(self):
        generated = Path(self.calculate()["report_path"])
        existing = generated.with_name("window-statistics-1b1047da59ca46f186c93ac5ad86f8d0.json")
        data = json.loads(generated.read_text(encoding="utf-8"))
        generated.rename(existing)
        existing.write_text(json.dumps(data, sort_keys=True, ensure_ascii=True), encoding="utf-8")
        saved = existing.read_bytes()
        assert (self.calculate()) == ({"report_path": str(existing), "reused": True})
        assert (list(self.output.iterdir())) == ([existing])
        assert (existing.read_bytes()) == (saved)

    @pytest.mark.parametrize('options', (('--confidence', '0.9'), ('--seed', '1'), ('--resamples', '40'), ('--block-size', '2'), ('--metric', 'latency_app_s')))
    def test_statistics_settings_and_source_changes_create_distinct_reports(self, options):
        first = Path(self.calculate()["report_path"])
        original = first.read_bytes()
        for other in (('--confidence', '0.9'), ('--seed', '1'), ('--resamples', '40'), ('--block-size', '2'), ('--metric', 'latency_app_s')):
            if other != options:
                self.calculate(*other)
        receipt = self.calculate(*options)
        assert not (receipt["reused"])
        assert (receipt["report_path"]) != (str(first))
        self.csv.write_text(self.csv.read_text(encoding="utf-8").replace("0.03", "0.04"), encoding="utf-8")
        assert not (self.calculate()["reused"])
        assert (len(list(self.output.glob("*.json")))) == (7)
        assert (first.read_bytes()) == (original)

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
        assert not (receipt["reused"])
        assert (receipt["report_path"]) != (str(original))
        assert (json.loads(original.read_text())["groups"][0]["mean"]) == (999)
        assert (corrupt.read_bytes()) == (b'{"schema_version":')

    def test_same_timestamp_never_overwrites_different_report(self):
        with patch("acprof.cli.stats.datetime") as clock:
            clock.now.return_value = datetime(2026, 9, 27, 12, 34, 56, 123456)
            first = Path(self.calculate()["report_path"])
            original = first.read_bytes()
            second = Path(self.calculate("--seed", "1")["report_path"])
        assert (first.name) == ("window-statistics-20260927-123456-123456.json")
        assert (second.name) == ("window-statistics-20260927-123456-123457.json")
        assert (first.read_bytes()) == (original)

    def test_concurrent_calculations_publish_only_one_report(self):
        command = [sys.executable, "-m", "acprof.cli.stats", *self.arguments()]
        processes = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                     for _ in range(2)]
        try:
            receipts = []
            for process in processes:
                stdout, stderr = process.communicate(timeout=20)
                assert (process.returncode) == (0), stderr
                receipts.append(json.loads(stdout.strip().removeprefix("ACPROF_STATS ")))
            assert (receipts[0]["report_path"]) == (receipts[1]["report_path"])
            assert (sorted(receipt["reused"] for receipt in receipts)) == ([False, True])
            assert (len(list(self.output.iterdir()))) == (1)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.communicate()

    def test_stdout_and_explicit_output_keep_their_existing_contract(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            assert (main([str(self.csv), "--resamples", "30"])) == (0)
        assert (json.loads(stdout.getvalue())["schema_version"]) == (1)
        target = self.directory / "chosen.json"
        with redirect_stdout(io.StringIO()):
            assert (main([str(self.csv), "--resamples", "30", "--output", str(target)])) == (0)
        original = target.read_bytes()
        with redirect_stderr(io.StringIO()), pytest.raises(SystemExit) as raised:
            main([str(self.csv), "--output", str(target)])
        assert (raised.value.code) == (2)
        assert (target.read_bytes()) == (original)

    def test_stats_and_tui_summary_preserve_output_quality_evidence(self):
        from acprof.quality import loading_quality
        from acprof.tui.diagnostics import result_summary_text, summarize_result_csv
        from acprof.tui.i18n import translate
        checks = loading_quality({"missing_keys": ["head.weight"]}, source="loader-log")
        (self.directory / "quality_checks.json").write_text(json.dumps({"schema_version": 1, "checks": checks}))
        stream = io.StringIO()
        with redirect_stdout(stream):
            assert (main([str(self.csv), "--metric", "latency_app_s", "--resamples", "30"])) == (0)
        report = json.loads(stream.getvalue())
        assert (report["quality_status"]) == ("blocked")
        assert not (report["auto_selection_eligible"])
        summary = result_summary_text(summarize_result_csv(self.csv), self.csv)
        assert ("weights_reinitialized") in (str(summary))
        assert ("loader-log") in (translate(summary, "en"))
