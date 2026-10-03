"""Report entry point, offline serialization and artifact protection."""
import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from acprof.cli.main import main


class TestReport:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.temporary = tmp_path
        self.root = Path(str(self.temporary))
        self.source = self.root / "result_all.csv"
        self.source.write_text(
            "cpu_cores,mem_cap_gb,gpu_mode,input_scale,repeat_idx,warmup,status,latency_app_p95_s\n"
            "2,4,off,32,0,0,ok,0.04\n", encoding="utf-8")

    def test_report_help_is_public_and_does_not_start_collection(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            with pytest.raises(SystemExit) as result:
                main(["report", "--help"])
        assert (result.value.code) == (0), stream.getvalue()
        assert ("--baseline") in (stream.getvalue())
        result = subprocess.run([sys.executable, "-c", (
            "import sys; from acprof.cli.main import main\n"
            "try: main(['report', '--help'])\n"
            "except SystemExit: pass\n"
            "assert 'plotly' not in sys.modules\n"
            "assert 'acprof.host.run_state' not in sys.modules\n"
        )], capture_output=True, text=True, check=False)
        assert (result.returncode) == (0), result.stderr

    def test_offline_report_escapes_data_and_protects_existing_output(self):
        name = "__STYLE__ </script><script>alert(1)</script>"
        (self.root / "static_meta.json").write_text(json.dumps({"model_name": name}))
        output = self.root / "report.html"
        with contextlib.redirect_stdout(io.StringIO()):
            assert (main(["report", str(self.source)])) == (0)
        content = output.read_text()
        assert ("</script><script>alert(1)</script>") not in (content)
        assert ("__STYLE__ \\u003c/script") in (content)
        assert ('<script src=') not in (content)
        assert ("Plotly") in (content)
        assert ("Permission is hereby granted") in (content)
        with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
            main(["report", str(self.source), "--output", str(output)])
        assert (output.read_text()) == (content)

    def test_baseline_is_unique_and_errors_do_not_publish_a_report(self):
        from acprof.analysis.model import load_analysis
        from acprof.plotting.report import report_payload
        model = load_analysis([self.source])
        config = model.configs[0]
        assert (report_payload(model, config["run_id"])["baseline"]) == (config["config_id"])
        with self.source.open("a") as stream:
            stream.write("4,4,off,32,0,0,ok,0.02\n")
        model = load_analysis([self.source])
        with pytest.raises(ValueError, match="唯一"):
            report_payload(model, model.configs[0]["run_id"])
        with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
            main(["report", str(self.source), "--baseline", "does-not-exist"])
        assert not ((self.root / "report.html").exists())

    @pytest.mark.skipif(not (os.name == "posix"), reason="Linux measurement lock")
    def test_report_refuses_to_run_during_a_formal_collection(self):
        from acprof.host.run_state import MeasurementLock
        with MeasurementLock(), contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit):
            main(["report", str(self.source)])
        assert not ((self.root / "report.html").exists())

    def test_malformed_csv_reports_an_error_without_a_traceback(self):
        self.source.write_text('cpu_cores,status,warmup\n"2,ok,0\n')
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream), pytest.raises(SystemExit):
            main(["report", str(self.source)])
        assert ("报告生成失败") in (stream.getvalue())
        assert ("Traceback") not in (stream.getvalue())
