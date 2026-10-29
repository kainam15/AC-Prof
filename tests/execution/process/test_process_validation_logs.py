"""Process validation reads complete result records without retaining diagnostic logs."""
import json
import tracemalloc
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.artifacts import file_sha256
from acprof.host.runtimes import validation
from acprof.host.runtimes.process_control import ManagedProcess

PREFIX = "ACPROF_RUNTIME_VALIDATION="


@pytest.fixture
def validation_log(tmp_path, monkeypatch):
    from acprof.host.model_store import store as model_store
    from acprof.host.runtimes import resources

    path = tmp_path / "validation.log"
    process = SimpleNamespace(log_path=path, wait=Mock(return_value=0), close=Mock())
    process.log_tail = MethodType(ManagedProcess.log_tail, process)
    lease = SimpleNamespace(pass_fds=(), close=Mock())
    limits = SimpleNamespace(path="", close=Mock())
    monkeypatch.setattr(model_store, "acquire_mount", lambda _: lease)
    monkeypatch.setattr(resources, "ProcessLimits", lambda *args: limits)
    monkeypatch.setattr(validation, "service_environment", lambda *args, **kwargs: {})
    monkeypatch.setattr(validation, "ManagedProcess", lambda *args, **kwargs: process)
    info = SimpleNamespace(python="fixture-python", runtime_environment={})

    def run():
        return validation.run_validation(None, info, "fixture.module", PREFIX, tmp_path)

    yield SimpleNamespace(path=path, run=run, process=process)
    process.close.assert_called_once()
    lease.close.assert_called_once()
    limits.close.assert_called_once()


@pytest.mark.parametrize("noise", ["long-lines", "many-records"])
def test_diagnostic_log_size_does_not_determine_parser_peak_memory(validation_log, noise):
    expected = {"status": "ok", "selected": "last result"}
    with validation_log.path.open("w", encoding="utf-8") as stream:
        if noise == "long-lines":
            block = "diagnostic " * 4096
            for _ in range(200):
                stream.write(block)
            stream.write("\n")
        else:
            record = PREFIX + json.dumps({"status": "ok", "diagnostic": "x" * 4096}) + "\n"
            for _ in range(2000):
                stream.write(record)
        stream.write(PREFIX + json.dumps(expected) + "\n")
        # A result need not occur near EOF; shutdown diagnostics may follow it.
        for _ in range(200):
            stream.write("shutdown " * 4096)
    before = file_sha256(validation_log.path)
    tracemalloc.start()
    try:
        observed = validation_log.run()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert observed == expected
    assert file_sha256(validation_log.path) == before
    assert peak_bytes < 2 * 1024 * 1024, f"log parsing allocated {peak_bytes:,} bytes"


@pytest.mark.parametrize("ending", ["\n", "\r\n", "\r", ""])
def test_complete_result_can_span_chunks_and_contain_unicode_line_separators(validation_log, ending):
    expected = {"status": "ok", "detail": "中\u0085文\u2028分\u2029隔" * 20000}
    record = PREFIX + json.dumps(expected, ensure_ascii=False) + ending
    validation_log.path.write_bytes(record.encode("utf-8"))
    assert validation_log.run() == expected


def test_invalid_latest_record_never_reuses_earlier_success(validation_log):
    validation_log.path.write_text(PREFIX + '{"status":"ok"}\n' + PREFIX + '{"status":')
    with pytest.raises(json.JSONDecodeError):
        validation_log.run()


def test_prefix_inside_a_long_diagnostic_line_is_not_a_result(validation_log):
    expected = {"status": "ok", "selected": "complete line"}
    content = (PREFIX + json.dumps(expected) + "\n" + "x" * 65536
               + PREFIX + '{"status":"error","error":"not a result"}\n')
    validation_log.path.write_text(content)
    assert validation_log.run() == expected


def test_failed_validation_keeps_record_error_and_raw_diagnostics(validation_log):
    validation_log.process.wait.return_value = 7
    content = "noise\n" + PREFIX + '{"status":"error","error":"fixture failed"}\ntrailing logs'
    validation_log.path.write_text(content)
    with pytest.raises(RuntimeError, match="fixture failed"):
        validation_log.run()
    assert validation_log.path.read_text() == content


def test_missing_result_retains_last_diagnostics_with_replacement_decoding(validation_log):
    validation_log.path.write_bytes(b"old noise\n" * 10000 + b"last\r\ndiagnostic\rwith replacement\xff")
    with pytest.raises(RuntimeError) as caught:
        validation_log.run()
    assert str(caught.value).endswith("last\ndiagnostic\nwith replacement\ufffd")
    assert len(str(caught.value)) == 4000
