"""Preparation evidence must survive failure without entering measurement."""
import io
import json
import sys
from unittest.mock import patch

import pytest

import tests.support.run_recovery_fixtures as recovery
from acprof.artifact_layout import ArtifactLayout
from acprof.host.preparation.collection_workflow import PreparationWorkflow
from acprof.host.preparation.preparation_evidence import PreparationEvidence
from acprof.host.preparation.preparation_log import PreparationLog, PreparationTee, redact_text


def test_split_credentials_and_urls_are_redacted_only_after_complete_lines():
    output, console, errors = io.StringIO(), io.StringIO(), []
    log = PreparationLog(output, errors.append)
    tee = PreparationTee(console, log, "stdout")
    for part in ('Authorization: Bearer ', 'split-token', '\n{"api_key": "split', '-key"}\n',
                 'https://user:password@example.test/model?signature=private#fragment\n'):
        tee.write(part)
        tee.flush()
    log.finish()
    saved = output.getvalue()
    assert "split-token" in console.getvalue()
    for secret in ("split-token", "split-key", "user:password", "signature", "private", "fragment"):
        assert secret not in saved
    assert "https://example.test/model" in saved
    assert "[REDACTED]" in saved
    assert log.byte_size == len(saved.encode())
    assert log.status == "saved"
    assert errors == []
    tee.close()
    assert not console.closed


def test_independent_stream_fragments_crlf_and_final_partial_line():
    output = io.StringIO()
    log = PreparationLog(output, lambda exc: pytest.fail(str(exc)))
    log.feed("stdout", "Bearer ")
    log.feed("stderr", "stderr\r")
    log.feed("stderr", "\nsecond\r\n")
    log.flush()
    assert output.getvalue() == "stderr\nsecond\n"
    log.feed("stdout", "credential")
    log.finish()
    log.finish()
    assert output.getvalue() == "stderr\nsecond\nBearer [REDACTED]\n"


def test_line_and_total_limits_count_utf8_and_never_emit_partial_lines():
    output = io.StringIO()
    log = PreparationLog(output, lambda exc: pytest.fail(str(exc)), max_bytes=10, max_line_bytes=6)
    log.feed("stdout", "你好\n")
    log.feed("stderr", "汉字太")
    log.feed("stderr", "长\n")
    log.feed("stdout", "ok\n")
    log.feed("stdout", "last\n\n")
    log.finish()
    assert output.getvalue() == "你好\nok\n"
    assert log.byte_size == 10
    assert log.omitted_lines == 3
    assert log.status == "truncated"


@pytest.mark.parametrize("failure", ["write", "flush", "short"])
def test_log_io_failure_and_error_callback_cannot_replace_work(failure):
    class Broken(io.StringIO):
        def write(self, value):
            if failure == "write":
                raise OSError("log disk unavailable")
            if failure == "short":
                return super().write(value[:1])
            return super().write(value)

        def flush(self):
            if failure == "flush":
                raise OSError("log flush unavailable")
            return super().flush()

    errors = []
    def report(exc):
        errors.append(exc)
        raise RuntimeError("report callback also failed")
    log = PreparationLog(Broken(), report)
    log.feed("stdout", "diagnostic\n")
    log.finish()
    assert log.status == "failed"
    assert len(errors) == 1


def test_concurrent_capture_keeps_one_shared_byte_budget():
    import time
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    class SlowWriter(io.StringIO):
        def write(self, text):
            time.sleep(0.02)  # Release the GIL between the capacity check and write.
            return super().write(text)

    output = SlowWriter()
    log = PreparationLog(output, lambda exc: pytest.fail(str(exc)), max_bytes=3)
    ready = Barrier(8)
    def record(index):
        ready.wait(timeout=5)
        log.feed("stdout" if index % 2 else "stderr", "ok\n")
    with ThreadPoolExecutor(max_workers=8) as workers:
        list(workers.map(record, range(8)))
    log.finish()
    log.feed("stdout", "late\n")
    assert output.getvalue() == "ok\n"
    assert log.byte_size == 3
    assert log.omitted_lines == 7


def test_summary_text_truncation_preserves_utf8_and_redacts_basic_auth():
    text = redact_text("Basic secret " + "模型" * 20, limit=30)
    assert "secret" not in text
    assert len(text.encode()) <= 30
    assert text.endswith("[truncated]")


def test_noninteractive_failure_saves_stage_and_restores_observer_and_streams(tmp_path):
    workflow = PreparationWorkflow()
    observed = []
    def previous(stage, status, fields):
        observed.append((stage, status))
    workflow.observer = previous
    evidence = PreparationEvidence(ArtifactLayout.for_new_run(tmp_path), "failure")
    stdout, stderr = sys.stdout, sys.stderr
    failure = RuntimeError("download https://user:pass@example.test/model?token=private")
    with pytest.raises(RuntimeError, match="download") as caught, evidence.capture(workflow):
        workflow.run("image", lambda: (_ for _ in ()).throw(failure))
    assert caught.value is failure
    assert sys.stdout is stdout
    assert sys.stderr is stderr
    assert workflow.observer is previous
    assert observed == [("image", "running"), ("image", "failed")]
    summary = json.loads(evidence.summary_path.read_text())
    assert summary["status"] == "failed"
    assert summary["last_stage"] == "image"
    assert summary["last_stage_status"] == "failed"
    assert summary["reason_code"] == "preparation_failed"
    assert summary["last_failure"]["type"] == "RuntimeError"
    assert summary["log"]["status"] == "saved"
    assert "private" not in evidence.summary_path.read_text()
    assert "user:pass" not in evidence.log_path.read_text()


@pytest.mark.parametrize(("reply", "reason"), [
    ('{"id":1,"action":"cancel"}\n', "cancelled_by_user"),
    ("", "controller_disconnected"),
])
def test_controller_cancel_and_disconnect_have_distinct_persisted_reasons(tmp_path, reply, reason):
    workflow = PreparationWorkflow(interactive=True)
    evidence = PreparationEvidence(ArtifactLayout.for_new_run(tmp_path), reason)
    with patch("sys.stdin", io.StringIO(reply)), pytest.raises(KeyboardInterrupt), evidence.capture(workflow):
        workflow.ask("runtime", "error", detail="validation failed")
    summary = json.loads(evidence.summary_path.read_text())
    assert summary["status"] == "cancelled"
    assert summary["reason_code"] == reason
    assert summary["last_stage"] == "runtime"


def test_failed_evidence_publication_preserves_original_exception(tmp_path):
    workflow = PreparationWorkflow()
    evidence = PreparationEvidence(ArtifactLayout.for_new_run(tmp_path), "io-failure")
    original = ValueError("original preparation error")
    stdout, stderr = sys.stdout, sys.stderr
    with patch("acprof.host.preparation.preparation_evidence.atomic_write_json", side_effect=OSError("disk unavailable")):
        with pytest.raises(ValueError, match="original preparation error") as caught, evidence.capture(workflow):
            raise original
    assert caught.value is original
    assert sys.stdout is stdout
    assert sys.stderr is stderr
    assert workflow.observer is None
    assert evidence.data["evidence_errors"] == ["OSError: disk unavailable"]


def test_log_finalization_failure_does_not_mask_preparation_error(tmp_path):
    evidence = PreparationEvidence(ArtifactLayout.for_new_run(tmp_path), "finalize-failure")
    original = RuntimeError("preparation operation failed")
    with patch.object(PreparationLog, "finish", side_effect=OSError("log finalization failed")):
        with pytest.raises(RuntimeError, match="preparation operation failed") as caught:
            with evidence.capture(PreparationWorkflow()):
                raise original
    assert caught.value is original
    summary = json.loads(evidence.summary_path.read_text())
    assert summary["status"] == "failed"
    assert summary["log"]["status"] == "failed"
    assert summary["evidence_errors"] == ["OSError: log finalization failed"]


class TestPreparationIntegration:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path):
        self.run = recovery.RunRecoveryFixture()
        self.run.build(request, tmp_path)

    def summary(self):
        state = json.loads((self.run.directory / ".acprof/run_state.json").read_text())
        attempt = state["attempts"][-1]
        path = self.run.directory / attempt["preparation_evidence"]
        return state, path, json.loads(path.read_text())

    def test_failed_attempt_is_preserved_and_resume_gets_new_evidence(self):
        def fail(*args, **kwargs):
            print("preparation failure evidence")
            raise RuntimeError("fixture image failed")
        with pytest.raises(SystemExit) as caught:
            self.run.invoke(image=fail)
        assert caught.value.code == 1
        first_state, first_path, first = self.summary()
        first_log = self.run.directory / first["log"]["path"]
        saved = first_log.read_bytes()
        assert first["status"] == first_state["status"] == "failed"
        assert first["last_stage"] == "image"
        assert self.run.calls == []
        self.run.invoke("--resume")
        state, second_path, second = self.summary()
        assert second_path != first_path
        assert second["attempt_id"] != first["attempt_id"]
        assert first_log.read_bytes() == saved
        assert second["status"] == "complete"
        assert state["status"] == "complete"

    def test_capture_is_closed_before_matrix_output(self):
        def image(*args, **kwargs):
            print("PREPARATION_SENTINEL")
            return self.run.image
        def case(request, **kwargs):
            print("MEASUREMENT_SENTINEL")
            return self.run.write_case(request, **kwargs)
        self.run.invoke(image=image, case=case)
        _, _, summary = self.summary()
        text = (self.run.directory / summary["log"]["path"]).read_text()
        assert "PREPARATION_SENTINEL" in text
        assert "MEASUREMENT_SENTINEL" not in text
        assert summary["status"] == "complete"
        assert summary["log"]["scope"] == "host_python_stdout_stderr"

    def test_rebuild_uses_one_attempt_and_retains_original_error(self):
        calls = []
        def image(*args, **kwargs):
            calls.append(kwargs["reuse_existing"])
            if len(calls) == 1:
                raise RuntimeError("rebuild fixture")
            return self.run.image
        with patch.dict("os.environ", {"ACPROF_INTERACTIVE_PREPARATION": "1"}), patch(
            "sys.stdin", io.StringIO('{"id":1,"action":"rebuild"}\n'),
        ):
            self.run.invoke(image=image)
        state, _, summary = self.summary()
        assert calls == [True, False]
        assert len(state["attempts"]) == 1
        assert summary["status"] == "complete"
        assert summary["stage_attempts"]["image"] == 2
        assert summary["last_failure"]["message"] == "rebuild fixture"
