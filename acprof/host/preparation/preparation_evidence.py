"""Attempt-scoped preparation evidence, closed before formal measurement."""
from __future__ import annotations

import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, TextIO, cast

from acprof.artifacts import atomic_write_json
from acprof.host.preparation.collection_workflow import PreparationWorkflow
from acprof.host.preparation.preparation_log import PreparationLog, PreparationTee, redact_text
from acprof.results.layout import ArtifactLayout


def _now() -> str:
    return datetime.now(UTC).isoformat()


class PreparationEvidence:
    """Best-effort diagnostics never replace the original preparation failure."""

    def __init__(self, layout: ArtifactLayout, attempt_id: str) -> None:
        self.summary_path = layout.contained(f"metadata/preparation/{attempt_id}.json")
        self.log_path = layout.contained(f"logs/preparation/{attempt_id}.log")
        self.relative_summary_path = self.summary_path.relative_to(layout.root).as_posix()
        self.log: PreparationLog | None = None
        self._log_stream: TextIO | None = None
        self._log_finalize_failed = False
        self._stderr: TextIO = sys.stderr
        self.data: dict[str, Any] = {
            "schema_version": 1, "scope": "preparation", "attempt_id": attempt_id,
            "status": "running", "started_at": _now(), "last_stage": None,
            "last_stage_status": None, "stage_attempts": {}, "reason_code": None,
            "exception": None, "last_failure": None, "evidence_errors": [],
            "log": {"path": self.log_path.relative_to(layout.root).as_posix(),
                    "scope": "host_python_stdout_stderr", "status": "unavailable",
                    "truncated": False, "omitted_lines": 0, "retained_bytes": 0},
            "recovery": {"retryability": "unknown"},
        }

    def _error(self, exc: Exception) -> None:
        detail = f"{type(exc).__name__}: {redact_text(str(exc))}"
        errors = self.data["evidence_errors"]
        if detail in errors or len(errors) >= 8:
            return
        errors.append(detail)
        try:
            self._stderr.write(f"[WARN] Preparation evidence is incomplete: {detail}\n")
            self._stderr.flush()
        except Exception:  # noqa: S110 - original failure is already in evidence_errors; stderr may be closed
            pass  # The original error and normal process cleanup still own the outcome.

    def _publish(self) -> None:
        self.data["updated_at"] = _now()
        if self.log is not None:
            self.data["log"].update(status="failed" if self._log_finalize_failed else self.log.status,
                                    truncated=bool(self.log.omitted_lines),
                                    omitted_lines=self.log.omitted_lines,
                                    retained_bytes=self.log.byte_size)
        try:
            atomic_write_json(self.summary_path, self.data)
        except Exception as exc:
            self._error(exc)

    def _observe(self, stage: str, status: str, fields: Mapping[str, object]) -> None:
        # Persist only explicitly selected values, never raw request fields or environments.
        self.data["last_stage"] = redact_text(stage, limit=128)
        self.data["last_stage_status"] = redact_text(status, limit=64)
        attempts = self.data["stage_attempts"]
        key = self.data["last_stage"]
        if status == "running" and (key in attempts or len(attempts) < 16):
            attempts[key] = attempts.get(key, 0) + 1
        if fields.get("exception_type"):
            self.data["last_failure"] = {
                "stage": self.data["last_stage"],
                "type": redact_text(str(fields["exception_type"]), limit=128),
                "message": redact_text(str(fields.get("exception_message", ""))),
            }
        if self.log is not None:
            self.log.flush()
        self._publish()

    @contextmanager
    def capture(self, workflow: PreparationWorkflow) -> Iterator[None]:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        self._stderr = original_stderr
        previous = workflow.observer
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            stream = self.log_path.open("x", encoding="utf-8", errors="backslashreplace")
            try:
                os.fchmod(stream.fileno(), 0o600)
            except Exception:
                stream.close()
                raise
            self._log_stream = stream
            self.log = PreparationLog(stream, self._error)
        except Exception as exc:
            self._error(exc)

        def observe(stage: str, status: str, fields: Mapping[str, object]) -> None:
            self._observe(stage, status, fields)
            if previous is not None:
                previous(stage, status, fields)

        workflow.observer = observe
        self._publish()
        if self.log is not None:
            sys.stdout = cast(TextIO, PreparationTee(original_stdout, self.log, "stdout"))
            sys.stderr = cast(TextIO, PreparationTee(original_stderr, self.log, "stderr"))
        try:
            yield
        except BaseException as exc:
            self.data["status"] = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
            self.data["reason_code"] = (
                getattr(exc, "reason_code", "interrupted") if isinstance(exc, KeyboardInterrupt)
                else "preparation_failed"
            )
            self.data["exception"] = {
                "type": type(exc).__name__, "message": redact_text(str(exc)),
            }
            if self.log is not None:
                self.log.feed("stderr", f"\n[{self.data['reason_code']}] {type(exc).__name__}: {exc}\n")
            raise
        else:
            self.data["status"] = "complete"
            self.data["reason_code"] = "preparation_complete"
        finally:
            # No proxy, file, observer or periodic callback survives into run_matrix.
            sys.stdout, sys.stderr = original_stdout, original_stderr
            workflow.observer = previous
            if self.log is not None:
                try:
                    self.log.finish()
                except Exception as error:
                    self._log_finalize_failed = True
                    self._error(error)
            if self._log_stream is not None:
                try:
                    try:
                        self._log_stream.flush()
                        os.fsync(self._log_stream.fileno())
                    finally:
                        self._log_stream.close()
                except Exception as error:
                    self._log_finalize_failed = True
                    self._error(error)
            self.data["ended_at"] = _now()
            self._publish()
