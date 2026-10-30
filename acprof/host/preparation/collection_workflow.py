"""Keep one collection alive through model review and preparation retries.

The child owns all model decisions. The TUI only renders bounded requests and
returns answers over stdin; there is no polling or control I/O in measurement.
"""
from __future__ import annotations

import copy
import os
import sys
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from acprof.artifacts import atomic_write_json, loads_finite_json, read_json_object, sync_directory
from acprof.events.preparation import MAX_MESSAGE, encode_event
from acprof.host.preparation.model_errors import ModelLookupError
from acprof.models.evidence import content_digest, pinned_revision


class RebuildEnvironment(Exception):
    """The user explicitly invalidated image, input and runtime preparation."""


class PreparationInterrupted(KeyboardInterrupt):
    """A specific control-plane interruption, still handled as KeyboardInterrupt."""

    def __init__(self, message: str, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def retain_validation_failure(output_dir: str) -> None:
    """Keep failed evidence before a retry overwrites the current report."""
    import shutil
    import uuid

    from acprof.artifact_layout import ArtifactLayout
    layout = ArtifactLayout.discover(output_dir)
    report = layout.path("runtime_validation.json")
    if not report.is_file():
        return
    current = read_json_object(report, label="runtime validation")
    if current.get("status") == "ok":
        return
    logs = layout.path("logs")
    archive = logs / "runtime-attempts" / uuid.uuid4().hex
    archive.mkdir(parents=True)
    files = [report, layout.path("model_resolution.json"), layout.path("input_scale_plan.json")]
    files.extend(logs / f"runtime_validation_{device}.log" for device in ("off", "on")
                 if device in current.get("devices", {}))
    for path in files:
        if path.is_file():
            destination = archive / path.name
            shutil.copy2(path, destination)
            with destination.open("rb") as stream:
                os.fsync(stream.fileno())
    for directory in (archive, archive.parent, logs, layout.root):
        sync_directory(directory)


class PreparationWorkflow:
    def __init__(self, *, interactive: bool = False, cache_dir: Path | None = None):
        self.interactive = interactive
        self.cache_dir = cache_dir
        self.request_id = 0
        self.rebuild_environment = False
        self.observer: Callable[[str, str, Mapping[str, object]], None] | None = None
        self.continue_on_task_failure = False

    @classmethod
    def from_environment(cls, output_dir: str):
        return cls(interactive=os.environ.get("ACPROF_INTERACTIVE_PREPARATION") == "1",
                   cache_dir=Path(output_dir).expanduser() / ".model-contracts" / "decisions")

    def emit(self, stage: str, status: str, **fields) -> None:
        if self.observer is not None:
            self.observer(stage, status, fields)
        if self.interactive:
            print(encode_event(stage, status, **fields), flush=True)

    def ask(self, stage: str, kind: str, **fields) -> dict:
        self.request_id += 1
        request = {"id": self.request_id, "kind": kind, **fields}
        self.emit(stage, "waiting" if kind == "review" else "failed", request=request)
        line = sys.stdin.readline(MAX_MESSAGE + 1)
        if not line:
            raise PreparationInterrupted("preparation controller disconnected", "controller_disconnected")
        if len(line) > MAX_MESSAGE:
            raise ValueError("preparation reply exceeds size limit")
        reply = loads_finite_json(line)
        if not isinstance(reply, dict) or type(reply.get("id")) is not int or reply["id"] != self.request_id:
            raise ValueError("stale or invalid preparation reply")
        if reply.get("action") == "cancel":
            raise PreparationInterrupted("preparation cancelled", "cancelled_by_user")
        if stage in {"image", "runtime"} and kind == "error" and reply.get("action") == "rebuild":
            self.rebuild_environment = True
            for invalidated in ("image", "input", "runtime"):
                self.emit(invalidated, "not_started")
            raise RebuildEnvironment()
        expected = ("confirm" if fields.get("resolved") else "answer") if kind == "review" else "retry"
        if reply.get("action") != expected:
            raise ValueError("unexpected preparation action")
        return reply

    def run(self, stage: str, operation, *args, **kwargs):
        while True:
            self.emit(stage, "running")
            try:
                value = operation(*args, **kwargs)
            except (Exception, SystemExit) as exc:
                if isinstance(exc, SystemExit) and exc.code in (0, None):
                    raise
                if self.observer is not None:
                    self.observer(stage, "failed", {"exception_type": type(exc).__name__,
                                                     "exception_message": str(exc)})
                detail = traceback.format_exc()[-24000:]
                if not self.interactive or self.continue_on_task_failure and stage == "runtime":
                    raise
                fields: dict[str, Any] = {"model_error": exc.to_dict()} if isinstance(exc, ModelLookupError) else {}
                from acprof.hub.huggingface.download import HfDownloadError, exception_chain
                download_error = next((cause.to_dict() for cause in exception_chain(exc)
                                       if isinstance(cause, HfDownloadError)), None)
                if download_error:
                    fields["download_error"] = download_error
                failure = getattr(exc, "failure", None)
                fields["failed_stage"] = getattr(failure, "stage", getattr(exc, "stage", stage))
                if failure is not None:
                    fields["failure"] = failure.to_dict()
                self.ask(stage, "error", detail=detail, **fields)
            else:
                self.emit(stage, "passed")
                return value

    @staticmethod
    def _explicit(task, args) -> None:
        for option, attribute in (("task", "pipeline_tag"), ("task_family", "task_family"),
                                  ("backend", "runtime_backend")):
            value = getattr(args, option, None)
            if value and getattr(task, attribute) != value:
                raise ValueError(f"resolved {attribute} conflicts with explicit --{option.replace('_', '-')}: {value}")

    @staticmethod
    def questions(task, args) -> list[dict]:
        from acprof.models.review import review_questions
        if task.metadata_errors:
            return []
        questions = review_questions(task)
        if any(item["path"] == "dependencies" and item.get("read_only") for item in questions):
            return []  # Access failures are retried, not accepted as user choices.
        if questions and not any(item.get("read_only") for item in questions):
            return questions
        resolution = task.model_resolution
        # Executable alternatives are handled once by select_tasks. Conflicting
        # observations are configuration problems, not a second task selector.
        if questions or resolution.get("status") in {"ambiguous", "needs_configuration"}:
            return [{"path": "model_spec", "value": "", "kind": "path",
                     "reason": "需要本地模型声明以补充接口信息。"}]
        return []

    @staticmethod
    def review_details(task, questions) -> dict:
        pending = {item["path"] for item in questions}
        values = {"Model": task.model_id, "Revision": task.model_revision,
                  "Task": task.pipeline_tag, "Backend": task.runtime_backend,
                  "Adapter / Interface": task.model_adapter}
        unresolved = task.model_resolution.get("status") in {"ambiguous", "needs_configuration", "selection_required"}
        if unresolved:
            for name in ("Task", "Backend", "Adapter / Interface"):
                values.pop(name)
        fields = {key: value for key, value in values.items() if value and value != "unknown" and key.lower() not in pending}
        summary = ("推理接口信息不足或相互矛盾，需要补充配置。" if questions else
                   "推理接口已确定，接下来将验证模型加载、推理与输出。")
        return {"fields": fields, "summary": summary}

    @staticmethod
    def _with_model_spec(task, path, args):
        from acprof.host.detect import detect_task
        if not isinstance(path, str) or not path:
            raise ValueError("model specification path is required")
        updated = copy.copy(args)
        updated.model_spec = path
        resolved = detect_task(model_id=task.model_id, revision=task.model_revision,
                              override_tag=updated.task, override_family=updated.task_family,
                              override_backend=updated.backend, model_spec_path=updated.model_spec)
        # Persist only resolver inputs, never write a final contract from the UI.
        args.model_spec = path
        return resolved

    def _apply(self, task, answers: dict, args):
        from acprof.host.detect import dependency_metadata
        from acprof.host.preparation.review import apply_review
        def lookup(repo, revision):
            return self.run("dependencies", dependency_metadata, repo, revision)
        if set(answers) == {"model_spec"}:
            result = self._with_model_spec(task, answers["model_spec"], args)
        else:
            result = apply_review(task, answers, resolve_repository=lookup)
        result.selected_tasks = []  # Reviewed configuration needs a fresh execution identity.
        PreparationWorkflow._explicit(result, args)
        return result

    def resolve(self, args, *, initial=None):
        from acprof.host.detect import detect_task
        from acprof.host.preparation.task_support import require_task_support

        def resolve_once():
            task = initial if initial is not None else detect_task(
                model_id=args.model, override_tag=args.task, override_family=args.task_family,
                override_backend=args.backend, model_spec_path=args.model_spec,
                **({"revision": args.revision} if args.revision else {}),
            )
            self._explicit(task, args)
            from acprof.host.preparation.task_selection import (
                review_selected_task_parameters,
                select_tasks,
            )
            task = select_tasks(task, args, self)
            self._explicit(task, args)
            if len(task.selected_tasks) > 1:
                # The entire collection is ready; only now ask for workload
                # parameters (once), not during repeated interface resolution.
                return review_selected_task_parameters(task, args, self)
            identity = {"model": task.model_id, "revision": task.model_revision,
                        "source": getattr(task, "model_source", "huggingface"),
                        "contract": task.model_resolution.get("contract", {}).get("cache_key"),
                        "evidence": task.model_resolution.get("provenance", {}).get("identity_sha256"),
                        "explicit": {name: getattr(args, name, None) for name in ("task", "task_family", "backend", "model_spec")}}
            cache = (self.cache_dir / (content_digest(identity) + ".json")
                     if self.interactive and self.cache_dir and initial is None and pinned_revision(task.model_revision)
                     and (identity["contract"] or identity["evidence"]) else None)
            decisions = []
            if cache and cache.is_file():
                try:
                    if cache.stat().st_size > MAX_MESSAGE:
                        raise ValueError("decision cache exceeds size limit")
                    saved = read_json_object(cache, label="decision cache")
                    if saved.get("schema_version") != 1 or saved.get("identity") != identity:
                        raise ValueError("decision cache identity changed")
                    restored = task
                    for answers in saved["decisions"]:
                        restored = self._apply(restored, answers, args)
                    restored = select_tasks(restored, args, self)
                    require_task_support(restored, batch_size=args.batch_size, devices=args.gpus.split(","))
                    task, decisions = restored, saved["decisions"]
                except (ValueError, KeyError, TypeError, OSError):
                    # A stale decision cannot force an incompatible route.
                    pass
            error = ""
            reviewed = False
            while self.interactive:
                questions = self.questions(task, args)
                if not questions and not reviewed:
                    break
                if not questions:
                    require_task_support(task, batch_size=args.batch_size, devices=args.gpus.split(","))
                reply = self.ask("resolution", "review", questions=questions,
                                 detail=error, resolved=not questions, **self.review_details(task, questions))
                if reply.get("action") == "confirm":
                    if questions:
                        raise ValueError("unresolved model cannot be confirmed")
                    break
                self.emit("resolution", "running")
                try:
                    answers = reply.get("answers")
                    if not isinstance(answers, dict) or set(answers) != {item["path"] for item in questions}:
                        raise ValueError("answers must cover exactly the unresolved fields")
                    task = select_tasks(self._apply(task, answers, args), args, self)
                except (ValueError, TypeError) as exc:
                    error = str(exc)
                    continue
                decisions.append(answers)
                reviewed, error = True, ""
            self._explicit(task, args)
            require_task_support(task, batch_size=args.batch_size, devices=args.gpus.split(","))
            if cache and decisions:
                atomic_write_json(cache, {"schema_version": 1, "identity": identity, "decisions": decisions})
            return review_selected_task_parameters(task, args, self)

        return self.run("resolution", resolve_once)
