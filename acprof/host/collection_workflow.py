"""Keep one collection alive through model review and preparation retries.

The child owns all model decisions. The TUI only renders bounded requests and
returns answers over stdin; there is no polling or control I/O in measurement.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys

from acprof.artifacts import atomic_write_json
from acprof.model_evidence import content_digest, pinned_revision
from acprof.preparation_events import MAX_MESSAGE, encode_event


class RebuildEnvironment(Exception):
    """The user explicitly invalidated image, input and runtime preparation."""


def retain_validation_failure(output_dir: str) -> None:
    """Keep failed evidence before a retry overwrites the current report."""
    import shutil
    import uuid
    from acprof.artifact_layout import ArtifactLayout
    layout = ArtifactLayout.discover(output_dir)
    report = layout.path("runtime_validation.json")
    if not report.is_file():
        return
    current = json.loads(report.read_text())
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
            shutil.copy2(path, archive / path.name)


class PreparationWorkflow:
    def __init__(self, *, interactive: bool = False, cache_dir: Path | None = None):
        self.interactive = interactive
        self.cache_dir = cache_dir
        self.request_id = 0
        self.rebuild_environment = False

    @classmethod
    def from_environment(cls, output_dir: str):
        return cls(interactive=os.environ.get("ACPROF_INTERACTIVE_PREPARATION") == "1",
                   cache_dir=Path(output_dir).expanduser() / ".model-contracts" / "decisions")

    def emit(self, stage: str, status: str, **fields) -> None:
        if self.interactive:
            print(encode_event(stage, status, **fields), flush=True)

    def ask(self, stage: str, kind: str, **fields) -> dict:
        self.request_id += 1
        request = {"id": self.request_id, "kind": kind, **fields}
        self.emit(stage, "waiting" if kind == "review" else "failed", request=request)
        line = sys.stdin.readline(MAX_MESSAGE + 1)
        if not line:
            raise KeyboardInterrupt("preparation controller disconnected")
        if len(line) > MAX_MESSAGE:
            raise ValueError("preparation reply exceeds size limit")
        reply = json.loads(line)
        if not isinstance(reply, dict) or type(reply.get("id")) is not int or reply["id"] != self.request_id:
            raise ValueError("stale or invalid preparation reply")
        if reply.get("action") == "cancel":
            raise KeyboardInterrupt("preparation cancelled")
        if stage == "runtime" and kind == "error" and reply.get("action") == "rebuild":
            self.rebuild_environment = True
            self.emit("runtime", "not_started")
            raise RebuildEnvironment()
        expected = "answer" if kind == "review" else "retry"
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
                detail = f"{type(exc).__name__}: {exc}"
                if not self.interactive:
                    raise
                self.ask(stage, "error", detail=detail)
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
        from acprof.model_review import review_questions
        if task.metadata_errors:
            return []
        questions = review_questions(task)
        if questions and not any(item.get("read_only") for item in questions):
            return questions
        resolution = task.model_resolution
        # Only evidence-backed, supported alternatives can be user choices.
        candidates = sorted({item["task"] for item in resolution.get("candidates", [])
                             if not item.get("unsupported_reason") and item.get("backend") != "unknown"})
        selection_only = not questions or all(item["path"] == "selection" for item in questions)
        if (selection_only and not args.task and not args.model_spec and len(candidates) > 1
                and resolution.get("status") in {"ambiguous", "needs_configuration"}):
            return [{"path": "task", "value": None, "options": candidates,
                     "reason": "; ".join(resolution.get("conflicts", []) + resolution.get("missing", []))}]
        return []

    def _apply(self, task, answers: dict, args):
        from acprof.host.detect import dependency_metadata
        from acprof.model_review import apply_review
        def lookup(repo, revision):
            return self.run("dependencies", dependency_metadata, repo, revision)
        if set(answers) != {"task"}:
            result = apply_review(task, answers, resolve_repository=lookup)
        else:
            from acprof.host.detect import read_model_source
            from acprof.model_contract import apply_model_contract
            from acprof.model_resolution import discover_model_candidates
            choices = PreparationWorkflow.questions(task, args)
            if not choices or answers["task"] not in choices[0].get("options", []):
                raise ValueError("select an available task")
            result = copy.deepcopy(task)
            result.model_resolution = discover_model_candidates(result, override_tag=answers["task"],
                                                                 override_backend=args.backend)
            apply_model_contract(result, lambda name: read_model_source(result.model_id, name, result.model_revision),
                                 override_tag=answers["task"], override_backend=args.backend,
                                 resolve_repository=lookup)
            result.model_resolution["user_decisions"] = {"revision": result.model_revision, "answers": answers}
        PreparationWorkflow._explicit(result, args)
        return result

    def resolve(self, args, *, initial=None):
        from acprof.host.detect import detect_task
        from acprof.host.task_support import require_task_support
        from acprof.host.model_inspection import explain_resolution

        def resolve_once():
            task = initial if initial is not None else detect_task(
                model_id=args.model, override_tag=args.task, override_family=args.task_family,
                override_backend=args.backend, model_spec_path=args.model_spec,
                **({"revision": args.revision} if args.revision else {}),
            )
            self._explicit(task, args)
            identity = {"model": task.model_id, "revision": task.model_revision,
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
                    saved = json.loads(cache.read_text())
                    if saved.get("schema_version") != 1 or saved.get("identity") != identity:
                        raise ValueError("decision cache identity changed")
                    restored = task
                    for answers in saved["decisions"]:
                        restored = self._apply(restored, answers, args)
                    require_task_support(restored, batch_size=args.batch_size)
                    task, decisions = restored, saved["decisions"]
                except (ValueError, KeyError, TypeError, OSError):
                    # A stale decision cannot force an incompatible route.
                    pass
            error = ""
            while self.interactive and (questions := self.questions(task, args)):
                reply = self.ask("resolution", "review", questions=questions,
                                 detail=error, summary=explain_resolution(task))
                self.emit("resolution", "running")
                try:
                    answers = reply.get("answers")
                    if not isinstance(answers, dict) or set(answers) != {item["path"] for item in questions}:
                        raise ValueError("answers must cover exactly the unresolved fields")
                    task = self._apply(task, answers, args)
                except (ValueError, TypeError) as exc:
                    error = str(exc)
                    continue
                decisions.append(answers)
                error = ""
            self._explicit(task, args)
            require_task_support(task, batch_size=args.batch_size)
            if cache and decisions:
                atomic_write_json(cache, {"schema_version": 1, "identity": identity, "decisions": decisions})
            return task

        return self.run("resolution", resolve_once)
