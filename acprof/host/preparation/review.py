"""Apply model-review decisions with host-owned resolution and downloads."""
from __future__ import annotations

import copy

from acprof.models.evidence import content_digest
from acprof.models.review import review_questions
from acprof.models.spec import validate_model_spec


def apply_review(task_info, answers: dict, *, resolve_repository=None):
    """Return a new task; failed/partial decisions leave the original untouched."""
    from acprof.host.preparation.task_support import require_task_support
    from acprof.models.candidates import apply_model_candidate, discover_model_candidates

    questions = review_questions(task_info)
    if any(item.get("read_only") for item in questions):
        raise ValueError("source or selection gaps require an explicit model spec/adapter")
    expected = {item["path"] for item in questions}
    if set(answers) != expected:
        raise ValueError("answers must cover exactly the unresolved fields")
    task = copy.deepcopy(task_info)
    if set(answers) == {"pipeline_task"}:
        from acprof.host.detect import dependency_metadata, read_model_source
        from acprof.models.contract import apply_model_contract
        if answers["pipeline_task"] not in questions[0].get("options", []):
            raise ValueError("select a declared Pipeline")
        previous = task.model_resolution["contract"]
        task.model_resolution = apply_model_candidate(task, discover_model_candidates(task))
        apply_model_contract(task, lambda name: read_model_source(task.model_id, name, task.model_revision),
                             resolve_repository=resolve_repository or dependency_metadata, selected_pipeline=answers["pipeline_task"])
        task.model_resolution["contract"]["reviews"] = [*previous.get("reviews", []),
            {"revision": task.model_revision, "answers": answers, "previous_cache_key": previous["cache_key"]}]
        if task.model_resolution["contract"]["status"] == "resolved":
            require_task_support(task)
        return task
    report = task.model_resolution["contract"]
    draft = copy.deepcopy(report["draft_spec"])
    for path, value in answers.items():
        if path == "dependencies":
            from acprof.host.detect import dependency_metadata
            from acprof.models.dependencies import resolve_dependencies
            if not isinstance(value, list) or len(value) > 16 or any(
                not isinstance(item, dict) or set(item) - {"repo_id", "role", "required"}
                or type(item.get("required", True)) is not bool
                or item.get("required", True) and (not isinstance(item.get("repo_id"), str)
                                                    or not isinstance(item.get("role"), str)) for item in value
            ):
                raise ValueError("dependency decisions require a bounded list of repo_id/role/required objects")
            candidates: list[dict[str, object]] = []
            for item in value:
                if not item.get("required", True):
                    continue
                # Confirming a conditional loader doesn't change the revision
                # its source will request. Conflicts still need a declaration.
                original = [candidate for candidate in report.get("dependency_candidates", [])
                            if candidate.get("repo_id") == item["repo_id"] and candidate.get("role") in {item["role"], "unknown"}
                            and candidate.get("activation") != "inactive"
                            and candidate.get("dependency_kind") != "main_model"] or [{}]
                candidates.extend({**item, "required": "candidate", "source": "user.review",
                                   "activation": "active", "dependency_kind": "external", "loader": candidate.get("loader"),
                                   "requested_revision": candidate.get("requested_revision", "main")}
                                  for candidate in original)
            value, errors = resolve_dependencies(candidates, resolve_repository or dependency_metadata)
            if errors:
                raise ValueError("; ".join(errors))
            report["reviewed_dependency_candidates"] = candidates
        target = draft
        *parts, key = path.split(".")
        for part in parts:
            target = target.setdefault(part, {})
        target[key] = copy.deepcopy(value)
    validate_model_spec(draft)
    # Decisions are resolver inputs. Never clear unresolved fields or force a
    # final status on the previous contract.
    from acprof.host.detect import read_model_source
    from acprof.models.contract import apply_model_contract
    previous = report
    task.model_spec = draft
    task.model_resolution = apply_model_candidate(task, discover_model_candidates(task))
    def source(name):
        if name not in task.repository_sources:
            task.repository_sources[name] = read_model_source(task.model_id, name, task.model_revision)
        return task.repository_sources[name]
    apply_model_contract(task, source, resolve_repository=resolve_repository,
                         selected_pipeline=draft.get("pipeline_task"))
    report = task.model_resolution["contract"]
    report["reviews"] = [*previous.get("reviews", []),
        {"revision": task.model_revision, "answers": copy.deepcopy(answers),
         "previous_cache_key": previous["cache_key"]}]
    for path, value in answers.items():
        report["fields"][path] = {"value": copy.deepcopy(draft["dependencies"] if path == "dependencies" else value),
                                  "state": "declared", "sources": ["user.review"]}
    report["cache_key"] = content_digest({"resolved": report["cache_key"], "reviews": report["reviews"]})
    if report["status"] == "resolved":
        require_task_support(task)
    return task
