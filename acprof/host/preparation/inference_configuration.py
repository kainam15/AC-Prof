"""Preparation-only inference request review and exact-content confirmation."""
from __future__ import annotations

import copy
import json
import shlex
from pathlib import Path
from typing import Any

from acprof.artifacts import atomic_write_json, loads_finite_json
from acprof.extensions.parameters import ParameterRequest
from acprof.failures import RuntimeFailure
from acprof.models.inference_configuration import (
    MAX_CONFIGURATION_BYTES,
    freeze_inference_configuration,
    validate_frozen_configuration,
)


class InferenceConfigurationReviewRequired(RuntimeError):
    """A normal pre-download stop until a concrete candidate is confirmed."""

    def __init__(self, *, identity_sha256: str, candidate_path: Path, requests_path: Path,
                 confirmation_mismatch: bool = False):
        self.identity_sha256 = identity_sha256
        self.candidate_path = candidate_path
        self.requests_path = requests_path
        self.confirmation_mismatch = confirmation_mismatch
        super().__init__("Inference confirmation is stale; review the current configuration."
                         if confirmation_mismatch else
                         "Review and confirm the inference configuration before collection.")


def read_inference_requests(path: str | Path) -> dict:
    """Read bounded finite JSON; the shared resolver owns its request schema."""
    source = Path(path).expanduser()
    try:
        with source.open("rb") as stream:
            content = stream.read(MAX_CONFIGURATION_BYTES + 1)
        if len(content) > MAX_CONFIGURATION_BYTES:
            raise ValueError(f"request exceeds {MAX_CONFIGURATION_BYTES} bytes")
        value = loads_finite_json(content.decode("utf-8"))
        ParameterRequest("explicit", value)  # Shared finite-value and depth limits.
        if type(value) is not dict:
            raise ValueError("request document must be an object")
        return value
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"inference_config {source}: {exc}") from exc


def validate_resume_inference_requests(frozen: dict, args: Any) -> None:
    """Compare supplied intent to frozen intent without rerunning Auto rules."""
    validate_frozen_configuration(frozen)
    digest = getattr(args, "confirm_inference", "")
    if digest and digest != frozen["identity_sha256"]:
        raise ValueError("confirm_inference differs from the frozen configuration")
    path = getattr(args, "inference_config", "")
    if not path:
        return
    supplied = read_inference_requests(path)
    if type(supplied.get("schema_version")) is not int or supplied["schema_version"] != 1:
        raise ValueError("inference_config requires schema_version 1")
    pending = [("requests", supplied, frozen["requests"])]
    while pending:
        field, requested, recorded = pending.pop()
        if isinstance(recorded, dict) and "mode" not in recorded:
            if type(requested) is not dict:
                raise ValueError(f"{field} must remain an object during resume")
            unknown = set(requested) - set(recorded)
            if unknown:
                raise ValueError(f"{field} contains unknown frozen paths: {sorted(unknown)}")
            pending.extend((f"{field}.{name}", value, recorded[name])
                           for name, value in requested.items())
            continue
        if isinstance(recorded, dict) and "mode" in recorded:
            ParameterRequest.from_dict(requested)
        # JSON preserves bool/int/float distinctions as well as exact request modes.
        if json.dumps(requested, sort_keys=True, allow_nan=False) != json.dumps(
            recorded, sort_keys=True, allow_nan=False,
        ):
            raise ValueError(f"{field} differs from the frozen request; create a new experiment")


def _fields(task_info: Any) -> dict:
    return {"Model": task_info.model_id, "Revision": task_info.model_revision}


def _questions(draft: dict) -> list[dict]:
    return draft.get("questions", []) + draft.get("parameter_advanced", [])


def _field_shapes(draft: dict) -> list[dict]:
    return [{name: question.get(name) for name in ("path", "parameter_spec", "options", "available")}
            for question in _questions(draft)]


def _routing_requests(requests: dict) -> dict:
    return {name: requests.get(name) for name in ("backend", "runtime_profile")}


def _retain_answers(draft: dict, answers: dict) -> dict:
    """Keep valid request envelopes editable after a value/constraint error."""
    if type(answers) is not dict:
        return draft
    updated = dict(draft)
    for section in ("questions", "parameter_advanced"):
        updated[section] = []
        for question in draft.get(section, []):
            field = dict(question)
            if question["path"] in answers:
                try:
                    field["request"] = ParameterRequest.from_dict(answers[question["path"]]).to_dict()
                except (TypeError, ValueError):
                    pass
            updated[section].append(field)
    return updated


def _confirmed_task(task_info: Any, candidate: dict, digest: str) -> Any:
    frozen = freeze_inference_configuration(candidate, digest)
    result = copy.copy(task_info)
    result.inference_configuration = frozen
    return result


def _write_review(candidate: dict, args: Any, *, mismatch: bool) -> None:
    digest = candidate["identity_sha256"]
    root = (Path(getattr(args, "output_dir", "results")).expanduser() /
            ".inference-configurations").resolve()
    root.mkdir(parents=True, exist_ok=True)
    candidate_path = root / f"{digest}.json"
    requests_path = root / f"{digest}.requests.json"
    atomic_write_json(candidate_path, candidate)
    atomic_write_json(requests_path, candidate["requests"])
    print("Inference confirmation is stale." if mismatch else "Inference configuration is ready for review.")
    print(f"Candidate: {candidate_path}")
    print(f"Requests: {requests_path}")
    print(f"Configuration SHA-256: {digest}")
    print("Review the candidate, then rerun the same command with " + shlex.join([
        "--inference-config", str(requests_path), "--confirm-inference", digest,
    ]))
    raise InferenceConfigurationReviewRequired(
        identity_sha256=digest, candidate_path=candidate_path, requests_path=requests_path,
        confirmation_mismatch=mismatch,
    )


def prepare_inference_configuration(
    task_info: Any, args: Any, workflow: Any, *, devices: list[str],
    workload: dict | None = None,
) -> Any:
    """Review before downloads; leave runtime validation to the frozen consumers."""
    from acprof.models.configuration_resolution import (
        build_configuration_draft,
        requests_from_answers,
        resolve_inference_configuration,
    )

    path = getattr(args, "inference_config", "")
    requests = read_inference_requests(path) if path else None
    draft = build_configuration_draft(
        task_info, devices=devices, requests=requests, workload=workload,
    )
    if not workflow.interactive:
        updated, candidate = resolve_inference_configuration(
            task_info, draft["requests"], devices=devices, workload=workload,
        )
        digest = getattr(args, "confirm_inference", "")
        if digest != candidate["identity_sha256"]:
            _write_review(candidate, args, mismatch=bool(digest))
        return _confirmed_task(updated, candidate, digest)

    detail = ""
    while True:
        reply = workflow.ask(
            "resolution", "review", purpose="inference_configuration", resolved=False,
            questions=draft.get("questions", []),
            parameter_advanced=draft.get("parameter_advanced", []),
            fields=draft.get("fields", _fields(task_info)), detail=detail,
            presets=draft.get("presets", []), selected_preset=draft["requests"].get("preset", "custom"),
            summary=draft.get("summary", ""),
        )
        answers = reply.get("answers", {})
        refreshed = None
        try:
            if type(answers) is dict and set(answers) == {"$preset"}:
                preset = next((item for item in draft.get("presets", [])
                               if item["id"] == answers["$preset"]), None)
                if preset is None:
                    raise ValueError("unknown inference configuration preset")
                draft = build_configuration_draft(
                    task_info, devices=devices, requests=preset["requests"], workload=workload,
                )
                detail = ""
                continue
            requests = requests_from_answers(draft, answers)
            refreshed = build_configuration_draft(
                task_info, devices=devices, requests=requests, workload=workload,
            )
            if (_routing_requests(requests) != _routing_requests(draft["requests"])
                    or _field_shapes(refreshed) != _field_shapes(draft)):
                draft, detail = refreshed, ""
                continue
            updated, candidate = resolve_inference_configuration(
                task_info, requests, devices=devices, workload=workload,
            )
        except (TypeError, ValueError, RuntimeFailure) as exc:
            draft = refreshed if refreshed is not None else _retain_answers(draft, answers)
            detail = str(exc)
            continue

        final = workflow.ask(
            "resolution", "review", purpose="inference_confirmation", resolved=True,
            questions=[], confirmation_digest=candidate["identity_sha256"],
            fields={**_fields(updated), "Backend": candidate["backend"]["id"],
                    "Runtime Profile": candidate["backend"]["runtime_profile"],
                    "Configuration SHA-256": candidate["identity_sha256"]},
            summary=json.dumps(candidate, ensure_ascii=False, allow_nan=False, indent=2),
        )
        if final["action"] == "edit":
            draft, detail = refreshed, ""
            continue
        return _confirmed_task(updated, candidate, final.get("confirmation_digest", ""))
