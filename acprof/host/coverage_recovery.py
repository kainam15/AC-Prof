"""Recover cleanup evidence written before a coverage attempt committed its row."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from acprof.artifacts import read_json_object, recorded_artifact_present
from acprof.host.container_lifecycle import cleanup_container_ids
from acprof.models.evidence import content_digest


def read_runtime_validation(path: Path) -> dict:
    validation = read_json_object(path, label="runtime validation")
    status = validation.get("status")
    devices = validation.get("devices", {})
    if not isinstance(status, str) or not status:
        raise ValueError(f"{path}: runtime validation status must be a nonempty string")
    if not isinstance(devices, dict) or any(not isinstance(value, dict) for value in devices.values()):
        raise ValueError(f"{path}: runtime validation devices must be an object of objects")
    return validation


def validation_cleanup_errors(validation: Mapping) -> list:
    return [value for value in (validation.get("cleanup_error"), *(
        device.get("cleanup_error") for device in validation.get("devices", {}).values())) if value]


def _check_cleanup_identity(path: Path, validation: dict, errors: list, model: dict) -> None:
    from acprof.analysis.compatibility import recorded_model_identity

    if not recorded_artifact_present(path):
        raise ValueError("model resolution is missing")
    resolution = read_json_object(path, label="model resolution")
    observed = resolution.get("runtime_validation")
    if not isinstance(observed, dict) or observed.get("report_sha256") != content_digest(validation):
        raise ValueError("model resolution does not bind this runtime validation report")
    identity = recorded_model_identity({}, resolution, {})
    snapshot = resolution.get("provenance", {}).get("sources", {}).get("repository_snapshot", {})
    source = model.get("source", "huggingface")
    if ((identity.model_id, identity.commit, identity.source) != (model["model_id"], model["revision"], source)
            or snapshot.get("source") != source):
        raise ValueError("recorded primary model does not match the frozen sample")
    _, unknown_id = cleanup_container_ids(errors)
    if unknown_id:
        raise ValueError("cleanup lacks a full immutable container ID")


def interrupted_cleanup(root: Path, attempt: dict, models: list[dict]) -> dict | None:
    """Read only the unfinished model identified by the original durable journal."""
    if "active_model_index" not in attempt:
        return None
    index = attempt["active_model_index"]
    if (type(index) is not int or index not in attempt["model_indices"] or attempt["status"] == "complete"
            or any(row["model_index"] == index for row in attempt["rows"])):
        raise ValueError(f"invalid active coverage model index in attempt {attempt['attempt_id']}")
    if attempt["configuration"].get("probe") != "full":
        return None
    attempt_path = Path("attempts") / f"attempt-{attempt['attempt_id']:06d}"
    directory = root.resolve() / attempt_path / f"model-{index:04d}"
    path = directory / "runtime_validation.json"
    if directory.resolve() != directory:
        raise ValueError(f"unsafe interrupted coverage cleanup evidence path: {directory}")
    if not recorded_artifact_present(path):
        return None
    model = models[index]
    errors = []
    issue = ""
    try:
        validation = read_runtime_validation(path)
        errors = validation_cleanup_errors(validation)
        if not errors and validation.get("cleanup_status") != "incomplete":
            return None
        _check_cleanup_identity(directory / "model_resolution.json", validation, errors, model)
    except (OSError, ValueError) as exc:
        # Without a trusted report and immutable ids, only an empty owner inventory is proof.
        issue = str(exc)
    return {
        "model_index": index, "model_id": model["model_id"], "revision": model["revision"],
        **({"requested_task_id": model["task_id"]} if "task_id" in model else {}),
        "source": model.get("source", "huggingface"), "weight": model["weight"], "resolved": False,
        "supported": False, "resolution_status": "unknown", "runtime_status": "error", "failed_stage": "cleanup",
        "cleanup_status": "incomplete", "cleanup_errors": [] if issue else errors,
        "attempt_id": attempt["attempt_id"], "cleanup_origin_attempt_id": attempt["attempt_id"],
        "interrupted_cleanup_artifact": str(path.relative_to(root.resolve())),
        **({"cleanup_identity_issue": issue} if issue else {}),
    }


def validate_cleanup_confirmation(attempt: dict, previous: dict) -> dict:
    """A proof releases only the unfinished model's previously recorded debt."""
    saved = attempt["active_cleanup_recovery"]
    origin_id = previous.get("cleanup_origin_attempt_id", previous.get("attempt_id"))
    if (not isinstance(saved, dict) or type(saved.get("model_index")) is not int
            or saved["model_index"] != attempt["active_model_index"]
            or type(saved.get("origin_attempt_id")) is not int or saved["origin_attempt_id"] != origin_id
            or previous.get("cleanup_status") != "incomplete" or origin_id >= attempt["attempt_id"]):
        raise ValueError(f"cleanup confirmation in attempt {attempt['attempt_id']} differs from the original debt")
    proof = saved.get("proof")
    if not isinstance(proof, dict):
        raise ValueError("active cleanup confirmation requires a proof object")
    issue = cleanup_proof_issue(previous, proof)
    if issue:
        raise ValueError(issue)
    return proof


def cleanup_proof_issue(previous: dict, proof: object) -> str:
    """Only a proof covering every original target releases its cleanup debt."""
    if not isinstance(proof, dict) or proof.get("status") != "complete":
        return "original cleanup has not been confirmed complete"
    ids, unknown_id = cleanup_container_ids(previous.get("cleanup_errors", []))
    verified = proof.get("verified_container_ids", [])
    if not isinstance(verified, list) or any(identifier not in verified for identifier in ids):
        return "cleanup confirmation is missing an original container ID"
    if (previous.get("interrupted_cleanup_artifact") or unknown_id) and proof.get("scoped_inventory_empty") is not True:
        return "cleanup confirmation does not prove an empty original Docker owner inventory"
    return ""
