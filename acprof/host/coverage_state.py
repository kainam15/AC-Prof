"""Durable coverage attempts; the overview is rebuilt from preserved attempt records."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from acprof.artifacts import atomic_write_json
from acprof.host.run_state import utc_now
from acprof.model_evidence import content_digest


def coverage_configuration(*, probe, resources, budgets, device):
    from acprof.hf_endpoints import hf_download_mode, hf_endpoints
    from acprof.host.execution_conditions import measurement_environment
    from acprof.host.run_state import host_identity
    from acprof.host.runtime_settings import runtime_build_overrides
    from acprof.installation import resource_root

    return {"probe": probe, "resources": resources, "budgets": budgets,
            "device": device, "host": host_identity(resource_root()),
            "container_owner_uid": os.getuid(),
            "runtime_environment": measurement_environment(),
            "runtime_build_overrides": runtime_build_overrides(),
            "preparation_sources": {"download_mode": hf_download_mode(), "endpoints": hf_endpoints()},
            "preparation_environment": {name: os.environ.get(name) for name in (
                "ACPROF_MAX_DOWNLOAD", "ACPROF_MODEL_STORE",
                "ACPROF_MODEL_STORE_MAX", "ACPROF_RUNTIME_IMAGE_SOURCE", "ACPROF_RUNTIME_REGISTRY",
                "DOCKER_HOST", "DOCKER_CONTEXT")}}


def confirm_previous_cleanup(row: dict, attempts: list[dict], configuration: dict) -> dict:
    """Confirm the original Docker scope before releasing a recorded cleanup block."""
    origin_id = row.get("cleanup_origin_attempt_id", row["attempt_id"])
    origin = next(attempt["configuration"] for attempt in attempts if attempt["attempt_id"] == origin_id)
    original_host = origin.get("host", {}).get("machine_id_sha256")
    if not original_host or original_host != configuration.get("host", {}).get("machine_id_sha256"):
        raise ValueError("cleanup recovery must use the original host")
    original_uid = origin.get("container_owner_uid")
    if type(original_uid) is not int or original_uid != configuration.get("container_owner_uid"):
        raise ValueError("cleanup recovery must use the original owner UID")
    previous = origin.get("preparation_environment", {})
    current = configuration.get("preparation_environment", {})
    if any(name not in previous or previous[name] != current.get(name) for name in ("DOCKER_HOST", "DOCKER_CONTEXT")):
        raise ValueError("cleanup recovery must use the original Docker endpoint/context")
    from acprof.host.command import run_command
    from acprof.host.container_lifecycle import recover_cleanup_debt
    from acprof.host.preflight import require_native_docker
    from acprof.host.run_state import MeasurementLock

    with MeasurementLock():
        require_native_docker()
        return recover_cleanup_debt(row.get("cleanup_errors", []), run_command)


def retain_cleanup_debt(previous: dict, row: dict, attempt_id: int) -> dict:
    """Replay old attempts conservatively; only recorded confirmation releases debt."""
    result = dict(row)
    if (previous.get("cleanup_status") == "incomplete"
            and row.get("cleanup_recovery", {}).get("status") != "complete"):
        result["cleanup_status"] = "incomplete"
        result["cleanup_origin_attempt_id"] = previous.get("cleanup_origin_attempt_id", previous["attempt_id"])
        result["cleanup_errors"] = list(previous.get("cleanup_errors", [])) + [
            error for error in row.get("cleanup_errors", []) if error not in previous.get("cleanup_errors", [])]
    elif row.get("cleanup_status") == "incomplete":
        result["cleanup_origin_attempt_id"] = attempt_id
    return result


def configuration_changes(before: dict, after: dict, prefix="") -> dict:
    changes = {}
    for name in sorted(before.keys() | after.keys()):
        key = f"{prefix}.{name}" if prefix else name
        left, right = before.get(name), after.get(name)
        if isinstance(left, dict) and isinstance(right, dict):
            changes.update(configuration_changes(left, right, key))
        elif left != right:
            changes[key] = {"before": left, "after": right}
    return changes


def verified(row: dict, probe: str) -> bool:
    return (row.get("cleanup_status") != "incomplete" and not row.get("failure")
            and row.get("resolved") is True and row.get("supported") is True
            and (probe == "none" or row.get("runtime_status") == "ok"))


class CoverageState:
    """Called under ResultDirectoryLock; no attempt from an earlier invocation is overwritten."""

    def __init__(self, root: Path, sample: dict, configuration: dict, *, resume: bool,
                 retry_failed: bool, retry_stages: tuple[str, ...], retry_reasons: tuple[str, ...]):
        self.root, self.sample = root, sample
        self.sample_hash = content_digest(sample)
        self.attempt = None
        self.rows: dict[int, dict] = {}
        retry = retry_failed or bool(retry_stages or retry_reasons)
        if resume:
            self.report = json.loads((root / "coverage.json").read_text())
            if not isinstance(self.report, dict) or self.report.get("schema_version") != 2:
                raise ValueError("coverage history lacks resumable configuration evidence; choose a new directory (schema v2 required)")
            frozen = json.loads((root / "sample.json").read_text())
            if self.report.get("sample_sha256") != self.sample_hash or content_digest(frozen) != self.sample_hash:
                raise ValueError("coverage frozen sample/revisions differ from the supplied sample")
            self.attempts = self._read_attempts()
            if (self.attempts and self.report.get("configuration") != self.attempts[0]["configuration"]
                    or self.report.get("sample_count") != len(sample["models"])):
                raise ValueError("coverage original configuration or sample count differs from attempt evidence")
            for attempt in self.attempts:
                for row in attempt["rows"]:
                    index = row["model_index"]
                    self.rows[index] = retain_cleanup_debt(self.rows.get(index, {}), row, attempt["attempt_id"])
        else:
            atomic_write_json(root / "sample.json", sample)
            self.report = {"schema_version": 2, "scope": "selected_sample_only; no_formal_measurement",
                           "sample_sha256": self.sample_hash, "sample_count": len(sample["models"]),
                           "probe": configuration["probe"], "resources": configuration["resources"],
                           "budgets": configuration["budgets"], "configuration": configuration,
                           "rows": [], "attempts": [], "status": "running"}
            self.attempts = []
            atomic_write_json(root / "coverage.json", self.report)
        if not isinstance(self.report.get("configuration"), dict):
            raise ValueError("coverage history is missing the original configuration")
        last = self.attempts[-1] if self.attempts else None
        pending = last is not None and last["status"] != "complete"
        unfinished = ([] if not pending else
                      [index for index in last["model_indices"]
                       if index not in {row["model_index"] for row in last["rows"]}])
        previous = last["configuration"] if last else self.report["configuration"]
        expected = previous if pending else self.report["configuration"]
        if not retry and configuration != expected:
            raise ValueError("coverage configuration changed; use explicit failed-model retry or a new directory")
        if retry:
            indices = []
            for index, row in sorted(self.rows.items()):
                if verified(row, self.report["probe"]):
                    continue
                failure = row.get("failure") or {}
                stages = set(str(row.get("failed_stage", "")).split(",")) | {failure.get("stage", "")}
                if row.get("cleanup_status") == "incomplete":
                    stages.add("cleanup")
                reason = failure.get("reason_code", row.get("reason_code", ""))
                if retry_stages and not stages.intersection(retry_stages):
                    continue
                if retry_reasons and reason not in retry_reasons:
                    continue
                indices.append(index)
            if not indices:
                raise ValueError("no matching failed models to retry")
        else:
            indices = unfinished if pending else [index for index in range(len(sample["models"])) if index not in self.rows]
        cleanup = {index for index, row in self.rows.items() if row.get("cleanup_status") == "incomplete"}
        if cleanup and (not retry or not cleanup.issubset(indices)):
            raise ValueError("coverage cleanup is incomplete; explicitly retry the cleanup failures before continuing")
        indices.sort(key=lambda index: (index not in cleanup, index))
        self.indices = indices
        if not indices and not pending:
            return
        # Never change the validation mode within one sample overview. Resource,
        # timeout, runtime and device changes belong to an explicit retry attempt.
        if configuration["probe"] != self.report["probe"]:
            raise ValueError("coverage probe configuration changed; choose a new directory")
        attempt_root = root / "attempts"
        attempt_root.mkdir(exist_ok=True)
        identifiers = [int(path.name.removeprefix("attempt-")) for path in attempt_root.iterdir()
                       if re.fullmatch(r"attempt-\d{6}", path.name)]
        identifier = max(identifiers, default=0) + 1
        self.directory = attempt_root / f"attempt-{identifier:06d}"
        self.directory.mkdir()
        self.attempt = {"schema_version": 1, "attempt_id": identifier,
                        "sample_sha256": self.sample_hash, "configuration": configuration,
                        "configuration_sha256": content_digest(configuration),
                        "configuration_changes": configuration_changes(previous, configuration),
                        "mode": "retry" if retry else "resume" if resume else "run",
                        "retry_stages": list(retry_stages), "retry_reasons": list(retry_reasons),
                        "model_indices": indices, "started_at": utc_now(), "status": "running", "rows": []}
        self.attempts.append(self.attempt)
        self.persist_attempt()

    def _read_attempts(self) -> list[dict]:
        attempts = []
        for path in sorted((self.root / "attempts").glob("attempt-??????/attempt.json")):
            payload = json.loads(path.read_text())
            if (not isinstance(payload, dict) or payload.get("schema_version") != 1
                    or type(payload.get("attempt_id")) is not int
                    or path.parent.name != f"attempt-{payload['attempt_id']:06d}"
                    or payload.get("sample_sha256") != self.sample_hash
                    or not isinstance(payload.get("configuration"), dict)
                    or payload.get("configuration_sha256") != content_digest(payload["configuration"])
                    or payload.get("status") not in {"running", "interrupted", "complete"}
                    or not isinstance(payload.get("model_indices"), list)
                    or not isinstance(payload.get("rows"), list)):
                raise ValueError(f"invalid coverage attempt evidence: {path}")
            indices = payload["model_indices"]
            if (any(type(index) is not int or not 0 <= index < len(self.sample["models"]) for index in indices)
                    or len(indices) != len(set(indices))):
                raise ValueError(f"invalid coverage attempt selection: {path}")
            seen = set()
            for row in payload["rows"]:
                index = row.get("model_index") if isinstance(row, dict) else None
                if type(index) is not int or index not in indices or index in seen:
                    raise ValueError(f"invalid coverage attempt row: {path}")
                model = self.sample["models"][index]
                if (any(row.get(key) != model[key] for key in ("model_id", "revision", "weight"))
                        or row.get("attempt_id") != payload["attempt_id"]
                        or row.get("configuration_sha256") != payload["configuration_sha256"]
                        or row.get("attempt_path") != str(path.parent.relative_to(self.root))):
                    raise ValueError(f"coverage attempt row differs from frozen sample: {path}")
                seen.add(index)
            if payload["status"] == "complete" and seen != set(indices):
                raise ValueError(f"coverage attempt claims completion with missing rows: {path}")
            attempts.append(payload)
        recorded = self.report.get("attempts", [])
        if (not isinstance(recorded, list) or any(not isinstance(item, dict) or item.get("attempt_id") not in {
                attempt["attempt_id"] for attempt in attempts} for item in recorded)
                or not attempts and self.report.get("rows")):
            raise ValueError("coverage attempt evidence is missing; refusing to repeat verified models")
        return attempts

    def persist_attempt(self):
        if self.attempt is not None:
            atomic_write_json(self.directory / "attempt.json", self.attempt)

    def start_model(self, index: int) -> Path:
        self.attempt["active_model_index"] = index
        self.persist_attempt()
        return self.directory / f"model-{index:04d}"

    def record(self, index: int, row: dict):
        row.update(retain_cleanup_debt(self.rows.get(index, {}), row, self.attempt["attempt_id"]))
        row.update(model_index=index, attempt_id=self.attempt["attempt_id"],
                   attempt_path=str(self.directory.relative_to(self.root)),
                   configuration_sha256=self.attempt["configuration_sha256"])
        self.attempt["rows"].append(row)
        self.attempt.pop("active_model_index", None)
        self.persist_attempt()  # Durable before publishing the replaceable overview.
        self.rows[index] = row

    def finish(self, status: str):
        if self.attempt is not None:
            self.attempt.update(status=status, finished_at=utc_now())
            self.persist_attempt()

    def overview(self) -> dict:
        self.report.update(rows=[self.rows[index] for index in sorted(self.rows)], attempts=self.attempts,
                           mixed_configurations=len({attempt["configuration_sha256"] for attempt in self.attempts}) > 1)
        self.report["status"] = "complete" if len(self.rows) == len(self.sample["models"]) else "incomplete"
        if self.attempt and self.attempt["status"] != "complete":
            self.report["status"] = self.attempt["status"]
        return self.report
