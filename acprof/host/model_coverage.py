"""Frozen model samples and coverage reports, independent of formal measurements."""
from __future__ import annotations

import logging
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from acprof.artifacts import atomic_write_json, read_json_object
from acprof.model_evidence import pinned_revision


def validate_sample(sample: dict) -> None:
    from huggingface_hub.utils import validate_repo_id
    if not isinstance(sample, dict) or sample.get("schema_version") != 1:
        raise ValueError("unsupported coverage sample schema")
    if not all(isinstance(sample.get(key), str) and sample[key] for key in ("sampling", "weight_basis")):
        raise ValueError("sample requires sampling and weight_basis")
    if not isinstance(sample.get("models"), list) or not sample["models"]:
        raise ValueError("coverage sample must contain models")
    seen = set()
    for item in sample["models"]:
        if not isinstance(item, dict):
            raise ValueError("model entries must be objects")
        validate_repo_id(item["model_id"])
        if not pinned_revision(item.get("revision")):
            raise ValueError("coverage requires full commit SHAs")
        identity = (item["model_id"], item["revision"])
        if identity in seen:
            raise ValueError("duplicate model/revision in sample")
        seen.add(identity)
        weight = item.get("weight")
        if type(weight) not in {float, int} or not math.isfinite(weight) or weight < 0:
            raise ValueError("coverage weights must be finite nonnegative numbers")
        reference = item.get("semantic_reference")
        if reference is not None and (not isinstance(reference, dict) or not isinstance(reference.get("task"), str)
                                      or not reference["task"] or not reference.get("source")):
            raise ValueError("semantic_reference needs an independently reviewed task and source")


def snapshot_sample(strata: list[str], limit: int) -> dict:
    from huggingface_hub import HfApi

    from acprof.hf_endpoints import hf_endpoints
    if limit <= 0:
        raise ValueError("limit must be positive")
    api = HfApi(endpoint=hf_endpoints()[0])
    models = {}
    for stratum in strata:
        task, separator, library = stratum.partition(":")
        if not separator or not task or not library:
            raise ValueError("stratum must be TASK:LIBRARY")
        selected = 0
        for info in api.list_models(pipeline_tag=task, filter=library, sort="downloads", limit=limit * 20, full=True):
            # Hub tag filtering can include wrappers tagged with multiple libraries.
            if info.library_name != library or info.pipeline_tag != task:
                continue
            revision = info.sha
            if not pinned_revision(revision):
                revision = api.model_info(info.id).sha
            models[(info.id, revision)] = {"model_id": info.id, "revision": revision,
                                          "weight": info.downloads or 0, "stratum": stratum}
            selected += 1
            if selected == limit:
                break
    sample = {"schema_version": 1, "sampling": "head_per_task_library; selected_sample_only",
              "captured_at": datetime.now(timezone.utc).isoformat(),
              "weight_basis": "Hub downloads at snapshot time; not a whole-Hub estimate",
              "strata": strata, "limit_per_stratum": limit, "models": list(models.values())}
    validate_sample(sample)
    return sample


def summarize(rows: list[dict], *, probe_requested: bool, sample_weight: float) -> dict:
    def weight(predicate):
        return sum(row["weight"] for row in rows if predicate(row))

    def rate(value, denominator=sample_weight):
        return value / denominator if denominator else None

    reviewed = [row for row in rows if row.get("semantic_correct") is not None]
    reviewed_weight = sum(row["weight"] for row in reviewed)
    return {
        "sample_weight": sample_weight, "completed_count": len(rows),
        "weighted_resolution_rate": rate(weight(lambda row: row["resolved"])),
        "weighted_support_rate": rate(weight(lambda row: row["supported"])),
        "weighted_abstain_rate": rate(weight(lambda row: row["resolution_status"] == "abstained")),
        "weighted_runtime_success_rate": rate(weight(lambda row: row["runtime_status"] == "ok")) if probe_requested else None,
        "weighted_resource_limited_rate": rate(weight(lambda row: row["runtime_status"] == "resource_limited" or
            (row.get("failure") or {}).get("reason_code") == "resource_limit")) if probe_requested else None,
        "weighted_access_denied_rate": rate(weight(lambda row: row["resolution_status"] == "access_denied")) if probe_requested else None,
        "semantic_reviewed_weight": reviewed_weight,
        "semantic_accuracy_on_reviewed_selections": rate(sum(row["weight"] for row in reviewed if row["semantic_correct"]), reviewed_weight),
        "semantic_wrong_selections": sum(row["semantic_correct"] is False for row in reviewed),
        "runtime_counts": dict(Counter(row["runtime_status"] for row in rows)),
        "failure_stages": dict(Counter(row["failed_stage"] for row in rows if row.get("failed_stage"))),
    }


class CoverageCleanupError(RuntimeError):
    """The failed model is durable, but an owned container prevents further work."""


_LOG = logging.getLogger(__name__)


def _read_runtime_validation_recovery(path: Path) -> dict:
    """Read only well-formed runtime evidence when preserving a primary failure."""
    validation = read_json_object(path, label="runtime validation")
    status = validation.get("status")
    devices = validation.get("devices", {})
    if not isinstance(status, str) or not status:
        raise ValueError(f"{path}: runtime validation status must be a nonempty string")
    if not isinstance(devices, dict) or any(not isinstance(value, dict) for value in devices.values()):
        raise ValueError(f"{path}: runtime validation devices must be an object of objects")
    return validation


def _evaluate_model(item: dict, output: Path, *, probe, cpus, memory_gb, gpu,
                    timeout_seconds, max_parameters, max_download_bytes) -> dict:
    from acprof.host.detect import detect_task
    from acprof.host.model_inspection import validate_model_runtime
    from acprof.host.task_support import require_task_support
    from acprof.model_contract import write_model_resolution

    row = {"model_id": item["model_id"], "revision": item["revision"], "weight": item["weight"],
           "resolved": False, "supported": False, "resolution_status": "error",
           "runtime_status": "not_requested" if probe == "none" else "not_run", "semantic_correct": None}
    stage = "resolution"
    task = None
    validation = None
    try:
        task = detect_task(item["model_id"], revision=item["revision"])
        if task.model_revision != item["revision"]:
            raise ValueError("resolver returned a different snapshot")
        write_model_resolution(task, output)
        row["selected_task"] = task.pipeline_tag
        row["resolved"] = task.model_resolution.get("status") not in {"ambiguous", "needs_configuration"}
        row["resolution_status"] = "resolved" if row["resolved"] else "abstained"
        if row["resolved"] and item.get("semantic_reference"):
            row["semantic_correct"] = task.pipeline_tag == item["semantic_reference"]["task"]
        stage = "support"
        require_task_support(task, devices=("gpu" if gpu else "cpu",) if probe == "full" else ())
        row["runtime_profile"] = task.runtime_profile_id
        write_model_resolution(task, output)
        row["supported"] = True
        if max_parameters is not None or max_download_bytes is not None:
            from acprof.resource_budget import assess_resource_budget
            plan = None
            if max_download_bytes is not None:
                from acprof.host.model_store import plan_model
                plan = plan_model(task)
            budget = assess_resource_budget(parameter_count=task.parameter_count, max_parameters=max_parameters,
                                            plan=plan, max_download_bytes=max_download_bytes)
            row["resource_preflight"] = budget
            if budget.get("failure"):
                from acprof.failures import Failure, RuntimeFailure
                raise RuntimeFailure(Failure(**budget["failure"]))
        if probe == "full":
            from acprof.host.automation import check_repository_access
            from acprof.model_spec import task_model_spec
            stage = "access"
            for repo_id in [task.model_id, *(dep["repo_id"] for dep in task_model_spec(task).get("dependencies", []))]:
                check_repository_access(repo_id)
            stage = "runtime"
            validation = validate_model_runtime(task, output, cpus=cpus, memory_gb=memory_gb,
                                                gpu=gpu, timeout_seconds=timeout_seconds, reuse_existing=True)
            row["runtime_status"] = validation["status"]
            row["quality_checks"] = [check for value in validation.get("devices", {}).values() for check in value.get("quality_checks", [])]
            failures = [value["failure"] for value in validation.get("devices", {}).values() if value.get("failure")]
            if failures:
                row["failure"] = failures[0]
    except (Exception, SystemExit) as exc:
        from acprof.failures import Failure, compatibility_status, failure_from_exception
        failure = failure_from_exception(exc, stage=getattr(exc, "stage", stage), device="gpu" if gpu else "cpu",
                                         runtime_profile=row.get("runtime_profile", ""))
        row["failure"] = failure.to_dict()
        row["reason_code"] = failure.reason_code
        if probe == "full":
            row["runtime_status"] = compatibility_status(failure)
        row.update(failed_stage=failure.stage, error_type=type(exc).__name__)
        if stage == "access":
            row["resolution_status"] = compatibility_status(failure)
        if stage == "runtime":
            path = output / "runtime_validation.json"
            row["runtime_status"] = "error"
            if path.is_file():
                try:
                    recovered_validation = _read_runtime_validation_recovery(path)
                    devices = recovered_validation.get("devices", {})
                    quality_checks = []
                    structured = []
                    failed_stages = []
                    for device in devices.values():
                        checks = device.get("quality_checks", [])
                        if not isinstance(checks, list) or any(not isinstance(check, dict) for check in checks):
                            raise ValueError(f"{path}: runtime validation quality_checks must be lists of objects")
                        quality_checks.extend(checks)
                        recorded_failure = device.get("failure")
                        if recorded_failure is not None:
                            if not isinstance(recorded_failure, dict):
                                raise ValueError(f"{path}: runtime validation failure must be an object")
                            structured.append(Failure(**recorded_failure).to_dict())
                        failed_stage = device.get("failed_stage")
                        if failed_stage is not None:
                            if not isinstance(failed_stage, str) or not failed_stage:
                                raise ValueError(f"{path}: runtime validation failed_stage must be a nonempty string")
                            failed_stages.append(failed_stage)
                except (OSError, TypeError, ValueError) as evidence_error:
                    _LOG.warning(
                        "runtime validation recovery evidence is unusable; preserving primary failure: %s",
                        evidence_error,
                    )
                else:
                    validation = recovered_validation
                    row["quality_checks"] = quality_checks
                    row["runtime_status"] = validation["status"]
                    if structured:
                        row["failure"] = structured[0]
                        row["reason_code"] = structured[0]["reason_code"]
                        row["runtime_status"] = compatibility_status(structured[0])
                    row["failed_stage"] = ",".join(failed_stages) or stage
    if validation is not None and validation.get("cleanup_status") == "incomplete":
        row["cleanup_status"] = "incomplete"
        row["cleanup_errors"] = [value for value in [validation.get("cleanup_error"), *(
            device.get("cleanup_error") for device in validation.get("devices", {}).values())] if value]
    if row["runtime_status"] == "resource_limited":
        row["failed_stage"] = "resource"
    if task is not None:
        write_model_resolution(task, output)
    return row


def run_sample(sample: dict, root: Path, *, probe: str = "none", cpus: int = 2, memory_gb: int = 4,
               gpu: bool = False, timeout_seconds: float = 300,
               max_parameters: int | None = None, max_download_bytes: int | None = None,
               resume: bool = False, retry_failed: bool = False,
               retry_stages=(), retry_reasons=()) -> dict:
    from contextlib import nullcontext

    from acprof.host.coverage_state import (
        CoverageState,
        confirm_previous_cleanup,
        coverage_configuration,
    )
    from acprof.host.gpu_device import gpu_device_scope, pin_gpu_device
    from acprof.host.run_state import MeasurementLock, ResultDirectoryLock

    validate_sample(sample)
    if (probe not in {"none", "full"} or type(cpus) is not int or cpus <= 0
            or type(memory_gb) is not int or memory_gb <= 0 or type(gpu) is not bool):
        raise ValueError("invalid coverage probe/resources")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout must be finite and positive")
    if any(value is not None and (type(value) is not int or value <= 0)
           for value in (max_parameters, max_download_bytes)):
        raise ValueError("resource budgets must be positive integers")
    retry_stages, retry_reasons = tuple(retry_stages or ()), tuple(retry_reasons or ())
    if any(not isinstance(value, str) or not value for value in (*retry_stages, *retry_reasons)):
        raise ValueError("retry selectors must be nonempty stage or reason codes")
    if (retry_failed or retry_stages or retry_reasons) and not resume:
        raise ValueError("failed-model retry requires --resume")
    root = Path(root)
    if resume:
        if not root.is_dir():
            raise ValueError("coverage resume requires an existing report directory")
    else:
        root.mkdir(parents=True, exist_ok=False)
    measurement_lock = MeasurementLock() if probe == "full" else nullcontext()
    with measurement_lock, ResultDirectoryLock(root), gpu_device_scope():
        device = pin_gpu_device() if gpu and probe == "full" else {}
        configuration = coverage_configuration(
            probe=probe, resources={"cpus": cpus, "memory_gb": memory_gb, "gpu": gpu,
                                    "timeout_seconds": timeout_seconds},
            budgets={"max_parameters": max_parameters, "max_download_bytes": max_download_bytes},
            device={"kind": "gpu" if gpu else "cpu", "uuid": device.get("uuid")})
        state = CoverageState(root, sample, configuration, resume=resume, retry_failed=retry_failed,
                              retry_stages=retry_stages, retry_reasons=retry_reasons)
        sample_weight = sum(item["weight"] for item in sample["models"])

        def save():
            from acprof.analysis.compatibility import write_compatibility_report
            report = state.overview()
            report["summary"] = summarize(report["rows"], probe_requested=probe == "full", sample_weight=sample_weight)
            atomic_write_json(root / "coverage.json", report)
            write_compatibility_report(root, report["rows"])
            return report

        save()
        try:
            for index in state.indices:
                output = state.start_model(index)
                previous = state.rows.get(index, {})
                recovery = None
                row = None
                if previous.get("cleanup_status") == "incomplete":
                    try:
                        recovery = confirm_previous_cleanup(previous, state.attempts, configuration)
                        if recovery.get("status") != "complete":
                            raise ValueError("previous cleanup has not been confirmed complete")
                    except (Exception, SystemExit) as exc:
                        detail = (exc.to_dict() if callable(getattr(exc, "to_dict", None)) else
                                  {"exception_type": type(exc).__name__, "detail": str(exc)})
                        row = {**previous, "runtime_status": "not_run", "failed_stage": "cleanup",
                               "cleanup_recovery": {"status": "incomplete", "error": detail},
                               "cleanup_errors": [*previous.get("cleanup_errors", []), detail]}
                if row is None:
                    row = _evaluate_model(sample["models"][index], output, probe=probe, cpus=cpus,
                                          memory_gb=memory_gb, gpu=gpu, timeout_seconds=timeout_seconds,
                                          max_parameters=max_parameters, max_download_bytes=max_download_bytes)
                    if recovery is not None:
                        row["cleanup_recovery"] = recovery
                state.record(index, row)
                save()
                if row.get("cleanup_status") == "incomplete":
                    raise CoverageCleanupError(
                        f"coverage cleanup incomplete for {row['model_id']}; stopped before the next model; "
                        f"run and cleanup evidence: {state.directory / 'attempt.json'}")
        except BaseException:
            state.finish("interrupted")
            save()
            raise
        state.finish("complete")
        return save()
