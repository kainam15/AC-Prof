"""Case archival, checkpoint recovery and cleanup debt outside measurement windows."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from acprof.artifacts import file_sha256, replace_file_durably, sync_directory_chain
from acprof.host.run_state import RunStateError, utc_now
from acprof.results.csv import read_result_csv

if TYPE_CHECKING:
    from acprof.host.run_state import RunState
    from acprof.results.layout import CaseArtifacts


def _validate_complete_case(
    state: RunState, path: Path, record: dict[str, Any],
    cpu: int | None, mem: int | None, gpu: str,
) -> None:
    if not path.is_file() or file_sha256(path) != record["sha256"]:
        raise RunStateError(f"已完成 case 的 CSV 缺失或改变：{path}")
    read_result_csv(path, expected=state.expected(cpu, mem, gpu))


def _recover_measurement_checkpoint(
    state: RunState, case: CaseArtifacts, record: dict[str, Any],
    cpu: int | None, mem: int | None, gpu: str,
) -> None:
    if not record.get("case_attempt_id") or record.get("measurement_status") == "complete":
        return
    from acprof.host.measurement.checkpoint import discover_measurement_checkpoint
    try:
        reference = discover_measurement_checkpoint(
            case,
            run_id=state.run_id,
            attempt_id=record["case_attempt_id"],
            expected=state.expected(cpu, mem, gpu),
        )
    except (OSError, ValueError, KeyError) as exc:
        raise RunStateError(f"已发布测量快照无法恢复：{exc}") from exc
    if reference is None:
        return
    record.update(
        status="measured",
        measurement_status="complete",
        processing_status="pending",
        measurement_checkpoint=reference,
        recovered_checkpoint_at=utc_now(),
    )
    state.save()


def _recover_process_cleanup(state: RunState, record: dict[str, Any]) -> None:
    if not record.get("cleanup_process") or record.get("cleanup_status") == "complete":
        return
    from acprof.host.runtimes.process_control import recover_process
    recover_process(record["cleanup_process"], directory=state.layout.path("logs"))
    record["cleanup_status"] = "complete"
    state.save()


def _archive_case_projection(
    state: RunState, case: CaseArtifacts, filename: str, record: dict[str, Any],
) -> str | None:
    candidates = [*case.temporary_files(), case.retained_requests, case.idle]
    existing = list(dict.fromkeys(
        candidate for candidate in candidates if candidate.exists() or candidate.is_symlink()
    ))
    if not existing:
        return None
    archived = record.get("status") == "archived" and bool(record.get("backup"))
    backup = (
        state.layout.contained(record["backup"])
        if archived
        else state.layout.path("interrupted_cases") / case.case_id / uuid4().hex
    )
    backup.mkdir(parents=True, exist_ok=bool(archived))
    backup_directories: set[Path] = set()
    for source in existing:
        state.artifact_path(str(source.relative_to(state.directory)))
        if source.is_symlink() or not source.is_file():
            raise RunStateError(f"不支持的 case 产物类型：{source}")
        target = backup / source.relative_to(state.directory)
        if archived:
            if not target.is_file() or file_sha256(target) != file_sha256(source):
                raise RunStateError(
                    f"Interrupted archive disagrees with its remaining projection: {source}"
                )
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        with target.open("rb") as stream:
            os.fsync(stream.fileno())
        backup_directories.add(target.parent)
    for directory in sorted(
        backup_directories, key=lambda path: len(path.parts), reverse=True,
    ):
        sync_directory_chain(directory, state.directory)
    backup_name = str(backup.relative_to(state.directory))
    state.data["cases"][filename] = {
        **record,
        "status": "archived",
        "backup": backup_name,
    }
    state.save()
    for source in existing:
        source.unlink()
    return backup_name


def _start_case_attempt(
    state: RunState, case: CaseArtifacts, path: Path, filename: str,
    record: dict[str, Any], archived_backup: str | None,
    cpu: int | None, mem: int | None, gpu: str,
) -> None:
    previous_attempts = list(record.get("previous_attempts", []))
    previous = {name: value for name, value in record.items() if name != "previous_attempts"}
    if previous:
        if archived_backup is not None:
            previous["backup"] = archived_backup
        previous_attempts.append(previous)
    state.data["cases"][filename] = {
        "status": "running",
        "started_at": utc_now(),
        "case_attempt_id": uuid4().hex,
        "measurement_status": "running",
        "processing_status": "pending",
        "resources": [cpu, mem, gpu],
        "previous_attempts": previous_attempts,
    }
    state.save()
    path.parent.mkdir(parents=True, exist_ok=True)
    from acprof.host.measurement.window_recovery import is_incremental, restore_case_windows
    if is_incremental(state.layout):
        count = (
            restore_case_windows(
                case, run_id=state.run_id, expected=state.expected(cpu, mem, gpu),
            )
            if state.layout.window_records_root(case.case_id).exists()
            else 0
        )
        state.data["cases"][filename]["restored_windows"] = count
        state.data["cases"][filename]["resume_semantics"] = (
            "window-v1; runtime segments have separate attempts"
        )
        state.save()


def prepare_case(
    state: RunState, filename: str, cpu: int | None, mem: int | None, gpu: str,
) -> str | None:
    path = (
        state.layout.case("", cpu, mem, gpu).csv
        if state.layout.layout_version == 2
        else state.artifact_path(filename)
    )
    filename = str(path.relative_to(state.directory))
    case = state.layout.case_from_csv(path)
    record = state.data["cases"].get(filename, {})
    if record.get("status") == "complete":
        _validate_complete_case(state, path, record, cpu, mem, gpu)
        return str(path)
    _recover_measurement_checkpoint(state, case, record, cpu, mem, gpu)
    if record.get("measurement_status") == "complete":
        state.recover_processing_checkpoint(str(path), cpu, mem, gpu)
        return None
    _recover_process_cleanup(state, record)
    archived_backup = _archive_case_projection(state, case, filename, record)
    _start_case_attempt(
        state, case, path, filename, record, archived_backup, cpu, mem, gpu,
    )
    return None


def recover_processing_checkpoint(
    state: RunState, path: str, cpu: int | None, mem: int | None, gpu: str,
) -> dict[str, Any] | None:
    checkpoint = state.processing_checkpoint(path, cpu, mem, gpu)
    if checkpoint is None:
        return None
    record = state.data["cases"][str(Path(path).relative_to(state.directory))]
    if record.get("cleanup_status") == "complete":
        return checkpoint
    if record.get("cleanup_process"):
        from acprof.host.runtimes.process_control import recover_process
        recover_process(record["cleanup_process"], directory=state.layout.path("logs"))
        recovery = {"kind": "process", "status": "complete"}
    else:
        from acprof.host.container_lifecycle import recover_cleanup_debt
        from acprof.system import command as host_command
        recovery = recover_cleanup_debt(
            [{"container_id": record.get("cleanup_container_id", "")}],
            host_command.run_command,
        )
    case = state.layout.case_from_csv(path)
    error = case.sidecar("cleanup_error")
    if error.exists():
        target = state.layout.contained(
            f"raw/cleanup/{case.case_id}/{record['case_attempt_id']}.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise RunStateError(f"Cleanup evidence already exists: {target}")
        replace_file_durably(error, target)
    record.update(cleanup_status="complete", cleanup_recovery=recovery)
    state.save()
    return checkpoint
