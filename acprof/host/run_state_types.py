"""Stable state fields guaranteed by the recovery boundary.

Opaque metadata remains object; these annotations do not validate persisted input.
Missing fields describe preparation, measurement and completion stages without
requiring historical snapshots to invent fields they never recorded.
"""
from typing import NotRequired, TypedDict


class AttemptState(TypedDict):
    attempt_id: str
    started_at: str
    pid: int
    resume: bool
    ended_at: NotRequired[str]
    outcome: NotRequired[str]
    preparation_backup: NotRequired[str]
    preparation_evidence: NotRequired[str]


class CaseState(TypedDict):
    status: str
    case_attempt_id: NotRequired[str]
    measurement_status: NotRequired[str]
    processing_status: NotRequired[str]
    cleanup_status: NotRequired[str]
    resources: NotRequired[list[int | str | None]]
    row_count: NotRequired[int]
    error_rows: NotRequired[int]
    processing_attempts: NotRequired[int]
    restored_windows: NotRequired[int]
    measurement_checkpoint: NotRequired[object]
    case_result: NotRequired[object]
    backup: NotRequired[str]
    previous_attempts: NotRequired[list["CaseState"]]
    cleanup_process: NotRequired[dict[object, object]]
    # These values are metadata, or are validated only in a later case stage.
    sha256: NotRequired[object]
    started_at: NotRequired[object]
    measured_at: NotRequired[object]
    recovered_checkpoint_at: NotRequired[object]
    completed_at: NotRequired[object]
    resume_semantics: NotRequired[object]
    cleanup_container_id: NotRequired[object]
    cleanup_recovery: NotRequired[object]


class MeasurementOptions(TypedDict):
    cpus: str
    mems: str
    gpus: str
    warmup: int
    repeat: int


class PlannedSnapshot(TypedDict):
    scales: list[float]
    source: str
    plan_file: NotRequired[str | None]
    workload: NotRequired[object]
    plan_sha256: NotRequired[str]


class RuntimeSnapshot(TypedDict):
    task: object
    image: object
    planned: PlannedSnapshot
    compute_plan: str
    execution_plan: str
    kind: NotRequired[str]


class RunStateSnapshot(TypedDict):
    schema_version: int
    layout_version: NotRequired[int]
    run_id: str
    status: str
    # Preparation has no measurement options yet. Open metadata keys and values
    # are not narrowed by the stored-state validator.
    options: object
    host: object
    attempts: list[AttemptState]
    cases: dict[str, CaseState]
    artifacts: dict[str, str]
    runtime: NotRequired[RuntimeSnapshot]
    created_at: NotRequired[object]
    updated_at: NotRequired[object]
    completed_at: NotRequired[object]
    matrix_plan_sha256: NotRequired[object]
    result_sha256: NotRequired[object]
    row_count: NotRequired[object]
    outcome: NotRequired[object]
