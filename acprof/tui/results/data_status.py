"""One cancellable snapshot for data quality and profiler status, without statistics."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from acprof.cancellation import check_cancelled
from acprof.messages import join_messages
from acprof.results.layout import resolve_result_file
from acprof.tui.i18n import error_message, message_id
from acprof.tui.results.profile_status import ProfilerOverview


@dataclass(frozen=True)
class ExperimentDataStatus:
    path: Path
    sha256: str
    formal_ok_rows: int
    audit: dict
    profiler_overview: ProfilerOverview | None
    profiler_error: str
    metadata_identity: tuple[tuple[str, str | None], ...] = ()
    html_path: Path | None = None
    html_state: str = "not_generated"


def inspect_experiment_data(source: Path, *, cancelled: Callable[[], bool] | None = None) -> ExperimentDataStatus:
    from acprof.analysis.audit import audit_result
    from acprof.artifacts import file_sha256
    from acprof.results.csv import (
        is_measured_success,
        read_result_csv_snapshot,
        verify_result_source,
    )
    from acprof.results.metadata import ResultMetadataSnapshot
    from acprof.tui.results.profile_status import inspect_profile_result
    from acprof.tui.results.report_files import existing_html_report, html_report_state

    check_cancelled(cancelled)
    path = resolve_result_file(source)
    snapshot = read_result_csv_snapshot(path, validate_measurements=False, cancelled=cancelled)
    if "status" not in snapshot.fields:
        raise ValueError(message_id("results.csv.invalid_header"))
    metadata = ResultMetadataSnapshot(path)
    audit = audit_result(snapshot.path, result_snapshot=snapshot, metadata_snapshot=metadata, verify_result_snapshot=False)
    check_cancelled(cancelled)
    overview, profile_error = None, ""
    try:
        overview = inspect_profile_result(snapshot.path, result_snapshot=snapshot, cancelled=cancelled)
    except (OSError, ValueError, RuntimeError) as exc:
        profile_error = error_message(exc)
    check_cancelled(cancelled)
    formal = sum(is_measured_success(row) and str(row.get("warmup", "0")).strip() == "0" for row in snapshot.rows)
    try:
        html = existing_html_report((source,))
        html_state = html_report_state(html, snapshot.path, snapshot.sha256, cancelled=cancelled) if html else "not_generated"
    except (OSError, ValueError):
        html, html_state = None, "damaged"
    try:
        if resolve_result_file(source).resolve() != snapshot.path:
            raise ValueError(f"result source changed: {source}")
        verify_result_source(snapshot.path, snapshot.sha256, cancelled=cancelled)
        for artifact, expected in metadata.observed_fingerprints().items():
            check_cancelled(cancelled)
            actual = file_sha256(artifact) if artifact.is_file() else None
            if actual != expected:
                raise ValueError(f"metadata changed: {artifact}")
    except (OSError, ValueError) as exc:
        raise ValueError(message_id("results.summary.snapshot_unverified", str(exc))) from exc
    identity = tuple(sorted((str(artifact), digest) for artifact, digest in metadata.observed_fingerprints().items()))
    return ExperimentDataStatus(snapshot.path, snapshot.sha256, formal, audit, overview, profile_error, identity, html, html_state)


def data_status_text(status: ExperimentDataStatus) -> str:
    from acprof.tui.results.reports import render_quality_summary

    audit = status.audit
    counts = audit["counts"]
    lines = [message_id("results.overview.data_counts", status.formal_ok_rows, counts["error"], counts["warn"], counts["warmup"]),
             message_id("results.overview.integrity", message_id("results.overview.verified" if audit["valid"] else "results.overview.issues")),
             render_quality_summary(audit)]
    if audit["issues"]:
        lines.append(message_id("results.overview.warnings", "; ".join(item["message"] for item in audit["issues"][:3])))
    return join_messages("\n", lines)
