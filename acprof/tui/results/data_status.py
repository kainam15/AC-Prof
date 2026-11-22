"""One cancellable snapshot for data quality and profiler status, without statistics."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from acprof.cancellation import cancellable, check_cancelled
from acprof.messages import join_messages
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
    from acprof.results.csv import (
        is_measured_success,
        preview_window_phase,
        read_result_csv_snapshot,
        verify_result_source,
    )
    from acprof.results.metadata import ResultMetadataSnapshot
    from acprof.tui.results.profile_status import inspect_profile_result
    from acprof.tui.results.report_files import existing_html_report, html_report_state

    check_cancelled(cancelled)
    metadata = ResultMetadataSnapshot(source, cancelled=cancelled)
    snapshot = read_result_csv_snapshot(metadata.source, validate_measurements=False, cancelled=cancelled)
    if snapshot.path != metadata.source:
        raise ValueError(message_id("results.summary.snapshot_unverified", f"result source changed: {source}"))
    if "status" not in snapshot.fields:
        raise ValueError(message_id("results.csv.invalid_header"))
    audit = audit_result(source, result_snapshot=snapshot, metadata_snapshot=metadata,
                         verify_result_snapshot=False, cancelled=cancelled)
    check_cancelled(cancelled)
    overview, profile_error = None, ""
    try:
        overview = inspect_profile_result(snapshot.path, result_snapshot=snapshot, cancelled=cancelled)
    except (OSError, ValueError, RuntimeError) as exc:
        profile_error = error_message(exc)
    check_cancelled(cancelled)
    formal = sum(is_measured_success(row) and preview_window_phase(row) == "0"
                 for row in cancellable(snapshot.rows, cancelled))
    try:
        html = existing_html_report((source,))
        html_state = html_report_state(html, snapshot.path, snapshot.sha256, cancelled=cancelled) if html else "not_generated"
    except (OSError, ValueError):
        html, html_state = None, "damaged"
    try:
        verify_result_source(snapshot.path, snapshot.sha256, cancelled=cancelled)
        metadata.verify_current(source, cancelled=cancelled)
        fingerprints = metadata.observed_fingerprints(cancelled=cancelled)
    except (OSError, ValueError) as exc:
        raise ValueError(message_id("results.summary.snapshot_unverified", str(exc))) from exc
    identity = tuple(sorted((str(artifact), digest) for artifact, digest in fingerprints.items()))
    return ExperimentDataStatus(snapshot.path, snapshot.sha256, formal, audit, overview, profile_error, identity, html, html_state)


def data_status_text(status: ExperimentDataStatus) -> str:
    from acprof.tui.results.reports import render_quality_summary

    audit = status.audit
    counts = audit["counts"]
    lines = [message_id("results.overview.data_counts", status.formal_ok_rows, counts["error"], counts["warn"], counts["warmup"]),
             message_id("results.overview.integrity", message_id("results.overview.verified" if audit["valid"] else "results.overview.issues")),
             render_quality_summary(audit)]
    from acprof.results.metrics import METRICS
    for key, title in (("missing_metrics", "results.overview.missing_metrics"),
                       ("warn_missing_metrics", "results.overview.warn_missing_metrics")):
        missing = audit.get(key)
        if not missing:
            continue
        by_reason: dict[str, list[str]] = {}
        for field, reasons in missing.items():
            for reason in reasons:
                by_reason.setdefault(reason, []).append(field)
        lines.append(message_id(title, len(missing)))
        for reason, fields in sorted(by_reason.items()):
            sources = ", ".join(sorted({METRICS[field].source for field in fields}))
            lines.append(message_id("results.overview.missing_reason", reason, len(fields), sources))
    if partial := audit.get("partial_metrics"):
        lines.append(message_id("results.overview.partial_metrics", ", ".join(sorted(partial))))
    diagnostics = audit.get("window_diagnostics", {})
    if pmu := diagnostics.get("pmu_multiplexing"):
        lines.append(message_id("results.overview.pmu_multiplexing", pmu["windows"], pmu["minimum_running_pct"]))
    if short := diagnostics.get("insufficient_windows"):
        lines.append(message_id("results.overview.insufficient_windows", short["configurations"], short["minimum_required"]))
    if audit["issues"]:
        lines.append(message_id("results.overview.warnings", "; ".join(item["message"] for item in audit["issues"][:3])))
    return join_messages("\n", lines)
