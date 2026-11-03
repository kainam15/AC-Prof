"""Pure availability and visibility rules for result-page actions."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from acprof.tui.commands import OperationState


@dataclass(frozen=True)
class ReportActionState:
    cancel_analysis_visible: bool
    cancel_analysis_enabled: bool
    analysis_export_csv_enabled: bool
    analysis_export_json_enabled: bool
    summary_enabled: bool
    plot_enabled: bool
    calculate_enabled: bool
    html_generate_enabled: bool
    report_open_enabled: bool
    html_file_visible: bool
    html_filename: str
    html_location: str
    hide_html_export_fields: bool
    html_remote_hint_visible: bool
    html_open_visible: bool
    html_open_enabled: bool
    html_copy_enabled: bool
    html_export_toggle_enabled: bool
    html_export_enabled: bool
    html_generate_visible: bool
    html_cancel_visible: bool
    html_cancel_enabled: bool


def report_action_state(
    operation: OperationState,
    *,
    source_ready: bool,
    formal_ok_rows: int | None,
    analysis_payload: Mapping[str, object],
    report_json_path: str,
    html_path: str,
    execution_kind: str,
    browser_available: bool,
) -> ReportActionState:
    can_analyze = source_ready and (formal_ok_rows is None or formal_ok_rows > 0)
    json_path = report_json_path.strip()
    selected_html = html_path.strip()
    has_html = bool(selected_html)
    has_payload = bool(analysis_payload)
    generating = execution_kind == "html-report"
    return ReportActionState(
        cancel_analysis_visible=execution_kind in {"stats", "compare"},
        cancel_analysis_enabled=operation.allows("stop"),
        analysis_export_csv_enabled=(
            has_payload and bool(analysis_payload.get("groups")) and operation.allows("analysis-export")
        ),
        analysis_export_json_enabled=has_payload and operation.allows("analysis-export"),
        summary_enabled=source_ready and operation.allows("summary"),
        plot_enabled=can_analyze and operation.allows("plot"),
        calculate_enabled=can_analyze and operation.allows("stats"),
        html_generate_enabled=can_analyze and operation.allows("html-report"),
        report_open_enabled=bool(json_path) and operation.allows("report"),
        html_file_visible=has_html,
        html_filename=Path(selected_html).name if has_html else "",
        html_location=selected_html,
        hide_html_export_fields=not has_html,
        html_remote_hint_visible=has_html and not browser_available,
        html_open_visible=has_html,
        html_open_enabled=has_html and operation.allows("html-open"),
        html_copy_enabled=has_html and operation.allows("html-open"),
        html_export_toggle_enabled=has_html and operation.allows("html-export"),
        html_export_enabled=has_html and operation.allows("html-export"),
        html_generate_visible=not generating,
        html_cancel_visible=generating,
        html_cancel_enabled=generating and operation.allows("stop"),
    )
