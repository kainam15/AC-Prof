"""Readable cache summaries; raw reports and deletion identities stay unchanged."""
from collections.abc import Callable
from datetime import datetime

from acprof.tui.i18n import message_id
from acprof.tui.ui.presentation import UNKNOWN, format_bytes
from acprof.tui.ui.view_models import FactRow, ViewSection


def _format_stamp(last_used: float | None, translate: Callable[[str], str]) -> str:
    if last_used is None:
        return translate(UNKNOWN)
    return datetime.fromtimestamp(last_used).astimezone().strftime("%Y-%m-%d %H:%M %z")


def _model_lines(model: dict, translate: Callable[[str], str]) -> list[str]:
    stamp = _format_stamp(model.get("last_used"), translate)
    return [
        str(model["model_id"]),
        translate(message_id("downloads.store.model_revision", model["model_revision"])),
        translate(message_id("downloads.store.model_files", translate(format_bytes(model.get("model_artifact_bytes"))))),
        translate(message_id("downloads.store.model_last_used", stamp)),
    ]


def model_store_facts(report: dict, preview: dict | None) -> ViewSection:
    if not report:
        return ViewSection("", ())
    capacity = format_bytes(report.get("capacity_bytes"))
    if "capacity_bytes" in report and report["capacity_bytes"] is None:
        capacity = message_id("common.value.unlimited")
    rows = [
        FactRow(message_id("downloads.store.fact.directory"), str(report.get("path", "")), copyable=bool(report.get("path"))),
        FactRow(message_id("downloads.store.fact.used"), format_bytes(report.get("total_bytes"))),
        FactRow(message_id("downloads.store.fact.free"), format_bytes(report.get("free_bytes"))),
        FactRow(message_id("downloads.store.fact.capacity"), capacity),
        FactRow(message_id("downloads.store.fact.models"), str(len(report["models"]))),
    ]
    if preview is not None:
        selected = preview["entries"]
        reclaim = preview["reclaimable_bytes"]
        rows.append(FactRow(
            message_id("downloads.store.fact.reclaimable"),
            format_bytes(reclaim),
            status="warning" if (selected or reclaim) else None,
        ))
    return ViewSection("", tuple(rows))


def model_store_rows(report: dict, preview: dict | None, translate: Callable[[str], str]) -> list[tuple[str, str, str, str, str]]:
    if not report:
        return []
    selected = set(preview["entries"]) if preview is not None else set()
    rows: list[tuple[str, str, str, str, str]] = []
    for model in report["models"]:
        if preview is None:
            state = translate(UNKNOWN)
        elif model["entry_id"] in selected:
            state = translate(message_id("downloads.store.preview_remove"))
        else:
            state = translate(message_id("downloads.store.preview_keep"))
        rows.append((
            str(model["model_id"]),
            str(model["model_revision"]),
            translate(format_bytes(model.get("model_artifact_bytes"))),
            _format_stamp(model.get("last_used"), translate),
            state,
        ))
    return rows


def model_store_prune_summary(report: dict, preview: dict, translate: Callable[[str], str]) -> str:
    """Describe exactly the approved entries, retaining identities absent from the scan."""
    entries = preview["entries"]
    models = {model["entry_id"]: model for model in report["models"]}
    lines = [
        translate(message_id("downloads.store.directory", report["path"])),
        translate(message_id("downloads.store.planned_removals", len(entries))),
        translate(message_id("downloads.store.reclaimable_space", translate(format_bytes(preview["reclaimable_bytes"])))),
        translate(message_id("downloads.store.prune_unreferenced_also" if entries else "downloads.store.prune_unreferenced_only")),
        translate(message_id("downloads.store.prune_revalidation_note")),
    ]
    if entries:
        lines.append(translate(message_id("downloads.store.redownload_note")))
    for entry in entries:
        lines.append("")
        if entry in models:
            lines.extend(_model_lines(models[entry], translate))
            lines.append(translate(message_id("downloads.store.entry_label", entry)))
        else:
            lines.append(translate(message_id("downloads.store.entry_not_in_summary", entry)))
    return "\n".join(lines)
