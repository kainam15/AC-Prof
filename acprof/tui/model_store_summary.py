"""Readable cache summaries; raw reports and deletion identities stay unchanged."""
from collections.abc import Callable
from datetime import datetime

from acprof.tui.i18n import message_id
from acprof.tui.presentation import UNKNOWN, format_bytes


def _model_lines(model: dict, translate: Callable[[str], str]) -> list[str]:
    last_used = model.get("last_used")
    stamp = (datetime.fromtimestamp(last_used).astimezone().strftime("%Y-%m-%d %H:%M %z")
             if last_used is not None else translate(UNKNOWN))
    return [
        str(model["model_id"]),
        translate(message_id("downloads.store.model_revision", model["model_revision"])),
        translate(message_id("downloads.store.model_files", translate(format_bytes(model.get("model_artifact_bytes"))))),
        translate(message_id("downloads.store.model_last_used", stamp)),
    ]


def model_store_summary(report: dict, preview: dict | None, translate: Callable[[str], str]) -> str:
    """Separate registered model sizes from shared physical cache usage."""
    if not report:
        return ""
    capacity = format_bytes(report.get("capacity_bytes"))
    if "capacity_bytes" in report and report["capacity_bytes"] is None:
        capacity = message_id("common.value.unlimited")
    models = report["models"]
    lines = [
        translate(message_id("downloads.store.used_space", translate(format_bytes(report.get("total_bytes"))))),
        translate(message_id("downloads.store.free_space", translate(format_bytes(report.get("free_bytes"))))),
        translate(message_id("downloads.store.capacity", translate(capacity))),
        translate(message_id("downloads.store.model_count", len(models))),
        "",
    ]
    selected = set(preview["entries"]) if preview is not None else set()
    if preview is None:
        lines.append(translate(message_id("downloads.store.preview_pending")))
    elif selected or preview["reclaimable_bytes"]:
        lines.extend([
            translate(message_id("downloads.store.planned_removals", len(selected))),
            translate(message_id("downloads.store.reclaimable_space", translate(format_bytes(preview["reclaimable_bytes"])))),
            translate(message_id("downloads.store.prune_revalidation_note")),
        ])
    else:
        lines.append(translate(message_id("downloads.store.nothing_to_prune")))
    lines.extend(["", translate(message_id("downloads.store.directory", report["path"])), ""])
    if not models:
        lines.append(translate(message_id("downloads.store.no_registered_models")))
    else:
        lines.append(translate(message_id("downloads.store.shared_weights_note")))
        for model in models:
            lines.extend(["", *_model_lines(model, translate)])
            if preview is not None:
                lines.append(translate(message_id("downloads.store.preview_remove" if model["entry_id"] in selected else "downloads.store.preview_keep")))
    return "\n".join(lines)


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
