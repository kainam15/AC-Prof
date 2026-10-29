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
        translate(message_id("ui.2326", model["model_revision"])),
        translate(message_id("ui.2327", translate(format_bytes(model.get("model_artifact_bytes"))))),
        translate(message_id("ui.2328", stamp)),
    ]


def model_store_summary(report: dict, preview: dict | None, translate: Callable[[str], str]) -> str:
    """Separate registered model sizes from shared physical cache usage."""
    if not report:
        return ""
    capacity = format_bytes(report.get("capacity_bytes"))
    if "capacity_bytes" in report and report["capacity_bytes"] is None:
        capacity = message_id("ui.0071")
    models = report["models"]
    lines = [
        translate(message_id("ui.2319", translate(format_bytes(report.get("total_bytes"))))),
        translate(message_id("ui.2320", translate(format_bytes(report.get("free_bytes"))))),
        translate(message_id("ui.2321", translate(capacity))),
        translate(message_id("ui.2323", len(models))),
        "",
    ]
    selected = set(preview["entries"]) if preview is not None else set()
    if preview is None:
        lines.append(translate(message_id("ui.2332")))
    elif selected or preview["reclaimable_bytes"]:
        lines.extend([
            translate(message_id("ui.2330", len(selected))),
            translate(message_id("ui.2329", translate(format_bytes(preview["reclaimable_bytes"])))),
            translate(message_id("ui.2338")),
        ])
    else:
        lines.append(translate(message_id("ui.2331")))
    lines.extend(["", translate(message_id("ui.2322", report["path"])), ""])
    if not models:
        lines.append(translate(message_id("ui.2324")))
    else:
        lines.append(translate(message_id("ui.2325")))
        for model in models:
            lines.extend(["", *_model_lines(model, translate)])
            if preview is not None:
                lines.append(translate(message_id("ui.2334" if model["entry_id"] in selected else "ui.2333")))
    return "\n".join(lines)


def model_store_prune_summary(report: dict, preview: dict, translate: Callable[[str], str]) -> str:
    """Describe exactly the approved entries, retaining identities absent from the scan."""
    entries = preview["entries"]
    models = {model["entry_id"]: model for model in report["models"]}
    lines = [
        translate(message_id("ui.2322", report["path"])),
        translate(message_id("ui.2330", len(entries))),
        translate(message_id("ui.2329", translate(format_bytes(preview["reclaimable_bytes"])))),
        translate(message_id("ui.2337" if entries else "ui.2336")),
        translate(message_id("ui.2338")),
    ]
    if entries:
        lines.append(translate(message_id("ui.2339")))
    for entry in entries:
        lines.append("")
        if entry in models:
            lines.extend(_model_lines(models[entry], translate))
            lines.append(translate(message_id("ui.2335", entry)))
        else:
            lines.append(translate(message_id("ui.2340", entry)))
    return "\n".join(lines)
