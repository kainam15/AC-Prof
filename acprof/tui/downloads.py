"""Download and capacity presentation from unmodified byte-valued reports."""
from __future__ import annotations

import json
from collections.abc import Callable

from acprof.download_events import DownloadProgress
from acprof.messages import message
from acprof.tui.i18n import message_id
from acprof.tui.presentation import UNKNOWN, format_byte_fields, format_bytes


def download_fields(report: dict) -> dict[str, str]:
    return {
        "预计下载": format_bytes(report.get("expected_download_bytes")),
        "可用空间": format_bytes(report.get("disk", {}).get("free_bytes")),
        "模型来源": report.get("model", {}).get("source") or "huggingface",
    }


def download_summary(report: dict, translate: Callable[[str], str]) -> str:
    model, disk = report.get("model", {}), report.get("disk", {})
    docker = report.get("docker_storage", {})
    budget = (UNKNOWN if "max_download_bytes" not in report else message_id('ui.0071')
              if report["max_download_bytes"] is None else format_bytes(report["max_download_bytes"]))
    docker_details = {key: value for key, value in docker.items()
                      if key not in {"docker_storage_total_bytes", "docker_storage_available_bytes_at_start"}}
    lines: list[str] = [
        message_id('ui.0069', format_bytes(report.get("expected_download_bytes"))),
        message_id('ui.0070', budget),
        message_id('ui.0072', report.get("model_store_path", UNKNOWN)),
        message_id('ui.0003'),
        message_id('ui.0073', format_bytes(model.get("total_bytes")),
                format_bytes(model.get("cached_bytes"))),
        message_id('ui.0074', model.get("endpoint", UNKNOWN)),
        message("Runtime: {0}", json.dumps(format_byte_fields(report.get("runtime", {}), translate), ensure_ascii=False)),
        message_id('ui.0075', format_bytes(disk.get("total_bytes")),
                format_bytes(disk.get("free_bytes"))),
        message_id('ui.0076', format_bytes(disk.get("reclaimable_bytes")),
                format_bytes(disk.get("remaining_bytes"))),
        message_id('ui.0077', format_bytes(docker.get("docker_storage_total_bytes"))),
        message_id('ui.0078', format_bytes(docker.get("docker_storage_available_bytes_at_start"))),
    ]
    if docker_details:
        lines.append(message_id('ui.0079', json.dumps(format_byte_fields(docker_details, translate), ensure_ascii=False)))
    lines.append(message_id('ui.0006'))
    if report.get("sources"):
        lines.append(json.dumps(report["sources"], ensure_ascii=False, indent=2))
    if report.get("network_probe"):
        lines.append(json.dumps(report["network_probe"], ensure_ascii=False, indent=2))
    return "\n".join(translate(line) for line in lines)


def download_result_summary(report: dict) -> str:
    return message_id('ui.0081', report["category"],
                   format_bytes(report["verified_new_payload_bytes"]), format_bytes(report["cache_savings_bytes"]),
                   format_bytes(report.get("wire_bytes")))


def live_download_summary(progress: DownloadProgress, translate: Callable[[str], str]) -> str:
    states = {"downloading": "ui.0938", "verifying": "ui.0939", "complete": "ui.0940",
              "failed": "ui.0941", "cancelled": "ui.0942"}
    percent = (f"{progress.completed_bytes / progress.total_bytes:.0%}"
               if progress.completed_bytes is not None and progress.total_bytes else UNKNOWN)
    speed = format_bytes(int(progress.rate_bytes_s)) + "/s" if progress.rate_bytes_s is not None else UNKNOWN
    lines = [message_id('ui.0943', progress.model, message_id(states[progress.state]), progress.attempt),
             message_id('ui.0944',
                     format_bytes(progress.completed_bytes), format_bytes(progress.total_bytes), percent,
                     progress.files_completed, progress.files_total, format_bytes(progress.cached_bytes)),
             message_id('ui.0945', format_bytes(progress.transferred_bytes), speed)]
    if progress.detail == 'retry_cache_baseline_unknown':
        lines.append(message_id('ui.0949'))
    elif progress.detail:
        lines.append(message_id('ui.0946', progress.detail))
    return "\n".join(translate(line) for line in lines)
