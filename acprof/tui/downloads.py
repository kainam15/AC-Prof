"""Download and capacity presentation from unmodified byte-valued reports."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

from acprof.messages import join_messages
from acprof.tui.i18n import message_id
from acprof.tui.presentation import UNKNOWN, format_bytes


def runtime_summary(runtime: dict) -> str:
    if runtime.get("environment_local"):
        state = message_id('ui.2101')
    elif runtime.get("kind") == "process":
        state = message_id('ui.2102')
    elif runtime.get("source_mode") == "build":
        state = message_id('ui.2103')
    elif runtime.get("source_mode") in {"auto", "pull"}:
        state = message_id('ui.2104')
    else:
        state = UNKNOWN
    ghcr = (message_id('ui.2105') if runtime.get("will_attempt_ghcr_pull") else
            message_id('ui.2106') if runtime.get("will_attempt_ghcr_pull") is False else UNKNOWN)
    return message_id('ui.2100', state, ghcr)


def probe_reason(reason: str | None) -> str:
    if reason and reason.startswith("http-"):
        return message_id('ui.2118', reason[5:])
    kinds = {"dns": 'ui.2119', "tls": 'ui.2120', "connect-timeout": 'ui.2121',
             "read-timeout": 'ui.2122', "timeout": 'ui.0873', "transfer-interrupted": 'ui.2124',
             "untrusted-host": 'ui.2125', "redirect-loop": 'ui.2126', "probe-budget": 'ui.2127',
             "offline": 'ui.2128', "unsupported-source": 'ui.2129', "source-policy": 'ui.2130',
             "manifest-unavailable": 'ui.2131', "source-unknown": 'ui.2132'}
    return message_id(kinds.get(reason or "", 'ui.2133'))


CATEGORIES = {"model": 'ui.0180', "python": 'ui.0029', "debian": 'ui.2109', "oci": 'ui.2110'}


def source_summary(report: dict, categories: tuple[str, ...]) -> list[str]:
    lines: list[str] = []
    for category in categories:
        sources = [item for item in report.get("sources", []) if item.get("category") == category]
        hosts = {urlsplit(item.get("url", "")).hostname for item in sources}
        origins = [*sorted(host for host in hosts if host), *([UNKNOWN] if None in hosts else [])]
        source = join_messages(", ", origins) if origins else message_id('ui.2111') if "sources" in report else UNKNOWN
        lines.append(message_id('ui.2112', message_id(CATEGORIES[category]), source))
    return lines


def network_summary(report: dict) -> list[str]:
    lines = [message_id('ui.0003'), *source_summary(report, ("model",))]
    probes = report.get("network_probe", [])
    if not probes:
        cached = (report.get("model", {}).get("total_bytes") is not None and
                  report.get("model", {}).get("cached_bytes") == report["model"]["total_bytes"])
        lines.append(message_id('ui.2113', message_id('ui.2114') if cached else message_id('ui.2115')))
    for probe in probes:
        label = message_id(CATEGORIES.get(probe.get("category", "model"), 'ui.0180'))
        state = message_id({"passed": 'ui.0808', "failed": 'ui.0771'}.get(probe.get("status"), 'ui.2136'))
        scope = message_id('ui.2137' if probe.get("method") == "HEAD+GET" else 'ui.2138')
        lines.append(message_id('ui.2116', label, state, scope,
                                urlsplit(probe.get("endpoint") or probe.get("source", "")).hostname or UNKNOWN,
                                probe.get("final_host") or UNKNOWN))
        if probe.get("reason"):
            lines.append(message_id('ui.2117', probe_reason(probe["reason"])))
        if probe.get("method") == "HEAD+GET":
            lines.append(message_id('ui.2139', probe.get("http_status") or UNKNOWN,
                                    format_bytes(probe.get("received_bytes")), probe.get("elapsed_seconds", UNKNOWN)))
    lines.extend([message_id('ui.2140', format_bytes(report.get("probe_payload_bytes")),
                             format_bytes(report.get("probe_byte_limit"))), message_id('ui.2141')])
    return lines


@dataclass(frozen=True)
class DownloadSection:
    key: str
    title: str
    lines: tuple[str, ...]


def download_sections(report: dict) -> tuple[DownloadSection, ...]:
    """Group the same report for the confirmation dialog and the settings summary."""
    model, disk = report.get("model", {}), report.get("disk", {})
    docker = report.get("docker_storage", {})
    budget = (UNKNOWN if "max_download_bytes" not in report else message_id('ui.0071')
              if report["max_download_bytes"] is None else format_bytes(report["max_download_bytes"]))
    overview = (
        message_id('ui.0069', format_bytes(report.get("expected_download_bytes"))),
        message_id('ui.0070', budget),
        message_id('ui.0073', format_bytes(model.get("total_bytes")),
                format_bytes(model.get("cached_bytes"))),
        message_id('ui.0074', model.get("endpoint", UNKNOWN)),
    )
    runtime = (
        runtime_summary(report.get("runtime", {})),
        *source_summary(report, ("python", "debian", "oci")),
    )
    storage = [
        message_id('ui.0075', format_bytes(disk.get("total_bytes")),
                format_bytes(disk.get("free_bytes"))),
        message_id('ui.0076', format_bytes(disk.get("reclaimable_bytes")),
                format_bytes(disk.get("remaining_bytes"))),
        message_id('ui.0077', format_bytes(docker.get("docker_storage_total_bytes"))),
        message_id('ui.0078', format_bytes(docker.get("docker_storage_available_bytes_at_start"))),
    ]
    if docker.get("docker_storage_filesystem") or docker.get("docker_storage_device"):
        storage.append(message_id('ui.2142', docker.get("docker_storage_filesystem") or UNKNOWN,
                                  docker.get("docker_storage_device") or UNKNOWN))
    storage.append(message_id('ui.0072', report.get("model_store_path", UNKNOWN)))
    return (
        DownloadSection("overview", message_id('ui.2144'), overview),
        DownloadSection("runtime", message_id('ui.2145'), runtime),
        DownloadSection("storage", message_id('ui.0301'), tuple(storage)),
        DownloadSection("network", message_id('ui.2146'), tuple(network_summary(report))),
    )


def download_summary(report: dict, translate: Callable[[str], str]) -> str:
    return "\n\n".join("\n".join(translate(line) for line in (section.title, *section.lines))
                       for section in download_sections(report))


def download_result_summary(report: dict) -> str:
    return message_id('ui.0081', report["category"],
                   format_bytes(report["verified_new_payload_bytes"]), format_bytes(report["cache_savings_bytes"]),
                   format_bytes(report.get("wire_bytes")))
