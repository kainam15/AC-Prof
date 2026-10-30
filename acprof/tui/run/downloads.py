"""Download and capacity presentation from unmodified byte-valued reports."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

from acprof.messages import join_messages
from acprof.tui.i18n import message_id
from acprof.tui.ui.presentation import UNKNOWN, format_bytes


def runtime_summary(runtime: dict) -> str:
    if runtime.get("environment_local"):
        state = message_id('downloads.runtime.reuse_environment')
    elif runtime.get("kind") == "process":
        state = message_id('downloads.runtime.prepare_process')
    elif runtime.get("source_mode") == "build":
        state = message_id('downloads.runtime.build_locally')
    elif runtime.get("source_mode") in {"auto", "pull"}:
        state = message_id('downloads.runtime.pull_image')
    else:
        state = UNKNOWN
    ghcr = (message_id('downloads.runtime.ghcr_required') if runtime.get("will_attempt_ghcr_pull") else
            message_id('downloads.runtime.ghcr_unneeded') if runtime.get("will_attempt_ghcr_pull") is False else UNKNOWN)
    return message_id('downloads.runtime.summary', state, ghcr)


def probe_reason(reason: str | None) -> str:
    if reason and reason.startswith("http-"):
        return message_id('downloads.error.http_status', reason[5:])
    kinds = {"dns": 'downloads.error.dns', "tls": 'downloads.error.tls', "connect-timeout": 'downloads.error.connect_timeout',
             "read-timeout": 'downloads.error.read_timeout', "timeout": 'common.error.request_timeout', "transfer-interrupted": 'downloads.error.incomplete_transfer',
             "untrusted-host": 'downloads.error.redirect_policy', "redirect-loop": 'downloads.error.redirect_loop', "probe-budget": 'downloads.preflight.budget_skipped',
             "offline": 'downloads.preflight.offline', "unsupported-source": 'downloads.preflight.unsupported_source', "source-policy": 'downloads.error.source_policy',
             "manifest-unavailable": 'downloads.error.manifest_unavailable', "source-unknown": 'downloads.preflight.entry_pending'}
    return message_id(kinds.get(reason or "", 'downloads.error.connection'))


CATEGORIES = {"model": 'common.field.model', "python": 'run.identity.python_dependencies', "debian": 'downloads.sources.system_dependencies', "oci": 'downloads.sources.docker_images'}


def source_summary(report: dict, categories: tuple[str, ...]) -> list[str]:
    lines: list[str] = []
    for category in categories:
        sources = [item for item in report.get("sources", []) if item.get("category") == category]
        hosts = {urlsplit(item.get("url", "")).hostname for item in sources}
        origins = [*sorted(host for host in hosts if host), *([UNKNOWN] if None in hosts else [])]
        source = join_messages(", ", origins) if origins else message_id('downloads.plan.no_downloads') if "sources" in report else UNKNOWN
        lines.append(message_id('downloads.sources.summary', message_id(CATEGORIES[category]), source))
    return lines


def network_summary(report: dict) -> list[str]:
    lines = [message_id('downloads.network.system_environment'), *source_summary(report, ("model",))]
    probes = report.get("network_probe", [])
    if not probes:
        cached = (report.get("model", {}).get("total_bytes") is not None and
                  report.get("model", {}).get("cached_bytes") == report["model"]["total_bytes"])
        lines.append(message_id('downloads.preflight.unverified_summary', message_id('downloads.preflight.cached_model') if cached else message_id('downloads.preflight.no_result')))
    for probe in probes:
        label = message_id(CATEGORIES.get(probe.get("category", "model"), 'common.field.model'))
        state = message_id({"passed": 'common.state.passed', "failed": 'common.state.failed'}.get(probe.get("status"), 'common.state.unverified'))
        scope = message_id('downloads.preflight.bounded_get' if probe.get("method") == "HEAD+GET" else 'downloads.preflight.metadata_only')
        lines.append(message_id('downloads.preflight.endpoint_summary', label, state, scope,
                                urlsplit(probe.get("endpoint") or probe.get("source", "")).hostname or UNKNOWN,
                                probe.get("final_host") or UNKNOWN))
        if probe.get("reason"):
            lines.append(message_id('downloads.preflight.reason', probe_reason(probe["reason"])))
        if probe.get("method") == "HEAD+GET":
            lines.append(message_id('downloads.preflight.transfer_summary', probe.get("http_status") or UNKNOWN,
                                    format_bytes(probe.get("received_bytes")), probe.get("elapsed_seconds", UNKNOWN)))
    lines.extend([message_id('downloads.preflight.budget_usage', format_bytes(report.get("probe_payload_bytes")),
                             format_bytes(report.get("probe_byte_limit"))), message_id('downloads.preflight.scope_note')])
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
    budget = (UNKNOWN if "max_download_bytes" not in report else message_id('common.value.unlimited')
              if report["max_download_bytes"] is None else format_bytes(report["max_download_bytes"]))
    overview = (
        message_id('downloads.size.expected_detail', format_bytes(report.get("expected_download_bytes"))),
        message_id('downloads.budget.effective', budget),
        message_id('downloads.size.total_and_cached', format_bytes(model.get("total_bytes")),
                format_bytes(model.get("cached_bytes"))),
        message_id('downloads.endpoint.detail', model.get("endpoint", UNKNOWN)),
    )
    runtime = (
        runtime_summary(report.get("runtime", {})),
        *source_summary(report, ("python", "debian", "oci")),
    )
    storage = [
        message_id('downloads.model_store.used_and_free', format_bytes(disk.get("total_bytes")),
                format_bytes(disk.get("free_bytes"))),
        message_id('downloads.storage.reclaimable_and_remaining', format_bytes(disk.get("reclaimable_bytes")),
                format_bytes(disk.get("remaining_bytes"))),
        message_id('downloads.docker.storage_used', format_bytes(docker.get("docker_storage_total_bytes"))),
        message_id('downloads.docker.storage_available', format_bytes(docker.get("docker_storage_available_bytes_at_start"))),
    ]
    if docker.get("docker_storage_filesystem") or docker.get("docker_storage_device"):
        storage.append(message_id('downloads.storage.docker_filesystem', docker.get("docker_storage_filesystem") or UNKNOWN,
                                  docker.get("docker_storage_device") or UNKNOWN))
    storage.append(message_id('downloads.model_store.path', report.get("model_store_path", UNKNOWN)))
    return (
        DownloadSection("overview", message_id('downloads.overview.title'), overview),
        DownloadSection("runtime", message_id('downloads.sources.section'), runtime),
        DownloadSection("storage", message_id('images.storage.title'), tuple(storage)),
        DownloadSection("network", message_id('downloads.preflight.title'), tuple(network_summary(report))),
    )


def download_summary(report: dict, translate: Callable[[str], str]) -> str:
    return "\n\n".join("\n".join(translate(line) for line in (section.title, *section.lines))
                       for section in download_sections(report))


def download_result_summary(report: dict) -> str:
    return message_id('downloads.result.transfer_summary', report["category"],
                   format_bytes(report["verified_new_payload_bytes"]), format_bytes(report["cache_savings_bytes"]),
                   format_bytes(report.get("wire_bytes")))
