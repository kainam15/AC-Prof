"""Acquire and verify repository payloads before publishing a Model Store view."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from acprof.host.model_store.views import snapshot_path
from acprof.hub.huggingface.auth import hf_token_for_url
from acprof.hub.huggingface.download import try_hf_endpoints
from acprof.hub.huggingface.endpoints import hf_endpoints
from acprof.hub.huggingface.transport import observed_hf_provenance, observed_hf_sources
from acprof.models.download_plan import DownloadPlan, seal_plan
from acprof.network_policy import DownloadSource

if TYPE_CHECKING:
    from acprof.host.preparation.download_progress import DownloadReporter


def _download_repository(
    root: Path, repo: Mapping[str, Any], progress: DownloadReporter, offline: bool, endpoint: str
) -> tuple[str, str]:
    from huggingface_hub import snapshot_download

    progress.start_attempt()
    if repo.get("source", "huggingface") == "modelscope":
        from acprof.hub.repository import modelscope_file, modelscope_snapshot

        if not offline:
            for record in repo["files"]:
                modelscope_file(
                    repo["model_id"],
                    record["path"],
                    repo["model_revision"],
                    root,
                    sha256=record["lfs_sha256"],
                )
        target = modelscope_snapshot(root, repo["model_id"], repo["model_revision"])
        return str(target), endpoint
    escape = {"[": "[[]", "*": "[*]", "?": "[?]"}
    progress_kwargs = {"tqdm_class": progress.tqdm_class()} if progress.enabled else {}
    target = snapshot_download(
        repo_id=repo["model_id"],
        revision=repo["model_revision"],
        cache_dir=str(root / "hf"),
        endpoint=endpoint,
        token=False if offline else hf_token_for_url(endpoint),
        allow_patterns=[
            "".join(escape.get(character, character) for character in record["path"])
            for record in repo["files"]
        ],
        local_files_only=offline,
        **progress_kwargs,
    )
    return target, endpoint


def prepare_repository(
    root: Path,
    key: str,
    repo: DownloadPlan | dict[str, Any],
    budget: DownloadSource,
    *,
    offline: bool,
    needs_publish: bool,
) -> tuple[Path, str]:
    from acprof.container.download_model import verify_download
    from acprof.host.preparation.download_progress import DownloadReporter

    source = repo.get("source", "huggingface")
    # Cached payloads can pass space checks while source metadata sizes remain unknown.
    # Preserve that pre-verification raw contract for the progress reporter.
    report_plan: Mapping[str, Any] = repo
    progress = DownloadReporter(
        repo["model_id"], source, report_plan["selected_bytes"], budget.cached_bytes, len(repo["files"])
    )
    try:
        if not needs_publish:
            progress.start_attempt()
            target, endpoint = str(snapshot_path(root, key, repo)), repo["endpoint"]
        elif offline or source == "modelscope":
            target, endpoint = _download_repository(root, repo, progress, offline, repo["endpoint"])
        else:
            choices = list(dict.fromkeys([repo["endpoint"], *hf_endpoints()]))
            target, endpoint = try_hf_endpoints(
                repo["model_id"],
                lambda endpoint: _download_repository(root, repo, progress, offline, endpoint),
                endpoints=choices,
            )
        progress.finish("verifying")
        # Verification happens before any runtime or formal measurement starts.
        verify_download(Path(target), repo)
    except BaseException as exc:
        progress.finish("cancelled" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed")
        raise
    progress.finish("complete")
    return Path(target), endpoint


def record_download_provenance(repo: DownloadPlan | dict[str, Any], *, endpoint: str, offline: bool) -> None:
    source = repo.get("source", "huggingface")
    repo["source"] = source
    repo["endpoint"] = endpoint
    repo["actual_source_hosts"] = (
        [] if offline or source == "modelscope" else observed_hf_sources()
    )
    repo["download_provenance"] = (
        {"schema_version": 1, "cache_hit": True, "public_egress": "externally-managed"}
        if offline
        else {
            "schema_version": 1,
            "public_egress": "externally-managed",
            "requests": [],
            "observation": "SDK redirects unobserved",
        }
        if source == "modelscope"
        else observed_hf_provenance()
    )
    events = repo["download_provenance"].get("requests", [])
    payloads = [
        event
        for event in events
        if event["method"] == "GET"
        and event["status"] in {200, 206}
        and "/api/" not in urlsplit(event["url"]).path
    ]
    repo["download_provenance"]["final_endpoint_type"] = (
        "cache" if offline else payloads[-1]["endpoint_type"] if payloads else "unknown"
    )
    repo["download_provenance"]["hub_endpoint"] = endpoint
    seal_plan(repo)
