"""Bound downloads before build/pull; unknown bytes never count as zero."""
from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from acprof.dependency_locks import content_digest
from acprof.host import command as host_command
from acprof.host.dependency_images import (
    DEFAULT_RUNTIME_REGISTRY,
    platform_fingerprint,
    registry_reference,
    runtime_fingerprint,
)
from acprof.network_policy import (
    DEPENDENCY_USER_AGENT,
    DownloadSource,
    enforce_download_budget,
    require_source_transition,
    summarize_downloads,
)
from acprof.runtime_profiles import environment_identity


class PolicyRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        require_source_transition(req.full_url, newurl)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and req.get_method() == "HEAD":
            # urllib otherwise turns a redirected HEAD into a payload-fetching GET.
            redirected.method = "HEAD"
        return redirected


def artifact_size(url: str) -> int | None:
    """HEAD only; no artifact GET is permitted during estimation."""
    opener = urllib.request.build_opener(PolicyRedirectHandler())
    try:
        request = urllib.request.Request(url, headers={"User-Agent": DEPENDENCY_USER_AGENT}, method="HEAD")
        with opener.open(request, timeout=8) as response:
            size = response.headers.get("Content-Length")
            return int(size) if size and size.isdecimal() else None
    except (OSError, ValueError):
        return None


def registry_manifest(reference: str) -> dict | None:
    """OCI metadata only; returns compressed upper bound and Docker config ID."""
    try:
        result = host_command.run_command(['docker', 'manifest', 'inspect', '--verbose', reference], timeout=20, check=False)
        if result.returncode:
            return None
        data = json.loads(result.stdout)
        if isinstance(data, list):
            matches = [entry for entry in data if entry.get("Descriptor", {}).get("platform", {}).get("architecture") == "amd64"
                       and entry.get("Descriptor", {}).get("platform", {}).get("os") == "linux"]
            if len(matches) != 1:
                return None
            data = matches[0]
        manifest = data.get("SchemaV2Manifest") or data.get("OCIManifest") or data
        layers, config = manifest["layers"], manifest["config"]
        return {"bytes": sum(item["size"] for item in [*layers, config]), "image_id": config["digest"]}
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired):
        return None


def runtime_sources(profile, project_dir: Path, *, inspect=None, size_probe=artifact_size,
                    manifest_probe=registry_manifest) -> tuple[list[DownloadSource], dict]:
    from acprof.host.runtime_images import inspect_identity
    inspect = inspect or inspect_identity
    identity = environment_identity(profile.environment, project_dir)
    policy = os.environ.get("ACPROF_RUNTIME_IMAGE_SOURCE", "auto")
    if policy not in {"auto", "pull", "build"}:
        raise ValueError("ACPROF_RUNTIME_IMAGE_SOURCE must be auto/pull/build")
    platform_key = platform_fingerprint(profile.environment.platform, project_dir)
    local = inspect(f"acprof-platform-{profile.environment.platform.platform_id}:{platform_key[:20]}")
    result = []
    summary = {"source_mode": policy, "platform_local": local is not None, "environment_local": False,
               "will_attempt_ghcr_pull": False, "alternatives": {}}
    platform_id = local["image_id"] if local else None
    if not local and policy != "build":
        ref = registry_reference("platform", platform_key)
        remote = manifest_probe(ref)
        summary["will_attempt_ghcr_pull"] |= ref.startswith("ghcr.io/")
        result.append(DownloadSource("oci", "https://" + ref, remote["bytes"] if remote else None,
                                     cache_status="miss", detail="platform; full compressed size upper bound"))
        platform_id = remote["image_id"] if remote else None
    if platform_id:
        environment_key = content_digest({"definition": runtime_fingerprint(profile.environment, project_dir),
                                          "platform_image_id": platform_id})
        local_env = inspect(f"acprof-runtime-env:{environment_key[:20]}")
        summary["environment_local"] = local_env is not None
        if local_env:
            return result, summary
        if policy != "build":
            ref = registry_reference("environment", environment_key)
            remote = manifest_probe(ref)
            summary["will_attempt_ghcr_pull"] |= ref.startswith("ghcr.io/")
            result.append(DownloadSource("oci", "https://" + ref, remote["bytes"] if remote else None,
                                         cache_status="miss", detail="environment; full compressed size upper bound"))
            summary["alternatives"]["local_build"] = "exact locks; cache state unknown until BuildKit runs"
            summary["alternatives"]["domestic_registry"] = os.environ.get("ACPROF_RUNTIME_REGISTRY") or "not configured"
            return result, summary
    elif policy != "build":
        result.append(DownloadSource("oci", "https://" + os.environ.get("ACPROF_RUNTIME_REGISTRY", DEFAULT_RUNTIME_REGISTRY),
                                     None, detail="environment reference depends on unavailable platform config digest"))
        return result, summary
    packages = identity["packages"]
    if local:
        parent_names = {item["name"] for item in identity["platform"]["packages"]}
        packages = [item for item in packages if item["name"] not in parent_names]
    if not local:
        base = profile.environment.platform.python_base_image
        if inspect(base) is None:
            remote = manifest_probe(base)
            result.append(DownloadSource("oci", "https://registry-1.docker.io/" + base,
                                         remote["bytes"] if remote else None, cache_status="miss", detail="Python base"))
    artifacts = [("python", item) for item in packages]
    if not local:
        artifacts += [("debian", item) for item in identity["platform"]["system"]["artifacts"]]
    recorded_sizes = {}
    for lock in (profile.environment.requirements_lock, profile.environment.platform.requirements_lock):
        metadata = (project_dir / lock).with_suffix(".artifacts.json")
        if metadata.is_file():
            recorded_sizes.update({(entry["url"], entry["sha256"]): entry.get("size")
                                   for entry in json.loads(metadata.read_text())["artifacts"]})
    with ThreadPoolExecutor(max_workers=8) as executor:
        sizes = list(executor.map(lambda pair: pair[1].get("size") or
            recorded_sizes.get((pair[1]["url"], pair[1]["sha256"])) or size_probe(pair[1]["url"]), artifacts))
    result += [DownloadSource(kind, item["url"], size, cache_status="unknown",
                  detail=f"{item['name']}; conservative full artifact; BuildKit cache verified during build")
               for (kind, item), size in zip(artifacts, sizes)]
    summary["alternatives"]["local_build_bytes"] = summarize_downloads(result)["expected_download_bytes"]
    return result, summary


def preflight(task, profile, project_dir, model_plan: dict, *, root=None) -> dict:
    from acprof.host.model_store import (
        model_sources,
        probe_model_download,
        require_space,
        store_root,
    )
    from acprof.network_policy import parse_bytes
    models = model_sources(model_plan, root)
    disk = require_space(model_plan, root)
    enforce_download_budget(summarize_downloads(models), os.environ.get("ACPROF_MAX_DOWNLOAD"))
    probes = probe_model_download(model_plan, root)
    models = model_sources(model_plan, root)  # Probe may have selected another Hub entry.
    sources, runtime = runtime_sources(profile, Path(project_dir))
    report = summarize_downloads([*models, *sources])
    report["runtime"] = runtime
    report["disk"] = disk
    report["network_probe"] = probes
    report["max_download_bytes"] = parse_bytes(os.environ.get("ACPROF_MAX_DOWNLOAD"))
    report["model_store_path"] = str(Path(root).expanduser().resolve() if root is not None else store_root())
    from acprof.host.static_metadata import _docker_storage_metadata
    report["docker_storage"] = _docker_storage_metadata()
    report["model"] = {"model_id": task.model_id, "revision": task.model_revision,
        "source": model_plan.get("source", "huggingface"),
        "total_bytes": model_plan.get("total_selected_bytes", model_plan["selected_bytes"]),
        "cached_bytes": sum(s.cached_bytes for s in models), "endpoint": model_plan["endpoint"]}
    report["metadata_traffic"] = "small API/HEAD/config requests precede bulk budget; no weights, wheel or OCI layers"
    print("[network-preflight] " + json.dumps(report, ensure_ascii=False), flush=True)
    enforce_download_budget(report, os.environ.get("ACPROF_MAX_DOWNLOAD"))
    if os.environ.get("ACPROF_INTERACTIVE_PREPARATION") == "1" and report["expected_download_bytes"] != 0:
        from acprof.host.collection_workflow import PreparationWorkflow
        workflow = PreparationWorkflow(interactive=True)
        summary = {key: report[key] for key in (
            "expected_download_bytes", "network", "public_egress", "network_probe", "runtime", "disk",
            "max_download_bytes", "model_store_path", "docker_storage", "model",
        )}
        workflow.ask("image", "review", resolved=True, questions=[], download_report=summary)
    return report
