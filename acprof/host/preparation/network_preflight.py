"""Bound downloads before build/pull; unknown bytes never count as zero."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from acprof.artifacts import read_json_object
from acprof.host.images.environment import (
    DEFAULT_RUNTIME_REGISTRY,
    platform_fingerprint,
    registry_reference,
    runtime_fingerprint,
)
from acprof.host.preparation.download_probe import failure_reason, probe_allowance, probe_record
from acprof.network_policy import (
    DEPENDENCY_USER_AGENT,
    DownloadSource,
    dependency_artifact_url,
    enforce_download_budget,
    is_direct_artifact_mirror,
    require_direct_artifact_mirror_redirect,
    require_source_transition,
    summarize_downloads,
)
from acprof.runtime.dependency_locks import content_digest
from acprof.runtime.profiles import environment_identity
from acprof.system import command as host_command


class PolicyRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        require_direct_artifact_mirror_redirect(req.full_url, newurl)
        require_source_transition(req.full_url, newurl)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and req.get_method() == "HEAD":
            # urllib otherwise turns a redirected HEAD into a payload-fetching GET.
            redirected.method = "HEAD"
        return redirected


def artifact_size(url: str) -> int | None:
    """HEAD only; no artifact GET is permitted during estimation."""
    return probe_artifact(url)[1]


def probe_artifact(url: str, *, category: str = "python") -> tuple[dict, int | None]:
    report = probe_record(category, url, "HEAD")
    report["scope"] = "host metadata only; BuildKit network and complete payload not verified"
    if not url:
        report["reason"] = "source-unknown"
        return report, None
    started = time.monotonic()
    handlers = [urllib.request.ProxyHandler({})] if is_direct_artifact_mirror(url) else []
    opener = urllib.request.build_opener(*handlers, PolicyRedirectHandler())
    try:
        request = urllib.request.Request(url, headers={"User-Agent": DEPENDENCY_USER_AGENT}, method="HEAD")
        with opener.open(request, timeout=8) as response:
            size = response.headers.get("Content-Length")
            report.update(status="passed", final_host=urlsplit(response.url).hostname, http_status=response.status)
            return report, int(size) if size and size.isdecimal() else None
    except (OSError, ValueError) as exc:
        report.update(status="failed", reason=failure_reason(exc, phase="HEAD"),
                      http_status=getattr(exc, "code", None),
                      final_host=urlsplit(getattr(exc, "url", "")).hostname)
        return report, None
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 4)


def probe_runtime_downloads(sources: list[DownloadSource], existing: list[dict] | None = None) -> list[dict]:
    """One representative HEAD per required package origin; OCI is inspected by Docker."""
    reports = []
    seen = {(item["category"], urlsplit(item["source"]).netloc) for item in existing or []}
    for source in sources:
        key = (source.category, urlsplit(source.url).netloc)
        if source.category not in {"python", "debian"} or source.cache_status == "hit" or key in seen:
            continue
        seen.add(key)
        report, _ = probe_artifact(source.url, category=source.category)
        reports.append(report)
    return reports


def registry_manifest(reference: str, *, evidence: list[dict] | None = None) -> dict | None:
    """OCI metadata only; returns compressed upper bound and Docker config ID."""
    entry = reference.split("/")[0]
    host = entry if "/" in reference and ("." in entry or ":" in entry or entry == "localhost") else "registry-1.docker.io"
    report = probe_record("oci", "https://" + host, "manifest")
    report.update(status="failed", reason="manifest-unavailable",
                  scope="Docker manifest only; layer transfer and final storage host not verified")
    started = time.monotonic()
    try:
        result = host_command.run_command(['docker', 'manifest', 'inspect', '--verbose', reference], timeout=20, check=False)
        if result.returncode:
            stderr = result.stderr.lower()
            status = re.search(r"\b(401|403|404|429|5\d\d)\b", stderr)
            if status:
                report.update(reason="http-" + status[1], http_status=int(status[1]))
            elif "timeout" in stderr or "timed out" in stderr:
                report["reason"] = "timeout"
            elif "certificate" in stderr or "tls" in stderr:
                report["reason"] = "tls"
            elif "no such host" in stderr or "lookup" in stderr:
                report["reason"] = "dns"
            return None
        data = json.loads(result.stdout)
        if isinstance(data, list):
            matches = [entry for entry in data if isinstance(entry, dict)
                       and entry.get("Descriptor", {}).get("platform", {}).get("architecture") == "amd64"
                       and entry.get("Descriptor", {}).get("platform", {}).get("os") == "linux"]
            if len(matches) != 1:
                return None
            data = matches[0]
        if not isinstance(data, dict):
            return None
        manifest = data.get("SchemaV2Manifest") or data.get("OCIManifest") or data
        if not isinstance(manifest, dict):
            return None
        layers, config = manifest.get("layers"), manifest.get("config")
        if not isinstance(layers, list) or not isinstance(config, dict):
            return None
        entries = [*layers, config]
        if any(not isinstance(item, dict) or type(item.get("size")) is not int or item["size"] < 0
               for item in entries):
            return None
        image_id = config.get("digest")
        if not isinstance(image_id, str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", image_id):
            return None
        report.update(status="passed", reason=None)
        return {"bytes": sum(item["size"] for item in entries), "image_id": image_id}
    except subprocess.TimeoutExpired:
        report["reason"] = "timeout"
        return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 4)
        if evidence is not None:
            evidence.append(report)


def _recorded_artifact_sizes(path: Path) -> dict[tuple[str, str], int | None]:
    metadata = read_json_object(path, label="dependency artifact metadata")
    if metadata.get("schema_version") != 1 or not isinstance(metadata.get("artifacts"), list):
        raise ValueError(f"{path}: invalid dependency artifact metadata schema")
    sizes: dict[tuple[str, str], int | None] = {}
    for entry in metadata["artifacts"]:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: dependency artifact metadata entries must be objects")
        url, digest, size = entry.get("url"), entry.get("sha256"), entry.get("size")
        if (not isinstance(url, str) or not url or not isinstance(digest, str) or not digest
                or (size is not None and (type(size) is not int or size < 0))):
            raise ValueError(f"{path}: invalid dependency artifact metadata entry")
        key = (url, digest)
        if key in sizes:
            raise ValueError(f"{path}: duplicate dependency artifact metadata entry")
        sizes[key] = size
    return sizes


def runtime_sources(profile, project_dir: Path, *, inspect=None, size_probe=artifact_size,
                    manifest_probe=None) -> tuple[list[DownloadSource], dict]:
    from acprof.host.images.runtime import inspect_identity
    inspect = inspect or inspect_identity
    evidence: list[dict] = []
    if manifest_probe is None:
        def manifest_probe(reference):
            return registry_manifest(reference, evidence=evidence)
    identity = environment_identity(profile.environment, project_dir)
    policy = os.environ.get("ACPROF_RUNTIME_IMAGE_SOURCE", "auto")
    if policy not in {"auto", "pull", "build"}:
        raise ValueError("ACPROF_RUNTIME_IMAGE_SOURCE must be auto/pull/build")
    platform_key = platform_fingerprint(profile.environment.platform, project_dir)
    local = inspect(f"acprof-platform-{profile.environment.platform.platform_id}:{platform_key[:20]}")
    result = []
    summary: dict[str, Any] = {"source_mode": policy, "platform_local": local is not None, "environment_local": False,
               "will_attempt_ghcr_pull": False, "alternatives": {}, "network_probe": evidence}
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
            recorded_sizes.update(_recorded_artifact_sizes(metadata))
    def resolve_size(pair):
        kind, item = pair
        size = item.get("size")
        if size is not None:
            return size
        size = recorded_sizes.get((item["url"], item["sha256"]))
        return size if size is not None else size_probe(dependency_artifact_url(kind, item["url"]))

    with ThreadPoolExecutor(max_workers=8) as executor:
        sizes = list(executor.map(resolve_size, artifacts))
    result += [DownloadSource(kind, dependency_artifact_url(kind, item["url"]), size, cache_status="unknown",
                  detail=f"{item['name']}; original lock URL unchanged; BuildKit cache verified during build")
               for (kind, item), size in zip(artifacts, sizes, strict=True)]
    summary["alternatives"]["local_build_bytes"] = summarize_downloads(result)["expected_download_bytes"]
    return result, summary


def process_sources(task, project_dir: Path) -> tuple[list[DownloadSource], dict]:
    """uv consumes the original lock URLs; BuildKit mirror options do not apply."""
    from acprof.host.runtimes.environment import environment_plan, verify_environment
    root, identity, declaration = environment_plan(task, project_dir)
    target = root / content_digest(identity)
    summary = {"kind": "process", "source_mode": "build", "environment_local": False,
               "will_attempt_ghcr_pull": False, "network_probe": []}
    if target.exists():
        manifest = read_json_object(target / "manifest.json", label="Process environment")
        if manifest["identity"] != identity:
            raise ValueError("cached Process environment identity changed")
        verify_environment(target, manifest)
        summary["environment_local"] = True
        return [], summary
    metadata = (project_dir / declaration.requirements_lock).with_suffix(".artifacts.json")
    sizes = _recorded_artifact_sizes(metadata) if metadata.is_file() else {}
    sources = [DownloadSource("python", item["url"], sizes.get((item["url"], item["sha256"])),
                              detail="uv exact lock; package cache unknown until installation")
               for item in identity["artifacts"]]
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("Process preparation requires uv")
    interpreter = host_command.run_command([uv, "python", "find", "--no-python-downloads", "--offline",
                                            declaration.platform.python_version], check=False, timeout=10)
    if interpreter.returncode:
        sources.append(DownloadSource("python", "", None,
                                      detail="uv Python interpreter source and size not verified"))
    return sources, summary


def preflight(task, profile, project_dir, model_plan: dict, *, root=None,
              runtime_plan: tuple[list[DownloadSource], dict] | None = None) -> dict:
    from acprof.host.model_store.store import (
        enforce_model_download_budgets,
        entry_key,
        model_sources,
        probe_model_download,
        require_space,
        store_lock,
        store_root,
    )
    from acprof.network_policy import parse_bytes
    entry_id = entry_key(task)
    models = model_sources(model_plan, root, entry_id=entry_id)
    enforce_model_download_budgets(summarize_downloads(models))
    sources, runtime = runtime_sources(profile, Path(project_dir)) if runtime_plan is None else runtime_plan
    planned = summarize_downloads([*models, *sources])
    enforce_download_budget(planned, os.environ.get("ACPROF_MAX_DOWNLOAD"))
    allowance = probe_allowance(summarize_downloads(models)["expected_download_bytes"], planned["expected_download_bytes"])
    print(f"[network-probe] HEAD + limited model GET; at most {allowance} payload bytes; "
          "no cache writes; complete downloads require confirmation in interactive mode", flush=True)
    probes = probe_model_download(model_plan, root, entry_id=entry_id, max_bytes=allowance)
    store = Path(root).expanduser().resolve() if root is not None else store_root()
    # A pristine store has no cached entries to race with GC. Keep bounded
    # preflight probes read-only rather than creating an empty cache lock.
    has_cache_state = store.exists() and any(store.iterdir())
    with (store_lock(store) if has_cache_state else nullcontext()):
        # Keep existing cache bytes and disk budget on one GC-stable snapshot.
        models = model_sources(model_plan, store, entry_id=entry_id)
        disk = require_space(models, store)
    probe_bytes = sum(item.get("received_bytes", 0) for item in probes)
    model_bytes = summarize_downloads(models)["expected_download_bytes"]
    enforce_model_download_budgets({"expected_download_bytes": None if model_bytes is None else model_bytes + probe_bytes})
    report = summarize_downloads([*models, *sources])
    report["runtime"] = runtime
    report["disk"] = disk
    report["network_probe"] = [*probes, *runtime.get("network_probe", []),
                               *probe_runtime_downloads(sources, runtime.get("network_probe"))]
    report["probe_payload_bytes"] = probe_bytes
    report["probe_byte_limit"] = allowance
    report["max_download_bytes"] = parse_bytes(os.environ.get("ACPROF_MAX_DOWNLOAD"))
    report["model_max_download_bytes"] = parse_bytes(os.environ.get("ACPROF_MODEL_MAX_DOWNLOAD"))
    report["model_store_path"] = str(store)
    from acprof.host.metadata.machine import docker_storage_metadata
    report["docker_storage"] = {} if runtime.get("kind") == "process" else docker_storage_metadata()
    report["model"] = {"model_id": task.model_id, "revision": task.model_revision,
        "source": model_plan.get("source", "huggingface"),
        "total_bytes": model_plan.get("total_selected_bytes", model_plan["selected_bytes"]),
        "cached_bytes": sum(s.cached_bytes for s in models), "endpoint": model_plan["endpoint"]}
    report["metadata_traffic"] = "API/HEAD/config metadata excluded; bounded model GET included in payload budget; no wheel or OCI layers"
    print("[network-preflight] " + json.dumps(report, ensure_ascii=False), flush=True)
    expected = report["expected_download_bytes"]
    enforce_download_budget({"expected_download_bytes": None if expected is None else expected + report["probe_payload_bytes"]},
                            os.environ.get("ACPROF_MAX_DOWNLOAD"))
    if os.environ.get("ACPROF_INTERACTIVE_PREPARATION") == "1" and report["expected_download_bytes"] != 0:
        from acprof.host.preparation.collection_workflow import PreparationWorkflow
        workflow = PreparationWorkflow(interactive=True)
        summary = {key: report[key] for key in (
            "expected_download_bytes", "network", "public_egress", "network_probe", "runtime", "disk",
            "max_download_bytes", "model_max_download_bytes", "model_store_path", "docker_storage", "model",
            "sources", "probe_payload_bytes", "probe_byte_limit",
        )}
        workflow.ask("image", "review", resolved=True, questions=[], download_report=summary)
    return report
