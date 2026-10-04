"""Source identity and ModelScope's public Hub SDK, outside measurement.

Sources never alias each other. ModelScope uses its own commit and per-file
SHA256; identical repository names are not evidence of identical artifacts.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

from acprof.container.model_files import ModelFilesError, safe_path

MODEL_SOURCES = ("huggingface", "modelscope")
MODELSCOPE_ENDPOINT = "https://modelscope.cn"


class ModelScopeRevisionError(ValueError):
    """The repository responded, but the requested Git ref was absent."""


def model_source(value: str | None = None) -> str:
    source = value or os.environ.get("ACPROF_MODEL_SOURCE", "huggingface")
    if source not in MODEL_SOURCES:
        raise ValueError("model source must be huggingface/modelscope")
    return source



def repository_context(info) -> dict:
    """Record original Hub observations before selection or user overrides."""
    def value(record, name):
        return record.get(name) if isinstance(record, dict) else getattr(record, name, None)

    transformers = getattr(info, "transformers_info", None)
    safetensors = getattr(info, "safetensors", None)
    config = getattr(info, "config", None)
    return {"schema_version": 1,
        "repository_files": sorted({safe_path(item.rfilename) for item in (info.siblings or [])}),
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "library_name": getattr(info, "library_name", None),
        "config": config if isinstance(config, dict) else {},
        "transformers_info": {key: field for key in ("auto_model", "pipeline_tag", "processor", "custom_class")
                              if isinstance(field := value(transformers, key), str)},
        "tags": list(getattr(info, "tags", None) or []),
        "card_data": {"license": value(getattr(info, "card_data", None), "license")},
        "safetensors": ({key: value(safetensors, key) for key in ("total", "parameters")}
                        if safetensors is not None else None)}


def modelscope_api():
    from modelscope_hub import HubApi
    return HubApi(endpoint=MODELSCOPE_ENDPOINT, token=os.environ.get("MODELSCOPE_API_TOKEN") or None)


def modelscope_revision(model_id: str, revision: str | None) -> str:
    from huggingface_hub.utils import validate_repo_id

    from acprof.host import command as host_command
    validate_repo_id(model_id)
    if revision and re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    ref = revision or "master"
    if not re.fullmatch(r"[\w./-]+", ref) or ".." in ref:
        raise ValueError("invalid ModelScope revision")
    # The file API's Revision field is the *last file commit*, not repo HEAD.
    # Resolve the actual Git ref, then give the same immutable commit to every
    # API/file request. ls-remote only transfers refs, never model payloads.
    try:
        result = host_command.run_command(["git", "ls-remote", "--exit-code",
            f"{MODELSCOPE_ENDPOINT}/{model_id}.git", f"refs/heads/{ref}",
            f"refs/tags/{ref}", f"refs/tags/{ref}^{{}}"], timeout=30,
            env_overrides={"GIT_TERMINAL_PROMPT": "0"}, check=False)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("ModelScope revision lookup timed out") from exc
    refs = {name: sha for line in result.stdout.splitlines()
            if len(parts := line.split()) == 2 for sha, name in [parts]
            if re.fullmatch(r"[0-9a-f]{40}", sha)}
    commit = refs.get(f"refs/tags/{ref}^{{}}") or refs.get(f"refs/tags/{ref}") or refs.get(f"refs/heads/{ref}")
    if result.returncode == 2 or result.returncode == 0 and not commit:
        raise ModelScopeRevisionError("ModelScope branch or tag does not exist")
    if result.returncode:
        raise ConnectionError("Cannot pin ModelScope revision; check network/Git authentication or supply a full commit SHA")
    return commit


def modelscope_info(model_id: str, revision: str | None = None):
    api = modelscope_api()
    commit = modelscope_revision(model_id, revision)
    files = api.legacy.list_repo_files(model_id, "model", revision=commit, recursive=True)
    repo = api.get_repo(model_id, "model")
    siblings = []
    for item in files:
        if item.get("Type") != "blob":
            continue
        name, size, sha256 = safe_path(item.get("Path")), item.get("Size"), item.get("Sha256")
        # Use the raw public legacy API: FileInfo maps absent sizes to zero.
        if not re.fullmatch(r"[0-9a-f]{64}", sha256 or "") or type(size) is not int or size < 0:
            raise ModelFilesError("ModelScope file metadata must contain size and SHA256")
        siblings.append(SimpleNamespace(rfilename=name, size=size,
                                        blob_id=None, lfs={"sha256": sha256}))
    tags = repo.tags or []
    library = "diffusers" if "library:diffusers" in tags else "transformers"
    return SimpleNamespace(sha=commit, siblings=siblings, pipeline_tag=(repo.tasks[0] if len(repo.tasks or []) == 1 else None),
        library_name=library, tags=tags, config={}, card_data={"license": repo.license},
        safetensors=None, transformers_info=None, source="modelscope")


def modelscope_snapshot(root: Path, model_id: str, revision: str) -> Path:
    from huggingface_hub.utils import validate_repo_id
    validate_repo_id(model_id)
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("ModelScope snapshot requires a full commit SHA")
    return root / "modelscope" / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision


def modelscope_file(model_id: str, name: str, revision: str, root: Path, *, sha256: str) -> str:
    name = safe_path(name)
    return str(modelscope_api().download_file(model_id, "model", name, revision=revision,
        local_dir=modelscope_snapshot(root, model_id, revision), expected_sha256=sha256))


def modelscope_metadata(model_id: str, name: str, revision: str | None, max_bytes: int) -> str:
    from acprof.host.model_store import store_root
    info = modelscope_info(model_id, revision)
    item = next((f for f in info.siblings if f.rfilename == name), None)
    if item is None or not 0 <= item.size <= max_bytes:
        raise ModelFilesError(f"ModelScope metadata missing or exceeds {max_bytes} bytes: {name}")
    return modelscope_file(model_id, name, info.sha, store_root(), sha256=item.lfs["sha256"])
