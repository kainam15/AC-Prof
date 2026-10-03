"""Materialize a pinned static source graph, never a model snapshot."""
from __future__ import annotations

import hashlib
import json
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from acprof.model_evidence import content_digest, pinned_revision
from acprof.model_metadata_analysis import MAX_SOURCE_FILES, MAX_TOTAL_SOURCE_BYTES
from acprof.model_source_analysis import MAX_SOURCE_BYTES, SOURCE_METADATA_FILES, parse_source
from acprof.model_spec import custom_code_files, task_model_spec

_WEIGHTS = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx"}


@dataclass(frozen=True)
class SourceBundle:
    root: Path
    identity: str
    files: tuple[str, ...]


def _source_name(name: str) -> bool:
    path = PurePosixPath(name)
    return (bool(name) and not path.is_absolute() and ".." not in path.parts
            and "\\" not in name and path.as_posix() == name
            and path.suffix == ".py" and not _WEIGHTS.intersection(s.lower() for s in path.suffixes))


@contextmanager
def source_bundle(task_info):
    """Only fetch missing declared files; incomplete graphs fail before any I/O."""
    from acprof.host.detect import read_model_source

    config = task_info.repository_metadata.get("config.json", task_info.model_config) or {}
    selected = task_model_spec(task_info).get("pipeline_task")
    if selected in config.get("custom_pipelines", {}):
        config = {**config, "custom_pipelines": {selected: config["custom_pipelines"][selected]}}
    contract = task_info.model_resolution.get("contract", {})
    graph = contract.get("fields", {}).get("custom_code", {}).get("value")
    graph = graph or task_info.model_resolution.get("source_graph")
    roots = custom_code_files(config)
    for name, metadata in task_info.repository_metadata.items():
        if name != "config.json" and isinstance(metadata, dict):
            roots.extend(custom_code_files(metadata))
    if roots and (not isinstance(graph, dict) or not graph.get("files")):
        raise ValueError("interface probe source graph incomplete")
    names = (graph or {}).get("files", [])
    if (not pinned_revision(task_info.model_revision) or not isinstance(names, list)
            or any(not isinstance(name, str) or not _source_name(name) for name in names)
            or len(names) > MAX_SOURCE_FILES or len(set(names)) != len(names)
            or not set(roots) <= set(names)
            or graph and graph.get("revision") != task_info.model_revision):
        raise ValueError("interface probe source graph incomplete or unsafe")
    sources = contract.get("sources", {}) if contract.get("fields", {}).get("custom_code") else (graph or {}).get("sources", {})
    if {name for name in sources if name.endswith(".py")} != set(names):
        raise ValueError("interface probe source graph incomplete: source dependency missing")
    if any(name not in sources or not sources[name].get("sha256") for name in names):
        raise ValueError("interface probe source graph incomplete: missing source identity")
    with tempfile.TemporaryDirectory(prefix="acprof-source-probe-") as directory:
        root = Path(directory)
        manifest = {}
        total = 0
        for name in names:
            text = task_info.repository_sources.get(name)
            if text is None:
                text = read_model_source(task_info.model_id, name, task_info.model_revision)
            data = text.encode("utf-8")
            total += len(data)
            if len(data) > MAX_SOURCE_BYTES or total > MAX_TOTAL_SOURCE_BYTES:
                raise ValueError("interface probe source size limit exceeded")
            digest = hashlib.sha256(data).hexdigest()
            if digest != sources[name]["sha256"]:
                raise ValueError(f"interface probe source changed since resolution: {name}")
            parse_source(text, name)
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            manifest[name] = digest
        # Parsed metadata and the reviewed declaration are small local text,
        # not requests for arbitrary repository assets or dependency weights.
        metadata = {name: value for name, value in task_info.repository_metadata.items() if name in SOURCE_METADATA_FILES}
        metadata["config.json"] = config
        if spec := task_model_spec(task_info):
            metadata["acprof_model.json"] = spec
        for name, value in metadata.items():
            data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            if len(data) > 1024 * 1024:
                raise ValueError("interface probe metadata size limit exceeded")
            (root / name).write_bytes(data)
            manifest[name] = hashlib.sha256(data).hexdigest()
        yield SourceBundle(root, content_digest({"model": task_info.model_id,
                           "revision": task_info.model_revision, "files": manifest}), tuple(sorted(manifest)))
