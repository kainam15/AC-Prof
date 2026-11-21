"""AC-Prof - HuggingFace Model Task Auto-Detection.

Collect Hub, repository and artifact evidence at a pinned revision, then select
an interface using explicit overrides or unambiguous metadata. No model code is
loaded on the host; runtime compatibility is verified separately in containers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from acprof.extensions import CATALOG, UnsupportedExtensionError
from acprof.host.preparation.model_errors import ModelLookupError, model_lookup_error
from acprof.hub.huggingface.auth import hf_token_for_url
from acprof.hub.huggingface.download import hf_endpoint_scope
from acprof.hub.repository import MetadataFile, modelscope_info_scope, read_metadata_file
from acprof.models.diffusion_pipelines import diffusers_index_task
from acprof.models.files import ModelFilesError
from acprof.models.metadata_validation import (
    MetadataValidationError,
    finite_metadata_diagnostics,
    module_config_files,
    parse_metadata_json,
)


@dataclass
class TaskInfo:
    model_id: str
    pipeline_tag: str
    task_family: str
    runtime_backend: str
    library_name: str
    model_revision: str
    detection_method: str  # "hub_api" / "config_infer" / "manual"
    model_source: str = "huggingface"
    requested_revision: str | None = None
    parameter_count: int | None = None
    parameter_bytes: int | None = None
    precision_dtype: str | None = None
    parameter_dtype_counts: dict[str, int] = field(default_factory=dict)
    quantized: bool | None = None
    quantization_method: str | None = None
    quantization_config: dict[str, Any] = field(default_factory=dict)
    model_license: str | None = None
    model_metadata_source: str | None = None
    model_config: dict[str, Any] = field(default_factory=dict)
    runtime_profile_id: str = ""
    model_adapter: str = "family-default"
    model_download_policy: str = "auto"
    repository_files: tuple[str, ...] = ()
    repository_metadata: dict[str, Any] = field(default_factory=dict)
    metadata_errors: tuple[str, ...] = ()
    metadata_diagnostics: tuple[dict[str, Any], ...] = ()
    metadata_sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_resolution: dict[str, Any] = field(default_factory=dict)
    model_spec: dict[str, Any] = field(default_factory=dict)
    hub_metadata: dict[str, Any] = field(default_factory=dict)
    model_store: dict[str, Any] = field(default_factory=dict)
    repository_sources: dict[str, str] = field(default_factory=dict, repr=False)
    task_discovery: dict[str, Any] = field(default_factory=dict)
    selected_tasks: list[dict[str, Any]] = field(default_factory=list)
    inference_configuration: dict[str, Any] = field(default_factory=dict)


def _architecture_metadata(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    keys = ("model_type", "architectures", "auto_map", "custom_pipelines", "transformers_version", "torch_dtype", "dtype",
            "quantization_config", "chronos_pipeline_class", "pretrained_cfg", "architecture",
            "task_specific_params", "is_encoder_decoder", "problem_type", "id2label")
    metadata = {key: config[key] for key in keys if key in config}
    if "pretrained_cfg" in metadata and "model_type" not in metadata:
        metadata["model_type"] = "timm_wrapper"
    return metadata


METADATA_MAX_BYTES = 1024 * 1024


def _download_metadata(
    model_id: str, name: str, revision: str | None = None, *,
    max_bytes: int = METADATA_MAX_BYTES, source: str | None = None,
) -> MetadataFile:
    """Bound pinned downloads and preserve the SDK's mutable-ref fallback cache."""
    from acprof.host.model_store.store import cached_metadata, store_root
    from acprof.hub.repository import model_source, modelscope_metadata
    source = model_source(source)
    cached = cached_metadata(model_id, name, revision, source=source, max_bytes=max_bytes)
    if cached is not None:
        return cached
    if source == "modelscope":
        return modelscope_metadata(model_id, name, revision, max_bytes, cache_root=store_root())
    from huggingface_hub import hf_hub_download

    from acprof.hub.huggingface.download import try_hf_endpoints
    from acprof.hub.huggingface.transport import configure_hf_transport
    from acprof.models.evidence import pinned_revision

    configure_hf_transport()
    def download(endpoint):
        size = None
        if revision is not None and pinned_revision(revision):
            metadata = hf_hub_download(repo_id=model_id, filename=name,
                                       revision=revision, endpoint=endpoint, token=hf_token_for_url(endpoint), dry_run=True)
            size = getattr(metadata, "file_size", None)
            if type(size) is not int or not 0 <= size <= max_bytes:
                raise ValueError(f"{name} size is unknown or exceeds {max_bytes} bytes")
            if getattr(metadata, "commit_hash", None) != revision:
                raise ValueError("metadata preflight returned a different model revision")
        # Mutable-ref fallback must let the SDK record its ref for offline use.
        # A dry-run followed by a SHA download would leave a fresh ref absent.
        path = hf_hub_download(repo_id=model_id, filename=name, revision=revision,
                               endpoint=endpoint, token=hf_token_for_url(endpoint))
        return read_metadata_file(path, max_bytes=max_bytes, expected_size=size)
    return try_hf_endpoints(model_id, download)


def _read_json_metadata(
    metadata: MetadataFile, *, max_bytes: int = METADATA_MAX_BYTES, filename: str | None = None,
) -> Any:
    if len(metadata.content) > max_bytes:
        raise ValueError(f"{metadata.path.name} exceeds {max_bytes} bytes")
    return parse_metadata_json(metadata.content, filename or metadata.path.name)


def _raw_metadata_source(metadata: MetadataFile, revision: str) -> dict[str, Any]:
    return {"sha256": hashlib.sha256(metadata.content).hexdigest(), "bytes": len(metadata.content),
            "revision": revision, "encoding": "raw"}


def _repository_json(metadata: MetadataFile, name: str, pending: list[str], sha256: str) -> Any:
    value = _read_json_metadata(metadata, filename=name)
    if name == "modules.json":
        pending.extend(path for path in module_config_files(value, sha256) if path not in pending)
    elif not isinstance(value, dict):
        raise MetadataValidationError([{"reason_code": "metadata_invalid", "file": name, "json_pointer": "",
                                        "raw_sha256": sha256, "detail": f"{name} must contain dict"}])
    return value


def _repository_metadata(
    model_id: str, revision: str, info: Any, lookup_diagnostics: list[str] | None = None,
) -> dict[str, Any]:
    """Read small, non-executable metadata at the same commit as the weights."""
    from acprof.models.artifact_layout import checkpoint_metadata_indexes
    from acprof.models.evidence import pinned_revision

    files = tuple(sorted({str(getattr(item, "rfilename", "")) for item in
                          (getattr(info, "siblings", None) or []) if getattr(item, "rfilename", "")}))
    metadata, errors, diagnostics, sources = {}, [], [], {}
    if files and not pinned_revision(revision):
        return {"repository_files": files, "repository_metadata": {},
                "metadata_errors": ("model revision must be a full commit SHA before metadata reads",),
                "metadata_diagnostics": (), "metadata_sources": {}}
    names = ["config.json", "model_index.json", "modules.json", "adapter_config.json",
             "acprof_model.json", "tokenizer_config.json", "processor_config.json", "preprocessor_config.json",
             "generation_config.json", "config_sentence_transformers.json", "sentence_bert_config.json"]

    def read_name(name: str) -> None:
        if files and name not in files:
            return
        try:
            downloaded = _download_metadata(model_id, name, revision)
            sources[name] = _raw_metadata_source(downloaded, revision)
            metadata[name] = _repository_json(downloaded, name, names, sources[name]["sha256"])
        except MetadataValidationError as exc:
            diagnostics.extend(exc.diagnostics)
            errors.append(f"{name}: {_format_failure(exc)}")
        except Exception as exc:
            if files:
                errors.append(f"{name}: {_format_failure(exc)}")
            else:
                _record_failure(lookup_diagnostics, "model_index", _format_failure(exc))

    if files:
        for name in names:
            read_name(name)
        try:
            indexes = checkpoint_metadata_indexes(set(files), metadata)
        except ValueError as exc:
            errors.append(f"checkpoint layout: {_format_failure(exc)}")
        else:
            for name in indexes:
                if name not in metadata:
                    read_name(name)
    elif (pinned_revision(revision) and getattr(info, "library_name", None) == "diffusers"
          and not getattr(info, "pipeline_tag", None)):
        # Older Hub responses can identify the library without listing files.
        # Bind the same bytes to the task hint and its evidence.
        read_name("model_index.json")
    return {"repository_files": files, "repository_metadata": metadata, "metadata_errors": tuple(errors),
            "metadata_diagnostics": tuple(diagnostics), "metadata_sources": sources}


def dependency_metadata(repo_id: str, revision: str) -> dict:
    """Resolve identity and lazily read bounded checkpoint metadata, never weights."""
    from huggingface_hub import HfApi

    from acprof.hub.huggingface.download import try_hf_endpoints
    try:
        from acprof.host.model_store.store import cached_model_info
        info = cached_model_info(repo_id, revision, source="huggingface")
        if info is None:
            from acprof.hub.huggingface.transport import configure_hf_transport
            configure_hf_transport()
            info = try_hf_endpoints(repo_id, lambda endpoint: HfApi(endpoint=endpoint, token=hf_token_for_url(endpoint)).model_info(
                repo_id, revision=revision, files_metadata=False))
        def read_json(name):
            from acprof.models.evidence import pinned_revision
            from acprof.models.files import safe_path
            if not pinned_revision(info.sha):
                raise ValueError("dependency metadata requires a fixed commit SHA")
            name = safe_path(name)
            metadata = _download_metadata(repo_id, name, info.sha, max_bytes=4 * 1024 * 1024, source="huggingface")
            return _read_json_metadata(metadata, max_bytes=4 * 1024 * 1024, filename=name)
        return {"revision": info.sha, "files": [item.rfilename for item in info.siblings or []],
                "read_json": read_json}
    except Exception as exc:
        raise OSError(_format_failure(exc)) from exc


def read_model_source(model_id: str, name: str, revision: str) -> str:
    from acprof.models.evidence import pinned_revision
    from acprof.models.inspection.ast_parser import MAX_SOURCE_BYTES
    if not pinned_revision(revision):
        raise ValueError("source analysis requires a fixed commit SHA")
    try:
        data = _download_metadata(model_id, name, revision, max_bytes=MAX_SOURCE_BYTES).content
        if len(data) > MAX_SOURCE_BYTES:
            raise ValueError(f"{name} exceeds {MAX_SOURCE_BYTES} bytes")
        return data.decode("utf-8")
    except Exception as exc:
        raise OSError(f"{name}: {_format_failure(exc)}") from exc


_HUB_DTYPE_NAMES = {
    "BOOL": "BOOL",
    "F64": "FP64",
    "F32": "FP32",
    "F16": "FP16",
    "BF16": "BF16",
    "F8_E5M2": "FP8_E5M2",
    "F8_E4M3": "FP8_E4M3",
    "I64": "INT64",
    "I32": "INT32",
    "I16": "INT16",
    "I8": "INT8",
    "U64": "UINT64",
    "U32": "UINT32",
    "U16": "UINT16",
    "U8": "UINT8",
}
_QUANTIZED_PRECISIONS = {
    "FP8_E5M2",
    "FP8_E4M3",
    "INT32",
    "UINT32",
    "INT16",
    "UINT16",
    "INT8",
    "UINT8",
    "INT4",
    "UINT4",
}
_QUANTIZATION_TAGS = {
    "4-bit",
    "8-bit",
    "aqlm",
    "awq",
    "bitsandbytes",
    "bnb-4bit",
    "compressed-tensors",
    "eetq",
    "fp8",
    "gguf",
    "gptq",
    "hqq",
    "int4",
    "int8",
    "quanto",
    "spqr",
}
_DTYPE_BYTE_WIDTHS = {
    "BOOL": 1,
    "FP64": 8,
    "FP32": 4,
    "FP16": 2,
    "BF16": 2,
    "FP8_E5M2": 1,
    "FP8_E4M3": 1,
    "INT64": 8,
    "INT32": 4,
    "INT16": 2,
    "INT8": 1,
    "UINT64": 8,
    "UINT32": 4,
    "UINT16": 2,
    "UINT8": 1,
}


def _normalize_dtype_name(raw_dtype: object) -> str:
    value = str(raw_dtype or "").strip().upper().replace("TORCH.", "")
    aliases = {
        "FLOAT": "FP32",
        "FLOAT32": "FP32",
        "FLOAT16": "FP16",
        "HALF": "FP16",
        "BFLOAT16": "BF16",
        "FLOAT64": "FP64",
        "DOUBLE": "FP64",
        "INT64": "INT64",
        "LONG": "INT64",
        "INT32": "INT32",
        "INT16": "INT16",
        "INT8": "INT8",
        "UINT8": "UINT8",
    }
    return _HUB_DTYPE_NAMES.get(value, aliases.get(value, value))


def _mapping_value(value: object, key: str) -> object:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _json_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        return dict(converted) if isinstance(converted, dict) else {}
    return {}


def _parameter_bytes_from_dtype_counts(
    dtype_counts: dict[str, int],
) -> int | None:
    """Return logical tensor payload bytes, or ``None`` if not exact."""
    if not dtype_counts:
        return None

    total = 0
    for dtype, count in dtype_counts.items():
        byte_width = _DTYPE_BYTE_WIDTHS.get(dtype)
        if byte_width is None or count < 0:
            return None
        total += count * byte_width
    return total


def _hub_model_metadata(info: object) -> dict[str, Any]:
    """Extract reproducible model-card metadata from a Hub ModelInfo object."""
    safetensors = getattr(info, "safetensors", None)
    raw_dtype_counts = _mapping_value(safetensors, "parameters")
    dtype_counts: dict[str, int] = {}
    if isinstance(raw_dtype_counts, dict):
        for raw_dtype, raw_count in raw_dtype_counts.items():
            try:
                count = int(raw_count)
            except (TypeError, ValueError, OverflowError):
                continue
            dtype = _normalize_dtype_name(raw_dtype)
            if dtype:
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + count

    parameter_count: int | None = None
    raw_parameter_count = _mapping_value(safetensors, "total")
    try:
        if raw_parameter_count is not None:
            if not isinstance(raw_parameter_count, (int, str, float)):
                raise ValueError("invalid model parameter count")
            parameter_count = int(raw_parameter_count)
    except (TypeError, ValueError, OverflowError):
        pass

    config = _json_mapping(getattr(info, "config", None))
    quantization_config = _json_mapping(config.get("quantization_config"))
    tags = {
        str(tag).strip().lower()
        for tag in (getattr(info, "tags", None) or [])
        if str(tag).strip()
    }

    precision_dtype: str | None = None
    if dtype_counts:
        # Small INT64/BOOL tensors are commonly non-weight buffers. The
        # dominant dtype is the most useful single precision label.
        precision_dtype = max(dtype_counts.items(), key=lambda item: item[1])[0]
    else:
        config_dtype = config.get("torch_dtype") or config.get("dtype")
        if config_dtype:
            precision_dtype = _normalize_dtype_name(config_dtype)

    quantization_method = (
        quantization_config.get("quant_method")
        or quantization_config.get("quantization_method")
        or quantization_config.get("quant_type")
    )
    matching_quantization_tags = sorted(tags & _QUANTIZATION_TAGS)
    if not quantization_method and matching_quantization_tags:
        quantization_method = matching_quantization_tags[0]

    if quantization_config or matching_quantization_tags:
        quantized: bool | None = True
    elif precision_dtype in _QUANTIZED_PRECISIONS:
        quantized = True
    elif precision_dtype:
        quantized = False
    else:
        quantized = None

    card_data = getattr(info, "card_data", None)
    model_license = _mapping_value(card_data, "license")
    if not model_license:
        license_tags = sorted(
            tag.split(":", 1)[1]
            for tag in tags
            if tag.startswith("license:") and ":" in tag
        )
        model_license = license_tags[0] if license_tags else None

    return {
        "parameter_count": parameter_count,
        "parameter_bytes": _parameter_bytes_from_dtype_counts(dtype_counts),
        "precision_dtype": precision_dtype,
        "parameter_dtype_counts": dtype_counts,
        "quantized": quantized,
        "quantization_method": (
            str(quantization_method) if quantization_method else None
        ),
        "quantization_config": quantization_config,
        "model_license": str(model_license) if model_license else None,
        "model_metadata_source": "huggingface_hub",
    }


def _format_failure(exc: Exception) -> str:
    from acprof.host.preparation.preparation_log import redact_text

    detail = redact_text(str(exc), limit=None).strip().replace("\n", " ")
    if detail:
        return f"{type(exc).__name__}: {detail}"
    return type(exc).__name__


def _record_failure(diagnostics: list[str] | None, stage: str, reason: str) -> None:
    if diagnostics is not None:
        diagnostics.append(f"{stage}: {reason}")


def _detect_from_hub(
    model_id: str,
    diagnostics: list[str] | None = None,
    *, revision: str | None = None, failures: list[Exception] | None = None,
) -> TaskInfo | None:
    """Level 1: Query HuggingFace Hub API."""
    try:
        from huggingface_hub import HfApi

        from acprof.host.model_store.store import cached_model_info
        from acprof.hub.huggingface.download import try_hf_endpoints
        from acprof.hub.huggingface.transport import configure_hf_transport
        from acprof.hub.repository import model_source, modelscope_info
        info = cached_model_info(model_id, revision, source=model_source())
        if info is None:
            if model_source() == "modelscope":
                info = modelscope_info(model_id, revision)
            else:
                configure_hf_transport()
                info = try_hf_endpoints(model_id, lambda endpoint: HfApi(endpoint=endpoint, token=hf_token_for_url(endpoint)).model_info(
                    model_id, revision=revision))
    except Exception as exc:
        _record_failure(diagnostics, "hub_api", _format_failure(exc))
        if failures is not None:
            failures.append(exc)
        return None

    pipeline_tag = getattr(info, "pipeline_tag", None)
    hub_pipeline_tag = pipeline_tag
    task_hint_source = "extension_catalog"
    library_name = getattr(info, "library_name", None) or ""
    sha = getattr(info, "sha", None) or "main"
    from acprof.models.evidence import pinned_revision
    if revision is not None and pinned_revision(revision) and sha != revision:
        raise ValueError("Hub returned a different commit than the requested revision")
    transformers_info = getattr(info, "transformers_info", None)
    transformers_info = {key: value for key in ("auto_model", "pipeline_tag", "processor", "custom_class")
                         if isinstance(value := getattr(transformers_info, key, None), str)} if not isinstance(transformers_info, dict) else {
                             key: value for key, value in transformers_info.items()
                             if key in {"auto_model", "pipeline_tag", "processor", "custom_class"} and isinstance(value, str)}
    repository = _repository_metadata(model_id, sha, info, diagnostics)
    hub_config = _architecture_metadata(getattr(info, "config", None))
    hub_diagnostics = finite_metadata_diagnostics(hub_config, "hub.config")
    if hub_diagnostics:
        repository["metadata_diagnostics"] += tuple(hub_diagnostics)
        repository["metadata_errors"] += (str(MetadataValidationError(hub_diagnostics)),)
    model_config = {} if hub_diagnostics else hub_config
    model_config.update(_architecture_metadata(repository["repository_metadata"].get("config.json")))
    if not pipeline_tag and library_name == "diffusers" and not repository["metadata_errors"]:
        index = repository["repository_metadata"].get("model_index.json")
        pipeline_tag = diffusers_index_task(index)
        if index is not None and pipeline_tag is None:
            _record_failure(diagnostics, "model_index", f"unsupported pipeline class: {index.get('_class_name')!r}")
        task_hint_source = "model_index"

    # A Hub 'pretraining' label is not a runnable inference protocol. An exact
    # registered checkpoint head can supply a stronger, executable task contract.
    # Preserve the original Hub label in hub_metadata for audit/provenance.
    if pipeline_tag and pipeline_tag not in CATALOG.task_families:
        inferred = CATALOG.infer_task(model_config.get('architectures') or (), '')
        if inferred and inferred in CATALOG.task_families:
            pipeline_tag = inferred
            task_hint_source = 'registered_architecture'

    # Keep raw metadata when unresolved so explicit CLI declarations can repair it.
    task_family = CATALOG.task_families.get(pipeline_tag or "")
    runtime_backend = "unknown"
    try:
        resolved = CATALOG.resolve(task=pipeline_tag, library=library_name, config=model_config, model_id=model_id)
        pipeline_tag, task_family, runtime_backend = resolved.task, resolved.family, resolved.backend
    except UnsupportedExtensionError as exc:
        _record_failure(diagnostics, "extension_resolution", str(exc))

    if not pipeline_tag:
        _record_failure(
            diagnostics,
            "hub_api",
            f"metadata returned no pipeline_tag (library_name={library_name or 'unknown'})",
        )

    model_metadata = _hub_model_metadata(info)
    if hub_diagnostics:
        # The raw Hub configuration is retained as invalid evidence, not passed
        # into executable configuration or a non-standard JSON report.
        model_metadata["quantization_config"] = {}

    return TaskInfo(
        model_id=model_id,
        pipeline_tag=pipeline_tag or "unknown",
        # Explicit Hub metadata must survive even without a collection adapter.
        # Falling back to a generic architecture suffix can misroute multimodal
        # models to NLP. CLI overrides still apply before the support check.
        task_family=task_family or "unknown",
        runtime_backend=runtime_backend,
        library_name=library_name,
        model_revision=sha,
        detection_method="hub_api",
        model_config=model_config,
        hub_metadata={"pipeline_tag": hub_pipeline_tag, "transformers_info": transformers_info,
                      "tags": list(getattr(info, "tags", None) or []),
                      "task_hint": pipeline_tag if pipeline_tag != hub_pipeline_tag else None,
                      "task_hint_source": task_hint_source},
        **repository,
        **model_metadata,
    )


def _cached_metadata_file(model_id: str, name: str, revision: str | None) -> MetadataFile | None:
    """Read a file from the selected source's cache without a network request."""
    from acprof.host.model_store.store import cached_metadata
    from acprof.hub.repository import model_source

    source = model_source()
    metadata = cached_metadata(model_id, name, revision, source=source, max_bytes=METADATA_MAX_BYTES)
    if metadata is not None or source != "huggingface":
        return metadata
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(model_id, name, revision=revision)
    return read_metadata_file(path, max_bytes=METADATA_MAX_BYTES) if isinstance(path, str) else None


def _fallback_config_info(
    model_id: str, requested_revision: str | None, filename: str, metadata: MetadataFile,
) -> TaskInfo:
    """Parse the same snapshot bytes that identify the fallback decision."""
    from acprof.models.evidence import pinned_revision

    revision = metadata.path.parent.name
    if not pinned_revision(revision) or metadata.path.parent.parent.name != "snapshots":
        raise ValueError("config fallback requires a pinned snapshot commit SHA")
    if pinned_revision(requested_revision) and revision != requested_revision:
        raise ValueError("config fallback returned a different model revision")
    libraries = {"model_index.json": "diffusers", "modules.json": "sentence-transformers", "config.json": "transformers"}
    source = _raw_metadata_source(metadata, revision)
    info = TaskInfo(
        model_id=model_id, pipeline_tag="unknown", task_family="unknown", runtime_backend="unknown",
        library_name=libraries[filename],
        model_revision=revision, detection_method="config_infer",
        metadata_sources={filename: source},
    )
    try:
        if filename == "modules.json":
            names = ["config.json", "config_sentence_transformers.json", "sentence_bert_config.json"]
            info.repository_metadata[filename] = _repository_json(metadata, filename, names, source["sha256"])
            for name in names:
                item = _cached_metadata_file(model_id, name, revision)
                if item is None:
                    continue
                suffix = ("snapshots", revision, *Path(name).parts)
                if item.path.parts[-len(suffix):] != suffix:
                    raise ModelFilesError(f"cached {name} belongs to a different model revision")
                source = info.metadata_sources[name] = _raw_metadata_source(item, revision)
                info.repository_metadata[name] = _repository_json(item, name, names, source["sha256"])
            info.model_config = _architecture_metadata(info.repository_metadata.get("config.json"))
        else:
            config_data = _read_json_metadata(metadata, filename=filename)
            if not isinstance(config_data, dict):
                raise ValueError(f"{filename} must contain dict")
            info.repository_metadata[filename] = config_data
    except MetadataValidationError as exc:
        info.metadata_errors = (str(exc),)
        info.metadata_diagnostics = tuple(exc.diagnostics)
    return info


def _detect_from_config(
    model_id: str,
    diagnostics: list[str] | None = None,
    *, requested_revision: str | None = None, failures: list[Exception] | None = None,
) -> TaskInfo | None:
    """Recover registered model interfaces from snapshot metadata."""
    stage = "model_index"
    try:
        for filename, stage in (("model_index.json", "model_index"), ("modules.json", "modules_json")):
            cached = _cached_metadata_file(model_id, filename, requested_revision)
            if cached is None:
                continue
            info = _fallback_config_info(model_id, requested_revision, filename, cached)
            if info.metadata_errors:
                _record_failure(diagnostics, stage, info.metadata_errors[0])
                return info
            if filename == "modules.json" or "_class_name" in info.repository_metadata[filename]:
                return info
            # An unrelated index does not identify a Diffusers pipeline.
            requested_revision = info.model_revision
        stage = "config_json"
        metadata = (_download_metadata(model_id, "config.json", requested_revision) if requested_revision
                    else _download_metadata(model_id, "config.json"))
        info = _fallback_config_info(model_id, requested_revision, "config.json", metadata)
    except Exception as exc:
        # A corrupt local manifest or component must also block explicit task selection.
        if stage != "config_json" and isinstance(exc, (ModelFilesError, OSError)):
            raise
        _record_failure(diagnostics, stage, _format_failure(exc))
        if failures is not None:
            failures.append(exc)
        return None

    if info.metadata_errors:
        _record_failure(diagnostics, stage, info.metadata_errors[0])
        return info
    config_data = info.repository_metadata["config.json"]
    if not config_data.get("architectures"):
        _record_failure(diagnostics, stage, "config.json has no architectures field")
    try:
        resolved = CATALOG.resolve(library="transformers", config=config_data, model_id=model_id)
    except UnsupportedExtensionError as exc:
        _record_failure(diagnostics, "architecture_infer", str(exc))
        return None
    info.pipeline_tag, info.task_family, info.runtime_backend = resolved.task, resolved.family, resolved.backend
    info.model_config = _architecture_metadata(config_data)
    return info


@hf_endpoint_scope()
@modelscope_info_scope()
def detect_task(
    model_id: str,
    override_tag: str | None = None,
    override_family: str | None = None,
    override_backend: str | None = None,
    model_spec_path: str | None = None,
    *, revision: str | None = None,
) -> TaskInfo:
    """Discover candidates and apply CLI overrides, checking declaration conflicts."""
    from huggingface_hub.errors import HFValidationError
    from huggingface_hub.utils import validate_repo_id

    try:
        validate_repo_id(model_id)
    except HFValidationError as exc:
        raise model_lookup_error(model_id, exc, revision=revision) from exc
    # Start with auto-detection
    diagnostics: list[str] = []
    failures: list[Exception] = []
    info = _detect_from_hub(model_id, diagnostics, revision=revision, failures=failures)
    if info is None:
        if failures:
            error = model_lookup_error(model_id, failures[0], revision=revision, diagnostics=diagnostics)
            if error.blocks_fallback:
                raise error from failures[0]
        info = _detect_from_config(model_id, diagnostics, requested_revision=revision, failures=failures)

    # If auto-detection failed entirely, require manual override
    if info is None:
        # A pinned cached config can recover transient Hub failures. Manual task
        # declarations cannot repair a failed repository lookup or grant access.
        if failures:
            errors = [(cause, model_lookup_error(model_id, cause, revision=revision, diagnostics=diagnostics))
                      for cause in failures]
            cause, error = next((item for item in errors if item[1].blocks_fallback),
                                next((item for item in errors if item[1].reason_code != "lookup_failed"), errors[0]))
            if error.reason_code != "lookup_failed" or not (override_tag or override_family or model_spec_path):
                raise error from cause
        if not override_tag and not override_family and not model_spec_path:
            raise ModelLookupError(model_id, "task_unresolved", revision=revision or "", detail="\n".join(diagnostics))
        info = TaskInfo(
            model_id=model_id,
            pipeline_tag=override_tag or "unknown",
            task_family=override_family or "unknown",
            runtime_backend=override_backend or "unknown",
            library_name="unknown",
            model_revision=revision or "main",
            detection_method="manual",
        )

    if model_spec_path:
        from acprof.models.spec import read_model_spec
        info.model_spec = read_model_spec(Path(model_spec_path).expanduser())
    from acprof.hub.repository import model_source
    info.model_source = model_source()
    info.requested_revision = revision
    if info.model_source == "modelscope":
        info.model_metadata_source = "modelscope_hub"
    if info.metadata_diagnostics:
        from acprof.failures import Failure
        from acprof.models.inspection.decision import record_resolution_failure

        info.model_resolution.update(
            schema_version=1, model_id=model_id, model_revision=info.model_revision,
            candidates=[], runtime_validation="not_run",
        )
        record_resolution_failure(info, Failure(
            "preflight", "metadata_invalid", "; ".join(info.metadata_errors),
            evidence={"diagnostics": list(info.metadata_diagnostics), "sources": info.metadata_sources},
        ))
        return info
    from acprof.models.candidates import apply_model_candidate, discover_model_candidates
    info.model_resolution = apply_model_candidate(info, discover_model_candidates(
        info, override_tag=override_tag, override_backend=override_backend))
    if not info.metadata_errors:
        from acprof.models.contract import apply_model_contract
        source_revision = info.model_revision
        def source_text(name):
            if name not in info.repository_sources:
                info.repository_sources[name] = read_model_source(model_id, name, source_revision)
            return info.repository_sources[name]
        apply_model_contract(info, source_text,
                             override_tag=override_tag, override_backend=override_backend,
                             resolve_repository=dependency_metadata)

    # Apply CLI overrides (highest priority, still checked against declarations).
    if override_tag:
        info.pipeline_tag = override_tag
        if override_tag in CATALOG.task_families:
            info.task_family = CATALOG.task_families[override_tag]
        info.detection_method = "manual"
    if override_family:
        info.task_family = override_family
        info.detection_method = "manual"
    if override_backend:
        info.runtime_backend = override_backend
        info.detection_method = "manual"

    return info
