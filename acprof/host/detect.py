"""AC-Prof - HuggingFace Model Task Auto-Detection.

Collect Hub, repository and artifact evidence at a pinned revision, then select
an interface using explicit overrides or unambiguous metadata. No model code is
loaded on the host; runtime compatibility is verified separately in containers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from acprof.extensions import CATALOG, UnsupportedExtensionError
from acprof.host.preparation.model_errors import (
    ModelLookupError,
    metadata_read_diagnostic,
    model_lookup_error,
)
from acprof.hub.huggingface.auth import hf_token_for_url
from acprof.hub.huggingface.download import hf_endpoint_scope
from acprof.hub.repository import MetadataFile, modelscope_info_scope, read_metadata_file
from acprof.models.diffusion_pipelines import diffusers_index_task
from acprof.models.files import ModelFilesError
from acprof.models.metadata_validation import (
    MetadataValidationError,
    finite_metadata_diagnostics,
    manifest_config_groups,
    parse_metadata_json,
)
from acprof.models.sentence_modules import read_config_groups


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
    parameter_bytes_reason: str | None = field(default=None, kw_only=True)
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


def _repository_json(
    metadata: MetadataFile, name: str, pending: list[tuple[str, ...]], sha256: str,
) -> Any:
    value = _read_json_metadata(metadata, filename=name)
    if name == "modules.json":
        pending.extend(manifest_config_groups(value, sha256))
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

    inventory = {str(item.rfilename) for item in (getattr(info, "siblings", None) or [])
                 if getattr(item, "rfilename", "")}
    files = tuple(sorted(inventory))
    metadata, errors, diagnostics, sources = {}, [], [], {}
    if files and not pinned_revision(revision):
        return {"repository_files": files, "repository_metadata": {},
                "metadata_errors": ("model revision must be a full commit SHA before metadata reads",),
                "metadata_diagnostics": (), "metadata_sources": {}}
    names = ["config.json", "model_index.json", "modules.json", "adapter_config.json",
             "acprof_model.json", "tokenizer_config.json", "processor_config.json", "preprocessor_config.json",
             "generation_config.json", "config_sentence_transformers.json"]
    if "modules.json" not in inventory:
        names.append("sentence_bert_config.json")

    groups: list[tuple[str, ...]] = [(name,) for name in names]

    def read_name(name: str) -> bool:
        if inventory and name not in inventory:
            return False
        try:
            downloaded = _download_metadata(model_id, name, revision)
            sources[name] = _raw_metadata_source(downloaded, revision)
            metadata[name] = _repository_json(downloaded, name, groups, sources[name]["sha256"])
            return bool(metadata[name])
        except MetadataValidationError as exc:
            diagnostics.extend(exc.diagnostics)
            errors.append(f"{name}: {_format_failure(exc)}")
        except Exception as exc:
            if files:
                errors.append(f"{name}: {_format_failure(exc)}")
                # Size, revision and integrity guards have their own validation errors.
                if name not in sources and not isinstance(exc, ModelFilesError) and type(exc) is not ValueError:
                    diagnostics.append(metadata_read_diagnostic(model_id, revision, name, exc))
            else:
                _record_failure(lookup_diagnostics, "model_index", _format_failure(exc))
        # A listed file failed to load. Its error takes precedence over later alternatives.
        return True

    if files:
        read_config_groups(groups, read_name)
        try:
            indexes = checkpoint_metadata_indexes(inventory, metadata)
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
    """Sum a complete dtype histogram, returning None if a byte width is unknown."""
    total = 0
    for dtype, count in dtype_counts.items():
        if count == 0:
            continue
        byte_width = _DTYPE_BYTE_WIDTHS.get(dtype)
        if byte_width is None or count < 0:
            return None
        total += count * byte_width
    return total


def _nonnegative_count(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str):
        digits = value.strip()
        if digits.isascii() and digits.isdecimal():
            try:
                return int(digits)
            except ValueError:
                # Python limits the length of decimal integer conversions.
                return None
    if isinstance(value, float) and value.is_integer() and 0 <= value < 2**53:
        return int(value)
    return None


def _hub_parameter_counts(safetensors: object) -> tuple[int | None, dict[str, int], int, str | None]:
    total = _nonnegative_count(_mapping_value(safetensors, "total"))
    raw_counts = _mapping_value(safetensors, "parameters")
    counts: dict[str, int] = {}
    if not isinstance(raw_counts, dict) or (not raw_counts and total != 0):
        return total, counts, 0, "Hub safetensors.parameters is missing or empty"
    unnamed_elements = 0
    problems = []
    for raw_dtype, raw_count in raw_counts.items():
        dtype = _normalize_dtype_name(raw_dtype) if isinstance(raw_dtype, str) else ""
        count = _nonnegative_count(raw_count)
        if not dtype:
            problems.append("missing dtype name")
            if count is not None:
                unnamed_elements += count
        elif count is None:
            problems.append(f"invalid count for {dtype}")
        else:
            counts[dtype] = counts.get(dtype, 0) + count
    if total is None:
        problems.append("missing or invalid total")
    elif not problems and (counted := sum(counts.values())) != total:
        problems.append(f"dtype counts sum to {counted}, declared total is {total}")
    return total, counts, unnamed_elements, "Hub safetensors: " + "; ".join(problems) if problems else None


@dataclass(frozen=True)
class _QuantizationDeclaration:
    section: str
    config: dict[str, Any] | None  # None means the declaration was ambiguous or malformed.


def _quantization_declaration(config: object) -> _QuantizationDeclaration | None:
    """Select top-level or unambiguous decoder evidence within one configuration."""
    if not isinstance(config, dict):
        return None
    owner = config
    section = ""
    top = config.get("quantization_config")
    if top is None or top == {}:
        names = [name for name in ("decoder", "generator", "text_config") if config.get(name) is not None]
        declared = [name for name in names
                    if isinstance(config[name], dict) and "quantization_config" in config[name]]
        if declared:
            if len(names) != 1:
                return _QuantizationDeclaration("", None)
            section = declared[0]
            owner = config[section]
    if "quantization_config" not in owner:
        return None
    value = owner["quantization_config"]
    if value is not None and not isinstance(value, dict) and not callable(getattr(value, "to_dict", None)):
        return _QuantizationDeclaration(section, None)
    return _QuantizationDeclaration(section, _json_mapping(value))


def _weight_metadata(
    config: dict[str, Any], *, tags: set[str], dtype_counts: dict[str, int] | None,
    declaration: _QuantizationDeclaration | None = None,
) -> dict[str, Any]:
    """A present but inconclusive histogram cannot be replaced by a config dtype."""
    declaration = declaration or _quantization_declaration(config)
    quantization_config = (declaration.config or {}) if declaration else {}
    precision_dtype: str | None = None
    if dtype_counts is not None:
        # A histogram with no elements or a tie has no single dominant dtype.
        largest = max(dtype_counts.values(), default=0)
        leaders = [dtype for dtype, count in dtype_counts.items() if count == largest]
        if largest > 0 and len(leaders) == 1:
            precision_dtype = leaders[0]
    else:
        config_dtype = config.get("dtype")
        if config_dtype is None:
            config_dtype = config.get("torch_dtype")
        if isinstance(config_dtype, str):
            dtype = _normalize_dtype_name(config_dtype)
            if dtype in _DTYPE_BYTE_WIDTHS or dtype in _QUANTIZED_PRECISIONS:
                precision_dtype = dtype

    quantization_method = None
    for key in ("quant_method", "quantization_method", "quant_type"):
        method = quantization_config.get(key)
        if isinstance(method, str) and method.strip():
            quantization_method = method.strip()
            break
    matching_quantization_tags = sorted(tags & _QUANTIZATION_TAGS)
    if not quantization_method and matching_quantization_tags:
        quantization_method = matching_quantization_tags[0]

    if quantization_config or matching_quantization_tags:
        quantized: bool | None = True
    elif precision_dtype in _QUANTIZED_PRECISIONS:
        quantized = True
    elif precision_dtype in _DTYPE_BYTE_WIDTHS and (declaration is None or declaration.config is not None):
        quantized = False
    else:
        quantized = None

    return {
        "precision_dtype": precision_dtype,
        "quantized": quantized,
        "quantization_method": quantization_method,
        "quantization_config": quantization_config,
    }


def _hub_model_metadata(
    info: object, *, config: dict[str, Any] | None = None, source: str = "huggingface",
    declaration: _QuantizationDeclaration | None = None,
) -> dict[str, Any]:
    """Extract Hub facts with the resolved model config when available."""
    if config is None:
        config = _json_mapping(getattr(info, "config", None))
    safetensors = getattr(info, "safetensors", None)
    parameter_count, dtype_counts, unnamed_elements, counts_reason = _hub_parameter_counts(safetensors)
    parameter_bytes = _parameter_bytes_from_dtype_counts(dtype_counts) if counts_reason is None else None
    bytes_reason = counts_reason
    if bytes_reason is None and parameter_bytes is None:
        unknown = sorted(dtype for dtype, count in dtype_counts.items() if count and dtype not in _DTYPE_BYTE_WIDTHS)
        bytes_reason = f"Unknown dtype byte width: {', '.join(unknown)}"

    tags = {
        str(tag).strip().lower()
        for tag in (getattr(info, "tags", None) or [])
        if str(tag).strip()
    }

    histogram = dtype_counts if counts_reason is None else {}
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
        "hub_metadata": {"unnamed_dtype_elements": unnamed_elements} if unnamed_elements else {},
        "parameter_count": parameter_count,
        "parameter_bytes": parameter_bytes,
        "parameter_bytes_reason": bytes_reason,
        "parameter_dtype_counts": dtype_counts,
        **_weight_metadata(config, tags=tags, dtype_counts=histogram if safetensors is not None else None,
                           declaration=declaration),
        "model_license": str(model_license) if model_license else None,
        "model_metadata_source": f"{source}_hub",
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
        source = model_source()
        info = cached_model_info(model_id, revision, source=source)
        if info is None:
            if source == "modelscope":
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
    raw_pinned_config = repository["repository_metadata"].get("config.json")
    pinned_quantization = _quantization_declaration(raw_pinned_config)
    raw_hub_config = getattr(info, "config", None) or {}
    hub_config = _architecture_metadata(raw_hub_config)
    hub_quantization = None if pinned_quantization else _quantization_declaration(raw_hub_config)
    hub_evidence = hub_config
    if hub_quantization and hub_quantization.section:
        section = hub_quantization.section
        hub_evidence = {**hub_config, section: {"quantization_config": raw_hub_config[section]["quantization_config"]}}
    hub_diagnostics = finite_metadata_diagnostics(hub_evidence, "hub.config")
    if hub_diagnostics:
        repository["metadata_diagnostics"] += tuple(hub_diagnostics)
        repository["metadata_errors"] += (str(MetadataValidationError(hub_diagnostics)),)
    model_config = {} if hub_diagnostics else hub_config
    pinned_config = _architecture_metadata(raw_pinned_config)
    if "dtype" in pinned_config or "torch_dtype" in pinned_config:
        # These are aliases for the same declaration, even across sources.
        model_config.pop("dtype", None)
        model_config.pop("torch_dtype", None)
    model_config.update(pinned_config)
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

    declaration = pinned_quantization or (None if hub_diagnostics else hub_quantization)
    model_metadata = _hub_model_metadata(info, config=model_config, source=source, declaration=declaration)
    model_metadata["hub_metadata"].update(
        pipeline_tag=hub_pipeline_tag, transformers_info=transformers_info,
        tags=list(getattr(info, "tags", None) or []),
        task_hint=pipeline_tag if pipeline_tag != hub_pipeline_tag else None,
        task_hint_source=task_hint_source,
    )

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
        **repository,
        **model_metadata,
    )


def _cached_metadata_file(model_id: str, name: str, revision: str | None) -> tuple[MetadataFile | None, bool]:
    """Return cached bytes and whether the Hub explicitly recorded that the file is absent."""
    from acprof.host.model_store.store import cached_metadata
    from acprof.hub.repository import model_source

    source = model_source()
    metadata = cached_metadata(model_id, name, revision, source=source, max_bytes=METADATA_MAX_BYTES)
    if metadata is not None or source != "huggingface":
        return metadata, False
    from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache

    path = try_to_load_from_cache(model_id, name, revision=revision)
    if isinstance(path, str):
        return read_metadata_file(path, max_bytes=METADATA_MAX_BYTES), False
    return None, path is _CACHED_NO_EXIST


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
            groups: list[tuple[str, ...]] = [("config.json",), ("config_sentence_transformers.json",)]
            info.repository_metadata[filename] = _repository_json(metadata, filename, groups, source["sha256"])

            def read_cached(name: str) -> bool:
                item, known_absent = _cached_metadata_file(model_id, name, revision)
                if item is None:
                    # An ordinary cache miss cannot rule out this higher-priority file.
                    return not known_absent
                suffix = ("snapshots", revision, *Path(name).parts)
                if item.path.parts[-len(suffix):] != suffix:
                    raise ModelFilesError(f"cached {name} belongs to a different model revision")
                record = info.metadata_sources[name] = _raw_metadata_source(item, revision)
                info.repository_metadata[name] = _repository_json(item, name, groups, record["sha256"])
                return bool(info.repository_metadata[name])

            read_config_groups(groups, read_cached)
            info.model_config = _architecture_metadata(info.repository_metadata.get("config.json"))
        else:
            info.repository_metadata[filename] = _repository_json(metadata, filename, [], source["sha256"])
    except MetadataValidationError as exc:
        info.metadata_errors = (str(exc),)
        info.metadata_diagnostics = tuple(exc.diagnostics)
    if config := info.repository_metadata.get("config.json"):
        return replace(info, **_weight_metadata(config, tags=set(), dtype_counts=None))
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
            cached, _ = _cached_metadata_file(model_id, filename, requested_revision)
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
    info.model_config = _architecture_metadata(config_data)
    if not config_data.get("architectures"):
        _record_failure(diagnostics, stage, "config.json has no architectures field")
    try:
        resolved = CATALOG.resolve(library="transformers", config=config_data, model_id=model_id)
    except UnsupportedExtensionError as exc:
        _record_failure(diagnostics, "architecture_infer", str(exc))
    else:
        info.pipeline_tag, info.task_family, info.runtime_backend = resolved.task, resolved.family, resolved.backend
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
    invalid = next((item for item in info.metadata_diagnostics if item["reason_code"] == "metadata_invalid"), None)
    if invalid:
        # Invalid content must not reach candidate discovery.
        info.model_resolution.update(
            schema_version=1, model_id=model_id, model_revision=info.model_revision,
            candidates=[], runtime_validation="not_run",
        )
    else:
        from acprof.models.candidates import apply_model_candidate, discover_model_candidates
        info.model_resolution = apply_model_candidate(info, discover_model_candidates(
            info, override_tag=override_tag, override_backend=override_backend))
    if info.metadata_diagnostics:
        from acprof.failures import Failure
        from acprof.models.inspection.decision import record_resolution_failure

        primary = invalid or info.metadata_diagnostics[0]
        record_resolution_failure(info, Failure(
            "preflight" if invalid else "resolution", primary["reason_code"], "; ".join(info.metadata_errors),
            evidence={"diagnostics": list(info.metadata_diagnostics), "sources": info.metadata_sources},
            retryability=primary.get("retryability", "not_retryable"), exception_type=primary.get("exception_type", ""),
        ))
        return info
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
