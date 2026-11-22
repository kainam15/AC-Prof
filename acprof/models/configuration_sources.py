"""Read-only configuration facts and default candidates from observed metadata.

Source discovery never selects values, resolves Auto or proves parameter
consumption. It uses the existing revision-bound metadata without performing I/O.
"""
from __future__ import annotations

import copy
import re
from collections.abc import Iterable
from typing import Any

from acprof.extensions.parameters import (
    ParameterSpec,
    parameter_spec_to_dict,
    validate_parameter_value,
)
from acprof.models.evidence import content_digest, pinned_revision
from acprof.models.metadata_validation import finite_metadata_diagnostics

_MISSING = object()
_OBJECT_FILES = frozenset({
    "config.json", "generation_config.json", "tokenizer_config.json",
    "processor_config.json", "preprocessor_config.json",
})


def _token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _json_diagnostics(value: Any, filename: str, raw_sha256: str | None) -> list[dict]:
    diagnostics = finite_metadata_diagnostics(value, filename, raw_sha256=raw_sha256)
    if diagnostics:
        return diagnostics  # Invalid documents must not reach copying or deeper traversal.

    def visit(item: Any, pointer: str) -> None:
        if type(item) is dict:
            for key, nested in item.items():
                if type(key) is not str:
                    diagnostics.append({
                        "reason_code": "metadata_invalid", "file": filename, "json_pointer": pointer,
                        "raw_sha256": raw_sha256, "detail": "JSON object keys must be strings",
                    })
                else:
                    visit(nested, pointer + "/" + _token(key))
        elif type(item) is list:
            for index, nested in enumerate(item):
                visit(nested, f"{pointer}/{index}")
        elif item is not None and type(item) not in (bool, int, float, str):
            diagnostics.append({
                "reason_code": "metadata_invalid", "file": filename, "json_pointer": pointer,
                "raw_sha256": raw_sha256, "detail": f"unsupported metadata value type: {type(item).__name__}",
            })

    try:
        visit(value, "")
    except RecursionError:
        if not any("nesting" in item["detail"] for item in diagnostics):
            diagnostics.append({
                "reason_code": "metadata_invalid", "file": filename, "json_pointer": "",
                "raw_sha256": raw_sha256, "detail": "metadata nesting exceeds the supported depth",
            })
    return diagnostics


def _source_document(task, source_id: str, value: Any, record: dict, kind: str, diagnostics: list) -> dict:
    revision = getattr(task, "model_revision", None)
    problems = []
    if not isinstance(record, dict):
        problems.append("source identity must be an object")
        record = {}
    raw_encoding = record.get("encoding") == "raw"
    raw = record.get("sha256") if raw_encoding and isinstance(record.get("sha256"), str) else None
    byte_count = record.get("bytes") if raw_encoding and type(record.get("bytes")) is int else None
    source = {
        "kind": kind, "model_source": getattr(task, "model_source", "huggingface"),
        "model_id": task.model_id, "revision": revision, "file": source_id,
        "recorded_revision": record.get("revision") if isinstance(record.get("revision"), str) else None,
        "state": "unknown" if value is _MISSING else "observed",
        "encoding": "raw" if raw_encoding else "canonical-json" if value is not _MISSING else "unknown",
        "raw_sha256": raw, "canonical_sha256": None,
        "bytes": byte_count if byte_count is not None and byte_count >= 0 else None,
    }
    if raw_encoding:
        if raw is None or re.fullmatch(r"[0-9a-f]{64}", raw) is None:
            problems.append("recorded raw SHA256 must be a lowercase 64-character digest")
        if record.get("revision") != revision or not pinned_revision(revision):
            problems.append("recorded source revision does not match the fixed model revision")
        if byte_count is None or byte_count < 0:
            problems.append("recorded raw byte count must be a nonnegative integer")
    for detail in problems:
        diagnostics.append({
            "reason_code": "metadata_invalid", "file": source_id, "json_pointer": "",
            "raw_sha256": raw, "detail": detail,
        })
    if value is not _MISSING:
        value_problems = _json_diagnostics(value, source_id, raw)
        if source_id in _OBJECT_FILES and type(value) is not dict:
            value_problems.append({
                "reason_code": "metadata_invalid", "file": source_id, "json_pointer": "",
                "raw_sha256": raw, "detail": "metadata document must be an object",
            })
        diagnostics.extend(value_problems)
        if not value_problems:
            source["canonical_sha256"] = content_digest(value)
    if any(item.get("file") == source_id and item.get("reason_code") == "metadata_invalid"
           for item in diagnostics):
        source["state"] = "invalid"
    return source


def _source_reference(source_id: str, source: dict, pointer: str) -> dict:
    return {
        "source_id": source_id, "file": source["file"], "json_pointer": pointer,
        "model_source": source["model_source"], "model_id": source["model_id"],
        "revision": source["revision"], "raw_sha256": source["raw_sha256"],
        "canonical_sha256": source["canonical_sha256"], "state": source["state"],
    }


def _pointer_value(document: Any, pointer: str) -> Any:
    value = document
    traversed = ""
    for encoded in pointer.split("/")[1:] if pointer else ():
        token = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict):
            if token not in value:
                return _MISSING
            value = value[token]
        elif isinstance(value, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", token) or int(token) >= len(value):
                return _MISSING
            value = value[int(token)]
        else:
            raise ValueError(f"{traversed or '/'} is not an object or list")
        traversed += "/" + encoded
    return value


def _semantic_task_ids(selected_task: dict | None) -> tuple[str, ...]:
    if selected_task is None:
        return ()
    profile = selected_task.get("execution_profile") or {}
    identifiers = profile.get("semantic_task_ids")
    if identifiers is None:
        semantic = selected_task.get("semantic_tasks")
        identifiers = [item["task_id"] for item in semantic] if semantic else [selected_task.get("task_id")]
    if not isinstance(identifiers, (list, tuple)) or any(type(value) is not str or not value for value in identifiers):
        raise ValueError("selected task must identify its semantic tasks")
    return tuple(dict.fromkeys(identifiers))


def _default_candidate(spec: ParameterSpec, value: Any, reference: dict, task_id: str | None,
                       diagnostics: list) -> dict:
    copied = copy.deepcopy(value)
    candidate = {"value": copied, "source": reference, "validation": "valid",
                 "validation_method": "parameter_value"}
    if task_id is not None:
        candidate["semantic_task_id"] = task_id
    try:
        validate_parameter_value(spec, copied, path=f"{reference['file']}:{reference['json_pointer'] or '/'}")
    except ValueError as exc:
        candidate["validation"] = "invalid"
        diagnostics.append({
            "reason_code": "metadata_invalid", "parameter": spec.name, "file": reference["file"],
            "json_pointer": reference["json_pointer"], "raw_sha256": reference["raw_sha256"],
            "detail": str(exc),
        })
    return candidate


def _unbound_attributes(source_id: str, source: dict, document: Any, consumed: set[str]) -> list[dict]:
    if "" in consumed:
        return []
    attributes = []

    def visit(value: Any, pointer: str) -> None:
        if pointer in consumed:
            return
        if any(bound.startswith(pointer + "/") for bound in consumed):
            if isinstance(value, dict):
                for key, nested in sorted(value.items()):
                    visit(nested, pointer + "/" + _token(key))
            elif isinstance(value, list):
                for index, nested in enumerate(value):
                    visit(nested, f"{pointer}/{index}")
            return
        attributes.append({
            "read_only": True, "value": copy.deepcopy(value),
            "source": _source_reference(source_id, source, pointer),
        })

    if isinstance(document, dict):
        for key, value in sorted(document.items()):
            visit(value, "/" + _token(key))
    else:
        visit(document, "")
    return attributes


def discover_configuration_sources(
    task_info, parameter_specs: Iterable[ParameterSpec], selected_task: dict | None = None,
) -> dict:
    """Return source facts and unselected default candidates, without changing inputs."""
    specs = tuple(parameter_specs)
    if any(not isinstance(spec, ParameterSpec) for spec in specs) or len({spec.name for spec in specs}) != len(specs):
        raise ValueError("configuration sources require unique parameter declarations")
    diagnostics = copy.deepcopy(list(getattr(task_info, "metadata_diagnostics", ()) or ()))
    # The reader also retains legacy "filename: detail" messages. Only surface those
    # here when a structured diagnosis for that file is absent.
    diagnosed = tuple(f"{item['file']}:" for item in diagnostics if item.get("file"))
    for detail in getattr(task_info, "metadata_errors", ()) or ():
        if not detail.startswith(diagnosed):
            diagnostics.append({"reason_code": "metadata_unavailable", "file": None, "json_pointer": "", "detail": detail})
    documents = dict(getattr(task_info, "repository_metadata", {}) or {})
    records = getattr(task_info, "metadata_sources", {}) or {}
    kinds = dict.fromkeys(documents, "repository_metadata")
    if ("config.json" not in documents and not records.get("config.json")
            and not any(item.get("file") == "config.json" for item in diagnostics)
            and getattr(task_info, "model_config", None)):
        documents["config.json"] = task_info.model_config
        kinds["config.json"] = "model_config"
    hub = getattr(task_info, "hub_metadata", None)
    if hub:
        documents["hub_metadata"] = hub
        kinds["hub_metadata"] = "hub_metadata"
    facts = {name: getattr(task_info, name) for name in (
        "parameter_count", "parameter_bytes", "parameter_dtype_counts", "precision_dtype",
        "quantized", "quantization_method", "quantization_config", "model_license",
    ) if getattr(task_info, name, None) not in (None, {}, "")}
    if facts:
        documents["model_facts"] = facts
        kinds["model_facts"] = getattr(task_info, "model_metadata_source", None) or "observed_model_facts"
    source_names = set(documents) | set(records) | {
        binding.filename for spec in specs if spec.enabled for binding in spec.source_bindings
    }
    sources = {
        name: _source_document(task_info, name, documents.get(name, _MISSING), records.get(name, {}),
                               kinds.get(name, "repository_metadata"), diagnostics)
        for name in sorted(source_names)
    }
    task_ids = _semantic_task_ids(selected_task)
    parameters = {}
    consumed: dict[str, set[str]] = {}
    for spec in sorted(specs, key=lambda item: item.name):
        if not spec.enabled:
            continue
        candidates = []
        invalid_source = False
        for binding in spec.source_bindings:
            source = sources[binding.filename]
            invalid_source |= source["state"] == "invalid"
            if source["state"] != "observed":
                continue
            locations = [
                ("/task_specific_params/" + _token(task_id) + binding.pointer, task_id) for task_id in task_ids
            ] if binding.task_scoped else [(binding.pointer, None)]
            for pointer, task_id in locations:
                try:
                    value = _pointer_value(documents[binding.filename], pointer)
                except ValueError as exc:
                    invalid_source = True
                    diagnostics.append({
                        "reason_code": "metadata_invalid", "parameter": spec.name, "file": binding.filename,
                        "json_pointer": pointer, "raw_sha256": source["raw_sha256"], "detail": str(exc),
                    })
                    continue
                if value is _MISSING:
                    continue
                if not spec.read_only:
                    consumed.setdefault(binding.filename, set()).add(pointer)
                reference = _source_reference(binding.filename, source, pointer)
                candidates.append(_default_candidate(spec, value, reference, task_id, diagnostics))
        if spec.read_only:
            # Declared model properties validate observed facts; defaults cannot
            # manufacture properties of a checkpoint that was not observed.
            continue
        if spec.default is not None:
            source_id = "parameter:" + spec.name
            declaration = parameter_spec_to_dict(spec)
            source = _source_document(task_info, source_id, declaration, {}, "parameter_declaration", diagnostics)
            # Extension defaults belong to a declaration, not the model snapshot.
            source.update(file=None, model_source=None, model_id=None, revision=None)
            sources[source_id] = source
            if source["state"] == "observed":
                candidates.append(_default_candidate(
                    spec, spec.default.value, _source_reference(source_id, source, "/default/value"), None, diagnostics,
                ))
            else:
                invalid_source = True
        parameters[spec.name] = {
            "scope": spec.scope, "default_candidates": candidates,
            "state": "invalid" if invalid_source or any(item["validation"] == "invalid" for item in candidates)
                     else "observed" if candidates else "unknown",
        }
    attributes = []
    for name, document in sorted(documents.items()):
        if sources[name]["state"] == "observed":
            attributes.extend(_unbound_attributes(name, sources[name], document, consumed.get(name, set())))
    return {
        "schema_version": 1, "evidence_level": "static", "runtime_validation": "not_run",
        "model": {"source": getattr(task_info, "model_source", "huggingface"), "model_id": task_info.model_id,
                  "revision": getattr(task_info, "model_revision", None)},
        "sources": sources, "attributes": attributes, "parameters": parameters, "diagnostics": diagnostics,
    }
