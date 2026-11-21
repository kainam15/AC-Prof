"""Validate JSON model metadata without changing the source configuration."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from acprof.models.files import ModelFilesError
from acprof.models.sentence_modules import module_config_groups

MAX_METADATA_DEPTH = 256


class MetadataValidationError(ValueError):
    """A metadata value is invalid; diagnostics preserve its source and location."""

    def __init__(self, diagnostics: list[dict[str, Any]]):
        self.diagnostics = diagnostics
        super().__init__("; ".join(
            f"{item['file']}:{item['json_pointer'] or '/'}: {item['detail']}"
            for item in diagnostics
        ))


def manifest_config_groups(modules: Any, raw_sha256: str) -> tuple[tuple[str, ...], ...]:
    """Validate every declared directory before returning any snapshot-relative reads."""
    def invalid(pointer: str, detail: str) -> MetadataValidationError:
        return MetadataValidationError([{"reason_code": "metadata_invalid", "file": "modules.json",
                                         "json_pointer": pointer, "raw_sha256": raw_sha256, "detail": detail}])

    if not isinstance(modules, list):
        raise invalid("", "modules.json must contain list")
    if len(modules) > 64:
        raise invalid("", "modules.json exceeds 64 modules")
    groups: list[tuple[str, ...]] = []
    for index, module in enumerate(modules):
        declaration = module if isinstance(module, dict) else {}
        try:
            groups.extend(module_config_groups(declaration.get("type"), declaration.get("path")))
        except ModelFilesError as exc:
            raise invalid(f"/{index}/path", "module path must stay inside the model snapshot") from exc
    return tuple(groups)


def finite_metadata_diagnostics(
    value: Any, filename: str, *, raw_sha256: str | None = None,
) -> list[dict[str, Any]]:
    """Reject non-finite or over-deep metadata with RFC 6901 field pointers."""
    diagnostics: list[dict[str, Any]] = []

    def visit(item: Any, pointer: str, depth: int) -> None:
        if depth > MAX_METADATA_DEPTH:
            diagnostics.append({
                "reason_code": "metadata_invalid", "file": filename, "json_pointer": pointer,
                "raw_sha256": raw_sha256, "detail": f"metadata nesting exceeds maximum depth {MAX_METADATA_DEPTH}",
            })
            return
        if isinstance(item, float) and not math.isfinite(item):
            literal = "NaN" if math.isnan(item) else "Infinity" if item > 0 else "-Infinity"
            diagnostics.append({
                "reason_code": "metadata_invalid", "file": filename,
                "json_pointer": pointer, "raw_sha256": raw_sha256,
                "value": literal, "detail": f"non-finite number {literal} is not valid JSON metadata",
            })
        elif isinstance(item, dict):
            for key, nested in item.items():
                token = str(key).replace("~", "~0").replace("/", "~1")
                visit(nested, f"{pointer}/{token}", depth + 1)
        elif isinstance(item, list):
            for index, nested in enumerate(item):
                visit(nested, f"{pointer}/{index}", depth + 1)

    try:
        visit(value, "", 0)
    except RecursionError:
        diagnostics.append({
            "reason_code": "metadata_invalid", "file": filename, "json_pointer": "",
            "raw_sha256": raw_sha256, "detail": "metadata nesting exceeds the supported depth",
        })
    return diagnostics


def validate_diffusers_declarations(config: dict[str, Any], *, raw_sha256: str | None = None) -> None:
    """Validate only the pipeline and component names consumed by static layout checks."""
    diagnostics: list[dict[str, Any]] = []

    def invalid(pointer: str, detail: str) -> None:
        diagnostics.append({
            "reason_code": "metadata_invalid", "file": "model_index.json",
            "json_pointer": pointer, "raw_sha256": raw_sha256, "detail": detail,
        })

    if "_class_name" in config and not isinstance(config["_class_name"], str):
        invalid("/_class_name", "Diffusers pipeline class name must be a string")
    for component, declaration in config.items():
        if component.startswith("_") or not isinstance(declaration, list) or len(declaration) != 2:
            continue
        if declaration == [None, None]:
            continue
        token = component.replace("~", "~0").replace("/", "~1")
        for index, value in enumerate(declaration):
            if not isinstance(value, str):
                invalid(
                    f"/{token}/{index}",
                    "Diffusers component declaration requires library and class strings, or [null, null]",
                )
    if diagnostics:
        raise MetadataValidationError(diagnostics)


def parse_metadata_json(content: bytes, filename: str) -> Any:
    """Parse bounded original bytes, retaining their hash for every diagnostic."""
    digest = hashlib.sha256(content).hexdigest()
    try:
        value = json.loads(content.decode("utf-8"))
    except (ValueError, RecursionError) as exc:
        raise MetadataValidationError([{
            "reason_code": "metadata_invalid", "file": filename, "json_pointer": "",
            "raw_sha256": digest, "exception_type": type(exc).__name__,
            "detail": "metadata nesting exceeds the supported depth" if isinstance(exc, RecursionError) else str(exc),
        }]) from exc
    diagnostics = finite_metadata_diagnostics(value, filename, raw_sha256=digest)
    if diagnostics:
        raise MetadataValidationError(diagnostics)
    if filename == "model_index.json":
        if not isinstance(value, dict):
            raise MetadataValidationError([{
                "reason_code": "metadata_invalid", "file": filename, "json_pointer": "",
                "raw_sha256": digest, "detail": "Diffusers pipeline index must be a JSON object",
            }])
        validate_diffusers_declarations(value, raw_sha256=digest)
    return value
