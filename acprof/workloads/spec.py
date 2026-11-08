"""Shared field checks for workload specifications.

Families own their schema, defaults, scale normalization and cross-field rules.
These checks neither coerce values nor mutate the parsed JSON object.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from typing import Any


def require_object(value: object, message: str) -> dict[str, Any]:
    """Require a JSON object while retaining the caller's error context."""
    if not isinstance(value, dict):
        raise ValueError(message)
    return value


def reject_unknown_keys(
    value: Mapping[str, Any], allowed: Set[str], message: str,
) -> None:
    """Report unsupported fields in deterministic order."""
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(message + ", ".join(unknown))


def require_schema_version(
    spec: Mapping[str, Any], family: str, *, default: int | None = None,
) -> None:
    """Check version 1; a default applies only when the field is omitted."""
    version = spec.get("schema_version", default)
    if version != 1 or isinstance(version, bool):
        raise ValueError(f"{family} workload schema_version must be 1")


def require_nonempty_string(value: object, location: str) -> str:
    """Reject blank strings without stripping meaningful surrounding spaces."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


def require_nonempty_list(value: object, message: str) -> list[Any]:
    """Check array shape; the family owns item checks and ordering policy."""
    if not isinstance(value, list) or not value:
        raise ValueError(message)
    return value
