"""Lightweight parameter declarations shared by extensions and configuration consumers.

These declarations describe reviewed consumers and validate values. They do not
apply defaults, resolve Auto, freeze configurations or prove runtime execution.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

KINDS = frozenset({"boolean", "integer", "float", "enum", "string", "list"})
SCOPES = frozenset({"model", "loading", "inference", "workload", "measurement"})
CONSUMER_STAGES = frozenset({"load", "preprocess", "predict", "workload", "measurement"})
VARIATIONS = frozenset({"task", "device", "request", "case"})
MAX_PARAMETER_JSON_DEPTH = 256


def _text(value: Any, path: str, *, empty: bool = False) -> None:
    if type(value) is not str or (not empty and not value.strip()):
        raise ValueError(f"{path} must be {'a string' if empty else 'a nonempty string'}")


def _strings(values: Any, path: str) -> None:
    if type(values) is not tuple:
        raise ValueError(f"{path} must be a tuple of strings")
    for value in values:
        _text(value, path)
    if len(values) != len(set(values)):
        raise ValueError(f"{path} contains duplicate values")


def _number(value: Any) -> bool:
    return type(value) is int or type(value) is float and math.isfinite(value)


def _same_value(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def _validate_json_value(value: Any, path: str) -> None:
    """Reject non-JSON values before requests or defaults become provenance."""
    def visit(item: Any, location: str, depth: int) -> None:
        if depth > MAX_PARAMETER_JSON_DEPTH:
            raise ValueError(f"{location} exceeds the maximum JSON depth of {MAX_PARAMETER_JSON_DEPTH}")
        if item is None or type(item) in (bool, int, str):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for index, nested in enumerate(item):
                visit(nested, f"{location}[{index}]", depth + 1)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for key, nested in item.items():
                visit(nested, f"{location}.{key}", depth + 1)
            return
        raise ValueError(f"{location} must contain finite JSON values")
    try:
        visit(value, path, 0)
    except RecursionError as exc:
        raise ValueError(f"{path} exceeds the supported JSON depth") from exc


@dataclass(frozen=True)
class ParameterDefault:
    """A wrapper distinguishes an absent default from an explicitly null one."""

    value: Any

    def __post_init__(self) -> None:
        _validate_json_value(self.value, "default.value")


@dataclass(frozen=True)
class ParameterConsumer:
    """The enclosing extension binds the consumer to its reviewed implementation."""

    stage: str
    argument: str

    def __post_init__(self) -> None:
        _text(self.stage, "consumer.stage")
        if self.stage not in CONSUMER_STAGES:
            raise ValueError(f"unknown parameter consumer stage: {self.stage}")
        _text(self.argument, "consumer.argument")
        if not all(part.isidentifier() for part in self.argument.split(".")):
            raise ValueError("consumer.argument must be a dotted parameter path")


@dataclass(frozen=True)
class ParameterSource:
    """A JSON pointer; task_scoped selects config.task_specific_params[task] first."""

    filename: str
    pointer: str
    task_scoped: bool = False

    def __post_init__(self) -> None:
        _text(self.filename, "source.filename")
        path = PurePosixPath(self.filename)
        if (path.is_absolute() or ".." in path.parts or "\\" in self.filename
                or "\0" in self.filename or str(path) != self.filename or self.filename == "."):
            raise ValueError("source.filename must stay inside the model snapshot")
        _text(self.pointer, "source.pointer", empty=True)
        if self.pointer and not self.pointer.startswith("/"):
            raise ValueError("source.pointer must be an RFC 6901 JSON pointer")
        for token in self.pointer.split("/"):
            index = 0
            while index < len(token):
                if token[index] == "~":
                    if index + 1 >= len(token) or token[index + 1] not in "01":
                        raise ValueError("source.pointer contains an invalid JSON pointer escape")
                    index += 1
                index += 1
        if type(self.task_scoped) is not bool:
            raise ValueError("source.task_scoped must be boolean")
        if self.task_scoped and self.filename != "config.json":
            raise ValueError("task-scoped sources must select config.json task_specific_params")


@dataclass(frozen=True)
class ParameterCondition:
    """Empty values means present; otherwise match one value with its JSON type."""

    parameter: str
    values: tuple[Any, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        _text(self.parameter, "condition.parameter")
        _text(self.reason, "condition.reason", empty=True)
        if type(self.values) is not tuple:
            raise ValueError("condition.values must be a tuple")
        for value in self.values:
            if value is not None and type(value) not in (bool, int, float, str):
                raise ValueError("condition.values must contain JSON scalar values")
            if type(value) is float and not math.isfinite(value):
                raise ValueError("condition.values must be finite")

    def matches(self, values: Mapping[str, Any]) -> bool:
        return self.parameter in values and (
            not self.values or any(_same_value(values[self.parameter], value) for value in self.values)
        )


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    kind: str = ""
    scope: str = "inference"
    consumers: tuple[ParameterConsumer, ...] = ()
    label: str = ""
    description: str = ""
    unit: str = ""
    advanced: bool = False
    enabled: bool = True
    read_only: bool = False
    nullable: bool = False
    default: ParameterDefault | None = None
    choices: tuple[Any, ...] = ()
    minimum: int | float | None = None
    maximum: int | float | None = None
    exclusive_minimum: bool = False
    exclusive_maximum: bool = False
    min_length: int | None = None
    max_length: int | None = None
    item_kind: str = ""
    min_items: int | None = None
    max_items: int | None = None
    requires: tuple[ParameterCondition, ...] = ()
    conflicts: tuple[ParameterCondition, ...] = ()
    tasks: tuple[str, ...] = ()
    backends: tuple[str, ...] = ()
    model_types: tuple[str, ...] = ()
    pipeline_classes: tuple[str, ...] = ()
    validation_method: str = ""
    variation: tuple[str, ...] = ()
    source_bindings: tuple[ParameterSource, ...] = ()

    def __post_init__(self) -> None:
        _validate_spec(self)


def _validate_spec(spec: ParameterSpec) -> None:
    path = f"parameter.{spec.name}"
    _text(spec.name, "parameter.name")
    if not spec.name.isidentifier():
        raise ValueError("parameter.name must be an identifier")
    for name in ("kind", "scope", "label", "description", "unit", "item_kind"):
        _text(getattr(spec, name), f"{path}.{name}", empty=True)
    for name in ("advanced", "enabled", "read_only", "nullable", "exclusive_minimum", "exclusive_maximum"):
        if type(getattr(spec, name)) is not bool:
            raise ValueError(f"{path}.{name} must be boolean")
    for name, cls in (("consumers", ParameterConsumer), ("source_bindings", ParameterSource),
                      ("requires", ParameterCondition), ("conflicts", ParameterCondition)):
        values = getattr(spec, name)
        if type(values) is not tuple or any(not isinstance(value, cls) for value in values):
            raise ValueError(f"{path}.{name} has invalid declarations")
    for name in ("tasks", "backends", "model_types", "pipeline_classes", "variation"):
        _strings(getattr(spec, name), f"{path}.{name}")
    _text(spec.validation_method, f"{path}.validation_method", empty=True)
    if spec.validation_method not in {
        "", "generate.kwargs", "pipeline.predict.kwargs", "pipeline.call.kwargs", "scheduler.class",
    }:
        raise ValueError(f"{path}.validation_method is unknown")
    if not set(spec.variation) <= VARIATIONS:
        raise ValueError(f"{path}.variation contains an unknown axis")
    if spec.scope not in SCOPES:
        raise ValueError(f"{path}.scope is unknown")
    if not spec.enabled and not spec.kind:
        # A named tombstone disables an inherited field; it is not a value spec.
        if (spec.default is not None or spec.choices or spec.consumers or spec.source_bindings
                or spec.requires or spec.conflicts or spec.item_kind
                or any(getattr(spec, name) is not None for name in
                       ("minimum", "maximum", "min_length", "max_length", "min_items", "max_items"))):
            raise ValueError(f"{path}: a disabled field cannot contain partial value declarations")
        return
    if spec.kind not in KINDS:
        raise ValueError(f"{path}.kind is unknown")
    if spec.enabled and not spec.read_only and not spec.consumers:
        raise ValueError(f"{path} requires a declared consumer")
    if spec.scope == "model" and not spec.read_only:
        raise ValueError(f"{path}: model properties must be read-only")
    if type(spec.choices) is not tuple:
        raise ValueError(f"{path}.choices must be a tuple")
    scalar_kind = spec.item_kind if spec.kind == "list" else spec.kind
    if scalar_kind == "enum":
        if not spec.choices:
            raise ValueError(f"{path}.choices must declare a nonempty enum")
    elif spec.choices:
        raise ValueError(f"{path}.choices is only valid for enum values")
    for index, value in enumerate(spec.choices):
        if type(value) not in (bool, int, float, str) or type(value) is float and not math.isfinite(value):
            raise ValueError(f"{path}.choices must contain finite JSON scalars")
        if any(_same_value(value, previous) for previous in spec.choices[:index]):
            raise ValueError(f"{path}.choices contains duplicate values")
    for name in ("minimum", "maximum"):
        value = getattr(spec, name)
        if value is not None and (scalar_kind not in {"integer", "float"} or not _number(value)):
            raise ValueError(f"{path}.{name} requires a finite numeric value")
    if spec.exclusive_minimum and spec.minimum is None or spec.exclusive_maximum and spec.maximum is None:
        raise ValueError(f"{path}: exclusive bounds require their corresponding numeric bound")
    if spec.minimum is not None and spec.maximum is not None and (
        spec.minimum > spec.maximum or spec.minimum == spec.maximum and (spec.exclusive_minimum or spec.exclusive_maximum)
    ):
        raise ValueError(f"{path}: numeric bounds have no valid interval")
    for low, high, kind in (("min_length", "max_length", "string"), ("min_items", "max_items", "list")):
        for name in (low, high):
            value = getattr(spec, name)
            applicable = scalar_kind == kind if kind == "string" else spec.kind == kind
            if value is not None and (not applicable or type(value) is not int or value < 0):
                raise ValueError(f"{path}.{name} requires a nonnegative integer for {kind}")
        if getattr(spec, low) is not None and getattr(spec, high) is not None and getattr(spec, low) > getattr(spec, high):
            raise ValueError(f"{path}: {low} exceeds {high}")
    if spec.kind == "list":
        if spec.item_kind not in KINDS - {"list"}:
            raise ValueError(f"{path}.item_kind must declare a scalar item type")
    elif spec.item_kind:
        raise ValueError(f"{path}.item_kind is only valid for lists")
    if spec.default is not None:
        if not isinstance(spec.default, ParameterDefault):
            raise ValueError(f"{path}.default must wrap an explicit value")
        validate_parameter_value(spec, spec.default.value, path=f"{path}.default.value")


def _validate_scalar(spec: ParameterSpec, kind: str, value: Any, path: str) -> None:
    valid = (
        type(value) is bool if kind == "boolean" else
        type(value) is int if kind == "integer" else
        _number(value) if kind == "float" else
        type(value) is str if kind == "string" else
        any(_same_value(value, choice) for choice in spec.choices) if kind == "enum" else False
    )
    if not valid:
        raise ValueError(f"{path} must be {'a finite number' if kind == 'float' else kind}")
    if kind in {"integer", "float"}:
        if spec.minimum is not None and (value < spec.minimum or spec.exclusive_minimum and value == spec.minimum):
            raise ValueError(f"{path} must be {'greater than' if spec.exclusive_minimum else 'at least'} {spec.minimum}")
        if spec.maximum is not None and (value > spec.maximum or spec.exclusive_maximum and value == spec.maximum):
            raise ValueError(f"{path} must be {'less than' if spec.exclusive_maximum else 'at most'} {spec.maximum}")
    if kind == "string":
        if spec.min_length is not None and len(value) < spec.min_length:
            raise ValueError(f"{path} is shorter than {spec.min_length}")
        if spec.max_length is not None and len(value) > spec.max_length:
            raise ValueError(f"{path} is longer than {spec.max_length}")


def validate_parameter_value(spec: ParameterSpec, value: Any, *, path: str | None = None) -> Any:
    """Validate one value without coercing it or deciding whether to apply it."""
    path = path or spec.name
    if not spec.enabled:
        raise ValueError(f"{path} is disabled")
    if value is None:
        if not spec.nullable:
            raise ValueError(f"{path} does not allow null")
        return value
    if spec.kind == "list":
        if type(value) is not list:
            raise ValueError(f"{path} must be a list")
        if spec.min_items is not None and len(value) < spec.min_items:
            raise ValueError(f"{path} must have at least {spec.min_items} items")
        if spec.max_items is not None and len(value) > spec.max_items:
            raise ValueError(f"{path} must have at most {spec.max_items} items")
        for index, item in enumerate(value):
            _validate_scalar(spec, spec.item_kind, item, f"{path}[{index}]")
    else:
        _validate_scalar(spec, spec.kind, value, path)
    return value


def merge_parameter_specs(
    *groups: Iterable[ParameterSpec], task: str | None = None, backend: str | None = None,
) -> tuple[ParameterSpec, ...]:
    """Whole declarations override by name; tombstones never expose inherited fields."""
    merged: dict[str, ParameterSpec] = {}
    for group in groups:
        seen: set[str] = set()
        for spec in group:
            if not isinstance(spec, ParameterSpec) or spec.name in seen:
                raise ValueError("parameter declarations must have unique names within each layer")
            seen.add(spec.name)
            merged[spec.name] = spec
    selected = tuple(spec for spec in merged.values() if spec.enabled
                     and (not task or not spec.tasks or task in spec.tasks)
                     and (not backend or not spec.backends or backend in spec.backends))
    by_name = {spec.name: spec for spec in selected}
    for spec in selected:
        for condition in (*spec.requires, *spec.conflicts):
            if condition.parameter == spec.name or condition.parameter not in by_name:
                raise ValueError(f"{spec.name}: condition references an unavailable parameter {condition.parameter}")
            for value in condition.values:
                validate_parameter_value(by_name[condition.parameter], value,
                                         path=f"{spec.name}.condition.{condition.parameter}")
    return deepcopy(selected)


def validate_parameter_values(
    specs: Iterable[ParameterSpec], values: Mapping[str, Any], *,
    explicit_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate a partial value map; never fill absent fields or apply defaults.

    Dependencies constrain explicit changes. A legal inactive model default may
    remain evidence when the caller supplies an explicit_names set excluding it.
    """
    selected = merge_parameter_specs(specs)
    by_name = {spec.name: spec for spec in selected}
    if not isinstance(values, Mapping) or any(type(name) is not str for name in values):
        raise ValueError("parameter values must be a mapping with string names")
    unknown = set(values) - set(by_name)
    if unknown:
        raise ValueError(f"unknown parameter fields: {sorted(unknown)}")
    if isinstance(explicit_names, str):
        raise ValueError("explicit parameter names must be a collection of names")
    try:
        explicit = set(values) if explicit_names is None else set(explicit_names)
    except TypeError as exc:
        raise ValueError("explicit parameter names must be a collection of names") from exc
    if not explicit <= set(values):
        raise ValueError("explicit parameter names must identify supplied values")
    for name, value in values.items():
        spec = by_name[name]
        validate_parameter_value(spec, value)
        if name in explicit and spec.read_only:
            raise ValueError(f"{name} is read-only")
        for condition in spec.requires:
            if name in explicit and not condition.matches(values):
                raise ValueError(f"{name}: {condition.reason}" if condition.reason else f"{name} requires {condition.parameter}")
        for condition in spec.conflicts:
            if (name in explicit or condition.parameter in explicit) and condition.matches(values):
                raise ValueError(f"{name}: {condition.reason}" if condition.reason else f"{name} conflicts with {condition.parameter}")
    return deepcopy(dict(values))


class _Absent(Enum):
    VALUE = "absent"


@dataclass(frozen=True)
class ParameterRequest:
    """Preserve explicit, auto and unset intent without authorizing execution."""

    mode: str
    value: Any = _Absent.VALUE

    def __post_init__(self) -> None:
        if type(self.mode) is not str or self.mode not in {"explicit", "auto", "unset"}:
            raise ValueError("parameter request mode must be explicit, auto or unset")
        if self.mode == "explicit" and self.value is _Absent.VALUE:
            raise ValueError("explicit parameter request requires a value, including explicit null")
        if self.mode != "explicit" and self.value is not _Absent.VALUE:
            raise ValueError(f"{self.mode} parameter request cannot contain a value")
        if self.mode == "explicit":
            _validate_json_value(self.value, "request.value")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ParameterRequest:
        if not isinstance(raw, dict) or set(raw) - {"mode", "value"} or "mode" not in raw:
            raise ValueError("parameter request must declare mode and optional explicit value")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, **({"value": deepcopy(self.value)} if self.mode == "explicit" else {})}


def validate_parameter_request(spec: ParameterSpec, request: ParameterRequest) -> ParameterRequest:
    if not spec.enabled or spec.read_only:
        raise ValueError(f"{spec.name} does not accept parameter requests")
    if request.mode == "explicit":
        validate_parameter_value(spec, request.value)
    return request


def parameter_spec_to_dict(spec: ParameterSpec) -> dict[str, Any]:
    """Return JSON-native declaration data, preserving the wrapped default value."""
    def native(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: native(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [native(item) for item in value]
        return value
    return native(asdict(spec))
