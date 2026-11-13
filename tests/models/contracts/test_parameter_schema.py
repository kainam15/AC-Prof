"""Parameter declarations preserve strict values, scope and user intent."""
from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from acprof.extensions import CATALOG, ExtensionCatalog, load_catalog
from acprof.extensions.parameters import (
    ParameterCondition,
    ParameterConsumer,
    ParameterDefault,
    ParameterRequest,
    ParameterSource,
    ParameterSpec,
    merge_parameter_specs,
    parameter_spec_to_dict,
    validate_parameter_request,
    validate_parameter_value,
    validate_parameter_values,
)
from acprof.extensions.schema import declaration_from_dict


def _spec(name="value", kind="integer", **kwargs):
    return ParameterSpec(name, kind, consumers=(ParameterConsumer("predict", f"params.{name}"),), **kwargs)


@pytest.mark.parametrize(("kind", "value", "options"), [
    ("boolean", True, {}),
    ("integer", 3, {"minimum": 1, "maximum": 3}),
    ("float", 1, {"minimum": 0, "exclusive_minimum": True}),
    ("enum", "eager", {"choices": ("eager", "sdpa")}),
    ("string", "en", {"min_length": 1, "max_length": 3}),
    ("list", [1, 2], {"item_kind": "integer", "minimum": 1, "min_items": 1, "max_items": 2}),
])
def test_parameter_types_preserve_values_without_coercion(kind, value, options):
    result = validate_parameter_value(_spec(kind=kind, **options), value)
    assert result is value


@pytest.mark.parametrize(("kind", "value", "options", "path"), [
    ("boolean", 1, {}, "run.value"),
    ("integer", True, {}, "run.value"),
    ("integer", "2", {}, "run.value"),
    ("integer", 0, {"minimum": 1}, "run.value"),
    ("float", True, {}, "run.value"),
    ("float", float("inf"), {}, "run.value"),
    ("float", float("nan"), {}, "run.value"),
    ("float", 0, {"minimum": 0, "exclusive_minimum": True}, "run.value"),
    ("float", 1.1, {"maximum": 1}, "run.value"),
    ("enum", True, {"choices": (1, 2)}, "run.value"),
    ("string", "long", {"max_length": 3}, "run.value"),
    ("list", (1, 2), {"item_kind": "integer"}, "run.value"),
    ("list", [1, True], {"item_kind": "integer"}, r"run.value\[1\]"),
    ("list", [float("nan")], {"item_kind": "float"}, r"run.value\[0\]"),
    ("list", [], {"item_kind": "string", "min_items": 1}, "run.value"),
    ("list", ["bad"], {"item_kind": "enum", "choices": ("good",)}, r"run.value\[0\]"),
])
def test_invalid_parameter_values_report_field_or_item_path(kind, value, options, path):
    with pytest.raises(ValueError, match=path):
        validate_parameter_value(_spec(kind=kind, **options), value, path="run.value")


def test_null_default_absence_and_explicit_intent_remain_distinct():
    absent = _spec(nullable=True)
    null_default = replace(absent, default=ParameterDefault(None))
    assert parameter_spec_to_dict(absent)["default"] is None
    assert parameter_spec_to_dict(null_default)["default"] == {"value": None}
    explicit = ParameterRequest.from_dict({"mode": "explicit", "value": None})
    assert validate_parameter_request(null_default, explicit).to_dict() == {"mode": "explicit", "value": None}
    assert ParameterRequest.from_dict({"mode": "unset"}).to_dict() == {"mode": "unset"}
    assert ParameterRequest.from_dict({"mode": "auto"}).to_dict() == {"mode": "auto"}
    with pytest.raises(ValueError, match="null"):
        validate_parameter_request(_spec(), explicit)
    default = _spec(default=ParameterDefault(3))
    assert validate_parameter_request(default, ParameterRequest("explicit", 3)).to_dict() == {"mode": "explicit", "value": 3}
    assert validate_parameter_request(default, ParameterRequest("auto")).to_dict() == {"mode": "auto"}


@pytest.mark.parametrize("kind", ["request", "default"])
def test_deep_parameter_json_is_rejected_with_a_path_without_mutation(kind):
    value = 7
    for _ in range(600):
        value = {"nested": value}
    with pytest.raises(ValueError, match="maximum JSON depth of 256") as caught:
        ParameterRequest("explicit", value) if kind == "request" else ParameterDefault(value)
    assert str(caught.value).startswith(f"{kind}.value.nested")
    original = value
    for _ in range(600):
        original = original["nested"]
    assert original == 7


@pytest.mark.parametrize("container", ["object", "array"])
def test_supported_parameter_json_depth_can_be_serialized(container):
    value = 7
    for _ in range(256):
        value = {"nested": value} if container == "object" else [value]
    request = ParameterRequest("explicit", value)
    result = request.to_dict()
    assert result["mode"] == "explicit"
    assert result["value"] is not value
    observed = result["value"]
    for _ in range(256):
        observed = observed["nested"] if container == "object" else observed[0]
    assert observed == 7
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    default = ParameterDefault(value)
    assert default.value is value
    assert json.loads(json.dumps(asdict(default), allow_nan=False))["value"] == value


@pytest.mark.parametrize("raw", [
    {"mode": "explicit"}, {"mode": "auto", "value": None},
    {"mode": "unset", "value": 3}, {"mode": "fallback"}, {"mode": "explicit", "value": float("nan")},
    {"mode": "explicit", "value": {"nested": [float("inf")]}},
    {"mode": "explicit", "value": (1, 2)}, {"mode": "auto", "unknown": 1},
])
def test_requests_reject_ambiguous_or_non_json_values(raw):
    with pytest.raises(ValueError, match="request"):
        ParameterRequest.from_dict(raw)


@pytest.mark.parametrize("kwargs", [
    {"kind": "enum", "choices": ()},
    {"kind": "enum", "choices": ("same", "same")},
    {"kind": "float", "minimum": float("inf")},
    {"kind": "integer", "minimum": True},
    {"kind": "float", "minimum": 2, "maximum": 1},
    {"kind": "float", "minimum": 1, "maximum": 1, "exclusive_maximum": True},
    {"kind": "list", "item_kind": "integer", "min_items": True},
    {"kind": "integer", "item_kind": "integer"},
    {"kind": "string", "scope": "model"},
    {"kind": "integer", "default": ParameterDefault("wrong")},
])
def test_inconsistent_declarations_are_rejected(kwargs):
    with pytest.raises(ValueError, match=r"parameter\.value"):
        _spec(**kwargs)


def test_writable_fields_need_consumers_and_model_facts_are_read_only():
    with pytest.raises(ValueError, match="consumer"):
        ParameterSpec("dtype", "enum", scope="loading", choices=("FP32",))
    with pytest.raises(ValueError, match="stage"):
        ParameterConsumer("unknown", "dtype")
    fact = ParameterSpec("layers", "integer", scope="model", read_only=True)
    assert validate_parameter_value(fact, 12) == 12
    with pytest.raises(ValueError, match="read-only"):
        validate_parameter_values((fact,), {"layers": 12})
    with pytest.raises(ValueError, match="unknown"):
        validate_parameter_values((_spec(),), {"unknown": 1})


def test_dependencies_constrain_explicit_changes_without_applying_defaults():
    sample = _spec("do_sample", "boolean", default=ParameterDefault(False))
    temperature = _spec("temperature", "float", minimum=0, exclusive_minimum=True,
                        requires=(ParameterCondition("do_sample", (True,), "请先开启随机采样"),))
    specs = (sample, temperature)
    values = {"do_sample": False, "temperature": 0.7}
    assert validate_parameter_values(specs, values, explicit_names=set()) == values
    assert validate_parameter_values(specs, values, explicit_names={"do_sample"}) == values
    with pytest.raises(ValueError, match="temperature.*随机采样"):
        validate_parameter_values(specs, values, explicit_names={"temperature"})
    with pytest.raises(ValueError, match="temperature.*随机采样"):
        validate_parameter_values(specs, {"temperature": 0.7})
    assert validate_parameter_values(specs, {"do_sample": True, "temperature": 0.7})["temperature"] == 0.7
    assert validate_parameter_values(specs, {}) == {}


def test_conflicts_are_symmetric_for_explicit_changes_and_missing_names_fail():
    first = _spec("first", conflicts=(ParameterCondition("second"),))
    second = _spec("second")
    with pytest.raises(ValueError, match="first.*second"):
        validate_parameter_values((first, second), {"first": 1, "second": 2}, explicit_names={"second"})
    with pytest.raises(ValueError, match="unavailable parameter"):
        merge_parameter_specs((first,))
    with pytest.raises(ValueError, match="supplied values"):
        validate_parameter_values((second,), {}, explicit_names={"second"})


def test_declaration_overrides_replace_whole_fields_and_disable_inherited_fields():
    base = _spec("size", minimum=1, maximum=8, default=ParameterDefault(4))
    backend = _spec("size", minimum=2, maximum=6, default=ParameterDefault(3))
    task = _spec("size", minimum=3, maximum=5, default=ParameterDefault(5))
    assert merge_parameter_specs((base,), (backend,), (task,))[0].default.value == 5
    assert merge_parameter_specs((base,), (ParameterSpec("size", enabled=False),)) == ()
    with pytest.raises(ValueError, match="unique names"):
        merge_parameter_specs((base, base))
    mutable = _spec("ids", "list", item_kind="integer", default=ParameterDefault([1, 2]))
    selected = merge_parameter_specs((mutable,))[0]
    selected.default.value.append(3)
    assert mutable.default.value == [1, 2]


def test_sources_are_bounded_json_pointers_without_expression_evaluation():
    source = ParameterSource("config.json", "/a~1b~0c", task_scoped=True)
    assert source.pointer == "/a~1b~0c"
    for filename, pointer, scoped, message in [
        ("../config.json", "/x", False, "source.filename"),
        ("config.json", "x", False, "source.pointer"),
        ("config.json", "/bad~2", False, "source.pointer"),
        ("generation_config.json", "/x", True, "task-scoped sources"),
    ]:
        with pytest.raises(ValueError, match=message):
            ParameterSource(filename, pointer, scoped)


def test_json_round_trip_preserves_null_defaults_and_rejects_unknown_fields():
    spec = _spec("optional", nullable=True, default=ParameterDefault(None))
    raw = parameter_spec_to_dict(spec)
    restored = declaration_from_dict(ParameterSpec, json.loads(json.dumps(raw, allow_nan=False)))
    assert restored == spec
    with pytest.raises(ValueError, match="unknown fields"):
        declaration_from_dict(ParameterSpec, {**raw, "typo": True})
    with pytest.raises(ValueError, match="finite"):
        ParameterDefault({"nested": float("inf")})


def test_catalog_exposes_only_parameters_for_the_selected_consumer():
    text = CATALOG.get_extension("nlp", "transformers_model", task="text-generation")
    assert [spec.name for spec in text.parameters] == [
        "max_new_tokens", "max_length", "do_sample", "temperature", "top_p", "num_beams",
    ]
    assert all(spec.scope == "inference" and spec.consumers[0].stage == "predict" for spec in text.parameters)
    assert all(spec.source_bindings[-1].task_scoped for spec in text.parameters)
    for backend, task in [
        ("transformers_pipeline", "fill-mask"),
        ("sentence_transformers", "feature-extraction"),
        ("cross_encoder", "text-ranking"),
    ]:
        assert CATALOG.get_extension("nlp", backend, task=task).parameters == ()


def test_catalog_overrides_and_nullable_defaults_use_existing_manifest_schema(tmp_path):
    original = CATALOG.get_extension("structured", "torchscript")
    value = _spec("optional", nullable=True, default=ParameterDefault(None))
    raw = asdict(replace(original, parameters=(value,), backend_parameters={
        "torchscript": (replace(value, default=ParameterDefault(4)),),
    }, task_parameters={"tabular-regression": (replace(value, default=ParameterDefault(None)),)}))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 2, "extensions": [raw]}), encoding="utf-8")
    catalog = load_catalog([path])
    selected = catalog.get_extension("structured", "torchscript", task="tabular-regression")
    assert selected.parameters[0].default.value is None
    assert catalog.get_extension("structured", "torchscript", task="tabular-classification").parameters[0].default.value == 4
    disabled = replace(original, parameters=(value,), backend_parameters={
        "torchscript": (ParameterSpec("optional", enabled=False),),
    })
    catalog = ExtensionCatalog()
    catalog.add(disabled)
    assert catalog.get_extension("structured", "torchscript", task="tabular-regression").parameters == ()


def test_catalog_rejects_parameter_routes_outside_the_declared_extension():
    original = CATALOG.get_extension("structured", "torchscript")
    with pytest.raises(ValueError, match="undeclared"):
        ExtensionCatalog().add(replace(original, backend_parameters={"invented": (_spec(),)}))
    with pytest.raises(ValueError, match="undeclared"):
        ExtensionCatalog().add(replace(original, parameters=(_spec(tasks=("invented",)),)))
