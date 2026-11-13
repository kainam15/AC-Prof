"""Configuration discovery preserves observed values without selecting or applying them."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from acprof.extensions.parameters import (
    ParameterConsumer,
    ParameterDefault,
    ParameterSource,
    ParameterSpec,
)
from acprof.models.configuration_sources import discover_configuration_sources

REVISION = "a" * 40


def _task(documents=None, **kwargs):
    raw_documents = documents or {}
    metadata = {
        name: json.loads(raw) if isinstance(raw, bytes) else copy.deepcopy(raw)
        for name, raw in raw_documents.items()
    }
    sources = {
        name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
               "revision": REVISION, "encoding": "raw"}
        for name, raw in raw_documents.items() if isinstance(raw, bytes)
    }
    values = {
        "model_id": "fixture/configuration", "model_source": "huggingface", "model_revision": REVISION,
        "repository_metadata": metadata, "metadata_sources": sources,
        "metadata_diagnostics": (), "metadata_errors": (), "model_config": {}, "hub_metadata": {},
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def _parameter(name="temperature", kind="float", **kwargs):
    values = {
        "name": name, "kind": kind, "consumers": (ParameterConsumer("predict", name),),
        "source_bindings": (ParameterSource("generation_config.json", "/" + name),),
    }
    values.update(kwargs)
    return ParameterSpec(**values)


def _attribute(report, source_id, pointer):
    return next(item for item in report["attributes"]
                if item.get("source", {}).get("source_id") == source_id
                and item["source"]["json_pointer"] == pointer)


def test_default_candidates_keep_each_source_without_applying_precedence():
    config = b'{ "hidden_size": 512, "temperature": 0.7, "task_specific_params": {"summarization": {"temperature": 0.6}} }'
    generation = b'{ "temperature": 0.8 }'
    task = _task({"config.json": config, "generation_config.json": generation})
    spec = _parameter(
        default=ParameterDefault(1.0),
        source_bindings=(
            ParameterSource("generation_config.json", "/temperature"),
            ParameterSource("config.json", "/temperature"),
            ParameterSource("config.json", "/temperature", task_scoped=True),
        ),
    )
    selected = {"task_id": "summarization", "config": {"generation": {"temperature": 0.99}}}
    before = copy.deepcopy((vars(task), asdict(spec), selected))

    report = discover_configuration_sources(task, (spec,), selected)
    assert report["evidence_level"] == "static"
    assert report["runtime_validation"] == "not_run"
    assert all(item["validation_method"] == "parameter_value"
               for item in report["parameters"]["temperature"]["default_candidates"])
    parameter = report["parameters"]["temperature"]
    assert [item["value"] for item in parameter["default_candidates"]] == [0.8, 0.7, 0.6, 1.0]
    assert "value" not in parameter
    assert "selected" not in parameter
    assert "supported" not in parameter
    assert "verified" not in parameter
    assert parameter["state"] == "observed"
    references = [item["source"] for item in parameter["default_candidates"]]
    assert references[0]["raw_sha256"] == hashlib.sha256(generation).hexdigest()
    assert references[1]["raw_sha256"] == hashlib.sha256(config).hexdigest()
    assert references[2]["json_pointer"] == "/task_specific_params/summarization/temperature"
    assert references[3]["raw_sha256"] is None
    assert references[3]["revision"] is None
    assert references[3]["model_id"] is None
    assert references[3]["source_id"] == "parameter:temperature"
    assert references[3]["json_pointer"] == "/default/value"
    assert (vars(task), asdict(spec), selected) == before


def test_canonical_fixture_and_hub_facts_do_not_claim_raw_bytes():
    task = _task(model_config={"model_type": "bert", "hidden_size": 768},
                 hub_metadata={"pipeline_tag": "fill-mask", "tags": ["transformers"]})
    report = discover_configuration_sources(task, ())
    config = report["sources"]["config.json"]
    assert config["kind"] == "model_config"
    assert config["encoding"] == "canonical-json"
    assert config["raw_sha256"] is None
    assert config["canonical_sha256"] == hashlib.sha256(
        b'{"hidden_size":768,"model_type":"bert"}'
    ).hexdigest()
    assert report["sources"]["hub_metadata"]["raw_sha256"] is None
    assert _attribute(report, "config.json", "/hidden_size")["read_only"] is True
    assert _attribute(report, "hub_metadata", "/pipeline_tag")["value"] == "fill-mask"


def test_missing_null_and_registered_null_default_are_distinct():
    task = _task({"generation_config.json": b'{"top_k":null}'})
    specs = (
        _parameter("top_k", "integer", nullable=True),
        _parameter("min_new_tokens", "integer"),
        _parameter("pad_token_id", "integer", nullable=True, default=ParameterDefault(None)),
    )
    parameters = discover_configuration_sources(task, specs)["parameters"]
    assert parameters["top_k"]["default_candidates"][0]["value"] is None
    assert parameters["top_k"]["default_candidates"][0]["validation"] == "valid"
    assert parameters["min_new_tokens"] == {
        "scope": "inference", "state": "unknown", "default_candidates": [],
    }
    pad = parameters["pad_token_id"]["default_candidates"]
    assert len(pad) == 1
    assert pad[0]["value"] is None
    assert pad[0]["source"]["source_id"] == "parameter:pad_token_id"


def test_read_only_properties_are_validated_but_never_become_tunable_defaults():
    task = _task({"config.json": b'{"hidden_size":512,"num_hidden_layers":"many"}'})
    specs = (
        ParameterSpec("hidden_size", "integer", scope="model", read_only=True,
                      source_bindings=(ParameterSource("config.json", "/hidden_size"),)),
        ParameterSpec("num_hidden_layers", "integer", scope="model", read_only=True,
                      source_bindings=(ParameterSource("config.json", "/num_hidden_layers"),)),
        ParameterSpec("vocab_size", "integer", scope="model", read_only=True, default=ParameterDefault(1000),
                      source_bindings=(ParameterSource("config.json", "/vocab_size"),)),
    )
    report = discover_configuration_sources(task, specs)
    assert report["parameters"] == {}
    assert _attribute(report, "config.json", "/hidden_size")["value"] == 512
    assert all(item["read_only"] for item in report["attributes"])
    assert any(item["parameter"] == "num_hidden_layers" and item["json_pointer"] == "/num_hidden_layers"
               for item in report["diagnostics"])
    assert not any(item.get("value") == 1000 for item in report["attributes"])


def test_unregistered_processor_values_remain_facts_and_bindings_are_explicit():
    task = _task({"preprocessor_config.json": b'{"image_mean":[0.1,0.2,0.3],"crop_size":{"height":224,"width":224}}'})
    unregistered = discover_configuration_sources(task, ())
    assert unregistered["parameters"] == {}
    assert _attribute(unregistered, "preprocessor_config.json", "/image_mean")["read_only"] is True
    spec = _parameter("image_mean", "list", scope="workload", item_kind="float",
                      consumers=(ParameterConsumer("preprocess", "image_mean"),),
                      source_bindings=(ParameterSource("preprocessor_config.json", "/image_mean"),))
    registered = discover_configuration_sources(task, (spec,))
    assert registered["parameters"]["image_mean"]["default_candidates"][0]["value"] == [0.1, 0.2, 0.3]
    assert _attribute(registered, "preprocessor_config.json", "/crop_size")["value"] == {"height": 224, "width": 224}
    assert not any(item["source"]["json_pointer"] == "/image_mean" for item in registered["attributes"])


@pytest.mark.parametrize(("value", "kind", "minimum"), [
    (True, "integer", None),
    ("0.7", "float", None),
    (-1, "integer", 0),
])
def test_invalid_parameter_defaults_report_path_without_coercion(value, kind, minimum):
    task = _task({"generation_config.json": {"temperature": value}})
    spec = _parameter(kind=kind, minimum=minimum)
    report = discover_configuration_sources(task, (spec,))
    parameter = report["parameters"]["temperature"]
    candidate = parameter["default_candidates"][0]
    assert type(candidate["value"]) is type(value)
    assert candidate["value"] == value
    assert candidate["validation"] == "invalid"
    assert parameter["state"] == "invalid"
    diagnostic = report["diagnostics"][0]
    assert diagnostic["parameter"] == "temperature"
    assert diagnostic["file"] == "generation_config.json"
    assert diagnostic["json_pointer"] == "/temperature"
    assert "generation_config.json:/temperature" in diagnostic["detail"]


@pytest.mark.parametrize("bad_value", [float("inf"), {"unsupported"}])
def test_non_json_metadata_remains_diagnostic_and_report_is_json_serializable(bad_value):
    task = _task({"generation_config.json": {"temperature": 0.7, "nested": {"a/b~c": bad_value}}})
    report = discover_configuration_sources(task, (_parameter(),))
    source = report["sources"]["generation_config.json"]
    assert source["state"] == "invalid"
    assert source["canonical_sha256"] is None
    assert report["parameters"]["temperature"]["default_candidates"] == []
    assert any(item["json_pointer"] == "/nested/a~1b~0c" for item in report["diagnostics"])
    json.dumps(report, allow_nan=False)


def test_mismatched_raw_revision_is_retained_as_invalid_evidence():
    task = _task({"generation_config.json": b'{"temperature":0.7}'})
    task.metadata_sources["generation_config.json"]["revision"] = "b" * 40
    report = discover_configuration_sources(task, (_parameter(),))
    source = report["sources"]["generation_config.json"]
    assert source["state"] == "invalid"
    assert source["recorded_revision"] == "b" * 40
    assert source["revision"] == REVISION
    assert report["parameters"]["temperature"]["state"] == "invalid"
    assert report["parameters"]["temperature"]["default_candidates"] == []


def test_missing_file_is_unknown_and_does_not_manufacture_default():
    report = discover_configuration_sources(_task(), (_parameter(),))
    assert report["sources"]["generation_config.json"]["state"] == "unknown"
    assert report["sources"]["generation_config.json"]["raw_sha256"] is None
    assert report["parameters"]["temperature"]["state"] == "unknown"


def test_bad_source_container_reports_binding_path():
    task = _task({"config.json": {"task_specific_params": "invalid"}})
    spec = _parameter(source_bindings=(ParameterSource("config.json", "/temperature", task_scoped=True),))
    report = discover_configuration_sources(task, (spec,), {"task_id": "summarization"})
    assert report["parameters"]["temperature"]["state"] == "invalid"
    assert report["diagnostics"][0]["json_pointer"] == "/task_specific_params/summarization/temperature"
    assert "/task_specific_params" in report["diagnostics"][0]["detail"]


def test_grouped_semantic_tasks_keep_separate_original_defaults_and_escaped_paths():
    task = _task({"config.json": {"task_specific_params": {
        "summary/a~b": {"temperature": 0.4}, "translation": {"temperature": 0.9},
    }}})
    spec = _parameter(source_bindings=(ParameterSource("config.json", "/temperature", task_scoped=True),))
    selected = {
        "task_id": "profile_group", "execution_profile": {"semantic_task_ids": ["summary/a~b", "translation"]},
        "config": {"generation": {"temperature": 42}},
    }
    report = discover_configuration_sources(task, (spec,), selected)
    candidates = report["parameters"]["temperature"]["default_candidates"]
    assert [(item["semantic_task_id"], item["value"]) for item in candidates] == [("summary/a~b", 0.4), ("translation", 0.9)]
    assert candidates[0]["source"]["json_pointer"] == "/task_specific_params/summary~1a~0b/temperature"
    assert candidates[1]["source"]["json_pointer"] == "/task_specific_params/translation/temperature"


def test_task_scoped_defaults_require_selection_and_array_pointers_are_supported():
    task = _task({"config.json": {"task_specific_params": {"summarization": {"temperature": 0.6}},
                                 "rules": [{"value": 0.2}, {"value": 0.3}]}})
    scoped = _parameter(source_bindings=(ParameterSource("config.json", "/temperature", task_scoped=True),))
    assert discover_configuration_sources(task, (scoped,))["parameters"]["temperature"]["state"] == "unknown"
    indexed = _parameter(source_bindings=(ParameterSource("config.json", "/rules/1/value"),))
    report = discover_configuration_sources(task, (indexed,))
    assert report["parameters"]["temperature"]["default_candidates"][0]["value"] == 0.3
    assert _attribute(report, "config.json", "/rules/0")["value"] == {"value": 0.2}


def test_returned_mutable_values_do_not_alias_inputs():
    task = _task({"preprocessor_config.json": {"image_mean": [0.1, 0.2], "crop_size": {"width": 224}}})
    spec = _parameter("image_mean", "list", item_kind="float", default=ParameterDefault([0.5]),
                      source_bindings=(ParameterSource("preprocessor_config.json", "/image_mean"),))
    report = discover_configuration_sources(task, (spec,))
    report["parameters"]["image_mean"]["default_candidates"][0]["value"].append(1.0)
    report["parameters"]["image_mean"]["default_candidates"][1]["value"].append(1.0)
    _attribute(report, "preprocessor_config.json", "/crop_size")["value"]["width"] = 1
    assert task.repository_metadata["preprocessor_config.json"] == {"image_mean": [0.1, 0.2], "crop_size": {"width": 224}}
    assert spec.default.value == [0.5]


def test_deep_valid_json_is_diagnosed_before_copying_read_only_attributes():
    raw = b'{"nested":' * 600 + b"1" + b"}" * 600
    task = _task({"config.json": raw})
    report = discover_configuration_sources(task, ())
    assert report["sources"]["config.json"]["state"] == "invalid"
    assert report["attributes"] == []
    diagnostic = report["diagnostics"][0]
    assert diagnostic["json_pointer"] == "/nested" * 257
    assert diagnostic["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    value = task.repository_metadata["config.json"]
    for _ in range(600):
        value = value["nested"]
    assert value == 1
    json.dumps(report, allow_nan=False)


def test_read_only_attributes_accept_metadata_at_the_depth_limit():
    raw = b'{"nested":' * 256 + b"1" + b"}" * 256
    report = discover_configuration_sources(_task({"config.json": raw}), ())
    assert report["sources"]["config.json"]["state"] == "observed"
    value = _attribute(report, "config.json", "/nested")["value"]
    for _ in range(255):
        value = value["nested"]
    assert value == 1
    assert report["diagnostics"] == []
    json.dumps(report, allow_nan=False)


def test_existing_invalid_source_diagnostics_prevent_architecture_fallback():
    task = _task(model_config={"model_type": "bert", "temperature": 0.5},
                 metadata_sources={"config.json": {"sha256": "c" * 64, "bytes": 99, "revision": REVISION, "encoding": "raw"}},
                 metadata_diagnostics=({"reason_code": "metadata_invalid", "file": "config.json", "json_pointer": "/bad",
                                        "raw_sha256": "c" * 64, "detail": "non-finite number"},))
    spec = _parameter(source_bindings=(ParameterSource("config.json", "/temperature"),))
    report = discover_configuration_sources(task, (spec,))
    assert report["sources"]["config.json"]["state"] == "invalid"
    assert report["parameters"]["temperature"]["default_candidates"] == []
    assert report["diagnostics"][0]["json_pointer"] == "/bad"
