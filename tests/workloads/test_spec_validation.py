"""Behavioral boundaries shared by, or intentionally different between, families."""

import hashlib
import json
import re

import pytest

from acprof.workloads.cv import CVWorkloadGenerator
from acprof.workloads.diffusion import DiffusionWorkloadGenerator
from acprof.workloads.multimodal import MultimodalWorkloadGenerator
from acprof.workloads.nlp import NLPWorkloadGenerator
from acprof.workloads.structured import StructuredWorkloadGenerator

GENERATORS = {
    "cv": (CVWorkloadGenerator, "image-classification"),
    "diffusion": (DiffusionWorkloadGenerator, "text-to-image"),
    "multimodal": (MultimodalWorkloadGenerator, "image-text-to-text"),
    "nlp": (NLPWorkloadGenerator, "text-generation"),
    "structured": (StructuredWorkloadGenerator, "tabular-classification"),
}
STRUCTURED_SHAPE_ERROR = (
    "structured workload spec must be an object with supported keys: "
    "feature_dim, input_scales, provenance, schema_version, seed, task"
)


def make_generator(tmp_path, family, spec):
    path = tmp_path / "workload.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    generator, task = GENERATORS[family]
    return generator("example/model", task, 1, workload_spec_path=str(path))


@pytest.mark.parametrize(
    ("family", "fields", "valid"),
    [
        ("cv", {}, True),
        ("diffusion", {}, True),
        ("structured", {}, False),
        ("multimodal", {}, False),
        ("nlp", {}, False),
        ("cv", {"schema_version": True}, False),
        ("diffusion", {"schema_version": True}, False),
        ("structured", {"schema_version": True}, False),
        # Existing equality-based manifests have not undergone a schema migration.
        ("multimodal", {"schema_version": True}, True),
        ("nlp", {"schema_version": True}, True),
    ],
)
def test_family_version_policy_is_not_silently_unified(tmp_path, family, fields, valid):
    if valid:
        make_generator(tmp_path, family, fields)
    else:
        with pytest.raises(ValueError, match="schema_version"):
            make_generator(tmp_path, family, fields)


@pytest.mark.parametrize("family", GENERATORS)
@pytest.mark.parametrize("version", [None, "1", 2, False])
def test_unsupported_explicit_versions_fail(tmp_path, family, version):
    with pytest.raises(ValueError, match="schema_version"):
        make_generator(tmp_path, family, {"schema_version": version})


@pytest.mark.parametrize("family", GENERATORS)
def test_numeric_version_one_keeps_existing_acceptance(tmp_path, family):
    make_generator(tmp_path, family, {"schema_version": 1.0})


@pytest.mark.parametrize(
    ("family", "message"),
    [
        ("cv", "CV workload spec must be an object"),
        ("diffusion", "diffusion workload spec must be an object"),
        ("multimodal", "multimodal workload manifest must be an object"),
        ("nlp", "NLP workload requires schema_version=1"),
        ("structured", STRUCTURED_SHAPE_ERROR),
    ],
)
@pytest.mark.parametrize("spec", [None, [], "wrong", True])
def test_manifest_shape_keeps_family_diagnostics(tmp_path, family, message, spec):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        make_generator(tmp_path, family, spec)


@pytest.mark.parametrize(
    ("family", "message"),
    [
        ("cv", "unknown CV workload keys: aaa, zzz"),
        ("diffusion", "unknown diffusion workload keys: aaa, zzz"),
        ("multimodal", "unknown multimodal workload keys: aaa, zzz"),
        ("nlp", "invalid NLP workload task or fields: ['aaa', 'zzz']"),
        ("structured", STRUCTURED_SHAPE_ERROR),
    ],
)
def test_unknown_fields_keep_sorted_diagnostics(tmp_path, family, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        make_generator(tmp_path, family, {"schema_version": 1, "zzz": 0, "aaa": 0})


@pytest.mark.parametrize("family", ["cv", "diffusion"])
@pytest.mark.parametrize("workload_id", [None, True, 1, "", " \t"])
def test_workload_identity_rejects_blank_or_nonstring_values(tmp_path, family, workload_id):
    with pytest.raises(ValueError, match="^workload_id must be a non-empty string$"):
        make_generator(tmp_path, family, {"workload_id": workload_id})


@pytest.mark.parametrize("family", ["cv", "diffusion"])
def test_workload_identity_preserves_significant_whitespace(tmp_path, family):
    generator = make_generator(tmp_path, family, {"workload_id": " named workload "})
    assert generator.plan_metadata()["workload_id"] == " named workload "


@pytest.mark.parametrize(
    ("family", "scales", "expected", "error"),
    [
        ("cv", [2, 1], [2.0, 1.0], None),
        ("diffusion", ["128", 64], [128, 64], None),
        ("multimodal", [448, 224, 448.0], [224.0, 448.0], None),
        ("structured", [1.0, 2], [1.0, 2.0], None),
        ("cv", [1, 1.0], None, "input_scales must not contain duplicates"),
        ("diffusion", ["64", 64.0], None, "input_scales must not contain duplicates"),
        ("structured", [2, 1], None, "input_scales must be strictly increasing"),
        ("structured", [1, 1.0], None, "input_scales must be strictly increasing"),
        ("cv", ["1"], None, "CV input scale must be a finite positive resolution multiplier"),
        ("multimodal", ["224"], None, "resolution_px must be a positive integer"),
    ],
)
def test_scale_conversion_order_and_duplicate_policy(tmp_path, family, scales, expected, error):
    spec = {"schema_version": 1, "input_scales": scales}
    if error:
        with pytest.raises(ValueError, match=f"^{re.escape(error)}$"):
            make_generator(tmp_path, family, spec)
    else:
        generator = make_generator(tmp_path, family, spec)
        assert generator.default_input_scales() == expected
        returned = generator.default_input_scales()
        returned.append(999)
        assert generator.default_input_scales() == expected


@pytest.mark.parametrize(
    ("family", "message"),
    [
        ("cv", "input_scales must be a non-empty array"),
        ("diffusion", "input_scales must be a non-empty resolution array"),
        ("multimodal", "input_scales must be a nonempty list"),
        ("structured", "input_scales must be a non-empty list"),
    ],
)
@pytest.mark.parametrize("scales", [None, [], "1", 1])
def test_invalid_scale_collections_keep_family_diagnostics(tmp_path, family, message, scales):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        make_generator(tmp_path, family, {"schema_version": 1, "input_scales": scales})


@pytest.mark.parametrize("family", ["cv", "diffusion", "multimodal", "structured"])
@pytest.mark.parametrize("scale", [True, float("nan"), float("inf"), -1])
def test_invalid_numeric_scales_do_not_become_valid_after_normalization(tmp_path, family, scale):
    with pytest.raises(ValueError, match="scale|resolution|input_scale"):
        make_generator(tmp_path, family, {"schema_version": 1, "input_scales": [scale]})


@pytest.mark.parametrize(
    ("family", "message"),
    [
        ("cv", "params must be an object"),
        ("diffusion", "params must be an object"),
        ("multimodal", "params must be an object"),
        ("nlp", "unsupported params for NLP workload task text-generation"),
    ],
)
@pytest.mark.parametrize("params", [None, [], "wrong"])
def test_params_shape_keeps_family_diagnostics(tmp_path, family, message, params):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        make_generator(tmp_path, family, {"schema_version": 1, "params": params})


@pytest.mark.parametrize(
    ("family", "params", "message"),
    [
        ("cv", {"num_workers": 2, "batch_size": 2}, "unsupported CV params: batch_size, num_workers"),
        ("diffusion", {"zzz": 2, "aaa": 2}, "unsupported diffusion params: aaa, zzz"),
    ],
)
def test_params_reject_reserved_or_unknown_fields_before_asset_reads(tmp_path, family, params, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        make_generator(
            tmp_path, family,
            {"schema_version": 1, "params": params, "image_path": "missing.png"},
        )


@pytest.mark.parametrize(
    ("family", "hash_field"),
    [
        ("cv", "workload_spec_sha256"),
        ("diffusion", "workload_spec_sha256"),
        ("multimodal", "manifest_sha256"),
        ("nlp", "workload_spec_sha256"),
        ("structured", "workload_spec_sha256"),
    ],
)
def test_manifest_identity_uses_original_bytes(tmp_path, family, hash_field):
    raw = b'{\n  "schema_version" : 1\n}\n'
    path = tmp_path / "source.json"
    path.write_bytes(raw)
    generator, task = GENERATORS[family]
    workload = generator("example/model", task, 1, workload_spec_path=str(path))
    assert workload.plan_metadata()[hash_field] == hashlib.sha256(raw).hexdigest()
    assert path.read_bytes() == raw


@pytest.mark.parametrize(("family", "scale"), [("cv", 0.125), ("multimodal", 64)])
def test_nested_params_are_copied_to_generated_payloads(tmp_path, family, scale):
    generator = make_generator(
        tmp_path, family, {"schema_version": 1, "params": {"custom": {"items": [1, 2]}}},
    )
    first = generator.generate(scale)
    first["params"]["custom"]["items"].append(3)
    metadata = generator.plan_metadata()
    metadata["params"]["custom"]["items"].append(4)
    assert generator.generate(scale)["params"]["custom"]["items"] == [1, 2]
