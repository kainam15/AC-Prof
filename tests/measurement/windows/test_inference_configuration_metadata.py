"""Static metadata stores confirmed intent separately from observed runtime facts."""
from __future__ import annotations

import copy
import json

import pytest

from acprof.artifacts import read_static_metadata
from acprof.host.metadata.static import StaticMeta, enrich_static_meta, write_static_meta_json
from tests.support.inference_configuration_fixture import frozen_task, verified_runtime_report


def metadata_fixture() -> StaticMeta:
    task = frozen_task()
    return StaticMeta(
        task.model_id, task.model_revision, "nlp", task.pipeline_tag, task.runtime_backend,
        "fixture:image", 1, "tokens", "fixture run", "https://example.invalid/model", "off",
        None, None, None, "fixture", "not_requested", "not_requested", "unknown", "unknown",
        inference_configuration=copy.deepcopy(task.inference_configuration),
    )


@pytest.mark.parametrize("replacement", [{}, None, []])
def test_enrichment_cannot_clear_or_replace_the_frozen_configuration(replacement):
    metadata = metadata_fixture()
    original = copy.deepcopy(metadata.inference_configuration)

    with pytest.raises(ValueError, match="frozen inference configuration|must be an object"):
        enrich_static_meta(metadata, {"inference_configuration": replacement})
    assert metadata.inference_configuration == original


def test_runtime_observations_do_not_rewrite_frozen_values_or_identity(tmp_path):
    task = frozen_task()
    metadata = metadata_fixture()
    expected = copy.deepcopy(metadata.inference_configuration)
    runtime = {"status": "ok", "devices": {"off": verified_runtime_report(task)}}

    enriched = enrich_static_meta(metadata, {"runtime_validation": runtime})
    path = tmp_path / "static_meta.json"
    write_static_meta_json(enriched, str(path))
    stored = json.loads(path.read_text())

    assert stored["schema_version"] == 8
    assert stored["inference_configuration"] == expected
    assert stored["inference_configuration"]["validation"]["runtime"] == "not_run"
    assert stored["runtime_validation"] == runtime
    assert metadata.runtime_validation == {}


def test_mutated_configuration_cannot_overwrite_a_published_static_file(tmp_path):
    metadata = metadata_fixture()
    path = tmp_path / "static_meta.json"
    write_static_meta_json(metadata, str(path))
    before = path.read_bytes()
    metadata.inference_configuration["loading"]["off"]["dtype"] = "FP16"

    with pytest.raises(ValueError, match="identity_sha256"):
        write_static_meta_json(metadata, str(path))
    assert path.read_bytes() == before


def test_legacy_static_schema_remains_unknown_without_migration(tmp_path):
    path = tmp_path / "static_meta.json"
    original = '{"schema_version":7,"model_name":"fixture/legacy"}'
    path.write_text(original)

    result = read_static_metadata(tmp_path, required=True)

    assert "inference_configuration" not in result
    assert result["schema_version"] == 7
    assert path.read_text() == original


def test_static_reader_rejects_mutated_configuration_without_rewriting(tmp_path):
    metadata = metadata_fixture()
    path = tmp_path / "static_meta.json"
    write_static_meta_json(metadata, str(path))
    payload = json.loads(path.read_text())
    payload["inference_configuration"]["loading"]["off"]["dtype"] = "FP16"
    path.write_text(json.dumps(payload))
    before = path.read_bytes()

    with pytest.raises(ValueError, match="identity_sha256"):
        read_static_metadata(tmp_path, required=True)
    assert path.read_bytes() == before
