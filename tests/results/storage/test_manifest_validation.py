"""Malformed manifests retain evidence and report the exact invalid field."""
from __future__ import annotations

import json
import re

import pytest

from acprof.analysis.audit import audit_result
from acprof.artifacts import read_json_object
from acprof.result_store.protocol import publication_lock, validate_manifest
from acprof.result_store.reader import read_snapshot
from acprof.result_store.writer import ResultWriter
from acprof.results.csv import ResultValidationError
from acprof.results.layout import ArtifactLayout
from tests.results.storage.test_sequential_result_reads import artifact_hashes
from tests.support.result_store_fixtures import assert_writer_lease_available

MALFORMED_FIELDS = [
    pytest.param(("state",), [], id="state-list"),
    pytest.param(("state",), {}, id="state-object"),
    pytest.param(("state",), None, id="state-null"),
    pytest.param(("state",), "unknown", id="state-unknown"),
    pytest.param(("layers",), [], id="layers-list"),
    pytest.param(("layers",), None, id="layers-null"),
    pytest.param(("layers",), {}, id="layers-missing"),
    pytest.param(("layers",), {"unknown": {}}, id="layers-unknown"),
    pytest.param(("layers", "summary"), [], id="layer-list"),
    pytest.param(("layers", "summary"), None, id="layer-null"),
    pytest.param(("layers", "summary"), {}, id="layer-empty"),
    pytest.param(("layers", "summary", "fields"), {}, id="layer-fields-object"),
    pytest.param(("layers", "summary", "fields"), [None], id="layer-fields-element"),
    pytest.param(("layers", "summary", "fields"), [], id="layer-fields-missing"),
    pytest.param(("layers", "summary", "path"), True, id="layer-path-type"),
    pytest.param(("layers", "summary", "path"), "wrong.csv", id="layer-path-value"),
    pytest.param(("layers", "summary", "sha256"), [], id="layer-hash-type"),
    pytest.param(("layers", "summary", "sha256"), "broken", id="layer-hash-value"),
    pytest.param(("wide_fields",), {}, id="wide-fields-object"),
    pytest.param(("wide_fields",), [None], id="wide-fields-element"),
    pytest.param(("schema_sha256",), [], id="schema-hash-type"),
    pytest.param(("schema_sha256",), "broken", id="schema-hash-value"),
    pytest.param(("run_id",), True, id="run-id-type"),
    pytest.param(("run_id",), "", id="run-id-empty"),
    *[
        pytest.param(path, value, id=f"{'-'.join(path)}-{label}")
        for path in (
            ("schema_version",), ("sequence",), ("row_count",),
            ("layers", "summary", "row_count"),
            ("layers", "summary", "committed_bytes"),
            ("layers", "summary", "revision"),
        )
        for label, value in (("bool", True), ("float", 3.0))
    ],
    *[
        pytest.param(path, value, id=f"{'-'.join(path)}-value")
        for path, value in (
            (("schema_version",), 4), (("sequence",), -1), (("row_count",), -1),
            (("layers", "summary", "row_count"), -1),
            (("layers", "summary", "committed_bytes"), 0),
            (("layers", "summary", "revision"), -1),
        )
    ],
]


@pytest.fixture
def store(tmp_path):
    layout = ArtifactLayout.for_new_run(tmp_path)
    layout.initialize()
    with ResultWriter(tmp_path) as writer:
        writer.update_rows([{
            "cpu_cores": "1", "mem_cap_gb": "4", "gpu_mode": "off",
            "input_scale": "1", "warmup": "0", "repeat_idx": "0",
            "run_id": writer.manifest["run_id"], "case_id": "1c_4g_off",
            "case_attempt_id": "a" * 32, "window_id": "synthetic-window",
            "status": "ok", "error": "", "environment_class": "unknown",
        }], reason="synthetic manifest boundary fixture")
    return layout


def change_manifest(layout, path, value):
    manifest = read_json_object(layout.result_csv)
    target = manifest
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    layout.result_csv.write_text(json.dumps(manifest))
    return manifest


@pytest.mark.parametrize(("path", "value"), MALFORMED_FIELDS)
def test_malformed_manifest_is_rejected_without_writes(store, path, value):
    change_manifest(store, path, value)
    before = artifact_hashes(store)
    with pytest.raises(ResultValidationError, match="result|layer"):
        read_snapshot(store.root)
    assert artifact_hashes(store) == before
    with pytest.raises(ResultValidationError, match="result|layer"):
        with ResultWriter(store.root):
            pytest.fail("malformed manifest acquired a recovered writer")
    assert artifact_hashes(store) == before
    report = audit_result(store.root)
    assert not report["valid"]
    assert report["counts"]["rows"] == 0
    assert any(issue["code"] == "invalid_csv" for issue in report["issues"])
    assert artifact_hashes(store) == before
    assert_writer_lease_available(store)
    with publication_lock(store, exclusive=True, blocking=False):
        pass


@pytest.mark.parametrize(("path", "value"), MALFORMED_FIELDS)
def test_manifest_diagnostic_identifies_invalid_field(store, path, value):
    manifest = change_manifest(store, path, value)
    before = artifact_hashes(store)
    with pytest.raises(ResultValidationError, match=re.escape(".".join(path))):
        validate_manifest(store, manifest)
    assert artifact_hashes(store) == before


def test_valid_manifest_remains_readable_auditable_and_recoverable(store):
    before = artifact_hashes(store)
    snapshot = read_snapshot(store.root)
    assert len(snapshot.rows) == 1
    assert snapshot.rows[0]["run_id"] == read_json_object(store.result_csv)["run_id"]
    assert audit_result(store.root)["valid"]
    with ResultWriter(store.root) as writer:
        assert len(writer.rows) == 1
    assert artifact_hashes(store) == before


def test_manifest_structure_is_checked_before_schema_hash(store):
    manifest = change_manifest(store, ("layers", "summary"), [])
    manifest["schema_sha256"] = "broken"
    with pytest.raises(ResultValidationError, match=r"layers\.summary"):
        validate_manifest(store, manifest)


def test_manifest_field_types_are_checked_before_state_values(store):
    manifest = change_manifest(store, ("wide_fields",), [None])
    manifest["state"] = "unknown"
    with pytest.raises(ResultValidationError, match="wide_fields"):
        validate_manifest(store, manifest)
