"""Typed result boundaries retain evidence and validate before publication."""
import copy
import json
import subprocess
import sys

import pytest

from acprof.artifacts import atomic_write_json
from acprof.config import CSV_FIELDS
from acprof.result_store.initialization import _validate_initialization_receipt
from acprof.result_store.protocol import (
    publication_lock,
    read_pending,
    save_commit,
    validate_pending_transaction,
)
from acprof.result_store.reader import read_snapshot
from acprof.result_store.writer import ResultWriter
from acprof.results.csv import ResultValidationError
from tests.results.storage.test_committed_digest import make_store
from tests.results.storage.test_incremental_results import receipt
from tests.results.storage.test_sequential_result_reads import artifact_hashes
from tests.support.result_store_fixtures import assert_writer_lease_available, leave_pending_append


@pytest.mark.parametrize("state", [[], True, "unsupported-state"])
def test_writer_validates_pending_before_publishing_any_bytes(tmp_path, state):
    layout = make_store(tmp_path)
    before = artifact_hashes(layout)
    with ResultWriter(tmp_path) as writer:
        with pytest.raises(ResultValidationError, match="state"):
            writer.update_rows([], reason="invalid state fixture", state=state)
    assert artifact_hashes(layout) == before
    assert_writer_lease_available(layout)
    with publication_lock(layout, exclusive=True, blocking=False):
        pass
    with ResultWriter(tmp_path) as writer:
        writer.update_rows([], reason="valid state control", state="collecting")
    assert read_snapshot(tmp_path).rows[0]["status"] == "error"


def test_optimized_python_keeps_pending_runtime_validation(tmp_path):
    layout = make_store(tmp_path)
    before = artifact_hashes(layout)
    program = """
import sys
from acprof.result_store.writer import ResultWriter
from acprof.results.csv import ResultValidationError
try:
    with ResultWriter(sys.argv[1]) as writer:
        writer.update_rows([], reason="optimized invalid state", state=[])
except ResultValidationError as error:
    if "state" not in str(error):
        raise
else:
    raise SystemExit("invalid pending state was published under python -O")
"""
    result = subprocess.run([sys.executable, "-O", "-c", program, str(tmp_path)],
                            check=False, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert artifact_hashes(layout) == before
    assert_writer_lease_available(layout)
    with publication_lock(layout, exclusive=True, blocking=False):
        pass


def test_typed_commit_retains_manifest_and_layer_extension_fields(tmp_path):
    layout = make_store(tmp_path)
    before = json.loads(layout.result_csv.read_bytes())
    after = copy.deepcopy({key: value for key, value in before.items() if key != "commit"})
    after["sequence"] += 1
    after["extension"] = {"source": "original evidence", "unknown": None}
    after["layers"]["summary"]["extension"] = {"unit": "unchanged"}
    manifest = save_commit(layout, before, after, [], transaction="extension-fixture")
    atomic_write_json(layout.result_csv, manifest)
    with ResultWriter(tmp_path) as writer:
        writer.update_rows([], reason="extension preservation", state="complete")
    committed = json.loads(layout.result_csv.read_bytes())
    journal = json.loads(layout.contained(committed["commit"]["path"]).read_bytes())
    for payload in (committed, journal["after"]):
        assert payload["extension"] == after["extension"]
        assert payload["layers"]["summary"]["extension"] == after["layers"]["summary"]["extension"]
    assert read_snapshot(tmp_path).rows[0]["status"] == "error"


def test_pending_narrowing_retains_unknown_evidence_keys(tmp_path, monkeypatch):
    layout = make_store(tmp_path)
    leave_pending_append(layout, receipt(tmp_path, 1), monkeypatch)
    pending = layout.result_store_path("pending.json")
    transaction = json.loads(pending.read_bytes())
    transaction["extension"] = {"raw": ["unknown", None, "NaN"]}
    pending.write_text(json.dumps(transaction))
    manifest = json.loads(layout.result_csv.read_bytes())
    before = artifact_hashes(layout)
    with publication_lock(layout, exclusive=False):
        assert read_pending(layout, manifest) == transaction
    assert artifact_hashes(layout) == before


@pytest.mark.parametrize("path", [(), ("before",), ("after",),
                                  ("before", "layers", "summary"), ("after", "layers", "summary")])
def test_pending_type_guard_rejects_non_string_field_names(tmp_path, monkeypatch, path):
    layout = make_store(tmp_path)
    leave_pending_append(layout, receipt(tmp_path, 1), monkeypatch)
    transaction = json.loads(layout.result_store_path("pending.json").read_bytes())
    target = transaction
    for key in path:
        target = target[key]
    target[1] = "not a protocol field name"
    manifest = json.loads(layout.result_csv.read_bytes())
    before = artifact_hashes(layout)
    with pytest.raises(ResultValidationError, match="field names must be strings"):
        validate_pending_transaction(transaction, layout, manifest)
    assert artifact_hashes(layout) == before


def test_receipt_type_guard_rejects_non_string_field_names():
    value = {"run_id": "valid-run", "fields": list(CSV_FIELDS), 1: "not a protocol field name"}
    with pytest.raises(ResultValidationError, match="field names must be strings"):
        _validate_initialization_receipt(value)
