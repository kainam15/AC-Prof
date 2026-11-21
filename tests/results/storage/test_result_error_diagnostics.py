"""Writer and recovery must agree with readers about diagnostics on error outcomes."""
import json
from uuid import uuid4

import pytest

from acprof.artifacts import atomic_write_json
from acprof.result_store.protocol import csv_bytes, reference
from acprof.result_store.recovery import projected_rows
from acprof.result_store.writer import ResultWriter
from acprof.results.csv import KEY_FIELDS, ResultValidationError, measurement_key, read_result_csv
from acprof.results.windows import content_hash
from tests.results.storage.test_incremental_results import initialized, receipt
from tests.results.storage.test_sequential_result_reads import artifact_hashes


@pytest.mark.parametrize("new", [False, True], ids=["update", "insert"])
@pytest.mark.parametrize("invalid", [{}, {"error": None}, {"error": " \t"}], ids=["absent", "null", "blank"])
def test_error_in_mixed_batch_leaves_valid_head_and_writer(tmp_path, new, invalid):
    layout = initialized(tmp_path)
    with ResultWriter(tmp_path) as writer:
        writer.append_window(receipt(tmp_path))
        if not new:
            writer.append_window(receipt(tmp_path, 1))
        original = read_result_csv(layout.result_csv)[1]
        before = artifact_hashes(layout)
        first = {name: original[0][name] for name in KEY_FIELDS}
        second = {**first, "repeat_idx": "1", "status": " ERROR ", **invalid}
        with pytest.raises(ResultValidationError, match="without an error diagnostic"):
            writer.update_rows([{**first, "latency_s": "7"}, second], reason="mixed")
        assert artifact_hashes(layout) == before
        assert list(writer.rows.values()) == original
        assert writer.update_rows([{**first, "latency_s": "7"}, {**second, "error": "timed out"}], reason="retry")
        assert writer.update_rows([{**second, "error": "retry timeout"}], reason="additional detail")
    rows = read_result_csv(layout.result_csv)[1]
    assert [(row["latency_s"], row["error"]) for row in rows] == [("7", ""), ("nan", "retry timeout")]


def test_sparse_update_uses_recorded_diagnostic(tmp_path):
    layout = initialized(tmp_path)
    with ResultWriter(tmp_path) as writer:
        writer.append_window(receipt(tmp_path))
        row = next(iter(writer.rows.values()))
        key = {name: row[name] for name in KEY_FIELDS}
        writer.update_rows([{**key, "error": "first failure"}], reason="add detail")
        writer.update_rows([{**key, "status": " ERROR "}], reason="set status")
        with pytest.raises(ResultValidationError, match="without an error diagnostic"):
            writer.update_rows([{**key, "error": ""}], reason="invalid removal")
    row = read_result_csv(layout.result_csv)[1][0]
    assert (row["status"], row["error"]) == (" ERROR ", "first failure")


def historical_commit(writer, row, source):
    """Install a correctly hashed single-row journal and projections, bypassing today's producer checks."""
    manifest = writer.manifest
    layers = {name: projected_rows(manifest, name, [row]) for name in manifest["layers"]}
    payloads = {name: csv_bytes(manifest["layers"][name]["fields"], rows, header=True) for name, rows in layers.items()}
    writer._commit(payloads, {name: len(rows) for name, rows in layers.items()},
                   [source], replace=True, row_count=1, state="processing")
    writer.rows[measurement_key(row)] = row


@pytest.mark.parametrize("corrected", [False, True])
def test_recovery_checks_outcome_of_latest_commit_before_repair(tmp_path, corrected):
    layout = initialized(tmp_path)
    with ResultWriter(tmp_path) as writer:
        writer.append_window(receipt(tmp_path))
        key, row = next(iter(writer.rows.items()))
        delta = {"status": "error", "error": ""}
        path = layout.result_update_path(uuid4().hex)
        atomic_write_json(path, {"schema_version": 1, "run_id": writer.manifest["run_id"], "reason": "legacy outcome",
                                 "changes": [{"key": list(key), "values": delta}]})
        historical_commit(writer, {**row, **delta}, {"kind": "derivation", "reference": reference(layout, path)})
        if corrected:
            row = next(iter(writer.rows.values()))
            writer.update_rows([{**row, "error": "later diagnosis"}], reason="corrected")
    summary = layout.layer_csv("summary")
    summary.unlink()  # Recreate a projection only when the final data are fit to publish.
    if corrected:
        with ResultWriter(tmp_path) as writer:
            assert next(iter(writer.rows.values()))["error"] == "later diagnosis"
        assert read_result_csv(layout.result_csv)[1][0]["error"] == "later diagnosis"
    else:
        before = artifact_hashes(layout)
        with pytest.raises(ResultValidationError, match="without an error diagnostic"), ResultWriter(tmp_path):
            pytest.fail("opened invalid committed state")
        assert artifact_hashes(layout) == before


@pytest.mark.parametrize("status", ["error", " ERROR "], ids=["always-invalid", "historically-accepted"])
def test_window_admission_and_replay_of_a_later_correction(tmp_path, status):
    layout = initialized(tmp_path)
    path = receipt(tmp_path)
    record = json.loads(path.read_bytes())
    record["row"].update(status=status, error=" ")
    record["row_sha256"] = content_hash(record["row"])
    del record["content_sha256"]
    record["content_sha256"] = content_hash(record)
    atomic_write_json(path, record)

    with ResultWriter(tmp_path) as writer:
        before = artifact_hashes(layout)
        with pytest.raises(ResultValidationError, match="without an error diagnostic"):
            writer.append_window(path)
        assert artifact_hashes(layout) == before
        row = record["row"]
        historical_commit(writer, row, {"kind": "window", "reference": reference(layout, path)})
        writer.update_rows([{**row, "error": "later diagnosis"}], reason="documented correction")
    if status == "error":
        with pytest.raises(ResultValidationError, match="without an error diagnostic"), ResultWriter(tmp_path):
            pytest.fail("reopened a receipt that the original window protocol already forbade")
    else:
        with ResultWriter(tmp_path) as writer:
            assert next(iter(writer.rows.values()))["error"] == "later diagnosis"
            assert not writer.append_window(path)
