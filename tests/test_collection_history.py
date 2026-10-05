import json
import tempfile
from pathlib import Path

import pytest

from acprof.host.collection_history import (
    COLLECTION_HISTORY_FIELDS,
    COLLECTION_HISTORY_SCHEMA_VERSION,
    append_collection_record,
    empty_collection_history,
    normalize_collection_history,
    write_collection_history_json,
)


def test_empty_document_has_stable_schema() -> None:
    payload = empty_collection_history()

    assert (payload["schema_version"]) == (COLLECTION_HISTORY_SCHEMA_VERSION)
    for field in COLLECTION_HISTORY_FIELDS:
        assert (payload[field]) == ([])


def test_append_validates_field_and_keeps_native_json_types() -> None:
    record = {"retry_rows": 21, "restored": True, "note": None}

    updated = append_collection_record(
        empty_collection_history(),
        "quality_retry_history",
        record,
    )

    assert (updated["quality_retry_history"]) == ([record])
    with pytest.raises(ValueError):
        append_collection_record(updated, "unknown_history", record)

def test_writer_rejects_nonfinite_provenance_numbers() -> None:
    payload = empty_collection_history()
    payload["quality_retry_history"].append({"retry_rows": float("nan")})

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "collection_history.json"
        with pytest.raises(ValueError, match="JSON compliant"):
            write_collection_history_json(payload, path)
        assert not path.exists()


def test_atomic_writer_emits_valid_json_without_temporary_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "collection_history.json"
        write_collection_history_json(empty_collection_history(), path)

        payload = json.loads(path.read_text(encoding="utf-8"))
        leftovers = list(path.parent.glob(".collection_history.json.*.tmp"))

    assert (payload) == (normalize_collection_history(payload))
    assert (leftovers) == ([])
