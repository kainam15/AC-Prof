"""Initialization never publishes a store with an unreadable receipt identity."""
import json

import pytest

from acprof.config import CSV_FIELDS
from acprof.result_store import initialization
from acprof.results.csv import ResultValidationError
from acprof.results.layers import read_result_layers
from acprof.results.layout import ArtifactLayout
from tests.results.storage.test_sequential_result_reads import artifact_hashes

_MISSING = object()


def initialize(layout, entry):
    if entry == "inspect":
        return initialization.initial_files(layout)
    if entry == "store":
        return initialization.initialize_results(layout)
    return layout.initialize()


@pytest.mark.parametrize("entry", ["inspect", "store", "layout"])
@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize("identity", [
    pytest.param(True, id="bool"),
    pytest.param(7, id="int"),
    pytest.param(1.5, id="float"),
    pytest.param(["run"], id="list"),
    pytest.param({"id": "run"}, id="dict"),
    pytest.param(None, id="null"),
    pytest.param("", id="empty"),
    pytest.param(_MISSING, id="missing"),
])
def test_invalid_receipt_identity_is_rejected_without_new_files(tmp_path, entry, published, identity):
    layout = ArtifactLayout.for_new_run(tmp_path)
    if published:
        layout.initialize()
    intent = layout.result_store_path("initialization.json")
    intent.parent.mkdir(parents=True, exist_ok=True)
    schema = {"fields": list(CSV_FIELDS)}
    if identity is not _MISSING:
        schema["run_id"] = identity
    intent.write_text(json.dumps(schema))
    before = artifact_hashes(layout)
    with pytest.raises(ResultValidationError, match="initialization receipt.*run_id"):
        initialize(layout, entry)
    assert artifact_hashes(layout) == before
    if not published:
        assert not layout.result_csv.exists()
        assert not layout.path("result_manifest.json").exists()
        assert not list(layout.root.rglob("*.csv"))
        assert not layout.result_store_path("commits").exists()


@pytest.mark.parametrize("entry", ["inspect", "store", "layout"])
def test_existing_store_does_not_silently_ignore_changed_receipt_identity(tmp_path, entry):
    layout = ArtifactLayout.for_new_run(tmp_path)
    layout.initialize()
    intent = layout.result_store_path("initialization.json")
    schema = json.loads(intent.read_bytes())
    schema["run_id"] = "another-run"
    intent.write_text(json.dumps(schema))
    before = artifact_hashes(layout)
    with pytest.raises(ResultValidationError, match="another run"):
        initialize(layout, entry)
    assert artifact_hashes(layout) == before


def test_interrupted_initialization_reuses_valid_receipt_and_zero_row_projections(tmp_path, monkeypatch):
    layout = ArtifactLayout.for_new_run(tmp_path)
    write = initialization.write_bytes
    synced = []

    def fail_after_first_header(path, raw):
        write(path, raw)
        synced.append(path)
        raise OSError("fixture initialization interrupted after header sync")

    with monkeypatch.context() as fault:
        fault.setattr(initialization, "write_bytes", fail_after_first_header)
        with pytest.raises(OSError, match="after header sync"):
            layout.initialize()
    assert len(synced) == 1
    intent = layout.result_store_path("initialization.json")
    original = intent.read_bytes()
    identity = json.loads(original)["run_id"]
    layout.initialize()
    assert intent.read_bytes() == original
    manifest = json.loads(layout.result_csv.read_bytes())
    assert manifest["run_id"] == identity
    assert manifest["sequence"] == manifest["row_count"] == 0
    fields, rows = read_result_layers(tmp_path)
    assert fields == list(CSV_FIELDS)
    assert rows == []
    before = artifact_hashes(layout)
    layout.initialize()
    assert artifact_hashes(layout) == before
