"""V3 commits bind derived and projected rows to the manifest's run."""
import pytest

from acprof.artifacts import atomic_write_json
from acprof.result_store import initialization
from acprof.result_store.protocol import csv_bytes, reference
from acprof.result_store.reader import read_snapshot
from acprof.result_store.recovery import projected_rows
from acprof.result_store.sources import EXACT_KEY_ENCODING
from acprof.result_store.writer import ResultWriter
from acprof.results.csv import ResultValidationError, measurement_key, read_result_csv_snapshot
from acprof.results.layout import ArtifactLayout
from tests.results.storage.test_result_row_assembly import row_store


def row(index=0, **values):
    return dict(cpu_cores="1", mem_cap_gb="4", gpu_mode="off", input_scale="64",
                warmup="0", repeat_idx=str(index), status="ok", error="", **values)


@pytest.fixture(params=("run_id", " run_id "), ids=("registered", "historical-spaces"))
def column(request):
    return request.param


@pytest.fixture
def store(tmp_path, column, monkeypatch):
    if column != "run_id":
        # Historical committed v3 projections retain their original wide fields.
        monkeypatch.setattr(initialization, "CSV_FIELDS", [
            column if field == "run_id" else field for field in initialization.CSV_FIELDS
        ])
    layout = ArtifactLayout.for_new_run(tmp_path)
    layout.initialize()
    return layout


def files(layout):
    return {p.relative_to(layout.root): p.read_bytes() for p in layout.root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("existing", [False, True], ids=["insert", "update"])
def test_foreign_derived_row_leaves_head_projections_and_journal_unchanged(store, column, existing):
    with ResultWriter(store.root) as writer:
        identity = writer.manifest["run_id"]
        if existing:
            writer.update_rows([row(**{column: identity})], reason="initial result")
        before = files(store)
        with pytest.raises(ResultValidationError, match="inconsistent run_id"):
            writer.update_rows([row(1, **{column: identity}), row(**{column: "another-run"})], reason="mixed inputs")
        assert files(store) == before
        assert writer.update_rows([row(1, **{column: identity})], reason="retry")


def write_historical_commit(writer, column, *, change=False):
    # Emit an internally hashed old transaction without going through today's input checks.
    manifest = writer.manifest
    source_row = row(**{column: "another-run"})
    wide = {name: source_row.get(name, "nan") for name in manifest["wide_fields"]}
    path = writer.layout.result_update_path("a" * 32)
    atomic_write_json(path, {"schema_version": 1, "run_id": manifest["run_id"], "reason": "historical derivation",
                             "measurement_key_encoding": EXACT_KEY_ENCODING, "insert": [] if change else [wide],
                             "changes": [{"key": list(measurement_key(wide)),
                                          "values": {column: "another-run"}}] if change else [], "evidence": []})
    layers = {name: projected_rows(manifest, name, [wide]) for name in manifest["layers"]}
    payloads = {name: csv_bytes(manifest["layers"][name]["fields"], rows, header=True)
                for name, rows in layers.items()}
    source = {"kind": "derivation", "reference": reference(writer.layout, path)}
    writer._commit(payloads, {name: len(rows) for name, rows in layers.items()},
                   [source], replace=True, row_count=1, state="processing")


@pytest.mark.parametrize("entry", ["read", "replay"])
@pytest.mark.parametrize("change", [False, True], ids=("insert", "change"))
def test_foreign_historical_derivation_is_rejected_during_read_and_replay(store, column, entry, change):
    with ResultWriter(store.root) as writer:
        if change:
            writer.update_rows([row(**{column: writer.manifest["run_id"]})], reason="initial result")
        write_historical_commit(writer, column, change=change)
    before = files(store)
    if entry == "read":
        with pytest.raises(ResultValidationError, match="inconsistent run_id"):
            read_snapshot(store.root)
    else:
        with (
            pytest.raises(ResultValidationError, match="inconsistent run_id|cannot change measurement identity"),
            ResultWriter(store.root),
        ):
            pytest.fail("writer accepted a foreign derivation")
    assert files(store) == before


def test_snapshot_binds_manifest_identity_for_rows_without_recorded_run_id(store, column):
    with ResultWriter(store.root) as writer:
        identity = writer.manifest["run_id"]
        writer.update_rows([row(), row(1, **{column: "null"}), row(2, **{column: "unknown"})], reason="historical rows")
    with ResultWriter(store.root) as replay:
        assert replay.manifest["run_id"] == identity
    snapshot = read_result_csv_snapshot(store.result_csv)
    assert [item[column] for item in snapshot.rows] == ["nan", "null", "unknown"]
    assert snapshot.run_id == identity


def test_snapshot_binds_manifest_identity_without_a_run_id_column(tmp_path):
    manifest, _ = row_store(tmp_path, 3, count=1, width=0)
    snapshot = read_result_csv_snapshot(tmp_path / "result_layers.json")
    assert "run_id" not in snapshot.fields
    assert snapshot.run_id == manifest["run_id"]
