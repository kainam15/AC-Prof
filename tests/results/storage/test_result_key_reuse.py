"""Reuse keys within one read, preserving identity checks at every boundary."""
import csv
import hashlib
import io
import json
from decimal import localcontext

import pytest
from tests.results.storage.test_result_row_assembly import row_store

from acprof.results import csv as result_csv
from acprof.analysis.audit import audit_result
from acprof.results.layout import ArtifactLayout
from acprof.results.layers import BASE_LAYERS, LAYER_FILES
from acprof.result_store import reader
from acprof.result_store.protocol import csv_bytes, save_commit


def rewrite_layer(root, manifest, layer, transform):
    """Publish explicitly synthetic corruption with matching content hashes."""
    info = manifest["layers"][layer]
    path = root / info["path"]
    rows = list(csv.DictReader(io.StringIO(path.read_text(), newline="")))
    transform(rows)
    raw = csv_bytes(info["fields"], rows, header=True)
    path.write_bytes(raw)
    info.update(sha256=hashlib.sha256(raw).hexdigest(), row_count=len(rows), committed_bytes=len(raw))
    before = manifest.copy()
    after = {key: value for key, value in manifest.items() if key != "commit"}
    after["sequence"] += 1
    committed = save_commit(ArtifactLayout(root), before, after, [], transaction="changed-fixture")
    (root / "result_layers.json").write_text(json.dumps(committed))
    return committed


@pytest.mark.parametrize("count", [2, 7])
def test_each_committed_row_is_keyed_once_before_joining(tmp_path, monkeypatch, count):
    manifest, expected = row_store(tmp_path, 3, count=count, width=2)
    calls = []
    original = result_csv.measurement_key

    def record(row):
        calls.append(row.copy())
        return original(row)

    monkeypatch.setattr(result_csv, "measurement_key", record)
    monkeypatch.setattr(reader, "measurement_key", record, raising=False)
    snapshot = reader.read_snapshot(tmp_path)
    assert snapshot.rows == expected
    assert len(calls) == sum(layer["row_count"] for layer in manifest["layers"].values())


@pytest.mark.parametrize("layer", ["summary", "resources", "ncu"])
def test_duplicate_keys_still_report_the_layer_and_original_row(tmp_path, layer):
    manifest, _ = row_store(tmp_path, 3, count=2, width=1)

    def duplicate(rows):
        rows[:] = [rows[0], {**rows[0], "cpu_cores": "2.00", "repeat_idx": "0.0"}]

    rewrite_layer(tmp_path, manifest, layer, duplicate)
    with pytest.raises(result_csv.ResultValidationError) as error:
        reader.read_snapshot(tmp_path)
    message = str(error.value)
    assert "duplicate measurement" in message
    assert f"{LAYER_FILES[layer]}:3:" in message
    assert "('2', '8', 'off', '1', '0', '0')" in message


@pytest.mark.parametrize("layer", BASE_LAYERS[1:])
def test_key_reuse_keeps_base_layer_order_checks(tmp_path, layer):
    manifest, _ = row_store(tmp_path, 3, count=2, width=1)
    rewrite_layer(tmp_path, manifest, layer, lambda rows: rows.reverse())
    with pytest.raises(result_csv.ResultValidationError, match=f"measurement order mismatch in {layer}"):
        reader.read_snapshot(tmp_path)


def test_key_reuse_keeps_orphan_profiler_checks(tmp_path):
    manifest, _ = row_store(tmp_path, 3, count=2, width=1)
    rewrite_layer(tmp_path, manifest, "ncu", lambda rows: rows[0].update(repeat_idx="2"))
    with pytest.raises(result_csv.ResultValidationError, match="orphan profiler row in ncu"):
        reader.read_snapshot(tmp_path)


def test_key_reuse_keeps_raw_identity_comparison(tmp_path):
    manifest, _ = row_store(tmp_path, 3, count=2, width=1)
    rewrite_layer(tmp_path, manifest, "resources", lambda rows: rows[0].update(cpu_cores="2.00"))
    with pytest.raises(result_csv.ResultValidationError, match="measurement identity mismatch in resources"):
        reader.read_snapshot(tmp_path)


@pytest.mark.parametrize("damage", ["duplicate", "invalid-key"])
def test_supplied_snapshot_rows_are_revalidated_after_mutation(tmp_path, damage):
    row_store(tmp_path, 3, count=2, width=1)
    snapshot = reader.read_snapshot(tmp_path)
    if damage == "duplicate":
        snapshot.rows[1]["repeat_idx"] = "0"
    else:
        snapshot.rows[1]["gpu_mode"] = "invalid"
    report = audit_result(snapshot.path, result_snapshot=snapshot, verify_result_snapshot=False)
    errors = [issue for issue in report["issues"] if issue["code"] == "invalid_csv"]
    assert not report["valid"]
    assert len(errors) == 1
    assert ("duplicate measurement" if damage == "duplicate" else "invalid gpu_mode") in errors[0]["message"]


def test_separate_reads_respect_their_decimal_context(tmp_path, monkeypatch):
    row_store(tmp_path, 3, count=123, width=0)
    calls = []
    original = result_csv.measurement_key

    def record(row):
        key = original(row)
        if row["repeat_idx"] == "122":
            calls.append(key[-1])
        return key

    monkeypatch.setattr(result_csv, "measurement_key", record)
    # A narrower precision makes repeat_idx=100 and 101 collide. That must
    # fail even if this exact file was read under a wider context before.
    with localcontext() as context:
        context.prec = 6
        assert len(reader.read_snapshot(tmp_path).rows) == 123
    assert calls
    assert set(calls) == {"122"}
    with localcontext() as context:
        context.prec = 2
        with pytest.raises(result_csv.ResultValidationError, match="duplicate measurement"):
            reader.read_snapshot(tmp_path)


def test_structural_only_validation_cannot_supply_measurement_keys(tmp_path):
    source = [{"historical": " keep original "}]
    validated = result_csv.validate_result_rows(["historical"], source, tmp_path / "legacy.csv",
                                                 validate_measurements=False)
    assert validated.rows == source
    assert validated.keys is None
    with pytest.raises(ValueError, match="expected measurements require measurement validation"):
        result_csv.validate_result_rows(["historical"], source, tmp_path / "legacy.csv",
                                        validate_measurements=False, expected=[])


def test_validation_keys_preserve_row_order_and_exact_plan_checks(tmp_path):
    fields = [*result_csv.KEY_FIELDS, "environment_class"]
    first = dict(zip(fields, ("2.00", "8", "off", "1e0", "0", "1", "unrecognized"), strict=True))
    second = {**first, "repeat_idx": "0"}
    expected = (("2", "8", "off", "1", "0", "1"), ("2", "8", "off", "1", "0", "0"))
    validated = result_csv.validate_result_rows(fields, [first, second], tmp_path / "source.csv", expected=expected)
    assert validated.keys == expected
    assert [row["repeat_idx"] for row in validated.rows] == ["1", "0"]
    assert all(row["environment_class"] == "unknown" for row in validated.rows)
    assert first["environment_class"] == "unrecognized"
    with pytest.raises(result_csv.ResultValidationError, match="missing=0, unexpected=1"):
        result_csv.validate_result_rows(fields, [first, second], tmp_path / "source.csv", expected=expected[:1])


def test_empty_validated_rows_have_an_empty_key_sequence(tmp_path):
    validated = result_csv.validate_result_rows(result_csv.KEY_FIELDS, [], tmp_path / "empty.csv", allow_empty=True)
    assert validated.rows == []
    assert validated.keys == ()
