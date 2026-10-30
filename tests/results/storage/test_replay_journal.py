"""Replay verified journals without retaining repeated manifest snapshots."""
from __future__ import annotations

import hashlib
import json
import tracemalloc
from pathlib import Path

import pytest
from tests.results.storage.test_incremental_results import receipt
from tests.results.storage.test_projection_recovery import committed_store

from acprof.results.layers import read_result_layers
from acprof.result_store.writer import ResultWriter


def state_history(root, checkpoints=128):
    layout = committed_store(root)
    with ResultWriter(root) as writer:
        for index in range(checkpoints):
            writer.update_rows([], reason="fixture state checkpoint",
                               state="processing" if index % 2 == 0 else "collecting")
    return layout


def test_replay_memory_does_not_retain_historical_manifest_copies(tmp_path):
    layout = state_history(tmp_path)
    commits = list(layout.result_store_path("commits").glob("*.json"))
    assert len(commits) == 130
    manifest = layout.result_csv.read_bytes()
    projections = {path: path.read_bytes() for path in tmp_path.rglob("*.csv")}
    tracemalloc.start()
    try:
        with ResultWriter(tmp_path) as writer:
            assert len(writer.rows) == 1
            assert writer.manifest["sequence"] == 129
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 2 * 1024 * 1024, f"replaying 130 commits allocated {peak:,} bytes for one result row"
    assert layout.result_csv.read_bytes() == manifest
    assert all(path.read_bytes() == raw for path, raw in projections.items())


def rewrite_history(layout, index, mutate):
    """Rehash a synthetic journal so semantic failures are not hidden by bad hashes."""
    previous = None
    for position, path in enumerate(sorted(layout.result_store_path("commits").glob("*.json"))):
        record = json.loads(path.read_bytes())
        record["previous"] = previous
        if position == index:
            mutate(record)
        raw = json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
        path.write_bytes(raw)
        previous = {"path": path.relative_to(layout.root).as_posix(), "sha256": hashlib.sha256(raw).hexdigest()}
    manifest = json.loads(layout.result_csv.read_bytes())
    manifest["commit"] = previous
    layout.result_csv.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


@pytest.mark.parametrize("commit_index", [0, 2], ids=["initial-commit", "historical-checkpoint"])
def test_corrupt_historical_commit_is_rejected_before_source_reads(tmp_path, monkeypatch, commit_index):
    layout = state_history(tmp_path, checkpoints=3)
    commits: list[Path] = sorted(layout.result_store_path("commits").glob("*.json"))
    commits[commit_index].write_bytes(b"{}")
    source_opens = []
    raw_root = layout.path("raw")
    before = layout.result_csv.read_bytes()

    def open_file(path: Path, mode="r", *args, **kwargs):
        if path.is_relative_to(raw_root):
            source_opens.append(path)
        return open(path, mode, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", open_file)
        with pytest.raises(ValueError, match="hash does not match expected SHA256"):
            with ResultWriter(tmp_path):
                pytest.fail("corrupt historical authority was accepted")
    assert source_opens == []
    assert layout.result_csv.read_bytes() == before


@pytest.mark.parametrize(("field", "value", "error"), [
    ("schema_version", 99, "broken result commit chain"),
    ("sequence", 99, "broken result commit chain"),
    ("run_id", "another-run", "broken result commit chain"),
    ("schema_sha256", "f" * 64, "broken result commit chain"),
    ("row_count", 2, "row count disagrees"),
    ("previous", None, "incomplete result commit chain"),
])
def test_historical_checks_remain_required_with_valid_reference_hashes(tmp_path, field, value, error):
    layout = state_history(tmp_path, checkpoints=3)

    def mutate(record):
        if field in {"schema_version", "previous"}:
            record[field] = value
        else:
            record["after"][field] = value

    rewrite_history(layout, 2, mutate)
    target = layout.layer_csv("energy")
    with target.open("ab") as stream:
        stream.write(b"uncommitted tail")
    before = target.read_bytes()
    manifest = layout.result_csv.read_bytes()
    recovering = ResultWriter(tmp_path)
    with pytest.raises(ValueError, match=error):
        with recovering:
            pytest.fail("invalid historical transaction was accepted")
    assert recovering.lease is None
    assert target.read_bytes() == before
    assert layout.result_csv.read_bytes() == manifest


@pytest.mark.parametrize("field", ["sources", "row_count"])
def test_missing_historical_replay_inputs_are_not_defaulted(tmp_path, field):
    layout = state_history(tmp_path, checkpoints=3)

    def mutate(record):
        (record if field == "sources" else record["after"]).pop(field)

    rewrite_history(layout, 2, mutate)
    with pytest.raises(KeyError, match=field):
        with ResultWriter(tmp_path):
            pytest.fail("missing historical replay input was accepted")


def test_replay_preserves_window_derivation_and_checkpoint_order(tmp_path):
    committed_store(tmp_path)
    second, third = receipt(tmp_path, 1), receipt(tmp_path, 2)
    with ResultWriter(tmp_path) as writer:
        first = dict(next(iter(writer.rows.values())), latency_s="0.25",
                     compute_profile_error_ncu='诊断,"quoted"\n')
        assert writer.update_rows([first], reason="fixture first derivation")
        assert writer.append_window(second)
        updated = [dict(row, latency_s="0.75" if row["repeat_idx"] == "0" else "1.25")
                   for row in writer.rows.values()]
        assert writer.update_rows(updated, reason="fixture later derivation")
        writer.update_rows([], reason="fixture checkpoint", state="collecting")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.csv")}
    for path in before:
        path.write_bytes(b"damaged projection")
    with ResultWriter(tmp_path) as recovered:
        assert all(path.read_bytes() == raw for path, raw in before.items())
        rows = list(recovered.rows.values())
        assert [row["repeat_idx"] for row in rows] == ["0", "1"]
        assert [row["latency_s"] for row in rows] == ["0.75", "1.25"]
        assert rows[0]["compute_profile_error_ncu"] == '诊断,"quoted"\n'
        assert not recovered.append_window(second)
        assert recovered.append_window(third)
    _, rows = read_result_layers(tmp_path)
    assert [row["repeat_idx"] for row in rows] == ["0", "1", "2"]
    assert [row["latency_s"] for row in rows] == ["0.75", "1.25", "nan"]
