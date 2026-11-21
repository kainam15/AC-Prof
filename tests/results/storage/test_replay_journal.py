"""Replay verified journals without retaining repeated manifest snapshots."""
from __future__ import annotations

import hashlib
import json
import tracemalloc
from decimal import localcontext
from pathlib import Path

import pytest
from tests.support.result_store_fixtures import assert_writer_lease_available
from tests.results.storage.test_incremental_results import initialized, receipt
from tests.results.storage.test_projection_recovery import committed_store
from tests.results.storage.test_sequential_result_reads import artifact_hashes

from acprof.result_store import writer as writer_module
from acprof.result_store.protocol import publication_lock

from acprof.results.csv import ResultValidationError
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
    assert_writer_lease_available(recovering.layout)
    assert target.read_bytes() == before
    assert layout.result_csv.read_bytes() == manifest


@pytest.mark.parametrize("field", ["sources", "row_count", "previous"])
def test_missing_historical_replay_inputs_are_not_defaulted(tmp_path, field):
    layout = state_history(tmp_path, checkpoints=3)

    def mutate(record):
        (record if field in {"sources", "previous"} else record["after"]).pop(field)

    rewrite_history(layout, 2, mutate)
    before = artifact_hashes(layout)
    with pytest.raises(ResultValidationError, match=field):
        with ResultWriter(tmp_path):
            pytest.fail("missing historical replay input was accepted")
    assert artifact_hashes(layout) == before
    assert_writer_lease_available(layout)
    with publication_lock(layout, exclusive=True, blocking=False):
        pass


@pytest.mark.parametrize(("field", "value", "error"), [
    ("previous", {}, "reference"),
    ("previous", [], "reference"),
    ("sources", {}, "sources"),
    ("sources", [None], "source"),
    ("sources", [{}], "source"),
    ("sources", [{"kind": ["window"]}], "source"),
    ("sources", [{"kind": "window"}], "reference"),
    ("sources", [{"kind": "window", "reference": {"path": 1, "sha256": "a" * 64}}], "reference"),
])
def test_historical_replay_structures_fail_before_projection_writes(tmp_path, field, value, error):
    layout = state_history(tmp_path, checkpoints=3)
    rewrite_history(layout, 2, lambda record: record.update({field: value}))
    with layout.layer_csv("energy").open("ab") as stream:
        stream.write(b"uncommitted tail")
    before = artifact_hashes(layout)
    with pytest.raises(ResultValidationError, match=error):
        with ResultWriter(tmp_path):
            pytest.fail("invalid historical replay structure was accepted")
    assert artifact_hashes(layout) == before
    assert_writer_lease_available(layout)
    with publication_lock(layout, exclusive=True, blocking=False):
        pass


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


def test_distinct_high_precision_windows_keep_their_own_derivations(tmp_path):
    initialized(tmp_path)
    scales = ("1.00000000000000000000000000001", "1.00000000000000000000000000002")
    rows = [dict(cpu_cores="1", mem_cap_gb="4", gpu_mode="off", input_scale=scale,
                 warmup="-0", repeat_idx="0", status="ok", error="", latency_app_s="3") for scale in scales]
    with ResultWriter(tmp_path) as writer:
        writer.update_rows(rows, reason="fixture observations")
        writer.update_rows([{**rows[1], "latency_app_s": "7"}], reason="fixture derivation")
    with ResultWriter(tmp_path) as writer:
        assert [row["latency_app_s"] for row in writer.rows.values()] == ["3", "7"]
    _, projected = read_result_layers(tmp_path)
    assert [(r["input_scale"], r["warmup"], r["latency_app_s"]) for r in projected] == [
        (scales[0], "-0", "3"), (scales[1], "-0", "7"),
    ]


@pytest.mark.parametrize(("precision", "old_scale"), [(28, "1.234567890123456789012345679"), (3, "1.23")])
def test_historical_rounded_derivation_replays_without_rewriting_authority(tmp_path, monkeypatch, precision, old_scale):
    layout = initialized(tmp_path)
    row = dict(cpu_cores="1", mem_cap_gb="4", gpu_mode="off", input_scale="1.23456789012345678901234567891",
               warmup="-0", repeat_idx="0", status="ok", error="", latency_app_s="3")
    publish = writer_module.atomic_write_json

    def write_historical(path, payload):
        if path.parent == layout.path("raw/result_updates"):
            payload = dict(payload)
            payload.pop("measurement_key_encoding", None)
            if payload["changes"]:
                payload["changes"] = [{"key": ["1", "4", "off", old_scale, "-0", "0"], "values": {"latency_app_s": "7"}}]
        return publish(path, payload)

    with localcontext(prec=precision), monkeypatch.context() as guard:
        guard.setattr(writer_module, "atomic_write_json", write_historical)
        with ResultWriter(tmp_path) as writer:
            writer.update_rows([row], reason="historical insert")
            writer.update_rows([{**row, "latency_app_s": "7"}], reason="historical update")
    history = {p: p.read_bytes() for p in layout.path("raw/result_updates").glob("*.json")}
    assert len(history) == 2
    with localcontext(prec=precision), ResultWriter(tmp_path) as writer:
        assert next(iter(writer.rows)) == ("1", "4", "off", row["input_scale"], "0", "0")
        assert next(iter(writer.rows.values()))["latency_app_s"] == "7"
    assert {p: p.read_bytes() for p in history} == history


@pytest.mark.parametrize(("encoding", "error"), [
    ("future-format", "unknown measurement key encoding"),
    (None, "ambiguous historical measurement key"),
    ("exact-decimal-v1", "invalid derived measurement key"),
])
def test_unresolvable_derivation_cannot_modify_a_published_row(tmp_path, monkeypatch, encoding, error):
    layout = initialized(tmp_path)
    base = dict(cpu_cores="1", mem_cap_gb="4", gpu_mode="off", warmup="0", repeat_idx="0",
                status="ok", error="", latency_app_s="3")
    rows = [dict(base, input_scale=scale) for scale in
            ("1.00000000000000000000000000001", "1.00000000000000000000000000002")]
    publish = writer_module.atomic_write_json

    def rewrite_change(path, payload):
        if path.parent == layout.path("raw/result_updates") and payload.get("changes"):
            payload = dict(payload)
            if encoding is None:
                payload.pop("measurement_key_encoding")
            else:
                payload["measurement_key_encoding"] = encoding
            payload["changes"] = [{"key": ["1", "4", "off", "1", "0", "0"], "values": {"latency_app_s": "9"}}]
        return publish(path, payload)

    with ResultWriter(tmp_path) as writer:
        writer.update_rows(rows, reason="fixture observations")
        manifest = layout.result_csv.read_bytes()
        before = {path: path.read_bytes() for path in tmp_path.rglob("*.csv")}
        monkeypatch.setattr(writer_module, "atomic_write_json", rewrite_change)
        with pytest.raises(ResultValidationError, match=error):
            writer.update_rows([{**rows[1], "latency_app_s": "9"}], reason="fixture update")
    assert layout.result_csv.read_bytes() == manifest
    assert {p: p.read_bytes() for p in before} == before
    with ResultWriter(tmp_path) as reader:
        assert [r["latency_app_s"] for r in reader.rows.values()] == ["3", "3"]
