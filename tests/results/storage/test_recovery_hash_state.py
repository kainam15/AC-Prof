"""Recovery keeps verified hash state without retaining every CSV projection."""
from __future__ import annotations

import hashlib
import json
import tracemalloc
from collections import Counter

import pytest
from tests.results.storage.test_incremental_results import initialized, receipt
from tests.results.storage.test_projection_recovery import committed_store

from acprof.result_store import metadata, recovery, writer as writer_module
from acprof.result_store.writer import ResultWriter


def diagnostic_store(root, *, repetitions=16384):
    layout = committed_store(root)
    detail = '诊断,"quoted"\n' * repetitions
    with ResultWriter(root) as writer:
        row = dict(next(iter(writer.rows.values())))
        for name in ("process_resource_error", "gpu_energy_fallback_reason",
                     "compute_profile_error_torch_profiler_eager", "compute_profile_error_ncu",
                     "compute_profile_error_nsys", "compute_profile_error_massif"):
            row[name] = detail
        row["workload_contract"] = json.dumps({"文字": detail}, ensure_ascii=False)
        writer.update_rows([row], reason="fixture diagnostic text")
    return layout


def test_recovery_releases_csv_buffers_before_metadata_recovery(tmp_path, monkeypatch):
    layout = diagnostic_store(tmp_path)
    manifest = layout.result_csv.read_bytes()
    committed_bytes = sum(info["committed_bytes"] for info in json.loads(manifest)["layers"].values())
    assert committed_bytes > 1024 * 1024
    observed = {}
    replay = recovery.replay
    recover_metadata = metadata.recover_metadata

    def replay_sources(*args):
        result = replay(*args)
        observed["after_replay"] = tracemalloc.get_traced_memory()[0]
        return result

    def recover_after_projections(*args):
        observed["after_projections"] = tracemalloc.get_traced_memory()[0]
        recover_metadata(*args)

    with monkeypatch.context() as guard:
        guard.setattr(recovery, replay.__name__, replay_sources)
        guard.setattr(metadata, recover_metadata.__name__, recover_after_projections)
        tracemalloc.start()
        try:
            with ResultWriter(tmp_path) as writer:
                assert len(writer.rows) == 1
        finally:
            tracemalloc.stop()
    assert set(observed) == {"after_replay", "after_projections"}
    retained = observed["after_projections"] - observed["after_replay"]
    assert retained < 512 * 1024, f"recovery retained {retained:,} bytes after checking {committed_bytes:,} CSV bytes"
    assert layout.result_csv.read_bytes() == manifest


@pytest.mark.parametrize("has_window", [False, True], ids=["headers", "committed-window"])
def test_opening_writer_hashes_each_complete_projection_once(tmp_path, monkeypatch, has_window):
    layout = committed_store(tmp_path) if has_window else initialized(tmp_path)
    manifest = json.loads(layout.result_csv.read_bytes())
    projections = {layer: layout.layer_csv(layer).read_bytes() for layer in manifest["layers"]}
    by_content = {raw: layer for layer, raw in projections.items()}
    assert len(by_content) == len(projections)
    calls = Counter()
    sha256 = hashlib.sha256

    def counted_sha256(raw=b"", *args, **kwargs):
        if isinstance(raw, bytes) and raw in by_content:
            calls[by_content[raw]] += 1
        return sha256(raw, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(hashlib, sha256().name, counted_sha256)
        with ResultWriter(tmp_path) as writer:
            for layer, raw in projections.items():
                expected = sha256(raw).hexdigest()
                assert writer.hashes[layer].hexdigest() == expected
                assert writer.manifest["layers"][layer]["sha256"] == expected
    assert calls == dict.fromkeys(projections, 1)


def hash_digests(hashes):
    return {layer: checksum.hexdigest() for layer, checksum in hashes.items()}


def assert_committed_hashes(writer):
    assert set(writer.hashes) == set(writer.manifest["layers"])
    for layer, info in writer.manifest["layers"].items():
        raw = writer.layout.layer_csv(layer).read_bytes()
        assert len(raw) == info["committed_bytes"]
        assert hashlib.sha256(raw).hexdigest() == info["sha256"]
        assert writer.hashes[layer].hexdigest() == info["sha256"]


@pytest.mark.parametrize("damage", ["none", "tail", "missing", "corrupt"])
def test_recovered_hashes_follow_append_replacement_and_reopen(tmp_path, damage):
    layout = diagnostic_store(tmp_path, repetitions=3)
    target = layout.layer_csv("ncu")
    expected = target.read_bytes()
    if damage == "tail":
        with target.open("ab") as stream:
            stream.write(b"uncommitted")
    elif damage == "missing":
        target.unlink()
    elif damage == "corrupt":
        target.write_bytes(b"modified")
    second, third = receipt(tmp_path, 1), receipt(tmp_path, 2)
    with ResultWriter(tmp_path) as writer:
        assert target.read_bytes() == expected
        assert_committed_hashes(writer)
        previous = dict(writer.hashes)
        previous_digests = hash_digests(previous)
        assert writer.append_window(second)
        assert hash_digests(previous) == previous_digests
        assert_committed_hashes(writer)
        row = dict(next(iter(writer.rows.values())), compute_profile_error_ncu="nan",
                   compute_profile_error_torch_profiler_eager='新诊断,"quoted"\n')
        assert writer.update_rows([row], reason="fixture profiler replacement")
        assert writer.manifest["layers"]["ncu"]["row_count"] == 0
        assert_committed_hashes(writer)
        assert writer.append_window(third)
        assert_committed_hashes(writer)
    with ResultWriter(tmp_path) as reopened:
        assert len(reopened.rows) == 3
        assert reopened.manifest["layers"]["ncu"]["row_count"] == 0
        assert_committed_hashes(reopened)


@pytest.mark.parametrize("operation", ["append", "replace"])
def test_failed_publication_does_not_advance_recovered_hashes(tmp_path, monkeypatch, operation):
    layout = committed_store(tmp_path)
    second = receipt(tmp_path, 1)
    before_manifest = layout.result_csv.read_bytes()

    def publish(active_writer):
        if operation == "append":
            return active_writer.append_window(second)
        row = dict(next(iter(active_writer.rows.values())), latency_s="0.2",
                   compute_profile_error_ncu="fixture profiler error")
        return active_writer.update_rows([row], reason="fixture failed replacement")

    def fail_commit(*_args, **_kwargs):
        raise OSError("fixture commit publication failed")

    with ResultWriter(tmp_path) as writer:
        previous = dict(writer.hashes)
        previous_digests = hash_digests(previous)
        with monkeypatch.context() as guard:
            guard.setattr(writer_module, writer_module.save_commit.__name__, fail_commit)
            with pytest.raises(OSError, match="fixture commit publication failed"):
                publish(writer)
        assert writer.failed
        assert hash_digests(writer.hashes) == previous_digests
        assert hash_digests(previous) == previous_digests
        assert layout.result_csv.read_bytes() == before_manifest
        assert layout.result_store_path("pending.json").is_file()
        with pytest.raises(ValueError, match="requires recovery"):
            publish(writer)
    with ResultWriter(tmp_path) as recovered:
        assert not layout.result_store_path("pending.json").exists()
        assert hash_digests(recovered.hashes) == previous_digests
        assert_committed_hashes(recovered)
        assert publish(recovered)
        assert not publish(recovered)
        assert_committed_hashes(recovered)


def test_metadata_failure_releases_writer_and_allows_fresh_hash_recovery(tmp_path, monkeypatch):
    layout = committed_store(tmp_path)
    before = layout.result_csv.read_bytes()
    writer = ResultWriter(tmp_path)

    def fail_metadata(*_args):
        raise OSError("fixture metadata recovery failed")

    with monkeypatch.context() as guard:
        guard.setattr(metadata, "recover_metadata", fail_metadata)
        with pytest.raises(OSError, match="fixture metadata recovery failed"):
            with writer:
                pytest.fail("incomplete recovery was accepted")
    assert writer.lease is None
    assert not hasattr(writer, "hashes")
    assert layout.result_csv.read_bytes() == before
    with writer as recovered:
        assert_committed_hashes(recovered)
        assert recovered.append_window(receipt(tmp_path, 1))
        assert_committed_hashes(recovered)
