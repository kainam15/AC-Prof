"""Result provenance hashes cover complete files without retaining their bytes."""
from __future__ import annotations

import hashlib
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.results.storage.test_incremental_results import initialized, receipt

from acprof.result_store.protocol import reference
from acprof.result_store.writer import ResultWriter


def evidence_file(layout, *, size=16 * 1024 * 1024):
    path = layout.path("raw/profiler-evidence.bin")
    block = bytes(range(256)) * 256
    checksum = hashlib.sha256()
    with path.open("wb") as stream:
        remaining = size
        while remaining:
            chunk = block[:min(len(block), remaining)]
            stream.write(chunk)
            checksum.update(chunk)
            remaining -= len(chunk)
    return path, {"path": path.relative_to(layout.root).as_posix(), "sha256": checksum.hexdigest()}


def populated_store(tmp_path):
    layout = initialized(tmp_path)
    with ResultWriter(tmp_path) as writer:
        writer.append_window(receipt(tmp_path))
    return layout


def traced_peak(action):
    tracemalloc.start()
    try:
        result = action()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return result, peak


@pytest.mark.parametrize("operation", ["reference", "publish", "replay"])
def test_large_evidence_hashing_has_bounded_memory(tmp_path, operation):
    layout = populated_store(tmp_path)
    path, expected = evidence_file(layout)
    if operation == "reference":
        result, peak = traced_peak(lambda: reference(layout, path))
        assert result == expected
    elif operation == "publish":
        with ResultWriter(tmp_path) as writer:
            row = dict(next(iter(writer.rows.values())), latency_s="0.25")
            changed, peak = traced_peak(lambda: writer.update_rows(
                [row], reason="fixture profiler evidence", evidence=[expected]))
            assert changed
            assert next(iter(writer.rows.values()))["latency_s"] == "0.25"
    else:
        with ResultWriter(tmp_path) as writer:
            row = dict(next(iter(writer.rows.values())), latency_s="0.25")
            writer.update_rows([row], reason="fixture profiler evidence", evidence=[expected])
        def recover():
            with ResultWriter(tmp_path) as recovered:
                return next(iter(recovered.rows.values()))
        result, peak = traced_peak(recover)
        assert result["latency_s"] == "0.25"
    assert peak < 4 * 1024 * 1024, f"{operation} allocated {peak:,} bytes for 16 MiB evidence"


@pytest.mark.parametrize("size", [0, 1, 1048575, 1048576, 1048577])
def test_reference_hash_covers_empty_binary_and_partial_chunks(tmp_path, size):
    layout = initialized(tmp_path)
    path, expected = evidence_file(layout, size=size)
    assert reference(layout, path) == expected


def observe_evidence(monkeypatch, path, *, fragment_size=None, fail_after=None, after_first=None):
    observed = SimpleNamespace(streams=[], received=0, reads=0)

    class Reader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def readinto(self, buffer):
            raw = self.read(len(buffer))
            buffer[:len(raw)] = raw
            return len(raw)

        def read(self, size=-1):
            assert 0 < size <= 1024 * 1024, "hashing made an unbounded read"
            if fail_after is not None and observed.reads >= fail_after:
                raise OSError("fixture evidence read failed")
            raw = self.stream.read(min(size, fragment_size) if fragment_size else size)
            observed.reads += 1
            observed.received += len(raw)
            if observed.reads == 1 and after_first is not None:
                after_first()
            return raw

    def open_file(source: Path, mode: str = "r", *args, **kwargs):
        stream = open(source, mode, *args, **kwargs)
        if source == path and mode == "rb":
            observed.streams.append(stream)
            return Reader(stream)
        return stream

    monkeypatch.setattr(Path, "open", open_file)
    return observed


def test_reference_short_reads_and_path_replacement_use_one_open_file(tmp_path, monkeypatch):
    layout = initialized(tmp_path)
    path, expected = evidence_file(layout, size=2 * 1024 * 1024 + 13)
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"replacement must not contribute to the original digest")
    observed = observe_evidence(monkeypatch, path, fragment_size=65521,
                                 after_first=lambda: replacement.replace(path))
    assert reference(layout, path) == expected
    assert observed.received == 2 * 1024 * 1024 + 13
    assert len(observed.streams) == 1
    assert observed.streams[0].closed
    assert not replacement.exists()


def test_reference_includes_growth_before_eof(tmp_path, monkeypatch):
    layout = initialized(tmp_path)
    path = layout.path("raw/growing-evidence.bin")
    path.write_bytes(b"original")

    def append_evidence():
        with path.open("ab") as stream:
            stream.write(b" extension")

    observed = observe_evidence(monkeypatch, path, after_first=append_evidence)
    assert reference(layout, path)["sha256"] == hashlib.sha256(b"original extension").hexdigest()
    assert observed.received == len(b"original extension")
    assert len(observed.streams) == 1
    assert observed.streams[0].closed


def test_evidence_read_failure_closes_stream_and_does_not_publish(tmp_path, monkeypatch):
    layout = populated_store(tmp_path)
    path, expected = evidence_file(layout, size=2 * 1024 * 1024)
    with ResultWriter(tmp_path) as writer:
        original_row = dict(next(iter(writer.rows.values())))
        before = {target: target.read_bytes() for target in
                  [layout.result_csv, *(layout.layer_csv(layer) for layer in writer.manifest["layers"])]}
        with monkeypatch.context() as guard:
            observed = observe_evidence(guard, path, fail_after=1)
            with pytest.raises(OSError, match="fixture evidence read failed"):
                writer.update_rows([dict(original_row, latency_s="0.25")], reason="fixture", evidence=[expected])
        assert len(observed.streams) == 1
        assert observed.streams[0].closed
        assert next(iter(writer.rows.values())) == original_row
        assert all(target.read_bytes() == raw for target, raw in before.items())
        assert not layout.result_store_path("pending.json").exists()
        assert writer.update_rows([dict(original_row, latency_s="0.25")], reason="fixture", evidence=[expected])


@pytest.mark.parametrize("phase", ["publish", "replay"])
@pytest.mark.parametrize("damage", ["missing", "empty", "last-byte"])
def test_changed_evidence_is_rejected_before_projection_writes(tmp_path, phase, damage):
    layout = populated_store(tmp_path)
    path, expected = evidence_file(layout, size=2 * 1024 * 1024 + 17)
    with ResultWriter(tmp_path) as writer:
        updated = dict(next(iter(writer.rows.values())), latency_s="0.25")
        if phase == "replay":
            writer.update_rows([updated], reason="fixture", evidence=[expected])
        before = {target: target.read_bytes() for target in
                  [layout.result_csv, *(layout.layer_csv(layer) for layer in writer.manifest["layers"])]}
        if damage == "missing":
            path.unlink()
        elif damage == "empty":
            path.write_bytes(b"")
        else:
            with path.open("r+b") as stream:
                stream.seek(-1, 2)
                stream.write(b"!")
        error = FileNotFoundError if damage == "missing" else ValueError
        if phase == "publish":
            with pytest.raises(error):
                writer.update_rows([updated], reason="fixture", evidence=[expected])
    if phase == "replay":
        recovering = ResultWriter(tmp_path)
        with pytest.raises(error):
            with recovering:
                pytest.fail("recovery accepted damaged evidence")
        assert recovering.lease is None
    assert all(target.read_bytes() == raw for target, raw in before.items())
