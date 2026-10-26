"""Bounded metadata reads allocate for actual data and keep byte identity."""
import hashlib
import json
import os
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import pytest

from acprof.artifacts import MAX_JSON_ARTIFACT_BYTES, read_json_object


def observe_reader(monkeypatch, path, *, chunk_limit=None, after_first=None, fail_after=None):
    observed = SimpleNamespace(streams=[], requests=[], sizes=[], offsets=[])

    class Reader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, size=-1):
            if fail_after is not None and len(observed.requests) >= fail_after:
                raise OSError("fixture read failure")
            observed.requests.append(size)
            data = self.stream.read(min(size, chunk_limit) if chunk_limit is not None else size)
            observed.sizes.append(len(data))
            observed.offsets.append(os.lseek(self.stream.fileno(), 0, os.SEEK_CUR))
            if len(observed.requests) == 1 and after_first is not None:
                after_first()
            return data

    def open_file(source: Path, mode: str = "r", *args, **kwargs):
        stream = open(source, mode, *args, **kwargs)
        if source == path and mode == "rb":
            observed.streams.append(stream)
            return Reader(stream)
        return stream

    monkeypatch.setattr(Path, "open", open_file)
    return observed


def test_small_metadata_does_not_allocate_the_full_byte_limit(tmp_path):
    path = tmp_path / "small.json"
    payload = {"model": "示例/model", "revision": "a" * 40, "enabled": True}
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8-sig")
    path.write_bytes(raw)
    expected_hash = hashlib.sha256(raw).hexdigest()
    tracemalloc.start()
    try:
        assert read_json_object(path, expected_sha256=expected_hash) == payload
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 256 * 1024, f"small artifact allocated {peak:,} bytes"
    assert path.read_bytes() == raw


def test_near_limit_metadata_does_not_retain_an_extra_full_copy(tmp_path):
    path = tmp_path / "near-limit.json"
    path.write_bytes(b"{}" + b" " * (MAX_JSON_ARTIFACT_BYTES - 2))
    tracemalloc.start()
    try:
        assert read_json_object(path) == {}
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # The raw bytes and decoded JSON text may coexist. Chunk accumulation must
    # not leave a third full copy alive during decoding.
    assert peak < 2 * MAX_JSON_ARTIFACT_BYTES + 512 * 1024


def test_oversize_read_stops_at_one_overflow_byte_without_prefetch(tmp_path, monkeypatch):
    path = tmp_path / "oversize.json"
    path.write_bytes(b"{}" + b" " * (MAX_JSON_ARTIFACT_BYTES + 65536))
    observed = observe_reader(monkeypatch, path)
    with pytest.raises(ValueError, match="metadata exceeds the 4 MiB read limit"):
        read_json_object(path, label="metadata", expected_sha256="0" * 64)
    assert sum(observed.sizes) == MAX_JSON_ARTIFACT_BYTES + 1
    assert observed.offsets[-1] == MAX_JSON_ARTIFACT_BYTES + 1
    assert all(stream.closed for stream in observed.streams)


@pytest.mark.parametrize("fragment_size", [1, 7, 65535])
def test_short_reads_and_split_utf8_keep_the_complete_verified_object(tmp_path, monkeypatch, fragment_size):
    path = tmp_path / "unicode.json"
    payload = {"文字": "多字节 \u2028 \u0085 \u2029" * 10}
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8-sig")
    path.write_bytes(raw)
    observed = observe_reader(monkeypatch, path, chunk_limit=fragment_size)
    assert read_json_object(path, expected_sha256=hashlib.sha256(raw).hexdigest()) == payload
    assert sum(observed.sizes) == len(raw)
    assert all(stream.closed for stream in observed.streams)


def test_growing_file_is_still_subject_to_the_receive_limit(tmp_path, monkeypatch):
    path = tmp_path / "growing.json"
    path.write_bytes(b"{}")

    def append_after_first_read():
        with path.open("ab") as stream:
            stream.write(b" " * (MAX_JSON_ARTIFACT_BYTES + 65536))

    observed = observe_reader(monkeypatch, path, after_first=append_after_first_read)
    with pytest.raises(ValueError, match="4 MiB read limit"):
        read_json_object(path)
    assert sum(observed.sizes) == MAX_JSON_ARTIFACT_BYTES + 1
    assert observed.offsets[-1] == MAX_JSON_ARTIFACT_BYTES + 1
    assert all(stream.closed for stream in observed.streams)


def test_hash_and_parse_use_one_open_file_even_when_path_is_replaced(tmp_path, monkeypatch):
    path = tmp_path / "source.json"
    replacement = tmp_path / "replacement.json"
    payload = {"text": "中" * 40000, "source": "original"}
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8-sig")
    path.write_bytes(raw)
    replacement.write_text('{"source":"replacement"}')
    observed = observe_reader(monkeypatch, path, after_first=lambda: replacement.replace(path))
    assert read_json_object(path, expected_sha256=hashlib.sha256(raw).hexdigest()) == payload
    assert len(observed.streams) == 1
    assert all(stream.closed for stream in observed.streams)
    assert json.loads(path.read_text()) == {"source": "replacement"}


def test_partial_io_failure_propagates_and_closes_the_file(tmp_path, monkeypatch):
    path = tmp_path / "broken.json"
    path.write_text('{"source":"partial"}')
    observed = observe_reader(monkeypatch, path, chunk_limit=3, fail_after=1)
    with pytest.raises(OSError, match="fixture read failure"):
        read_json_object(path)
    assert all(stream.closed for stream in observed.streams)


def test_explicit_unbounded_records_keep_large_payload_and_hash_validation(tmp_path):
    path = tmp_path / "transaction.json"
    raw = b'{"payload":"' + b"x" * MAX_JSON_ARTIFACT_BYTES + b'"}'
    path.write_bytes(raw)
    payload = read_json_object(path, bounded=False, expected_sha256=hashlib.sha256(raw).hexdigest())
    assert payload == {"payload": "x" * MAX_JSON_ARTIFACT_BYTES}
    with pytest.raises(ValueError, match="hash does not match"):
        read_json_object(path, bounded=False, expected_sha256="0" * 64)
