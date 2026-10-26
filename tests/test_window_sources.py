"""Window sources must consume the receipt bytes named by their commit hash."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
from test_incremental_results import initialized, receipt

from acprof.artifact_layout import ArtifactLayout
from acprof.result_store.protocol import reference
from acprof.result_store.sources import window_row
from acprof.result_store.writer import ResultWriter


def encoded_record(record):
    def checksum(value):
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(raw).hexdigest()

    record["row_sha256"] = checksum(record["row"])
    record["request_sha256"] = checksum(record["request"])
    record["content_sha256"] = checksum({key: value for key, value in record.items() if key != "content_sha256"})
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


@pytest.fixture(params=[1, 2], ids=["legacy-receipt", "sequenced-receipt"])
def window_source(tmp_path, request):
    layout = initialized(tmp_path)
    path = receipt(tmp_path)
    record = json.loads(path.read_bytes())
    if request.param == 2:
        record.update(schema_version=2, sequence=1,
                      projection_semantics="immutable_authority_before_result_projection")
        path.write_bytes(encoded_record(record))
    return layout, path, record


def observe_reads(monkeypatch, paths, *, after_read=None):
    counts = Counter()
    watched = {path.resolve() for path in paths}

    class Reader:
        def __init__(self, stream, path):
            self.stream = stream
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, *args):
            raw = self.stream.read(*args)
            counts[self.path] += 1
            if after_read is not None:
                after_read(self.path, counts[self.path])
            return raw

    def open_file(path: Path, mode: str = "r", *args, **kwargs):
        stream = open(path, mode, *args, **kwargs)
        resolved = path.resolve()
        return Reader(stream, resolved) if resolved in watched and "r" in mode else stream

    monkeypatch.setattr(Path, "open", open_file)
    return counts


def test_row_comes_from_the_hashed_receipt_when_path_is_replaced(window_source, tmp_path, monkeypatch):
    layout, path, record = window_source
    source = {"kind": "window", "reference": reference(layout, path)}
    telemetry = layout.window_evidence_path("telemetry", record["identity"])
    replacement = tmp_path / "replacement.json"
    record["row"]["latency_app_s"] = "9.9"
    replacement.write_bytes(encoded_record(record))

    def replace_after_first_read(read_path, count):
        if read_path == path and count == 1:
            replacement.replace(path)

    counts = observe_reads(monkeypatch, [path, telemetry], after_read=replace_after_first_read)
    row, identity = window_row(layout, source)
    assert row["latency_app_s"] == "0.1"
    assert identity[-1] == "case_scale1_r0"
    assert not replacement.exists(), "the replacement must occur during receipt validation"
    assert counts == {path: 1, telemetry: 1}


def test_telemetry_is_verified_against_the_loaded_receipt(window_source, tmp_path, monkeypatch):
    layout, path, record = window_source
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(path.read_bytes())
    record["telemetry"]["sha256"] = "0" * 64
    path.write_bytes(encoded_record(record))
    source = {"kind": "window", "reference": reference(layout, path)}
    original = ArtifactLayout.window_evidence_path

    def replace_after_path_check(self: ArtifactLayout, kind, identity):
        expected = original(self, kind, identity)
        if self == layout and kind == "telemetry":
            replacement.replace(path)
        return expected

    monkeypatch.setattr(ArtifactLayout, "window_evidence_path", replace_after_path_check)
    with pytest.raises(ValueError, match="telemetry hash mismatch"):
        window_row(layout, source)
    assert not replacement.exists()


def test_append_validates_each_window_once(window_source, monkeypatch):
    layout, path, record = window_source
    telemetry = layout.window_evidence_path("telemetry", record["identity"])
    with ResultWriter(layout.root) as writer:
        counts = observe_reads(monkeypatch, [path, telemetry])
        assert writer.append_window(path)
        assert next(iter(writer.rows.values()))["latency_app_s"] == "0.1"
        # Reference creation reads once, then validation hashes and parses once.
        assert counts == {path: 2, telemetry: 1}


def test_null_reference_hash_cannot_disable_verification(window_source):
    layout, path, _ = window_source
    source = {"kind": "window", "reference": {**reference(layout, path), "sha256": None}}
    with pytest.raises(ValueError, match="SHA256"):
        window_row(layout, source)


def test_hash_mismatch_is_rejected_before_json_or_telemetry_reads(window_source, monkeypatch):
    layout, path, record = window_source
    source = {"kind": "window", "reference": reference(layout, path)}
    telemetry = layout.window_evidence_path("telemetry", record["identity"])
    path.write_bytes(b"invalid JSON")
    counts = observe_reads(monkeypatch, [path, telemetry])
    with pytest.raises(ValueError, match="receipt SHA256 mismatch"):
        window_row(layout, source)
    assert counts == {path: 1}


@pytest.mark.parametrize("variant", ["receipt-location", "receipt-identity", "telemetry-location", "telemetry-symlink"])
def test_source_paths_are_checked_before_reading_evidence(window_source, tmp_path, monkeypatch, variant):
    layout, path, record = window_source
    if variant == "receipt-location":
        relocated = tmp_path / path.name
        path.rename(relocated)
        path = relocated
    elif variant == "receipt-identity":
        record["identity"]["case_id"] = "2c_4g_off"
        path.write_bytes(encoded_record(record))
    elif variant == "telemetry-location":
        record["telemetry"]["path"] = str(tmp_path / "unrelated.json")
        path.write_bytes(encoded_record(record))
    else:
        telemetry = layout.window_evidence_path("telemetry", record["identity"])
        target = tmp_path / "aliased.json"
        telemetry.rename(target)
        telemetry.symlink_to(target)
    source = {"kind": "window", "reference": reference(layout, path)}

    def allow_receipt_read_only(read_path: Path, mode: str = "r", *args, **kwargs):
        assert read_path.resolve() == path.resolve(), "evidence was read before the receipt paths were checked"
        return open(read_path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", allow_receipt_read_only)
    with pytest.raises(ValueError, match="outside raw/windows|path disagrees|symlinks"):
        window_row(layout, source)


@pytest.mark.parametrize("damage", ["missing", "hash", "identity", "schema"])
def test_source_still_rejects_damaged_telemetry(window_source, damage):
    layout, path, record = window_source
    telemetry = layout.window_evidence_path("telemetry", record["identity"])
    if damage == "missing":
        telemetry.unlink()
        message = "invalid window record"
    elif damage == "hash":
        telemetry.write_bytes(b"{}")
        message = "telemetry hash mismatch"
    else:
        evidence = json.loads(telemetry.read_bytes())
        if damage == "identity":
            evidence["identity"]["window_id"] = "different-window"
            message = "telemetry identity mismatch"
        else:
            evidence["schema_version"] = 99
            message = "unsupported telemetry schema"
        raw = json.dumps(evidence).encode()
        telemetry.write_bytes(raw)
        record["telemetry"]["sha256"] = hashlib.sha256(raw).hexdigest()
        path.write_bytes(encoded_record(record))
    source = {"kind": "window", "reference": reference(layout, path)}
    with pytest.raises(ValueError, match=message):
        window_row(layout, source)


def test_receipt_hash_covers_raw_utf8_and_crlf_without_a_metadata_size_cap(window_source):
    layout, path, record = window_source
    raw = json.dumps(record, ensure_ascii=False, indent=2).replace("\n", "\r\n").encode()
    path.write_bytes(raw + b" " * (4 * 1024 * 1024))
    source = {"kind": "window", "reference": reference(layout, path)}
    assert window_row(layout, source)[0] == record["row"]


def test_append_rejects_a_replaced_reference_without_publishing(window_source, tmp_path, monkeypatch):
    layout, path, record = window_source
    original = path.read_bytes()
    replacement = tmp_path / "replacement.json"
    record["row"]["latency_app_s"] = "9.9"
    replacement.write_bytes(encoded_record(record))

    def replace_after_reference(read_path, count):
        if read_path == path and count == 1:
            replacement.replace(path)

    with ResultWriter(layout.root) as writer:
        before = {target: target.read_bytes() for target in
                  [layout.result_csv, *(layout.layer_csv(layer) for layer in writer.manifest["layers"])]}
        with monkeypatch.context() as guard:
            observe_reads(guard, [path], after_read=replace_after_reference)
            with pytest.raises(ValueError, match="receipt SHA256 mismatch"):
                writer.append_window(path)
        assert not replacement.exists()
        assert writer.rows == writer.windows == {}
        assert all(target.read_bytes() == raw for target, raw in before.items())
        assert not layout.result_store_path("pending.json").exists()
        path.write_bytes(original)
        assert writer.append_window(path)
        assert next(iter(writer.rows.values()))["latency_app_s"] == "0.1"
