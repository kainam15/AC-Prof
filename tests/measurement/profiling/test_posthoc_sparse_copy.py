"""Preserve forensic bytes without allocating sparse holes during backup or restore."""
from __future__ import annotations

import errno
import hashlib
import os
import stat

import pytest

from acprof.host.posthoc import copying, storage
from acprof.result_csv import read_result_csv
from tests.measurement.profiling.test_posthoc_backup import backup_context


def sparse_tail(path, size=16 * 1024 * 1024):
    with path.open("ab") as stream:
        stream.truncate(size)
        stream.write(b"forensic-tail-not-a-CSV-row")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o640)
    os.utime(path, ns=(1600000000000000000, 1600000000123456789))
    assert path.stat().st_blocks * 512 < path.stat().st_size // 4
    return path


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def assert_sparse_copy(source, destination):
    original, copied = source.stat(), destination.stat()
    assert copied.st_size == original.st_size
    assert checksum(destination) == checksum(source)
    assert stat.S_IMODE(copied.st_mode) == stat.S_IMODE(original.st_mode)
    assert copied.st_mtime_ns == original.st_mtime_ns
    assert copied.st_ino != original.st_ino
    assert copied.st_blocks * 512 < original.st_size // 4


def test_posthoc_backup_keeps_sparse_tail_without_materializing_zeros(tmp_path):
    context = backup_context(tmp_path, "v3")
    expected = read_result_csv(context.result_csv)
    source = sparse_tail(context.result_dir / "performance.csv")
    before = checksum(source)
    backup = storage.create_backup(context)
    assert_sparse_copy(source, backup / "performance.csv")
    assert read_result_csv(backup / "result_layers.json") == expected
    assert checksum(source) == before


def test_posthoc_restore_keeps_sparse_tail_and_metadata(tmp_path):
    source = tmp_path / "backup.csv"
    source.write_bytes(b"original prefix\n")
    sparse_tail(source)
    destination = tmp_path / "result.csv"
    destination.write_bytes(b"replaced result")
    storage._restore_from_backup(destination, source)
    assert_sparse_copy(source, destination)
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("operation", ["backup", "restore"])
@pytest.mark.parametrize("cancelled", [False, True])
def test_sparse_copy_failure_preserves_original_and_cleans_partial_output(tmp_path, monkeypatch, operation, cancelled):
    context = backup_context(tmp_path, "v3")
    source = sparse_tail(context.result_dir / "performance.csv")
    previous = storage.create_backup(context)
    expected = checksum(source)
    failure = KeyboardInterrupt("copy interrupted") if cancelled else OSError(errno.ENOSPC, "copy storage full")

    def fail_write(*_args):
        raise failure

    monkeypatch.setattr(copying.os, "write", fail_write)
    if operation == "backup":
        with pytest.raises(type(failure), match="copy interrupted|copy storage full"):
            storage.create_backup(context)
        assert list(previous.parent.iterdir()) == [previous]
        assert checksum(previous / "performance.csv") == expected
    else:
        destination = tmp_path / "result.csv"
        destination.write_bytes(b"original destination")
        with pytest.raises(type(failure), match="copy interrupted|copy storage full"):
            storage._restore_from_backup(destination, source)
        assert destination.read_bytes() == b"original destination"
        assert not list(tmp_path.glob("*.tmp"))
    assert checksum(source) == expected
