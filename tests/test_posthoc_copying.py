"""Sparse copying preserves bytes, bounds buffers, and exposes storage failures."""
from __future__ import annotations

import errno
import os
import shutil

import pytest
from test_posthoc_sparse_copy import assert_sparse_copy, checksum, sparse_tail

from acprof.host.posthoc import copying


@pytest.fixture
def sparse_source(tmp_path):
    path = tmp_path / "source"
    path.write_bytes(b"prefix\x00\xff\n")
    return sparse_tail(path, size=4 * 1024 * 1024)


@pytest.mark.parametrize("positions", [[], [0], [2 * 1024 * 1024], [0, 4 * 1024 * 1024 - 16]])
def test_holes_at_all_positions_keep_size_bytes_and_metadata(tmp_path, positions):
    source, destination = tmp_path / "source", tmp_path / "copy"
    with source.open("wb") as stream:
        stream.truncate(4 * 1024 * 1024)
        for offset in positions:
            stream.seek(offset)
            stream.write(b"binary\x00\xffpayload!")
    source.chmod(0o640)
    destination.write_bytes(b"stale output contents")
    copying.copy_backup_file(source, destination)
    assert_sparse_copy(source, destination)


@pytest.mark.parametrize("contents", [b"", b"dense\x00\xffpayload" * 100])
def test_ordinary_files_keep_the_standard_copy_path(tmp_path, monkeypatch, contents):
    source, destination = tmp_path / "source", tmp_path / "copy"
    source.write_bytes(contents)
    copied = []
    copy = copying.shutil.copy2

    def observe(*args):
        copied.append(args)
        return copy(*args)

    monkeypatch.setattr(copying.shutil, "copy2", observe)
    copying.copy_backup_file(source, destination)
    assert copied == [(source, destination)]
    assert destination.read_bytes() == contents


@pytest.mark.parametrize("operation", [os.SEEK_DATA, os.SEEK_HOLE])
@pytest.mark.parametrize("error_number", [errno.EINVAL, errno.ENOTSUP, errno.ENOSYS])
def test_unsupported_initial_probe_falls_back_before_destination_changes(
    tmp_path, monkeypatch, sparse_source, operation, error_number,
):
    destination = tmp_path / "copy"
    destination.write_bytes(b"prior destination")
    seek, copy = copying.os.lseek, copying.shutil.copy2
    fallbacks = []

    def unavailable(fd, offset, whence):
        if whence == operation:
            raise OSError(error_number, "extent seeking unavailable")
        return seek(fd, offset, whence)

    def fallback(source, target):
        assert target.read_bytes() == b"prior destination"
        fallbacks.append(target)
        return copy(source, target)

    monkeypatch.setattr(copying.os, "lseek", unavailable)
    monkeypatch.setattr(copying.shutil, "copy2", fallback)
    copying.copy_backup_file(sparse_source, destination)
    assert fallbacks == [destination]
    assert checksum(destination) == checksum(sparse_source)


def test_missing_extent_api_keeps_copy2_semantics(tmp_path, monkeypatch, sparse_source):
    destination = tmp_path / "copy"
    monkeypatch.delattr(copying.os, "SEEK_HOLE")
    copying.copy_backup_file(sparse_source, destination)
    assert checksum(destination) == checksum(sparse_source)


def test_filesystem_reporting_only_data_keeps_the_standard_copy_path(tmp_path, monkeypatch, sparse_source):
    destination = tmp_path / "copy"
    seek, copy = copying.os.lseek, copying.shutil.copy2
    copied = []

    def no_holes(fd, offset, whence):
        if whence == os.SEEK_DATA:
            return offset
        if whence == os.SEEK_HOLE:
            return sparse_source.stat().st_size
        return seek(fd, offset, whence)

    def observe(*args):
        copied.append(args)
        return copy(*args)

    monkeypatch.setattr(copying.os, "lseek", no_holes)
    monkeypatch.setattr(copying.shutil, "copy2", observe)
    copying.copy_backup_file(sparse_source, destination)
    assert copied == [(sparse_source, destination)]
    assert checksum(destination) == checksum(sparse_source)


@pytest.mark.parametrize("bounds", [(-1, 4096), (0, 0), (0, 8 * 1024 * 1024)])
def test_invalid_extent_bounds_are_rejected(tmp_path, monkeypatch, sparse_source, bounds):
    destination = tmp_path / "copy"
    monkeypatch.setattr(copying.os, "lseek", lambda _fd, _offset, whence: bounds[0 if whence == os.SEEK_DATA else 1])
    with pytest.raises(OSError, match="invalid or changed"):
        copying.copy_backup_file(sparse_source, destination)
    assert not destination.exists()


@pytest.mark.parametrize("error_number", [errno.EIO, errno.ENOSPC, errno.EACCES])
def test_real_probe_errors_propagate_without_a_fallback(tmp_path, monkeypatch, sparse_source, error_number):
    destination = tmp_path / "copy"
    destination.write_bytes(b"prior destination")

    def failed_seek(*_args):
        raise OSError(error_number, "storage failure")

    def must_not_fallback(*_args):
        pytest.fail("real storage error was hidden by fallback")

    monkeypatch.setattr(copying.os, "lseek", failed_seek)
    monkeypatch.setattr(copying.shutil, "copy2", must_not_fallback)
    with pytest.raises(OSError, match="storage failure") as caught:
        copying.copy_backup_file(sparse_source, destination)
    assert caught.value.errno == error_number
    assert destination.read_bytes() == b"prior destination"


def test_mid_copy_seek_failure_is_not_restarted(tmp_path, monkeypatch, sparse_source):
    destination = tmp_path / "copy"
    seek = copying.os.lseek

    def fail_later(fd, offset, whence):
        if whence == os.SEEK_DATA and offset:
            raise OSError(errno.EINVAL, "later extent failed")
        return seek(fd, offset, whence)

    def must_not_fallback(*_args):
        pytest.fail("partial copy was silently restarted")

    monkeypatch.setattr(copying.os, "lseek", fail_later)
    monkeypatch.setattr(copying.shutil, "copy2", must_not_fallback)
    with pytest.raises(OSError, match="later extent failed"):
        copying.copy_backup_file(sparse_source, destination)


def test_short_reads_and_writes_keep_bounded_buffers(tmp_path, monkeypatch, sparse_source):
    with sparse_source.open("r+b") as stream:
        stream.write(b"\xff" * (2 * 1024 * 1024))
    destination = tmp_path / "copy"
    read, write = copying.os.read, copying.os.write
    requests = []

    def short_read(fd, size):
        requests.append(size)
        return read(fd, min(size, 32768))

    def short_write(fd, data):
        return write(fd, data[:8192])

    monkeypatch.setattr(copying.os, "read", short_read)
    monkeypatch.setattr(copying.os, "write", short_write)
    copying.copy_backup_file(sparse_source, destination)
    assert max(requests) == 1024 * 1024
    assert checksum(destination) == checksum(sparse_source)


@pytest.mark.parametrize("operation", ["read", "write"])
def test_zero_progress_is_rejected(tmp_path, monkeypatch, sparse_source, operation):
    destination = tmp_path / "copy"
    if operation == "read":
        monkeypatch.setattr(copying.os, "read", lambda *_args: b"")
    else:
        monkeypatch.setattr(copying.os, "write", lambda *_args: 0)
    with pytest.raises(OSError, match="ended inside|no write progress"):
        copying.copy_backup_file(sparse_source, destination)


def test_source_shrink_is_not_published_as_a_complete_copy(tmp_path, monkeypatch, sparse_source):
    destination = tmp_path / "copy"
    read = copying.os.read

    def shrink_after_read(fd, size):
        block = read(fd, size)
        with sparse_source.open("r+b") as stream:
            stream.truncate(0)
        return block

    monkeypatch.setattr(copying.os, "read", shrink_after_read)
    with pytest.raises(OSError, match="changed during copy"):
        copying.copy_backup_file(sparse_source, destination)


@pytest.mark.parametrize("alias", [False, True])
def test_same_file_and_hardlinks_never_truncate_the_source(tmp_path, sparse_source, alias):
    destination = tmp_path / "alias" if alias else sparse_source
    if alias:
        destination.hardlink_to(sparse_source)
    before = checksum(sparse_source)
    with pytest.raises(shutil.SameFileError, match="same file"):
        copying.copy_backup_file(sparse_source, destination)
    assert checksum(sparse_source) == before
