"""HTML exports publish complete bytes and never remove competing destinations."""

import errno
import os
import stat
from pathlib import Path

import pytest

from acprof.host import run_state
from acprof.tui import report_files


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.html"
    path.write_bytes(b"<html>\r\n" + bytes(range(256)) * 8192 + b"</html>\r\n")
    return path


def test_export_is_not_visible_until_all_bytes_have_been_copied(source, tmp_path, monkeypatch):
    destination = tmp_path / "export.html"
    original_copy = report_files.shutil.copyfileobj

    def checked_copy(reader, writer, **kwargs):
        writer.write(reader.read(13))
        writer.flush()
        assert not destination.exists(), "a partial HTML export was already published"
        original_copy(reader, writer, **kwargs)

    monkeypatch.setattr(report_files.shutil, "copyfileobj", checked_copy)
    report_files.export_html_file(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert set(tmp_path.iterdir()) == {source, destination}


def test_export_creates_parent_directories_and_private_file(source, tmp_path):
    destination = tmp_path / "new" / "nested" / "export.html"
    report_files.export_html_file(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_mode & 0o777 == 0o600
    assert list(destination.parent.iterdir()) == [destination]


def test_export_bounds_source_reads_and_excludes_collection(source, tmp_path, monkeypatch):
    original_open = Path.open
    reads = []

    class BoundedReader:
        def __enter__(self):
            self.stream = original_open(source, "rb")
            return self

        def read(self, size=-1):
            assert 0 < size <= 1024 * 1024
            with pytest.raises(RuntimeError):
                with run_state.MeasurementLock():
                    pytest.fail("collection entered during export")
            reads.append(size)
            return self.stream.read(size)

        def __exit__(self, *_exc):
            self.stream.close()

    def checked_open(path, *args, **kwargs):
        if path == source:
            return BoundedReader()
        return original_open(path, *args, **kwargs)

    destination = tmp_path / "export.html"
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "open", checked_open)
        report_files.export_html_file(source, destination)
    assert len(reads) >= 3
    assert destination.read_bytes() == source.read_bytes()


def test_existing_destination_is_rejected_before_copying(source, tmp_path, monkeypatch):
    destination = tmp_path / "export.html"
    destination.write_bytes(b"previous report")

    def forbidden(*_args, **_kwargs):
        pytest.fail("an existing destination still triggered a source read")

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "open", forbidden)
        with pytest.raises(FileExistsError):
            report_files.export_html_file(source, destination)
    assert destination.read_bytes() == b"previous report"
    assert set(tmp_path.iterdir()) == {source, destination}


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "dangling_symlink"])
@pytest.mark.parametrize("copy_fails", [False, True])
def test_export_preserves_entries_created_during_copy(source, tmp_path, monkeypatch, kind, copy_fails):
    destination = tmp_path / "export.html"
    linked = tmp_path / "keep.html"
    linked.write_bytes(b"linked report")
    entries = []
    failure = OSError(errno.EIO, "injected copy failure")

    def compete(reader, writer, **_kwargs):
        writer.write(reader.read(13))
        if destination.exists():
            destination.unlink()
        if kind == "file":
            destination.write_bytes(b"competing report")
            destination.chmod(0o600)
        elif kind == "directory":
            destination.mkdir()
            (destination / "keep.txt").write_text("keep")
        else:
            destination.symlink_to(linked.name if kind == "symlink" else "missing.html")
        entries.append(destination.lstat())
        if copy_fails:
            raise failure
        writer.write(reader.read())

    monkeypatch.setattr(report_files.shutil, "copyfileobj", compete)
    with pytest.raises(OSError, match="injected copy failure" if copy_fails else "File exists") as caught:
        report_files.export_html_file(source, destination)
    if copy_fails:
        assert caught.value is failure
    else:
        assert isinstance(caught.value, FileExistsError)
    assert destination.lstat().st_ino == entries[0].st_ino
    assert destination.lstat().st_mode == entries[0].st_mode
    if kind == "file":
        assert destination.read_bytes() == b"competing report"
    elif kind == "directory":
        assert (destination / "keep.txt").read_text() == "keep"
    else:
        assert destination.readlink() == Path(linked.name if kind == "symlink" else "missing.html")
    assert linked.read_bytes() == b"linked report"
    assert not list(tmp_path.glob(".export.html.*.tmp"))


@pytest.mark.parametrize("phase", ["copy", "interrupt", "file_sync", "publish"])
def test_failure_before_publication_leaves_no_export(source, tmp_path, monkeypatch, phase):
    destination = tmp_path / "export.html"
    original = source.read_bytes()
    failure = KeyboardInterrupt("interrupted copy") if phase == "interrupt" else OSError(errno.EIO, phase)

    def fail(*_args, **_kwargs):
        raise failure

    if phase in {"copy", "interrupt"}:
        def partial(_reader, writer, **_kwargs):
            writer.write(b"partial")
            raise failure
        monkeypatch.setattr(report_files.shutil, "copyfileobj", partial)
    else:
        monkeypatch.setattr(os, "fsync" if phase == "file_sync" else "link", fail)
    with pytest.raises(type(failure)) as caught:
        report_files.export_html_file(source, destination)
    assert caught.value is failure
    assert list(tmp_path.iterdir()) == [source]
    assert source.read_bytes() == original
    with run_state.MeasurementLock():
        pass


def test_directory_sync_failure_retains_only_the_complete_export(source, tmp_path, monkeypatch):
    destination = tmp_path / "export.html"
    sync = os.fsync

    def fail_directory(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EIO, "directory sync failed")
        sync(fd)

    monkeypatch.setattr(os, "fsync", fail_directory)
    with pytest.raises(OSError, match="directory sync failed"):
        report_files.export_html_file(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert set(tmp_path.iterdir()) == {source, destination}


@pytest.mark.parametrize("system", ["Darwin", "Windows"])
def test_offline_export_does_not_request_a_linux_lock(source, tmp_path, monkeypatch, system):
    from acprof import platform

    monkeypatch.setattr(platform.platform, "system", lambda: system)

    def forbidden():
        pytest.fail("non-Linux export requested Linux measurement exclusion")

    monkeypatch.setattr(run_state, "MeasurementLock", forbidden)
    destination = tmp_path / "export.html"
    report_files.export_html_file(source, destination)
    assert destination.read_bytes() == source.read_bytes()
