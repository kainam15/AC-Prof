"""Private settings failures preserve errors, ownership and process values."""
import errno
import os
import stat
from pathlib import Path
from typing import TextIO

import pytest

from acprof.host.env_utils import load_project_env, save_project_env


class ClosingStream:
    """Delegate actual writes and optionally fail after closing the real file."""

    def __init__(self, stream: TextIO, error: BaseException | None):
        self.stream = stream
        self.descriptor = stream.fileno()
        self.error = error

    def fileno(self) -> int:
        return self.stream.fileno()

    def write(self, text: str) -> int:
        return self.stream.write(text)

    def flush(self) -> None:
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()
        if self.error is not None:
            raise self.error

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def track_streams(monkeypatch, close_error=None):
    opened: list[ClosingStream] = []
    original = os.fdopen

    def fdopen(*args, **kwargs):
        stream = ClosingStream(original(*args, **kwargs), close_error)
        opened.append(stream)
        return stream

    monkeypatch.setattr(os, "fdopen", fdopen)
    return opened


def fail_temp_cleanup(monkeypatch, error):
    original = os.unlink

    def unlink(path, *args, **kwargs):
        if Path(path).name.startswith(".env.local.") and str(path).endswith(".tmp"):
            raise error
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)


def assert_descriptor_closed(descriptor):
    with pytest.raises(OSError, match="Bad file descriptor") as caught:
        os.fstat(descriptor)
    assert caught.value.errno == errno.EBADF


def assert_streams_closed(streams):
    assert streams
    for stream in streams:
        assert stream.stream.closed
        assert_descriptor_closed(stream.descriptor)


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_open_failure_closes_raw_descriptor(tmp_path, monkeypatch, error_type, cleanup_fails):
    primary = error_type("stream open failed")
    descriptors = []
    environment = {"HF_TOKEN": "previous"}

    def fail_open(descriptor, *_, **__):
        descriptors.append(descriptor)
        raise primary

    monkeypatch.setattr(os, "fdopen", fail_open)
    if cleanup_fails:
        fail_temp_cleanup(monkeypatch, PermissionError("temporary unlink failed"))
    try:
        with pytest.raises(error_type, match="stream open failed") as caught:
            save_project_env(tmp_path, {"HF_TOKEN": "hf_testonly"}, environ=environment)
        assert caught.value is primary
        assert environment == {"HF_TOKEN": "previous"}
        assert not (tmp_path / ".env.local").exists()
        assert bool(list(tmp_path.glob("*.tmp"))) == cleanup_fails
        if cleanup_fails:
            assert "temporary unlink failed" in "\n".join(primary.__notes__)
        assert len(descriptors) == 1
        assert_descriptor_closed(descriptors[0])
    finally:
        # Reclaim the known descriptor when exercising the original leaking code.
        for descriptor in descriptors:
            try:
                os.fstat(descriptor)
            except OSError as error:
                if error.errno != errno.EBADF:
                    raise
            else:
                os.close(descriptor)


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
@pytest.mark.parametrize("cleanup", ["close", "unlink", "both"])
def test_sync_failure_preserves_primary_and_all_cleanup_errors(tmp_path, monkeypatch, error_type, cleanup):
    primary = error_type("file sync failed")
    close_fails = cleanup in {"close", "both"}
    unlink_fails = cleanup in {"unlink", "both"}
    streams = track_streams(monkeypatch, OSError("stream close failed") if close_fails else None)
    if unlink_fails:
        fail_temp_cleanup(monkeypatch, PermissionError("temporary unlink failed"))

    def fail_sync(_):
        raise primary

    monkeypatch.setattr(os, "fsync", fail_sync)
    environment = {"HF_TOKEN": "previous"}
    with pytest.raises(error_type, match="file sync failed") as caught:
        save_project_env(tmp_path, {"HF_TOKEN": "hf_testonly"}, environ=environment)

    assert caught.value is primary
    notes = "\n".join(primary.__notes__)
    assert ("stream close failed" in notes) == close_fails
    assert ("temporary unlink failed" in notes) == unlink_fails
    assert_streams_closed(streams)
    assert environment == {"HF_TOKEN": "previous"}
    assert not (tmp_path / ".env.local").exists()
    leftovers = list(tmp_path.glob("*.tmp"))
    assert bool(leftovers) == unlink_fails
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in leftovers)


@pytest.mark.parametrize("cleanup", ["close", "unlink"])
def test_cleanup_cancellation_retains_preceding_save_failure(tmp_path, monkeypatch, cleanup):
    primary = OSError("file sync failed")
    cancellation = KeyboardInterrupt("cleanup cancelled")
    streams = track_streams(monkeypatch, cancellation if cleanup == "close" else None)
    if cleanup == "unlink":
        fail_temp_cleanup(monkeypatch, cancellation)

    def fail_sync(_):
        raise primary

    monkeypatch.setattr(os, "fsync", fail_sync)
    environment = {}
    with pytest.raises(KeyboardInterrupt) as caught:
        save_project_env(tmp_path, {"HF_TOKEN": "hf_testonly"}, environ=environment)

    assert caught.value is cancellation
    assert cancellation.__cause__ is primary
    assert "file sync failed" in "\n".join(cancellation.__notes__)
    assert_streams_closed(streams)
    assert not (tmp_path / ".env.local").exists()
    assert environment == {}


@pytest.mark.parametrize("failed_target", [".env.local.bak", ".env.local"])
def test_publish_failure_preserves_existing_config_and_environment(tmp_path, monkeypatch, failed_target):
    target = tmp_path / ".env.local"
    backup = tmp_path / ".env.local.bak"
    original = "# keep\nHF_TOKEN=previous\n"
    target.write_text(original)
    backup.write_text("older backup\n")
    primary = OSError("config publish failed")
    original_replace = os.replace
    streams = track_streams(monkeypatch)

    def replace(source, destination, *args, **kwargs):
        if Path(destination).name == failed_target:
            raise primary
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    fail_temp_cleanup(monkeypatch, PermissionError("temporary unlink failed"))
    environment = {"HF_TOKEN": "previous"}
    with pytest.raises(OSError, match="config publish failed") as caught:
        save_project_env(tmp_path, {"HF_TOKEN": "hf_testonly"}, environ=environment)

    assert caught.value is primary
    assert "temporary unlink failed" in "\n".join(primary.__notes__)
    assert target.read_text() == original
    assert backup.read_text() == ("older backup\n" if failed_target == backup.name else original)
    assert environment == {"HF_TOKEN": "previous"}
    assert_streams_closed(streams)
    leftovers = list(tmp_path.glob("*.tmp"))
    assert len(leftovers) == 1
    assert leftovers[0].stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("stage", ["new", "backup", "update"])
def test_directory_sync_failure_retains_published_file_and_primary_error(tmp_path, monkeypatch, stage):
    target = tmp_path / ".env.local"
    backup = tmp_path / ".env.local.bak"
    original = "HF_TOKEN=previous\n"
    if stage != "new":
        target.write_text(original)
    primary = OSError("directory sync failed")
    original_sync, original_close = os.fsync, os.close
    failed_descriptors = []

    def sync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            if stage == "backup" or "hf_testonly" in target.read_text():
                failed_descriptors.append(descriptor)
                raise primary
        return original_sync(descriptor)

    def close(descriptor):
        original_close(descriptor)
        if descriptor in failed_descriptors:
            raise OSError("directory close failed")

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "close", close)
    environment = {"HF_TOKEN": "previous"}
    with pytest.raises(OSError, match="directory sync failed") as caught:
        save_project_env(tmp_path, {"HF_TOKEN": "hf_testonly"}, environ=environment)

    assert caught.value is primary
    assert "directory close failed" in "\n".join(primary.__notes__)
    assert environment == {"HF_TOKEN": "previous"}
    loaded = {}
    load_project_env(tmp_path, environ=loaded)
    assert loaded["HF_TOKEN"] == ("previous" if stage == "backup" else "hf_testonly")
    if stage != "new":
        assert backup.read_text() == original
        assert backup.stat().st_mode & 0o777 == 0o600
    if stage != "backup":
        assert target.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob("*.tmp"))
    assert len(failed_descriptors) == 1
    assert_descriptor_closed(failed_descriptors[0])


def test_close_failure_alone_aborts_publication(tmp_path, monkeypatch):
    primary = OSError("stream close failed")
    streams = track_streams(monkeypatch, primary)
    environment = {}
    with pytest.raises(OSError, match="stream close failed") as caught:
        save_project_env(tmp_path, {"HF_TOKEN": "hf_testonly"}, environ=environment)
    assert caught.value is primary
    assert_streams_closed(streams)
    assert not list(tmp_path.iterdir())
    assert environment == {}
