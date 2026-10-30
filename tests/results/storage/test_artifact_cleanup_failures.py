"""Artifact cleanup preserves the primary exception and all committed content."""
import asyncio
import errno
import os
import sys
from concurrent.futures import CancelledError
from contextlib import nullcontext
from pathlib import Path

import pytest

from acprof import artifacts
from acprof.artifacts import atomic_write, atomic_write_binary
from tests.results.storage.test_result_generations import _publish


@pytest.mark.parametrize("write", [atomic_write, atomic_write_binary])
@pytest.mark.parametrize("failure_type", [OSError, KeyboardInterrupt])
def test_atomic_write_keeps_failure_when_temporary_removal_fails(tmp_path, monkeypatch, write, failure_type):
    target = tmp_path / "artifact"
    target.write_bytes(b"original")
    failure = failure_type("original artifact write failure")
    temporary = []

    def fail_write(stream):
        stream.write(b"partial" if write is atomic_write_binary else "partial")
        raise failure

    def fail_remove(path, *args, **kwargs):
        temporary.append(path)
        raise PermissionError("artifact temporary removal denied")

    monkeypatch.setattr(Path, "unlink", fail_remove)
    with pytest.raises(failure_type) as caught:
        write(target, fail_write)
    assert caught.value is failure
    assert target.read_bytes() == b"original"
    assert len(temporary) == 1
    assert temporary[0].is_file()
    assert str(temporary[0]) in "\n".join(failure.__notes__)


def test_generation_cleanup_cannot_hide_failed_publication(tmp_path, monkeypatch):
    from acprof.results import layers

    _publish(tmp_path, 1)
    before = layers.read_result_layers(tmp_path)
    manifest = (tmp_path / "result_layers.json").read_bytes()
    failure = KeyboardInterrupt("cancelled generation write")
    stages = []

    def fail_write(*_args, **_kwargs):
        raise failure

    def fail_remove(path, *_args, **_kwargs):
        stages.append(path)
        raise PermissionError("generation cleanup denied")

    monkeypatch.setattr(layers, "atomic_write", fail_write)
    monkeypatch.setattr(layers.shutil, "rmtree", fail_remove)
    with pytest.raises(KeyboardInterrupt) as caught:
        _publish(tmp_path, 2)
    assert caught.value is failure
    assert layers.read_result_layers(tmp_path) == before
    assert (tmp_path / "result_layers.json").read_bytes() == manifest
    assert str(stages[0]) in "\n".join(failure.__notes__)


def test_successful_write_does_not_hide_its_cleanup_failure(tmp_path, monkeypatch):
    """An outer handled exception is unrelated to this successful publication."""
    target = tmp_path / "artifact"
    failure = PermissionError("temporary link removal denied")

    def fail_remove(_path, *args, **kwargs):
        raise failure

    monkeypatch.setattr(Path, "unlink", fail_remove)
    try:
        raise ValueError("previously handled error")
    except ValueError:
        with pytest.raises(PermissionError) as caught:
            atomic_write(target, lambda stream: stream.write("complete"), overwrite=False)
    assert caught.value is failure
    assert target.read_text() == "complete"
    assert "temporary artifact" in "\n".join(failure.__notes__)


FAILURES = [OSError, CancelledError, asyncio.CancelledError, KeyboardInterrupt, SystemExit]


@pytest.mark.parametrize("failure_type", FAILURES)
def test_directory_sync_preserves_failure_when_close_also_fails(tmp_path, monkeypatch, failure_type):
    failure = failure_type("directory synchronization failed")
    close = os.close
    descriptors = []

    def fail_sync(descriptor):
        raise failure

    def fail_close(descriptor):
        descriptors.append(descriptor)
        close(descriptor)
        raise OSError("directory close failed")

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "fsync", fail_sync)
        scoped.setattr(os, "close", fail_close)
        with pytest.raises(failure_type) as caught:
            artifacts.sync_directory(tmp_path)
    assert caught.value is failure
    assert len(descriptors) == 1
    with pytest.raises(OSError, match="Bad file descriptor") as closed:
        os.fstat(descriptors[0])
    assert closed.value.errno == errno.EBADF
    notes = "\n".join(failure.__notes__)
    assert "directory close failed" in notes
    assert str(tmp_path) in notes


def test_directory_close_failure_is_reported_after_successful_sync(tmp_path, monkeypatch):
    close = os.close
    failure = OSError("directory close failed")
    calls = []

    def fail_close(descriptor):
        calls.append(descriptor)
        close(descriptor)
        raise failure

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "close", fail_close)
        with pytest.raises(OSError, match="directory close failed") as caught:
            artifacts.sync_directory(tmp_path)
    assert caught.value is failure
    assert len(calls) == 1
    assert str(tmp_path) in "\n".join(failure.__notes__)


class ClosingTemporary:
    def __init__(self, stream, failure):
        self.stream = stream
        self.failure = failure
        self.close_calls = 0

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def close(self):
        self.close_calls += 1
        self.stream.close()
        raise self.failure


@pytest.mark.parametrize("publisher", [atomic_write, atomic_write_binary])
@pytest.mark.parametrize("failure_type", FAILURES)
def test_atomic_write_preserves_failure_across_close_and_removal(tmp_path, monkeypatch, publisher, failure_type):
    target = tmp_path / "artifact"
    target.write_bytes(b"original")
    failure = failure_type("artifact serialization failed")
    temporary = artifacts.tempfile.NamedTemporaryFile
    streams = []

    def open_temporary(**kwargs):
        stream = ClosingTemporary(temporary(**kwargs), OSError("temporary stream close failed"))
        streams.append(stream)
        return stream

    def fail_write(stream):
        stream.write(b"partial" if publisher is atomic_write_binary else "partial")
        raise failure

    def fail_remove(_path, *_args, **_kwargs):
        raise PermissionError("temporary removal failed")

    with monkeypatch.context() as scoped:
        scoped.setattr(artifacts.tempfile, "NamedTemporaryFile", open_temporary)
        scoped.setattr(Path, "unlink", fail_remove)
        with pytest.raises(failure_type) as caught:
            publisher(target, fail_write)
    assert caught.value is failure
    assert target.read_bytes() == b"original"
    assert streams[0].stream.closed
    assert streams[0].close_calls == 1
    assert Path(streams[0].name).is_file()
    notes = "\n".join(failure.__notes__)
    assert "temporary stream close failed" in notes
    assert "temporary removal failed" in notes
    assert streams[0].name in notes


@pytest.mark.parametrize("publisher", [atomic_write, atomic_write_binary])
def test_temporary_close_failure_prevents_publication(tmp_path, monkeypatch, publisher):
    target = tmp_path / "artifact"
    target.write_bytes(b"original")
    failure = OSError("temporary stream close failed")
    temporary = artifacts.tempfile.NamedTemporaryFile
    streams = []

    def open_temporary(**kwargs):
        stream = ClosingTemporary(temporary(**kwargs), failure)
        streams.append(stream)
        return stream

    monkeypatch.setattr(artifacts.tempfile, "NamedTemporaryFile", open_temporary)
    with pytest.raises(OSError, match="temporary stream close failed") as caught:
        publisher(target, lambda stream: stream.write(b"new" if publisher is atomic_write_binary else "new"))
    assert caught.value is failure
    assert target.read_bytes() == b"original"
    assert streams[0].stream.closed
    assert streams[0].close_calls == 1
    assert list(tmp_path.iterdir()) == [target]
    assert streams[0].name in "\n".join(failure.__notes__)


@pytest.mark.parametrize("action", ["verify", "export"])
def test_results_cli_includes_cleanup_diagnostics(tmp_path, monkeypatch, capsys, action):
    from acprof.cli import results
    from acprof.host import run_state
    from acprof.results import layers

    failure = OSError("result operation failed")
    failure.add_note("Cleanup failed for publication lock stream: close failed")

    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(run_state, "offline_analysis_lock", nullcontext)
    monkeypatch.setattr(layers, "read_result_layers" if action == "verify" else "export_result_layers", fail)
    arguments = ["acprof results", action, str(tmp_path)]
    if action == "export":
        arguments.append(str(tmp_path / "export.csv"))
    monkeypatch.setattr(sys, "argv", arguments)
    assert results.main() == 1
    captured = capsys.readouterr()
    assert "result operation failed" in captured.err
    assert "publication lock stream: close failed" in captured.err
    assert not captured.out
