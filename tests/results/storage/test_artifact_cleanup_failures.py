"""Artifact cleanup preserves the primary exception and all committed content."""
from pathlib import Path

import pytest

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
