"""A visible snapshot must have durable directory entries before the root can reference it."""
import json
import os
from contextlib import ExitStack
from pathlib import Path

import pytest

from acprof.artifacts import file_sha256
from acprof.result_store.writer import ResultWriter
from acprof.results.csv import ResultValidationError
from acprof.results.layers import GENERATIONS_DIR
from tests.results.storage.test_incremental_results import initialized, receipt


def installed_files(directory: Path) -> dict[Path, tuple[int, int, str]]:
    return {path: (path.stat().st_ino, path.stat().st_mtime_ns, file_sha256(path))
            for path in directory.rglob("*") if path.is_file()}


@pytest.mark.parametrize("directory", ["generation", "snapshot"])
@pytest.mark.parametrize("reopen", [False, True], ids=["same-writer", "new-writer"])
def test_freeze_retries_unconfirmed_directory_sync(tmp_path, monkeypatch, directory, reopen):
    layout = initialized(tmp_path)
    source = receipt(tmp_path)
    parent = layout.root / GENERATIONS_DIR if directory == "generation" else layout.result_store_path("snapshots")
    fsync = os.fsync
    attempts = 0

    def fail_directory_sync(fd):
        nonlocal attempts
        if parent.is_dir() and os.path.samestat(os.fstat(fd), parent.stat()):
            attempts += 1
            if attempts <= 2:
                raise OSError("fixture snapshot directory sync failed")
        fsync(fd)

    with ExitStack() as owner:
        writer = owner.enter_context(ResultWriter(tmp_path))
        writer.append_window(source)
        head = layout.result_csv.read_bytes()
        monkeypatch.setattr(os, "fsync", fail_directory_sync)
        published: dict[Path, tuple[int, int, str]] = {}
        for _ in range(2):
            with pytest.raises(OSError, match="fixture snapshot directory sync failed"):
                writer.freeze()
            assert layout.result_csv.read_bytes() == head
            current = installed_files(parent)
            assert current
            if published:
                assert current == published
            published = current
            if reopen:
                owner.close()
                writer = owner.enter_context(ResultWriter(tmp_path))
        snapshot = writer.freeze()

    assert attempts == 3
    assert installed_files(parent) == published
    frozen = json.loads(layout.result_csv.read_bytes())
    assert frozen["state"] == "complete"
    manifest = layout.root / frozen["snapshot"]["path"]
    assert json.loads(manifest.read_bytes()) == snapshot
    assert file_sha256(manifest) == frozen["snapshot"]["sha256"]


@pytest.mark.parametrize(("directory", "error"), [
    ("generation", "modified immutable result layer"), ("snapshot", "immutable snapshot manifest changed"),
])
def test_freeze_validates_reused_content_before_sync(tmp_path, monkeypatch, directory, error):
    layout = initialized(tmp_path)
    with ResultWriter(tmp_path) as writer:
        writer.append_window(receipt(tmp_path))
        snapshot = writer.freeze()
        head = layout.result_csv.read_bytes()
        if directory == "generation":
            parent = layout.root / GENERATIONS_DIR
            artifact = layout.root / snapshot["layers"]["summary"]["path"]
        else:
            parent = layout.result_store_path("snapshots")
            artifact = layout.root / json.loads(head)["snapshot"]["path"]
        artifact.write_bytes(b"{}")
        fsync = os.fsync

        def check_sync(fd):
            assert not os.path.samestat(os.fstat(fd), parent.stat()), "accepted a damaged immutable artifact"
            fsync(fd)

        monkeypatch.setattr(os, "fsync", check_sync)
        with pytest.raises(ResultValidationError, match=error):
            writer.freeze()
        assert artifact.read_bytes() == b"{}"
        assert layout.result_csv.read_bytes() == head
