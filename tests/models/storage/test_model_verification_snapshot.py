"""Model verification must not certify stale bytes after a concurrent cache change."""
import builtins
import copy
import hashlib
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from acprof.container.download_model import verify_download
from acprof.container.model_files import ModelFilesError, plan_download, validate_plan

CONTENT = b"trusted model payload"
SHA256 = hashlib.sha256(CONTENT).hexdigest()


def make_plan(names):
    return plan_download(model_id="example/snapshot", revision="a" * 40, family="nlp",
        backend="transformers_pipeline", policy="full",
        files={name: {"size": len(CONTENT), "lfs_sha256": SHA256} for name in names},
        read_json=lambda _: {})


class ObservedReader:
    def __init__(self, stream, *, on_read=None, on_close=None):
        self.stream = stream
        self.on_read = on_read
        self.on_close = on_close

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def __enter__(self):
        self.stream.__enter__()
        return self

    def __exit__(self, *args):
        result = self.stream.__exit__(*args)
        if self.on_close is not None and args[0] is None:
            self.on_close()
        return result

    def read(self, size=-1):
        assert 0 <= size <= 4 * 1024 * 1024, "payload reads must stay bounded"
        chunk = self.stream.read(size)
        if self.on_read is not None:
            self.on_read(chunk)
        return chunk


def observe_reads(monkeypatch, paths, *, on_read=None, on_close=None):
    original = builtins.open
    paths = set(paths)

    def opening(file, mode="r", *args, **kwargs):
        stream = original(file, mode, *args, **kwargs)
        if not isinstance(file, int) and Path(file) in paths and mode == "rb":
            return ObservedReader(stream, on_read=on_read, on_close=on_close)
        return stream

    # Path.open uses io.open; the ordinary open function uses builtins.open.
    # Observe actual reads rather than prescribing either implementation.
    monkeypatch.setattr(builtins, "open", opening)
    monkeypatch.setattr(io, "open", opening)


@pytest.mark.parametrize("change", ["rewrite", "replace", "truncate", "grow"])
def test_file_changed_while_hashing_is_rejected_without_updating_plan(tmp_path, monkeypatch, change):
    path = tmp_path / "model.bin"
    path.write_bytes(CONTENT)
    plan = make_plan([path.name])
    before = copy.deepcopy(plan)
    changed = False

    def mutate(chunk):
        nonlocal changed
        if not chunk or changed:
            return
        changed = True
        if change == "replace":
            replacement = tmp_path / "replacement"
            replacement.write_bytes(b"x" * len(CONTENT))
            replacement.replace(path)
        else:
            path.write_bytes({"rewrite": b"x" * len(CONTENT), "truncate": b"", "grow": CONTENT + b"unexpected"}[change])

    observe_reads(monkeypatch, [path], on_read=mutate)
    with pytest.raises(ModelFilesError, match="changed|mismatch"):
        verify_download(tmp_path, plan)
    assert changed
    assert plan == before
    validate_plan(plan)


@pytest.mark.parametrize("link", ["hardlink", "symlink"])
@pytest.mark.parametrize("restore_mtime", [False, True])
def test_changed_shared_payload_does_not_reuse_a_stale_hash(tmp_path, monkeypatch, link, restore_mtime):
    first, second = tmp_path / "a.bin", tmp_path / "z.bin"
    first.write_bytes(CONTENT)
    if link == "hardlink":
        os.link(first, second)
    else:
        second.symlink_to(first.name)
    original = first.stat()
    plan = make_plan([first.name, second.name])
    before = copy.deepcopy(plan)
    changed = False

    def mutate():
        nonlocal changed
        if changed:
            return
        changed = True
        first.write_bytes(b"x" * len(CONTENT))
        if restore_mtime:
            os.utime(first, ns=(original.st_atime_ns, original.st_mtime_ns))

    observe_reads(monkeypatch, [first], on_close=mutate)
    with pytest.raises(ModelFilesError, match="changed|mismatch"):
        verify_download(tmp_path, plan)
    assert changed
    assert plan == before
    validate_plan(plan)


@pytest.mark.parametrize("link", ["hardlink", "symlink"])
def test_stable_shared_payload_is_hashed_only_once(tmp_path, monkeypatch, link):
    first, second = tmp_path / "a.bin", tmp_path / "z.bin"
    first.write_bytes(CONTENT)
    if link == "hardlink":
        os.link(first, second)
    else:
        second.symlink_to(first.name)
    plan = make_plan([first.name, second.name])
    bytes_read = 0

    def count(chunk):
        nonlocal bytes_read
        bytes_read += len(chunk)

    observe_reads(monkeypatch, [first, second], on_read=count)
    verified = verify_download(tmp_path, plan)
    validate_plan(verified)
    assert bytes_read == len(CONTENT)
    assert [item["sha256"] for item in verified["files"]] == [SHA256, SHA256]
    assert verified["selected_bytes"] == 2 * len(CONTENT)


@pytest.mark.parametrize("change", ["rewrite", "replace", "remove"])
def test_earlier_file_changed_while_verifying_later_file_is_rejected(tmp_path, monkeypatch, change):
    first, second = tmp_path / "a.bin", tmp_path / "z.bin"
    first.write_bytes(CONTENT)
    second.write_bytes(CONTENT)
    plan = make_plan([first.name, second.name])
    before = copy.deepcopy(plan)
    changed = False

    def mutate(chunk):
        nonlocal changed
        if chunk and not changed:
            changed = True
            if change == "remove":
                first.unlink()
            elif change == "replace":
                replacement = tmp_path / "replacement"
                replacement.write_bytes(CONTENT)
                replacement.replace(first)
            else:
                first.write_bytes(b"x" * len(CONTENT))

    observe_reads(monkeypatch, [second], on_read=mutate)
    with pytest.raises(ModelFilesError, match="changed"):
        verify_download(tmp_path, plan)
    assert changed
    assert plan == before


@pytest.mark.parametrize("size", [0, 4 * 1024 * 1024, 4 * 1024 * 1024 + 17])
@pytest.mark.parametrize("identity", ["lfs", "git"])
def test_payload_hashes_keep_exact_bytes_at_chunk_boundaries(tmp_path, monkeypatch, size, identity):
    content = b"x" * size
    path = tmp_path / "model.bin"
    path.write_bytes(content)
    metadata = {"size": size}
    if identity == "lfs":
        metadata["lfs_sha256"] = hashlib.sha256(content).hexdigest()
    else:
        metadata["blob_id"] = hashlib.sha1(b"blob " + str(size).encode("ascii") + bytes([0]) + content).hexdigest()
    plan = plan_download(model_id="example/snapshot", revision="a" * 40, family="nlp",
        backend="transformers_pipeline", policy="full",
        files={path.name: metadata}, read_json=lambda _: {})
    observe_reads(monkeypatch, [path])
    verified = verify_download(tmp_path, plan)
    validate_plan(verified)
    assert verified["selected_bytes"] == size
    assert verified["files"][0]["sha256"] == hashlib.sha256(content).hexdigest()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes require POSIX")
def test_fifo_payload_is_rejected_without_waiting_for_a_writer(tmp_path):
    path = tmp_path / "model.bin"
    os.mkfifo(path)
    code = '''
import sys
from acprof.container.download_model import verify_download
from acprof.container.model_files import ModelFilesError, plan_download
plan = plan_download(model_id="example/fifo", revision="a" * 40, family="nlp",
    backend="transformers_pipeline", policy="full",
    files={"model.bin": {"size": 0}}, read_json=lambda _: {})
try:
    verify_download(sys.argv[1], plan)
except ModelFilesError as exc:
    assert "regular file" in str(exc)
else:
    raise AssertionError("FIFO was accepted as a model payload")
'''
    completed = subprocess.run([sys.executable, "-c", code, str(tmp_path)],
                               text=True, capture_output=True, timeout=5)
    assert completed.returncode == 0, completed.stderr
