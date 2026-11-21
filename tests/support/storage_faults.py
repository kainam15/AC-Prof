"""Inject durability failures at the filesystem boundary, after file writes."""
import os
import stat
from pathlib import Path


def fail_directory_sync(monkeypatch, published_path, original_content, first_error, *, later_error=None):
    """Fail publication sync once; optionally fail every recovery directory sync."""
    original = os.fsync
    interrupted = False

    def sync(descriptor):
        nonlocal interrupted
        if (stat.S_ISDIR(os.fstat(descriptor).st_mode)
                and (interrupted or published_path.read_bytes() != original_content)):
            if not interrupted:
                interrupted = True
                raise first_error
            if later_error is not None:
                raise later_error
        return original(descriptor)

    monkeypatch.setattr(os, "fsync", sync)


def fail_publication(monkeypatch, path, error, *, attempt, repeat=False):
    """Count both exclusive first publication and replacement of an owned report."""
    writes = []

    def checked(operation):
        def publish(source, destination, *args, **kwargs):
            if Path(destination) == path:
                writes.append(path)
                if len(writes) == attempt or (repeat and len(writes) > attempt):
                    raise error
            return operation(source, destination, *args, **kwargs)
        return publish

    monkeypatch.setattr(os, "link", checked(os.link))
    monkeypatch.setattr(os, "replace", checked(os.replace))
    return writes
