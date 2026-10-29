"""Build and inspect immutable loader views without choosing cache or lease policy."""
from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
from pathlib import Path

import acprof.artifacts as artifact_io
from acprof.container.model_files import PLAN_FILENAME
from acprof.host.model_store_paths import StorePaths, store_paths


def snapshot_path(root: Path, key: str, repo: dict) -> Path:
    return root / 'entries' / key / 'hf' / ('models--' + repo['model_id'].replace('/', '--')) / 'snapshots' / repo['model_revision']


def _ref_matches(paths: StorePaths, relative: Path, revision: str) -> bool:
    info = paths.file_stat(relative)
    if not stat.S_ISREG(info.st_mode) or info.st_size != len(revision):
        return False
    with paths.directory(relative.parent) as directory:
        descriptor = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=directory)
        with os.fdopen(descriptor, 'rb', buffering=0) as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                return False
            return stream.read(len(revision) + 1) == revision.encode('ascii')


def view_complete(root: Path, key: str, repositories: list[dict]) -> bool:
    """Inspect layout and refs only; the caller separately verifies payload hashes."""
    try:
        with store_paths(root, lambda: None) as paths:
            for repo in repositories:
                snapshot = snapshot_path(Path(), key, repo)
                if not _ref_matches(paths, snapshot.parent.parent / 'refs/main', repo['model_revision']):
                    return False
                for record in repo['files']:
                    relative = snapshot / record['path']
                    if not stat.S_ISLNK(paths.file_stat(relative).st_mode):
                        return False
                    with paths.directory(relative.parent) as directory:
                        target = Path(os.readlink(relative.name, dir_fd=directory))
                    if target.is_absolute():
                        return False
                    absolute = Path(os.path.abspath(paths.root / relative.parent / target))
                    resolved = absolute.resolve(strict=True)
                    if absolute != resolved or not any(resolved.is_relative_to(paths.root / category)
                                                       for category in ('hf', 'modelscope')):
                        return False
                    info = resolved.stat()
                    if not stat.S_ISREG(info.st_mode):
                        return False
                    if info.st_size != record['size']:
                        raise ValueError(f"Model Store view file size mismatch: {record['path']}")
            paths.check_root()
        return True
    except FileNotFoundError:
        return False


def make_traversable(path: Path) -> None:
    """Grant directory search without exposing listings or relaxing file modes."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        mode = stat.S_IMODE(os.fstat(descriptor).st_mode)
        if mode & 0o111 != 0o111:
            os.fchmod(descriptor, mode | 0o111)
    finally:
        os.close(descriptor)


def _write_text(stream, content: str) -> None:
    stream.write(content)


def _sync_directory_tree(root: Path) -> None:
    directories = [root, *(path for path in root.rglob('*') if path.is_dir() and not path.is_symlink())]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        artifact_io.sync_directory(directory)


def _publish_entry(temporary: Path, destination: Path) -> None:
    previous = None
    if destination.exists():
        previous = Path(tempfile.mkdtemp(prefix='.preparing-', dir=destination.parent))
        try:
            destination.rename(previous)
        except BaseException:
            previous.rmdir()
            raise
    try:
        temporary.rename(destination)
    except BaseException:
        if previous is not None:
            previous.rename(destination)
            artifact_io.sync_directory(destination.parent)
        raise
    # Keep the previous view available for explicit cleanup if durability is uncertain.
    artifact_io.sync_directory(destination.parent)
    if previous is not None:
        shutil.rmtree(previous)


def publish_view(root: Path, key: str, plan: dict, repositories: list[dict], targets: list[Path]) -> None:
    """Caller owns the store lock and exclusive entry lease until publication completes."""
    root = root.resolve()
    entries = root / 'entries'
    entries.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.preparing-', dir=entries) as directory:
        temporary = Path(directory)
        for repo, source in zip(repositories, targets, strict=True):
            cache = temporary / 'hf' / ('models--' + repo['model_id'].replace('/', '--'))
            snapshot = cache / 'snapshots' / repo['model_revision']
            for record in repo['files']:
                link = snapshot / record['path']
                link.parent.mkdir(parents=True, exist_ok=True)
                blob = (source / record['path']).resolve()
                if not blob.is_relative_to(root):
                    raise ValueError('Model Store snapshot escaped its store root')
                link.symlink_to(os.path.relpath(blob, link.parent))
            (cache / 'refs').mkdir(exist_ok=True)
            revision = repo['model_revision']
            artifact_io.atomic_write(cache / 'refs/main', lambda stream: _write_text(stream, revision))
        payload = json.dumps(plan, ensure_ascii=False, indent=2) + '\n'
        artifact_io.atomic_write(temporary / PLAN_FILENAME, lambda stream: _write_text(stream, payload))
        make_traversable(temporary)
        _sync_directory_tree(temporary)
        # Keep both views at the same depth, preserving relative payload links.
        _publish_entry(temporary, entries / key)
