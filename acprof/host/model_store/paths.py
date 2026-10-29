"""Directory-anchored filesystem access for Model Store garbage collection."""
from __future__ import annotations

import errno
import os
import shutil
import stat
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from fnmatch import fnmatchcase
from pathlib import Path

_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


class StorePaths:
    def __init__(self, root: Path, descriptor: int, check_cancelled: Callable[[], None]):
        self.root = root
        self.descriptor = descriptor
        self.check_cancelled = check_cancelled

    @staticmethod
    def _relative(path: Path) -> Path:
        if path.is_absolute() or '..' in path.parts:
            raise ValueError(f'Model Store cleanup path must stay inside its root: {path}')
        return path

    @contextmanager
    def directory(self, relative: Path) -> Iterator[int]:
        relative = self._relative(relative)
        with ExitStack() as stack:
            descriptor = self.descriptor
            try:
                for part in relative.parts:
                    descriptor = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
                    stack.callback(os.close, descriptor)
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ValueError(f'Model Store directory is a symlink or non-directory: {relative}') from exc
                raise
            yield descriptor

    def check_root(self) -> None:
        info = self.root.lstat()
        if not stat.S_ISDIR(info.st_mode) or not os.path.samestat(info, os.fstat(self.descriptor)):
            raise ValueError('Model Store root changed or became a symlink during cleanup')

    def scan(self, relative: Path, *, missing_ok: bool = False) -> Iterator[tuple[Path, os.stat_result]]:
        with ExitStack() as stack:
            try:
                descriptor = stack.enter_context(self.directory(relative))
            except FileNotFoundError:
                if missing_ok:
                    return  # Optional cache categories may not exist yet.
                raise
            with os.scandir(descriptor) as entries:
                for entry in entries:
                    self.check_cancelled()
                    yield relative / entry.name, entry.stat(follow_symlinks=False)

    def directories(self, relative: Path, pattern: str = '*', *, missing_ok: bool = False) -> Iterator[Path]:
        for path, info in self.scan(relative, missing_ok=missing_ok):
            if not fnmatchcase(path.name, pattern):
                continue
            if stat.S_ISLNK(info.st_mode):
                raise ValueError(f'Model Store directory must not be a symlink: {path}')
            if stat.S_ISDIR(info.st_mode):
                yield path

    def walk(self, relative: Path, *, missing_ok: bool = False) -> Iterator[tuple[Path, os.stat_result]]:
        with ExitStack() as stack:
            try:
                descriptor = stack.enter_context(self.directory(relative))
            except FileNotFoundError:
                if missing_ok:
                    return
                raise
            yield from self._walk(relative, descriptor)

    def _walk(self, relative: Path, descriptor: int) -> Iterator[tuple[Path, os.stat_result]]:
        with os.scandir(descriptor) as entries:
            for entry in entries:
                self.check_cancelled()
                path = relative / entry.name
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    child = os.open(entry.name, _DIRECTORY_FLAGS, dir_fd=descriptor)
                    try:
                        if not os.path.samestat(info, os.fstat(child)):
                            raise OSError(errno.ESTALE, 'Model Store directory changed during cleanup', str(path))
                        yield from self._walk(path, child)
                    finally:
                        os.close(child)
                else:
                    if stat.S_ISLNK(info.st_mode):
                        try:
                            target = os.stat(entry.name, dir_fd=descriptor)
                        except FileNotFoundError:
                            target = None
                        if target is not None and stat.S_ISDIR(target.st_mode):
                            raise ValueError(f'Model Store directory must not be a symlink: {path}')
                    yield path, info

    def file_stat(self, relative: Path) -> os.stat_result:
        self._relative(relative)
        with self.directory(relative.parent) as descriptor:
            return os.stat(relative.name, dir_fd=descriptor, follow_symlinks=False)

    def validate_deletions(self, entries: list[Path], blobs: list[Path]) -> None:
        self.check_root()
        for entry in entries:
            self.check_cancelled()
            with self.directory(entry):
                pass
        for blob in blobs:
            self.check_cancelled()
            if not stat.S_ISREG(self.file_stat(blob).st_mode):
                raise ValueError(f'Model Store cleanup file is a symlink or non-regular file: {blob}')
        self.check_root()

    def remove_tree(self, relative: Path) -> None:
        self._relative(relative)
        with self.directory(relative.parent) as descriptor:
            shutil.rmtree(relative.name, dir_fd=descriptor)

    def unlink(self, relative: Path) -> None:
        self._relative(relative)
        with self.directory(relative.parent) as descriptor:
            os.unlink(relative.name, dir_fd=descriptor)

    def unlink_if_broken(self, relative: Path) -> None:
        self._relative(relative)
        with self.directory(relative.parent) as descriptor:
            if not stat.S_ISLNK(os.stat(relative.name, dir_fd=descriptor, follow_symlinks=False).st_mode):
                return
            try:
                os.stat(relative.name, dir_fd=descriptor)
            except FileNotFoundError:
                os.unlink(relative.name, dir_fd=descriptor)


@contextmanager
def store_paths(root: Path, check_cancelled: Callable[[], None]) -> Iterator[StorePaths]:
    # The configured root may itself be a symlink. All descendants are opened
    # one component at a time with O_NOFOLLOW, anchored to this directory.
    root = root.resolve()
    descriptor = os.open(root, _DIRECTORY_FLAGS)
    try:
        yield StorePaths(root, descriptor, check_cancelled)
    finally:
        os.close(descriptor)
