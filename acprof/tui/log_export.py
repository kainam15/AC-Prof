"""Manual TUI exports, independent of experiment artifacts and config storage."""
from __future__ import annotations

import errno
from datetime import datetime
from itertools import count
from pathlib import Path
from threading import Event

from acprof.tui.i18n import error_message, message_id


def default_log_export_dir() -> Path:
    return Path.home() / "AC-Prof" / "exports" / "logs"


def default_log_export_filename() -> str:
    return datetime.now().strftime("acprof-tui-%Y-%m-%d_%H-%M-%S.log")


def export_directory(value: str) -> Path:
    """Validate syntax without probing or creating a directory."""
    if not value.strip() or "\0" in value:
        raise ValueError(message_id('logs.export.absolute_directory_required'))
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        raise ValueError(message_id('logs.export.absolute_directory_required'))
    return path


def log_export_path(directory: str, filename: str) -> Path:
    path = export_directory(directory)
    name = filename.strip()
    if not name or name in {".", ".."} or any(char in name for char in ("/", "\\", "\0")):
        raise ValueError(message_id('logs.export.invalid_filename'))
    return path / name


def write_log_export(path: Path, text: str) -> Path:
    """Exclusively create a UTF-8 file; collisions get a readable suffix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for number in count():
        candidate = path if not number else path.with_name(f"{path.stem}-{number}{path.suffix}")
        try:
            stream = candidate.open("x", encoding="utf-8", newline="")
        except FileExistsError:
            continue
        try:
            with stream:
                stream.write(text)
        except (OSError, UnicodeError):
            # Remove only the file this attempt created, never a collision.
            candidate.unlink()
            raise
        return candidate
    raise AssertionError("unreachable")


def list_export_directories(
    directory: Path, *, nearest_parent: bool, cancelled: Event,
) -> tuple[Path, tuple[Path, ...]]:
    """Read one level in a worker; surface permission and filesystem errors."""
    while True:
        try:
            entries = []
            for path in directory.iterdir():
                if cancelled.is_set():
                    return directory, ()
                if path.is_dir():
                    entries.append(path)
            return directory, tuple(sorted(entries, key=lambda path: path.name.casefold()))
        except FileNotFoundError:
            if not nearest_parent or directory == directory.parent:
                raise
            directory = directory.parent


def log_export_error(error: Exception) -> str:
    if isinstance(error, OSError):
        if error.errno in {errno.EACCES, errno.EPERM}:
            return message_id('logs.export.permission_denied', str(error))
        if error.errno == errno.ENOSPC:
            return message_id('logs.export.disk_full', str(error))
    return error_message(error)
