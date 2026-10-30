"""Explicit file actions for completed HTML artifacts; no analysis or inference."""
from __future__ import annotations

import errno
import shutil
import webbrowser
from pathlib import Path
from typing import BinaryIO

from acprof.artifacts import atomic_write_binary
from acprof.results.layout import ArtifactLayout
from acprof.tui.i18n import message_id


def existing_html_report(inputs: tuple[Path, ...]) -> Path | None:
    """Find the newest conventional report for the first selected experiment."""
    first = inputs[0]
    layout = ArtifactLayout.discover(first) if first.is_dir() else ArtifactLayout.from_csv(first)
    candidates = [layout.root / "report.html", *layout.path("analysis").glob("report*.html")]
    files = [path for path in candidates if path.is_file()]
    return max(files, key=lambda path: path.stat().st_mtime_ns) if files else None


def require_html_file(path: Path) -> None:
    if path.suffix.lower() != ".html" or not path.is_file():
        raise ValueError(message_id('results.html.file_missing', path))


def open_html_file(path: Path) -> None:
    require_html_file(path)
    if not webbrowser.open(path.resolve().as_uri(), new=2):
        raise RuntimeError(message_id('results.html.browser_unavailable'))


def export_html_file(source: Path, destination: Path) -> None:
    """Copy a complete offline report, refusing every existing destination."""
    require_html_file(source)
    if destination.suffix.lower() != ".html":
        raise ValueError(message_id('results.html.invalid_export_extension'))
    from acprof.host.run_state import offline_analysis_lock

    with offline_analysis_lock():
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(errno.EEXIST, "File exists", str(destination))

        def copy(target: BinaryIO) -> None:
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, target, length=1024 * 1024)

        atomic_write_binary(destination, copy, default_mode=0o600, overwrite=False)
