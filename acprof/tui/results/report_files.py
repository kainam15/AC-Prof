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
    from acprof.results.layout import resolve_single_result_source
    first = inputs[0]
    if first.is_dir():
        first = resolve_single_result_source(first)
    layout = ArtifactLayout.discover(first) if first.is_dir() else ArtifactLayout.from_csv(first)
    candidates = [layout.root / "report.html", *layout.path("analysis").glob("report*.html")]
    files = [path for path in candidates if path.is_file()]
    return max(files, key=lambda path: path.stat().st_mtime_ns) if files else None


def html_report_state(report: Path, source: Path, digest: str, *, cancelled=None) -> str:
    """Verify embedded provenance in the background; legacy HTML stays unknown.

    Read only up to the report-data script, with a bounded document budget.
    Additional report inputs are verified too, so a combined report cannot be
    labelled current solely because its first experiment still matches.
    """
    import json
    from html.parser import HTMLParser

    from acprof.artifacts import file_sha256
    from acprof.cancellation import check_cancelled
    from acprof.results.csv import verify_result_source

    class SourcesParser(HTMLParser):
        collecting = False
        done = False

        def __init__(self):
            super().__init__()
            self.parts: list[str] = []

        def handle_starttag(self, tag, attrs):
            if tag == "script" and dict(attrs).get("id") == "report-data":
                self.collecting = True

        def handle_data(self, data):
            if self.collecting:
                self.parts.append(data)

        def handle_endtag(self, tag):
            if tag == "script" and self.collecting:
                self.collecting = False
                self.done = True

    parser = SourcesParser()
    try:
        with report.open(encoding="utf-8") as stream:
            for _ in range(1024):  # 64 MiB; larger historical reports remain unverified.
                check_cancelled(cancelled)
                chunk = stream.read(65536)
                if not chunk:
                    break
                parser.feed(chunk)
                if parser.done:
                    break
        if not parser.done:
            return "unverified"
        payload = json.loads("".join(parser.parts))
        sources = payload.get("sources") if isinstance(payload, dict) else None
        if not isinstance(sources, list) or not sources:
            return "unverified"
        matched = False
        for item in sources:
            check_cancelled(cancelled)
            if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item.get("sha256"):
                return "unverified"
            path = Path(item["path"])
            if not path.is_absolute():
                return "unverified"
            if path.resolve() == source.resolve():
                matched = True
                if item["sha256"] != digest:
                    return "stale"
            else:
                verify_result_source(path, item["sha256"], cancelled=cancelled)
            fingerprints = item.get("metadata_sha256")
            if not isinstance(fingerprints, dict):
                return "unverified"
            for name, expected in fingerprints.items():
                check_cancelled(cancelled)
                artifact = Path(name)
                # Report provenance is data, never authority to inspect arbitrary files.
                if not artifact.is_absolute() or not artifact.resolve().is_relative_to(path.parent.resolve()):
                    return "unverified"
                if (file_sha256(artifact) if artifact.is_file() else None) != expected:
                    return "stale"
        return "current" if matched else "stale"
    except json.JSONDecodeError:
        return "damaged"
    except (OSError, ValueError):
        return "stale" if parser.done else "damaged"


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
