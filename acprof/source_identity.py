"""Explicit source scopes shared by resume identity and the service build context."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
from typing import Iterable


_SERVICE_DIRECTORIES = {"container", "extensions", "workloads"}
_PRESENTATION_DIRECTORIES = {"tui", "plotting", "analysis"}
_PRESENTATION_COMMANDS = {"tui.py", "plot.py", "stats.py", "audit.py", "compare.py"}


def _package_sources(root: Path) -> list[Path]:
    package = root / "acprof"
    return sorted(path for path in package.rglob("*") if path.is_file()
                  and path.suffix in {".py", ".json"}
                  and not {"_bundle", "__pycache__"}.intersection(path.relative_to(package).parts))


def measurement_sources(root: str | Path) -> list[Path]:
    root = Path(root)
    paths = []
    for path in _package_sources(root):
        parts = path.relative_to(root / "acprof").parts
        if parts[0] in _PRESENTATION_DIRECTORIES:
            continue
        if parts[0] == "cli" and path.name in _PRESENTATION_COMMANDS:
            continue
        paths.append(path)
    # The built-in audio fixture and its provenance affect generated inputs.
    paths.extend(path for path in (root / "assets").rglob("*") if path.is_file())
    return sorted(paths)


def service_sources(root: str | Path) -> list[Path]:
    root = Path(root)
    return [path for path in _package_sources(root)
            if len(path.relative_to(root / "acprof").parts) == 1
            or path.relative_to(root / "acprof").parts[0] in _SERVICE_DIRECTORIES]


def source_fingerprint(root: str | Path, paths: Iterable[Path], *, scope: str) -> str:
    root = Path(root)
    digest = hashlib.sha256(scope.encode())
    for path in sorted(paths):
        name = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def service_context_files(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted([*service_sources(root), root / "dockerfiles/runtime-final.Dockerfile",
                   root / "LICENSE", root / "NOTICE", *sorted((root / "licenses").rglob("*.txt"))])


def stage_service_context(root: str | Path, destination: str | Path) -> None:
    root, destination = Path(root), Path(destination)
    for source in service_context_files(root):
        target = destination / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
