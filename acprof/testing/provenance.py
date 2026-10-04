"""Content identities for test evidence, independent of Git or checkout location."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from acprof.source_identity import source_fingerprint

ROOT = Path(__file__).resolve().parents[2]
_IGNORED = {"__pycache__", "_bundle", ".git", ".venv", ".pytest_cache", ".mypy_cache"}


def _raise_walk_error(error: OSError) -> None:
    raise error


def _files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    paths: list[Path] = []
    for directory, children, files in os.walk(root, followlinks=False, onerror=_raise_walk_error):
        children[:] = sorted(name for name in children if name not in _IGNORED)
        paths.extend(Path(directory) / name for name in sorted(files)
                     if not name.endswith((".pyc", ".pyo"))
                     and name != ".env" and not name.startswith(".env."))
    return paths


def capture_provenance(test_paths: Iterable[Path], *, root: Path = ROOT) -> dict:
    """Hash explicit project inputs and external fixture suites before/after execution."""
    root = root.resolve()
    sources = [path for name in ("acprof", "scripts", "packaging", "dockerfiles", "assets",
                                 "docs", "examples", ".github/workflows") for path in _files(root / name)]
    sources.extend(path for name in ("pyproject.toml", "README.md", "setup.sh", ".gitignore")
                   if (path := root / name).is_file())
    locks = _files(root / "requirements")
    tests = _files(root / "tests")
    external = sorted({path.resolve().parent for path in test_paths
                       if not path.resolve().is_relative_to(root)})
    # Relative names and bytes survive independent checkout/container mount paths.
    external_roots = [path for path in external
                      if not any(path != parent and path.is_relative_to(parent) for parent in external)]
    external_hashes = [source_fingerprint(path, [file for file in _files(path) if file.suffix == ".py"],
                                         scope="external-test-source-v1") for path in external_roots]
    return {
        "schema_version": 1,
        "source_sha256": source_fingerprint(root, sources, scope="test-project-source-v1"),
        "tests_sha256": source_fingerprint(root, tests, scope="test-suite-source-v1:" + ":".join(external_hashes)),
        "locks_sha256": source_fingerprint(root, locks, scope="test-lock-source-v1"),
    }
