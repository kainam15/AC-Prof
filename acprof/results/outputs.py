"""Keep derived outputs out of authoritative paths of their input experiments."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from acprof.results.layers import GENERATIONS_DIR, LAYER_FILES
from acprof.results.layout import (
    DIRECT_EXPERIMENT_FILES,
    METADATA_FILES,
    ArtifactLayout,
    expand_result_sources,
)

# Missing evidence and selectors are meaningful. Creating a directory with the
# name of a file is just as disruptive as publishing an unrelated file there.
_RESERVED = frozenset(Path(name) for name in (
    *DIRECT_EXPERIMENT_FILES, *METADATA_FILES, *LAYER_FILES.values(), GENERATIONS_DIR,
    "quality_checks.json", "hardware_conditions.json", "load.json", "sweep.json",
    ".acprof", ".acprof-result.lock", "result.lock", "metadata/task_runs",
    "metadata/preparation", "metadata/hardware_conditions.json", *(f"metadata/{name}" for name in METADATA_FILES),
))
_RESERVED_NAMES = frozenset(path.name for path in _RESERVED)


def _entry_roots(entry: Path) -> set[Path]:
    if entry.is_dir():
        return {entry.resolve()}
    # An explicit CSV alias can have evidence alongside both names. Archived
    # exports may refer to input files that can no longer be opened.
    return {ArtifactLayout.root_for_csv(path).resolve() for path in (entry, entry.resolve())}


def _source_roots(sources: Iterable[str | Path], candidates: set[Path],
                  expanded_sources: Iterable[Path] | None) -> set[Path]:
    roots: set[Path] = set()
    for source in sources:
        entry = Path(source).expanduser()
        initial = _entry_roots(entry)
        roots.update(initial)
        if expanded_sources is None and any(path.is_relative_to(root) for path in candidates for root in initial):
            for child in expand_result_sources([entry]):
                if child != entry:
                    roots.update(_entry_roots(child))
    for child in expanded_sources or ():
        roots.update(_entry_roots(child))
    # A parent output directory may have expanded into children of several collections.
    for root in tuple(roots):
        if root.parent.parent.name == "tasks":
            collection = root.parent.parent.parent
            if (collection / "metadata" / "task_runs").is_dir():
                roots.add(collection)
    return roots


def output_ancestors(destination: Path) -> Iterator[Path]:
    """Paths which a writer or its recursive mkdir can occupy."""
    final = destination.resolve()
    yield final
    yield from final.parents
    # mkdir(parents=True) may create a prefix before '..' even when it is
    # absent from the resolved destination (result_manifest.json/../reports/).
    for part in destination.parents:
        if part.name == "..":
            prefix = part.parent.resolve()
            yield prefix
            yield from prefix.parents


def _output_entry(path: Path) -> Path:
    # mkdir may make missing/../link reachable. Resolve its parent separately
    # so that a dangling link at the entry itself remains visible.
    return path.parent.resolve() / path.name if ".." in path.parts else path


def validate_comparison_destinations(destinations: Iterable[str | Path]) -> None:
    """Check explicit exports and generated sidecars as one set of new files."""
    files: set[Path] = set()
    directories: set[Path] = set()
    checked: set[Path] = set()
    for destination in destinations:
        path = Path(destination).absolute()
        for parent in reversed(path.parents):
            if parent in checked:
                continue
            entry = _output_entry(parent)
            if (entry.is_symlink() or entry.exists()) and not entry.is_dir():
                raise ValueError(f"output parent must be a directory: {entry}")
            checked.add(parent)
        entry = _output_entry(path)
        if entry.is_symlink() or entry.exists():
            raise ValueError(f"output must be a new file: {entry}")
        paths = output_ancestors(path)
        final = next(paths)
        if final in files:
            raise ValueError(f"output destinations must be distinct: {final}")
        files.add(final)
        directories.update(paths)
    if conflicts := files.intersection(directories):
        raise ValueError(f"output destination is also required as a directory: {min(conflicts)}")


def validate_analysis_destinations(sources: Iterable[str | Path], destinations: Iterable[str | Path], *,
                                   wide_export: Path | None = None,
                                   expanded_sources: Iterable[Path] | None = None) -> None:
    """Protect selectors and evidence, even when publishing to currently absent names.

    Already expanded sources preserve each consumer's collection discovery rules.
    Only the wide CSV exporter can produce result_all.csv next to its own layers.
    """
    candidates = set()
    allowed = wide_export.resolve() if wide_export is not None else None
    for destination in destinations:
        paths = output_ancestors(Path(destination))
        final = next(paths)
        if final != allowed and final.name in _RESERVED_NAMES:
            candidates.add(final)
        candidates.update(path for path in paths if path.name in _RESERVED_NAMES)
    if not candidates:
        return
    for root in _source_roots(sources, candidates, expanded_sources):
        for path in candidates:
            if path.is_relative_to(root) and path.relative_to(root) in _RESERVED:
                raise ValueError(f"output cannot create or overwrite a reserved experiment artifact: {path}")
