"""Share metadata observations within one read-only result operation."""
from __future__ import annotations

from pathlib import Path

from acprof.artifacts import JsonObjectSnapshot, read_json_object_snapshot
from acprof.results.layout import ArtifactLayout


class ResultMetadataSnapshot:
    """Retain the first value, absence or read failure for each artifact.

    Payloads are shared read-only. Create a new instance for each operation;
    this is not an atomic snapshot across files or a replacement for the
    independent comparison's initial and final source fingerprints.
    """

    def __init__(self, source: str | Path):
        path = self._result_path(source)
        self.source = path.resolve()
        self.layout = ArtifactLayout.from_csv(path)
        self._presence: dict[str, bool | OSError | ValueError] = {}
        self._records: dict[str, JsonObjectSnapshot | OSError | ValueError] = {}

    @staticmethod
    def _result_path(source: str | Path) -> Path:
        path = Path(source).expanduser()
        return path / "result_layers.json" if path.is_dir() else path

    def validate_source(self, source: str | Path) -> None:
        if self._result_path(source).resolve() != self.source:
            raise ValueError("result metadata snapshot belongs to a different path")

    def exists(self, name: str) -> bool:
        if name not in self._presence:
            try:
                self._presence[name] = self.layout.path(name).exists()
            except (OSError, ValueError) as error:
                self._presence[name] = error
        present = self._presence[name]
        if isinstance(present, (OSError, ValueError)):
            raise present
        return present

    def read(self, name: str, *, label: str | None = None) -> JsonObjectSnapshot:
        if name not in self._records:
            try:
                if not self.exists(name):
                    raise FileNotFoundError(f"metadata artifact missing: {name}")
                self._records[name] = read_json_object_snapshot(self.layout.path(name), label=label or name)
            except (OSError, ValueError) as error:
                self._records[name] = error
        record = self._records[name]
        if isinstance(record, (OSError, ValueError)):
            raise record
        return record
