"""Share metadata observations within one read-only result operation."""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

from acprof.artifacts import (
    JsonObjectSnapshot,
    read_json_object_snapshot,
    recorded_artifact_present,
)
from acprof.cancellation import cancellable, check_cancelled
from acprof.hardware_conditions import decode_hardware_conditions
from acprof.results.layout import ArtifactLayout, resolve_result_file


def _same_error(previous: object, current: BaseException) -> bool:
    return type(previous) is type(current) and str(previous) == str(current)


def _resolved(path: Path) -> Path:
    try:
        return path.resolve()
    except RuntimeError as error:  # Python 3.12 raises RuntimeError on symlink loops.
        raise ValueError(f"cannot resolve metadata source {path}: {error}") from error


def _matches_content(path: Path, content: bytes, cancelled: Callable[[], bool] | None) -> bool:
    """Compare up to the retained length and one overflow byte, allowing short raw reads."""
    with path.open("rb", buffering=0) as stream, memoryview(content) as expected:
        offset = 0
        while True:
            check_cancelled(cancelled)
            chunk = stream.read(min(64 * 1024, len(expected) - offset + 1))
            check_cancelled(cancelled)
            if not chunk:
                return chunk == b"" and offset == len(expected)
            end = offset + len(chunk)
            if chunk != expected[offset:end]:
                return False
            offset = end


class ResultMetadataSnapshot:
    """Retain the first value, absence or read failure for each artifact.

    Payloads are shared read-only. Create a new instance for each operation;
    this is not an atomic snapshot across files or a replacement for the
    independent comparison's initial and final source fingerprints.
    """

    def __init__(self, source: str | Path, *, cancelled: Callable[[], bool] | None = None):
        self.result_path = resolve_result_file(source, cancelled=cancelled)
        self.source = _resolved(self.result_path)
        self.layout = ArtifactLayout.from_csv(self.result_path, cancelled=cancelled)
        self._metadata_root = _resolved(self.layout.root)
        self._presence: dict[str, bool | OSError | ValueError] = {}
        self._records: dict[str, JsonObjectSnapshot | OSError | ValueError] = {}
        self._failed_labels: dict[str, str] = {}

    def validate_source(self, source: str | Path, *, cancelled: Callable[[], bool] | None = None) -> Path:
        """Return the selected file after checking that it matches this snapshot."""
        path = resolve_result_file(source, cancelled=cancelled)
        layout = ArtifactLayout.from_csv(path, cancelled=cancelled)
        if _resolved(path) != self.source or _resolved(layout.root) != self._metadata_root:
            raise ValueError(f"result metadata snapshot belongs to a different path: {source}")
        if layout.layout_version != self.layout.layout_version:
            raise ValueError(f"result metadata layout changed: {path}")
        return path

    def _observe_presence(self, name: str) -> bool | OSError | ValueError:
        try:
            return recorded_artifact_present(self.layout.path(name))
        except (OSError, ValueError) as error:
            return error

    def exists(self, name: str) -> bool:
        if name not in self._presence:
            self._presence[name] = self._observe_presence(name)
        present = self._presence[name]
        if isinstance(present, (OSError, ValueError)):
            raise present
        return present

    def read(self, name: str, *, label: str | None = None,
             cancelled: Callable[[], bool] | None = None) -> JsonObjectSnapshot:
        check_cancelled(cancelled)
        if name not in self._records:
            try:
                if not self.exists(name):
                    raise FileNotFoundError(f"metadata artifact missing: {name}")
                self._records[name] = self._read_record(name, label or name, cancelled)
            except (OSError, ValueError) as error:
                self._records[name] = error
                self._failed_labels[name] = label or name
        record = self._records[name]
        if isinstance(record, (OSError, ValueError)):
            raise record
        return record

    def _read_record(self, name: str, label: str, cancelled: Callable[[], bool] | None) -> JsonObjectSnapshot:
        path = self.layout.path(name)
        decode = decode_hardware_conditions if path.name == "hardware_conditions.json" else None
        return read_json_object_snapshot(path, label=label, cancelled=cancelled, decode=decode,
                                         preserve_input_scales=path.name == "input_scale_plan.json")

    def verify_current(self, source: str | Path, *, cancelled: Callable[[], bool] | None = None) -> None:
        """Raise the first verification error for callers requiring a stable source."""
        error = next(self.verification_errors(source, cancelled=cancelled), None)
        if error is not None:
            raise error

    def verification_errors(self, source: str | Path, *, cancelled: Callable[[], bool] | None = None,
                            ) -> Iterator[OSError | ValueError]:
        """Check selection, then each observed artifact in its original read order."""
        try:
            self.validate_source(source, cancelled=cancelled)
        except (OSError, ValueError) as error:
            check_cancelled(cancelled)
            yield error
            return
        for name, present in cancellable(self._presence.items(), cancelled):
            try:
                self._verify_observation(name, present, cancelled)
            except (OSError, ValueError) as error:
                check_cancelled(cancelled)
                yield error
        check_cancelled(cancelled)

    def _verify_observation(self, name: str, present: bool | OSError | ValueError,
                            cancelled: Callable[[], bool] | None) -> None:
        current = self._observe_presence(name)
        if isinstance(current, (OSError, ValueError)):
            if _same_error(present, current):
                return
            raise ValueError(f"metadata changed: {name}: {current}") from current
        if current is not present:
            raise ValueError(f"metadata presence changed: {name}")
        record = self._records.get(name)
        if current and record is not None and not self._record_unchanged(name, record, cancelled):
            raise ValueError(f"metadata changed: {self.layout.path(name)}")

    def _record_unchanged(self, name: str, record: JsonObjectSnapshot | OSError | ValueError,
                          cancelled: Callable[[], bool] | None) -> bool:
        path = self.layout.path(name)
        check_cancelled(cancelled)
        if isinstance(record, JsonObjectSnapshot):
            return _matches_content(path, record.content, cancelled)
        try:
            self._read_record(name, self._failed_labels[name], cancelled)
        except (OSError, ValueError) as error:
            if _same_error(record, error):
                return True
            raise ValueError(f"metadata changed: {path}: {error}") from error
        return False

    def verify_observed_sources(self, fingerprints: Mapping[Path, str | None]) -> None:
        """Bind successfully parsed bytes to the caller's initial fingerprints.

        Read failures remain cached and reported by the consumers; they supply
        no parsed evidence to verify. This does not reopen documents or replace
        the caller's final check of every source, including absent files.
        """
        for name, record in self._records.items():
            if isinstance(record, JsonObjectSnapshot):
                path = self.layout.path(name)
                if path not in fingerprints or hashlib.sha256(record.content).hexdigest() != fingerprints[path]:
                    raise ValueError(f"metadata content differs from its initial fingerprint: {path}")

    def observed_fingerprints(self, *, cancelled: Callable[[], bool] | None = None) -> dict[Path, str | None]:
        """Identify parsed bytes and observed absences for later source checks.

        Failed reads supply diagnostics, not parsed evidence or a fingerprint.
        """
        fingerprints: dict[Path, str | None] = {
            self.layout.path(name): None for name, present in cancellable(self._presence.items(), cancelled)
            if present is False
        }
        for name, record in cancellable(self._records.items(), cancelled):
            if isinstance(record, JsonObjectSnapshot):
                fingerprints[self.layout.path(name)] = hashlib.sha256(record.content).hexdigest()
        return fingerprints
