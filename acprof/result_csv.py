"""结果 CSV 的结构、测量唯一键与完整性校验，不初始化采集依赖。"""
from __future__ import annotations

import csv
import hashlib
import io
import struct
import sys
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from itertools import product
from pathlib import Path
from typing import cast

from acprof.artifacts import atomic_write
from acprof.config import CSV_FIELDS
from acprof.metric_registry import order_csv_fields

KEY_FIELDS = ("cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", "warmup", "repeat_idx")
MeasurementKey = tuple[str, ...]
# Python 3.12 accepts a C long; later versions accept a Py_ssize_t.
_CSV_FIELD_LIMIT = min(sys.maxsize, (1 << (8 * struct.calcsize("l") - 1)) - 1)


@dataclass(frozen=True)
class ResultCsvSnapshot:
    """One parsed result file together with the digest of its exact source bytes."""

    path: Path
    fields: list[str]
    rows: list[dict[str, str]]
    sha256: str


@dataclass(frozen=True)
class ValidatedResultRows:
    """Rows and keys produced together for one private read.

    Structural-only reads have no measurement keys. These keys are not a
    validation certificate for rows exposed to a consumer and later mutated.
    """

    rows: list[dict[str, str]]
    keys: tuple[MeasurementKey, ...] | None


class ResultValidationError(ValueError):
    """输入不完整或无法唯一归属到计划中的测量。"""


def allow_large_csv_fields() -> None:
    """Accept full diagnostics without lowering the shared interpreter limit.

    csv has no per-reader limit. Do not restore it after a read: another
    concurrent result reader may still be consuming a large field.
    """
    if csv.field_size_limit() < _CSV_FIELD_LIMIT:
        csv.field_size_limit(_CSV_FIELD_LIMIT)


def require_current_fields(fields: Iterable[str]) -> None:
    """拒绝已被明确归因字段替代的旧列，不猜测其测量来源。"""
    retired = {
        "energy_iters", "avg_power_total_w", "peak_power_total_w", "energy_total_j",
        "avg_power_eff_w", "peak_power_eff_w", "energy_eff_j", "model_mflop_per_request",
        "compute_mflops_app", "compute_mflops", "compute_profile_tool", "compute_profile_error",
    }
    unsupported = retired.intersection(fields)
    if unsupported:
        raise ResultValidationError(f"unsupported retired CSV fields: {sorted(unsupported)}; regenerate current results")


def measurement_key(row: Mapping[str, object]) -> MeasurementKey:
    values = []
    for field in KEY_FIELDS:
        value = str(row.get(field, "")).strip()
        if (field in {"cpu_cores", "mem_cap_gb"} and row.get("runtime_kind") == "process"
                and value.lower() in {"", "none", "nan"}):
            values.append("inherited")
            continue
        if field == "gpu_mode":
            if value not in ("off", "on"):
                raise ResultValidationError(f"invalid gpu_mode: {value!r}")
        else:
            try:
                number = Decimal(value)
            except InvalidOperation as exc:
                raise ResultValidationError(f"invalid {field}: {value!r}") from exc
            if not number.is_finite() or number < 0:
                raise ResultValidationError(f"invalid {field}: {value!r}")
            if field in ("cpu_cores", "mem_cap_gb", "input_scale") and number <= 0:
                raise ResultValidationError(f"invalid {field}: {value!r}")
            if field in ("warmup", "repeat_idx") and number != number.to_integral_value():
                raise ResultValidationError(f"invalid {field}: {value!r}")
            if field == "warmup" and number not in (0, 1):
                raise ResultValidationError(f"invalid warmup: {value!r}")
            value = str(number.normalize())
        values.append(value)
    return tuple(values)


def expected_measurements(cpus: Sequence[int | None], mems: Sequence[int | None], gpus: Sequence[str],
                          scales: Sequence[float], warmup: int, repeat: int) -> set[MeasurementKey]:
    return {
        measurement_key({**dict(zip(KEY_FIELDS, (cpu, mem, gpu, str(float(scale)), is_warmup, index), strict=True)),
                         **({"runtime_kind": "process"} if cpu is None or mem is None else {})})
        for cpu, mem, gpu, scale in product(cpus, mems, gpus, scales)
        for is_warmup, count in ((1, warmup), (0, repeat))
        for index in range(count)
    }


_HASH_BUFFER_BYTES = 2**18


def _check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise CancelledError()


def _cancellable_lines(stream: Iterable[str], cancelled: Callable[[], bool]) -> Iterable[str]:
    for line in stream:
        _check_cancelled(cancelled)
        yield line


class _DigestingRawReader(io.RawIOBase):
    """Forward raw reads while hashing the exact bytes consumed by TextIOWrapper."""

    def __init__(self, stream, digest, cancelled: Callable[[], bool] | None = None):
        super().__init__()
        self._stream = stream
        self._digest = digest
        self._cancelled = cancelled

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int | None:
        _check_cancelled(self._cancelled)
        size = self._stream.readinto(buffer)
        if size:
            self._digest.update(memoryview(buffer)[:size])
        return size


def parse_result_stream(stream: Iterable[str], path: Path, expected: Iterable[MeasurementKey] | None, *,
                        allow_empty=False, validate_measurements=True,
                      ) -> tuple[list[str], ValidatedResultRows]:
    allow_large_csv_fields()
    reader = csv.DictReader(stream, strict=True)
    fields = list(reader.fieldnames or [])
    return fields, validate_result_rows(fields, reader, path, expected=expected, allow_empty=allow_empty,
                                       validate_measurements=validate_measurements)


def validate_result_rows(fields: Sequence[str], source_rows: Iterable[dict], path: Path, *,
                         expected: Iterable[MeasurementKey] | None = None, allow_empty=False,
                         validate_measurements=True) -> ValidatedResultRows:
    """Validate parsed rows without modifying a consumer's source snapshot.

    Historical report input can retain its original values with structural
    validation only. Audits and measurement consumers always use the default
    identity, retired-field and diagnostic checks.
    """
    if not validate_measurements and expected is not None:
        raise ValueError("expected measurements require measurement validation")
    if validate_measurements:
        require_current_fields(fields)
    if not fields or len(fields) != len(set(fields)):
        raise ResultValidationError(f"missing or duplicate CSV columns: {path}")
    missing = set(KEY_FIELDS) - set(fields)
    if validate_measurements and missing:
        raise ResultValidationError(f"missing identity columns {sorted(missing)}: {path}")
    rows = []
    keys: dict[MeasurementKey, None] = {}
    for index, row in enumerate(source_rows, 2):
        if None in row or any(value is None for value in row.values()):
            raise ResultValidationError(f"malformed CSV row: {path}:{index}")
        if validate_measurements:
            try:
                key = measurement_key(row)
            except ResultValidationError as exc:
                raise ResultValidationError(f"{path}:{index}: {exc}") from exc
            if key in keys:
                raise ResultValidationError(f"duplicate measurement: {path}:{index}: {key}")
            if row.get("status", "").strip().lower() == "error" and not row.get("error", "").strip():
                raise ResultValidationError(f"status=error without an error diagnostic: {path}:{index}")
            keys[key] = None
            if row.get("environment_class") not in {"native_linux", "wsl2", "vm", "cloud", "container_host", "unknown"}:
                row = {**row, "environment_class": "unknown"}
        rows.append(cast(dict[str, str], row))
    if not rows and not allow_empty:
        raise ResultValidationError(f"empty case CSV (no measurement rows): {path}")
    if expected is not None:
        planned = set(expected)
        if keys.keys() != planned:
            raise ResultValidationError(
                f"measurement plan mismatch: {path}; missing={len(planned - keys.keys())}, "
                f"unexpected={len(keys.keys() - planned)}"
            )
    return ValidatedResultRows(rows, tuple(keys) if validate_measurements else None)


def _file_sha256(path: Path, *, cancelled: Callable[[], bool] | None = None) -> str:
    _check_cancelled(cancelled)
    digest = hashlib.sha256()
    buffer = bytearray(_HASH_BUFFER_BYTES)
    view = memoryview(buffer)
    with path.open("rb", buffering=0) as stream:
        while True:
            _check_cancelled(cancelled)
            size = stream.readinto(buffer)
            if size is None:
                raise BlockingIOError("CSV hash read would block")
            if size == 0:
                break
            digest.update(view[:size])
    return digest.hexdigest()


def read_result_csv(path: str | Path, *, expected: Iterable[MeasurementKey] | None = None
                    ) -> tuple[list[str], list[dict[str, str]]]:
    path = Path(path)
    if path.name == "result_layers.json":
        from acprof.result_layers import read_result_layers
        fields, rows = read_result_layers(path.parent)
        if expected is not None and {measurement_key(row) for row in rows} != set(expected):
            raise ResultValidationError("layered measurement plan mismatch")
        return fields, rows
    with path.open(encoding="utf-8-sig", newline="") as stream:
        fields, validated = parse_result_stream(stream, path, expected)
        return fields, validated.rows


class _CsvRowEncoder:
    """Let csv.writer return one encoded record without retaining a text buffer."""

    @staticmethod
    def write(text: str) -> bytes:
        return text.encode("utf-8")


class _CsvRawReader(io.RawIOBase):
    """Drain private parsed rows through a single encoded record at a time."""

    def __init__(self, fields: list[str], rows: list[dict[str, str]]):
        super().__init__()
        self._rows = deque(rows)
        self._pending = b""
        self._offset = 0
        csv_writer = csv.DictWriter(_CsvRowEncoder(), fieldnames=fields)
        self._writer: csv.DictWriter[str] | None = csv_writer
        try:
            self._pending = csv_writer.writeheader()
        except BaseException:
            self.close()
            raise

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        if self.closed or self._writer is None:
            raise ValueError("I/O operation on closed result view")
        view = memoryview(buffer).cast("B")
        copied = 0
        try:
            while copied < len(view):
                if self._offset == len(self._pending):
                    self._pending = b""
                    self._offset = 0
                    if not self._rows:
                        break
                    self._pending = self._writer.writerow(self._rows.popleft())
                count = min(len(view) - copied, len(self._pending) - self._offset)
                view[copied:copied + count] = self._pending[self._offset:self._offset + count]
                copied += count
                self._offset += count
        except BaseException:
            self.close()
            raise
        return copied

    def close(self) -> None:
        self._rows.clear()
        self._pending = b""
        self._writer = None
        super().close()


def open_result_text(path: str | Path):
    """Open a CSV or a forward-only logical text stream of a layered experiment.

    Layers are verified before returning. Parsed rows are released as consumed;
    only a record and I/O buffers are serialized ahead of the consumer.
    An explicit unbounded read still materializes the requested complete text.
    No complete wide CSV is written to the file system by readers.
    """
    allow_large_csv_fields()
    path = Path(path)
    if path.is_dir():
        path /= "result_layers.json"
    if path.name != "result_layers.json":
        return path.open(encoding="utf-8-sig", newline="")
    fields, rows = read_result_csv(path)
    return io.TextIOWrapper(io.BufferedReader(_CsvRawReader(fields, rows)), encoding="utf-8", newline="")


def read_result_csv_snapshot(path: str | Path, *, expected: Iterable[MeasurementKey] | None = None,
                             validate_measurements: bool = True,
                             cancelled: Callable[[], bool] | None = None,
                             ) -> ResultCsvSnapshot:
    """Parse once while hashing the exact bytes; retain raw historical rows on request.

    Disabling measurement validation only affects flat CSV input. Layered
    storage always verifies its manifest, committed content and row identities.
    Cancellation is checked per raw buffer/line for flat files and around the
    existing verified layer read for manifests.
    """
    _check_cancelled(cancelled)
    if not validate_measurements and expected is not None:
        raise ValueError("expected measurements require measurement validation")
    path = Path(path).resolve()
    if path.name == "result_layers.json":
        from acprof.artifacts import read_json_object
        if read_json_object(path).get("schema_version") == 3:
            from acprof.result_store.reader import read_snapshot
            snapshot = read_snapshot(path.parent)
            if expected is not None and {measurement_key(row) for row in snapshot.rows} != set(expected):
                raise ResultValidationError("layered measurement plan mismatch")
            _check_cancelled(cancelled)
            return snapshot
        from acprof.result_layers import read_result_layers
        layer_digest = _file_sha256(path, cancelled=cancelled)
        fields, rows = read_result_layers(path.parent, expected_manifest_sha256=layer_digest)
        if expected is not None and {measurement_key(row) for row in rows} != set(expected):
            raise ResultValidationError("layered measurement plan mismatch")
        verify_result_source(path, layer_digest, cancelled=cancelled)
        _check_cancelled(cancelled)
        return ResultCsvSnapshot(path=path, fields=fields, rows=rows, sha256=layer_digest)
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as raw:
        buffered = io.BufferedReader(_DigestingRawReader(raw, digest, cancelled), buffer_size=_HASH_BUFFER_BYTES)
        with io.TextIOWrapper(buffered, encoding="utf-8-sig", newline="") as stream:
            lines = _cancellable_lines(stream, cancelled) if cancelled is not None else stream
            fields, validated = parse_result_stream(lines, path, expected,
                                                   allow_empty=not validate_measurements,
                                                   validate_measurements=validate_measurements)
    _check_cancelled(cancelled)
    return ResultCsvSnapshot(path=path, fields=fields, rows=validated.rows, sha256=digest.hexdigest())


def result_csv_snapshot_unchanged(snapshot: ResultCsvSnapshot) -> bool:
    """Verify the current file bytes still match a previously parsed snapshot."""
    if _file_sha256(snapshot.path) != snapshot.sha256:
        return False
    if snapshot.path.name == "result_layers.json":
        from acprof.result_layers import read_result_layers
        try:
            fields, rows = read_result_layers(snapshot.path.parent)
        except (ValueError, OSError, csv.Error):
            return False
        return (fields == snapshot.fields and rows == snapshot.rows
                and _file_sha256(snapshot.path) == snapshot.sha256)
    return True


def verify_result_source(artifact: str | Path, expected_sha256: str | None, *,
                         cancelled: Callable[[], bool] | None = None) -> None:
    """Recheck source bytes and referenced layers without rebuilding current tables.

    The caller has already parsed its snapshot. Schema v3 checks committed
    prefixes under the publication lock; uncommitted tails are not results.
    Legacy v1 retains its imported-CSV dependency validation.
    """
    _check_cancelled(cancelled)
    artifact = Path(artifact)
    current = _file_sha256(artifact, cancelled=cancelled) if artifact.exists() else None
    if current != expected_sha256:
        raise ResultValidationError(f"result source changed: {artifact}")
    if artifact.name != "result_layers.json" or expected_sha256 is None:
        return
    from acprof.artifacts import read_json_object
    manifest = read_json_object(artifact, label="result layers manifest", expected_sha256=expected_sha256)
    if manifest.get("schema_version") == 3:
        from acprof.artifact_layout import ArtifactLayout
        from acprof.result_store.protocol import publication_lock, validate_layer_files
        layout = ArtifactLayout.discover(artifact.parent)
        with publication_lock(layout, exclusive=False):
            validate_layer_files(layout, manifest)
    else:
        from acprof.result_layers import verify_result_layer_files
        verify_result_layer_files(artifact.parent, manifest)
    if _file_sha256(artifact, cancelled=cancelled) != expected_sha256:
        raise ResultValidationError("result manifest changed during verification")


def merge_result_csvs(paths: Sequence[str], destination: str, *,
                     expected: Iterable[MeasurementKey] | None = None) -> int:
    if not paths:
        raise ResultValidationError("no case files to merge")
    resolved = [Path(path).resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise ResultValidationError("duplicate case CSV input path")
    if Path(destination).resolve() in resolved:
        raise ResultValidationError("final CSV cannot also be an input case")
    rows, keys = [], set()
    environments: set[str] = set()
    runtimes: set[tuple[str, str]] = set()
    runtime_ids: set[str] = set()
    fields = list(CSV_FIELDS)
    for path in resolved:
        if not path.is_file():
            raise ResultValidationError(f"missing case CSV: {path}")
        source_fields, source_rows = read_result_csv(path)
        environments.update(row["environment_class"] for row in source_rows)
        runtimes.update((row.get("runtime_kind", "docker"),
                         row.get("resource_scope", "container_cgroup")) for row in source_rows)
        runtime_ids.update(str(row["runtime_id"]) for row in source_rows if str(row.get("runtime_id", "")).lower() not in {"", "nan", "none"})
        if len(runtimes) > 1 or len(runtime_ids) > 1:
            raise ResultValidationError("cannot merge different execution runtimes, identities or metric scopes")
        if len(environments) > 1:
            raise ResultValidationError(f"cannot merge result environments: {sorted(environments)}")
        fields.extend(field for field in source_fields if field not in fields)
        for row in source_rows:
            key = measurement_key(row)
            if key in keys:
                raise ResultValidationError(f"duplicate measurement across case CSVs: {key}")
            keys.add(key)
            rows.append(row)
    fields = order_csv_fields(fields)
    if expected is not None:
        planned = set(expected)
        if keys != planned:
            raise ResultValidationError(
                f"measurement plan mismatch: missing={len(planned - keys)}, unexpected={len(keys - planned)}"
            )
    if Path(destination).name == "result_layers.json":
        from acprof.artifacts import read_json_object
        if Path(destination).is_file() and read_json_object(destination).get("schema_version") == 3:
            from acprof.result_store.writer import ResultWriter
            with ResultWriter(Path(destination).parent) as writer:
                writer.update_rows(rows, reason="validated case consolidation", state="processing")
        else:
            from acprof.result_layers import publish_result_rows
            publish_result_rows(fields, rows, Path(destination).parent)
    else:
        def write(stream):
            writer = csv.DictWriter(stream, fieldnames=fields, restval="nan")
            writer.writeheader()
            writer.writerows(rows)
        atomic_write(destination, write)
    return len(rows)
