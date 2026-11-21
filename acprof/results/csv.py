"""结果 CSV 的结构、测量唯一键与完整性校验，不初始化采集依赖。"""
from __future__ import annotations

import csv
import hashlib
import io
import struct
import sys
from collections import deque
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from decimal import InvalidOperation
from itertools import dropwhile, product, zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, cast

from acprof.artifacts import atomic_write, file_sha256
from acprof.cancellation import cancellable, check_cancelled
from acprof.config import CSV_FIELDS
from acprof.results import is_inferred_result
from acprof.results.measurement_plan import (
    MeasurementKey,
    RecordedMeasurementPlan,
    decimal_coordinate,
    decimal_key_text,
)
from acprof.results.metrics import order_csv_fields

if TYPE_CHECKING:
    from acprof.result_store.writer import ResultWriter

KEY_FIELDS = ("cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", "warmup", "repeat_idx")
type ExpectedMeasurements = RecordedMeasurementPlan | Iterable[MeasurementKey]
MISSING_CSV_TEXT = frozenset({"", "nan", "none", "null", "n/a"})
_MISSING_RUN_IDS = MISSING_CSV_TEXT | {"unknown"}
# Python 3.12 accepts a C long; later versions accept a Py_ssize_t.
_CSV_FIELD_LIMIT = min(sys.maxsize, (1 << (8 * struct.calcsize("l") - 1)) - 1)


def result_row_status(row: Mapping[str, object]) -> str:
    """Interpret CSV status consistently without changing the retained row."""
    return str(row.get("status", "")).strip().lower()


def is_measured_success(row: Mapping[str, object]) -> bool:
    """Successful observations exclude inferred values; callers select warmup scope."""
    return result_row_status(row) == "ok" and not is_inferred_result(row)


def csv_diagnostic_text(value: object) -> str:
    """Read a CSV diagnostic without treating a missing marker as an error.

    Match whole tokens only, preserving actual diagnostic text. JSON profiler
    plans have their own error contract and must not use CSV missing markers.
    """
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in MISSING_CSV_TEXT else text


@dataclass(frozen=True)
class ResultCsvSnapshot:
    """One parsed result file together with the digest of its exact source bytes."""

    path: Path
    fields: list[str]
    rows: list[dict[str, str]]
    sha256: str
    run_id: str | None = None  # Recorded by the v3 root manifest, even with no rows.


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


class ResultRunIdentityError(ResultValidationError):
    """Recorded result identities contradict one another or are malformed."""


def recorded_run_id(value: object) -> str | None:
    """Return an explicit run ID, preserving case and ignoring historical missing markers."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ResultRunIdentityError(f"invalid run_id: expected a string, observed {value!r}")
    text = value.strip()
    return None if text.lower() in _MISSING_RUN_IDS else text


def run_id_column(fields: Iterable[str]) -> str | None:
    """Locate the original identity header in a CSV that may carry surrounding spaces."""
    names = [field for field in fields if field.strip() == "run_id"]
    if len(names) > 1:
        raise ResultRunIdentityError(f"duplicate CSV columns for run_id: {names!r}")
    return names[0] if names else None


def require_matching_run_id(expected: object, observed: object, *, source: str) -> None:
    """Reject conflicting recorded IDs while allowing historical rows without one."""
    expected_id, observed_id = recorded_run_id(expected), recorded_run_id(observed)
    if expected_id is not None and observed_id is not None and observed_id != expected_id:
        raise ResultRunIdentityError(f"{source}: inconsistent run_id: expected {expected_id!r}, observed {observed_id!r}")


def require_expected_measurements(actual: Collection[MeasurementKey], expected: ExpectedMeasurements | None, *,
                                  path: Path | None = None, layered: bool = False,
                                  cancelled: Callable[[], bool] | None = None) -> None:
    """Check a compact plan or legacy iterable against strictly validated, unique keys."""
    if expected is None:
        return
    if isinstance(expected, RecordedMeasurementPlan):
        coverage = expected.coverage(actual, cancelled=cancelled)
        missing, unexpected = coverage["missing"], coverage["unexpected"]
    else:
        planned = set(cancellable(expected, cancelled))
        matched = sum(key in planned for key in cancellable(actual, cancelled))
        missing, unexpected = len(planned) - matched, len(actual) - matched
    if missing or unexpected:
        if layered and not isinstance(expected, RecordedMeasurementPlan):
            raise ResultValidationError("layered measurement plan mismatch")
        location = f" {path};" if path is not None else ""
        prefix = "layered measurement plan mismatch" if layered else "measurement plan mismatch"
        raise ResultValidationError(f"{prefix}:{location} missing={missing}, unexpected={unexpected}")


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


def numeric_coordinate_key(field: str, value: str) -> str:
    """Validate one numeric measurement coordinate without rounding it through float."""
    try:
        number = decimal_coordinate(value)
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
    return decimal_key_text(number)


def preview_window_phase(row: Mapping[str, object]) -> str | None:
    """Read an exact phase for a partial CSV preview, defaulting an absent column to formal."""
    raw = str(row.get("warmup", "0")).strip()
    if raw in ("0", "1"):
        return raw
    try:
        return numeric_coordinate_key("warmup", raw)
    except ResultValidationError:
        return None


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
            value = numeric_coordinate_key(field, value)
        values.append(value)
    return tuple(values)


def expected_measurements(cpus: Sequence[int | None], mems: Sequence[int | None], gpus: Sequence[str],
                          scales: Sequence[float], warmup: int, repeat: int) -> set[MeasurementKey]:
    return {
        measurement_key({**dict(zip(KEY_FIELDS, (cpu, mem, gpu, scale, is_warmup, index), strict=True)),
                         **({"runtime_kind": "process"} if cpu is None or mem is None else {})})
        for cpu, mem, gpu, scale in product(cpus, mems, gpus, scales)
        for is_warmup, count in ((1, warmup), (0, repeat))
        for index in range(count)
    }


_HASH_BUFFER_BYTES = 2**18


class _DigestingRawReader(io.RawIOBase):
    """Forward raw reads while hashing the exact bytes consumed by TextIOWrapper."""

    def __init__(self, stream, digest, cancelled: Callable[[], bool] | None = None, *, capture: BinaryIO | None = None):
        super().__init__()
        self._stream = stream
        self._digest = digest
        self._capture = capture
        self._cancelled = cancelled

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int | None:
        check_cancelled(self._cancelled)
        size = self._stream.readinto(buffer)
        if size:
            chunk = memoryview(buffer)[:size]
            self._digest.update(chunk)
            if self._capture is not None and self._capture.write(chunk) != size:
                raise OSError("short write while capturing case CSV")
        return size


def parse_result_stream(stream: Iterable[str], path: Path, expected: ExpectedMeasurements | None, *,
                        allow_empty=False, validate_measurements=True,
                        cancelled: Callable[[], bool] | None = None,
                      ) -> tuple[list[str], ValidatedResultRows]:
    allow_large_csv_fields()
    reader = csv.DictReader(cancellable(stream, cancelled), strict=True)
    fields = list(reader.fieldnames or [])
    return fields, validate_result_rows(fields, reader, path, expected=expected, allow_empty=allow_empty,
                                       validate_measurements=validate_measurements, cancelled=cancelled)


def validate_result_rows(fields: Sequence[str], source_rows: Iterable[dict], path: Path, *,
                         expected: ExpectedMeasurements | None = None, allow_empty=False,
                         validate_measurements=True, cancelled: Callable[[], bool] | None = None) -> ValidatedResultRows:
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
    for index, row in enumerate(cancellable(source_rows, cancelled), 2):
        if None in row or any(value is None for value in row.values()):
            raise ResultValidationError(f"malformed CSV row: {path}:{index}")
        if validate_measurements:
            try:
                key = measurement_key(row)
            except ResultValidationError as exc:
                raise ResultValidationError(f"{path}:{index}: {exc}") from exc
            if key in keys:
                raise ResultValidationError(f"duplicate measurement: {path}:{index}: {key}")
            if result_row_status(row) == "error" and not row.get("error", "").strip():
                raise ResultValidationError(f"status=error without an error diagnostic: {path}:{index}")
            keys[key] = None
            if row.get("environment_class") not in {"native_linux", "wsl2", "vm", "cloud", "container_host", "unknown"}:
                row = {**row, "environment_class": "unknown"}
        rows.append(cast(dict[str, str], row))
    if not rows and not allow_empty:
        raise ResultValidationError(f"empty case CSV (no measurement rows): {path}")
    require_expected_measurements(keys.keys(), expected, path=path, cancelled=cancelled)
    return ValidatedResultRows(rows, tuple(keys) if validate_measurements else None)


def _read_flat_csv(path: Path, expected: ExpectedMeasurements | None = None) -> tuple[list[str], ValidatedResultRows]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return parse_result_stream(stream, path, expected)


def read_result_csv(path: str | Path, *, expected: ExpectedMeasurements | None = None
                    ) -> tuple[list[str], list[dict[str, str]]]:
    path = Path(path)
    if path.name == "result_layers.json":
        from acprof.results.layers import read_result_layers
        return read_result_layers(path.parent, expected=expected)
    fields, validated = _read_flat_csv(path, expected)
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


def open_result_text(path: str | Path, *, inspect_header: Callable[[Sequence[str]], None] | None = None):
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
        with ExitStack() as opening:
            stream = opening.enter_context(path.open(encoding="utf-8-sig", newline=""))
            if inspect_header is not None:
                # Plot inputs skip initial blank lines; preserve the original stream for its parser.
                reader = csv.reader(dropwhile(str.isspace, stream), skipinitialspace=True)
                inspect_header(next(reader, []))
                stream.seek(0)
            opening.pop_all()
            return stream
    fields, rows = read_result_csv(path)
    if inspect_header is not None:
        inspect_header(fields)
    return io.TextIOWrapper(io.BufferedReader(_CsvRawReader(fields, rows)), encoding="utf-8", newline="")


def read_hashed_flat_csv(path: Path, *, expected: ExpectedMeasurements | None = None,
                         validate_measurements: bool = True, cancelled: Callable[[], bool] | None = None,
                         capture: BinaryIO | None = None) -> tuple[list[str], ValidatedResultRows, str]:
    """Validate, hash and optionally copy the same raw CSV stream."""
    check_cancelled(cancelled)
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as raw:
        reader = _DigestingRawReader(raw, digest, cancelled, capture=capture)
        buffered = io.BufferedReader(reader, buffer_size=_HASH_BUFFER_BYTES)
        with io.TextIOWrapper(buffered, encoding="utf-8-sig", newline="") as stream:
            fields, validated = parse_result_stream(stream, path, expected,
                                                   allow_empty=not validate_measurements,
                                                   validate_measurements=validate_measurements, cancelled=cancelled)
    check_cancelled(cancelled)
    return fields, validated, digest.hexdigest()


def read_result_csv_snapshot(path: str | Path, *, expected: ExpectedMeasurements | None = None,
                             validate_measurements: bool = True,
                             cancelled: Callable[[], bool] | None = None,
                             ) -> ResultCsvSnapshot:
    """Parse once while hashing the exact bytes; retain raw historical rows on request.

    Disabling measurement validation only affects flat CSV input. Layered
    storage always verifies its manifest, committed content and row identities.
    Cancellation reaches raw buffers, text lines, layer joins and publication
    lock waits. It never returns partially verified rows.
    """
    check_cancelled(cancelled)
    if not validate_measurements and expected is not None:
        raise ValueError("expected measurements require measurement validation")
    path = Path(path).resolve()
    if path.name == "result_layers.json":
        from acprof.artifacts import read_json_object
        if read_json_object(path, cancelled=cancelled).get("schema_version") == 3:
            from acprof.result_store.reader import read_snapshot
            snapshot = read_snapshot(path.parent, expected=expected, cancelled=cancelled)
            check_cancelled(cancelled)
            return snapshot
        from acprof.results.layers import read_result_layers
        layer_digest = file_sha256(path, cancelled=cancelled)
        fields, rows = read_result_layers(path.parent, expected_manifest_sha256=layer_digest, expected=expected,
                                         cancelled=cancelled)
        verify_result_source(path, layer_digest, cancelled=cancelled)
        check_cancelled(cancelled)
        return ResultCsvSnapshot(path=path, fields=fields, rows=rows, sha256=layer_digest)
    fields, validated, sha256 = read_hashed_flat_csv(
        path, expected=expected, validate_measurements=validate_measurements, cancelled=cancelled,
    )
    return ResultCsvSnapshot(path=path, fields=fields, rows=validated.rows, sha256=sha256)


def result_csv_snapshot_unchanged(snapshot: ResultCsvSnapshot, *, verify_rows: bool = True,
                                  cancelled: Callable[[], bool] | None = None) -> bool:
    """Verify source content, retaining row comparison for externally mutable snapshots.

    Only an owner that has just parsed and exclusively holds the rows may omit
    their second parsing. Committed bytes and pending replacements are always checked.
    """
    if file_sha256(snapshot.path, cancelled=cancelled) != snapshot.sha256:
        return False
    if snapshot.path.name == "result_layers.json":
        from acprof.artifacts import read_json_object
        from acprof.results.layers import read_result_layers
        try:
            if not verify_rows:
                verify_result_source(snapshot.path, snapshot.sha256, cancelled=cancelled)
                return True
            manifest = read_json_object(snapshot.path, expected_sha256=snapshot.sha256, cancelled=cancelled)
            manifest_id = manifest.get("run_id") if manifest.get("schema_version") == 3 else None
            if snapshot.run_id != manifest_id:
                return False
            fields, rows = read_result_layers(snapshot.path.parent, expected_manifest_sha256=snapshot.sha256,
                                              cancelled=cancelled)
        except (ValueError, OSError, csv.Error):
            return False
        row_pairs = zip_longest(rows, snapshot.rows)
        return (fields == snapshot.fields and len(rows) == len(snapshot.rows)
                and all(current == retained for current, retained in cancellable(row_pairs, cancelled))
                and file_sha256(snapshot.path, cancelled=cancelled) == snapshot.sha256)
    return snapshot.run_id is None


def verify_result_source(artifact: str | Path, expected_sha256: str | None, *,
                         cancelled: Callable[[], bool] | None = None) -> None:
    """Recheck source bytes and referenced layers without rebuilding current tables.

    The caller has already parsed its snapshot. Schema v3 checks committed
    prefixes under the publication lock; uncommitted tails are not results.
    Legacy v1 retains its imported-CSV dependency validation.
    """
    check_cancelled(cancelled)
    artifact = Path(artifact)
    current = file_sha256(artifact, cancelled=cancelled) if artifact.exists() else None
    if current != expected_sha256:
        raise ResultValidationError(f"result source changed: {artifact}")
    if artifact.name != "result_layers.json" or expected_sha256 is None:
        return
    from acprof.artifacts import read_json_object
    manifest = read_json_object(artifact, label="result layers manifest", expected_sha256=expected_sha256,
                                cancelled=cancelled)
    if manifest.get("schema_version") == 3:
        from acprof.result_store.protocol import (
            check_pending_replacement,
            publication_lock,
            validate_layer_files,
        )
        from acprof.results.layout import ArtifactLayout
        layout = ArtifactLayout.discover(artifact.parent)
        with publication_lock(layout, exclusive=False, cancelled=cancelled):
            check_pending_replacement(layout, manifest, cancelled=cancelled)
            validate_layer_files(layout, manifest, cancelled=cancelled)
    else:
        from acprof.results.layers import verify_result_layer_files
        verify_result_layer_files(artifact.parent, manifest, cancelled=cancelled)
    if file_sha256(artifact, cancelled=cancelled) != expected_sha256:
        raise ResultValidationError("result manifest changed during verification")


def _read_merge_input(path: Path, expected_sha256: str | None
                      ) -> tuple[list[str], list[dict[str, str]], Iterable[MeasurementKey]]:
    if not path.is_file():
        raise ResultValidationError(f"missing case CSV: {path}")
    if path.name == "result_layers.json":
        if expected_sha256 is not None:
            raise ResultValidationError(f"expected a flat completed case CSV: {path}")
        fields, rows = read_result_csv(path)
        return fields, rows, (measurement_key(row) for row in rows)
    if expected_sha256 is None:
        fields, validated = _read_flat_csv(path)
    else:
        fields, validated, digest = read_hashed_flat_csv(path)
        if digest != expected_sha256:
            raise ResultValidationError(f"completed case CSV changed: {path}")
    assert validated.keys is not None  # Flat case reads always validate measurements.
    return fields, validated.rows, validated.keys


def _read_case_csvs(paths: Sequence[str], destination: str, expected: ExpectedMeasurements | None,
                    case_sha256s: Mapping[Path, str] | None) -> tuple[list[str], list[dict[str, str]]]:
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
        digest = case_sha256s.get(path) if case_sha256s is not None else None
        source_fields, source_rows, source_keys = _read_merge_input(path, digest)
        environments.update(row["environment_class"] for row in source_rows)
        runtimes.update((row.get("runtime_kind", "docker"),
                         row.get("resource_scope", "container_cgroup")) for row in source_rows)
        runtime_ids.update(str(row["runtime_id"]) for row in source_rows if str(row.get("runtime_id", "")).lower() not in {"", "nan", "none"})
        if len(runtimes) > 1 or len(runtime_ids) > 1:
            raise ResultValidationError("cannot merge different execution runtimes, identities or metric scopes")
        if len(environments) > 1:
            raise ResultValidationError(f"cannot merge result environments: {sorted(environments)}")
        fields.extend(field for field in source_fields if field not in fields)
        for key, row in zip(source_keys, source_rows, strict=True):
            if key in keys:
                raise ResultValidationError(f"duplicate measurement across case CSVs: {key}")
            keys.add(key)
            rows.append(row)
    fields = order_csv_fields(fields)
    require_expected_measurements(keys, expected)
    if case_sha256s is not None:
        for path in resolved:
            if path not in case_sha256s:
                raise ResultValidationError(f"no completed case CSV recorded for: {path}")
    return fields, rows


def publish_merged_cases(paths: Sequence[str], destination: str, *, scope: ExitStack,
                         expected: ExpectedMeasurements | None = None, expected_run_id: str | None = None,
                         case_sha256s: Mapping[Path, str] | None = None) -> tuple[int, ResultWriter | None]:
    """Validate the inputs first, then publish and hold any v3 writer in the caller's scope."""
    fields, rows = _read_case_csvs(paths, destination, expected, case_sha256s)
    if Path(destination).name == "result_layers.json":
        from acprof.artifacts import read_json_object
        if Path(destination).is_file() and read_json_object(destination).get("schema_version") == 3:
            from acprof.result_store.writer import ResultWriter
            writer = scope.enter_context(ResultWriter(Path(destination).parent, expected_run_id=expected_run_id))
            writer.update_rows(rows, reason="validated case consolidation", state="processing")
            return len(rows), writer
        from acprof.results.layers import publish_result_rows
        publish_result_rows(fields, rows, Path(destination).parent)
    else:
        def write(stream):
            writer = csv.DictWriter(stream, fieldnames=fields, restval="nan")
            writer.writeheader()
            writer.writerows(rows)
        atomic_write(destination, write)
    return len(rows), None


def merge_result_csvs(paths: Sequence[str], destination: str, *,
                      expected: ExpectedMeasurements | None = None) -> int:
    with ExitStack() as scope:
        count, _writer = publish_merged_cases(paths, destination, scope=scope, expected=expected)
        return count
