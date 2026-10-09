"""结果 CSV 的结构、测量唯一键与完整性校验，不初始化采集依赖。"""
from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from itertools import product
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from acprof.artifacts import atomic_write
from acprof.config import CSV_FIELDS
from acprof.metric_registry import order_csv_fields

KEY_FIELDS = ("cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", "warmup", "repeat_idx")
MeasurementKey = tuple[str, ...]


@dataclass(frozen=True)
class ResultCsvSnapshot:
    """One parsed result file together with the digest of its exact source bytes."""

    path: Path
    fields: list[str]
    rows: list[dict[str, str]]
    sha256: str


class ResultValidationError(ValueError):
    """输入不完整或无法唯一归属到计划中的测量。"""


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


def expected_measurements(cpus: Sequence[int], mems: Sequence[int], gpus: Sequence[str],
                          scales: Sequence[float], warmup: int, repeat: int) -> set[MeasurementKey]:
    return {
        measurement_key(dict(zip(KEY_FIELDS, (cpu, mem, gpu, str(float(scale)), is_warmup, index))))
        for cpu, mem, gpu, scale in product(cpus, mems, gpus, scales)
        for is_warmup, count in ((1, warmup), (0, repeat))
        for index in range(count)
    }


_HASH_BUFFER_BYTES = 2**18


class _DigestingRawReader(io.RawIOBase):
    """Forward raw reads while hashing the exact bytes consumed by TextIOWrapper."""

    def __init__(self, stream, digest):
        super().__init__()
        self._stream = stream
        self._digest = digest

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int | None:
        size = self._stream.readinto(buffer)
        if size:
            self._digest.update(memoryview(buffer)[:size])
        return size


def _parse_result_csv(stream: Iterable[str], path: Path, expected: Iterable[MeasurementKey] | None
                      ) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(stream, strict=True)
    fields = reader.fieldnames or []
    require_current_fields(fields)
    if not fields or len(fields) != len(set(fields)):
        raise ResultValidationError(f"missing or duplicate CSV columns: {path}")
    missing = set(KEY_FIELDS) - set(fields)
    if missing:
        raise ResultValidationError(f"missing identity columns {sorted(missing)}: {path}")
    rows, keys = [], set()
    for index, row in enumerate(reader, 2):
        if None in row or any(value is None for value in row.values()):
            raise ResultValidationError(f"malformed CSV row: {path}:{index}")
        try:
            key = measurement_key(row)
        except ResultValidationError as exc:
            raise ResultValidationError(f"{path}:{index}: {exc}") from exc
        if key in keys:
            raise ResultValidationError(f"duplicate measurement: {path}:{index}: {key}")
        if row.get("status", "").strip().lower() == "error" and not row.get("error", "").strip():
            raise ResultValidationError(f"status=error without an error diagnostic: {path}:{index}")
        keys.add(key)
        if row.get("environment_class") not in {"native_linux", "wsl2", "vm", "cloud", "container_host"}:
            row["environment_class"] = "unknown"
        rows.append(row)
    if not rows:
        raise ResultValidationError(f"empty case CSV (no measurement rows): {path}")
    if expected is not None:
        planned = set(expected)
        if keys != planned:
            raise ResultValidationError(
                f"measurement plan mismatch: {path}; missing={len(planned - keys)}, "
                f"unexpected={len(keys - planned)}"
            )
    return fields, rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    buffer = bytearray(_HASH_BUFFER_BYTES)
    view = memoryview(buffer)
    with path.open("rb", buffering=0) as stream:
        while True:
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
        return _parse_result_csv(stream, path, expected)


def open_result_text(path: str | Path):
    """Open a CSV or an in-memory logical view of a layered experiment.

    No complete wide CSV is written to the file system by readers.
    """
    path = Path(path)
    if path.is_dir():
        path /= "result_layers.json"
    if path.name != "result_layers.json":
        return path.open(encoding="utf-8-sig", newline="")
    fields, rows = read_result_csv(path)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    stream.seek(0)
    return stream


def read_result_csv_snapshot(path: str | Path, *, expected: Iterable[MeasurementKey] | None = None
                             ) -> ResultCsvSnapshot:
    """Parse the CSV once while hashing its exact source bytes."""
    path = Path(path).resolve()
    if path.name == "result_layers.json":
        fields, rows = read_result_csv(path, expected=expected)
        digest = _file_sha256(path)
        if not result_csv_snapshot_unchanged(ResultCsvSnapshot(path, fields, rows, digest)):
            raise ResultValidationError("layered results changed during read")
        return ResultCsvSnapshot(path=path, fields=fields, rows=rows, sha256=digest)
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as raw:
        buffered = io.BufferedReader(_DigestingRawReader(raw, digest), buffer_size=_HASH_BUFFER_BYTES)
        with io.TextIOWrapper(buffered, encoding="utf-8-sig", newline="") as stream:
            fields, rows = _parse_result_csv(stream, path, expected)
    return ResultCsvSnapshot(path=path, fields=fields, rows=rows, sha256=digest.hexdigest())


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
        return fields == snapshot.fields and rows == snapshot.rows
    return True


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
    environments = set()
    fields = list(CSV_FIELDS)
    for path in resolved:
        if not path.is_file():
            raise ResultValidationError(f"missing case CSV: {path}")
        source_fields, source_rows = read_result_csv(path)
        environments.update(row["environment_class"] for row in source_rows)
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
        from acprof.result_layers import publish_result_rows
        publish_result_rows(fields, rows, Path(destination).parent)
    else:
        def write(stream):
            writer = csv.DictWriter(stream, fieldnames=fields, restval="nan")
            writer.writeheader()
            writer.writerows(rows)
        atomic_write(destination, write)
    return len(rows)
