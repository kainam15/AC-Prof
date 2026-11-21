import csv
import json
import math
import os
import sys
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import cached_property
from itertools import zip_longest
from typing import Any, SupportsFloat, SupportsIndex, TextIO, cast

from acprof.artifacts import (
    atomic_write,
    loads_finite_json,
    read_json_object,
    read_static_metadata,
)
from acprof.latency_slo import latency_slo_threshold
from acprof.packet.protocol import require_packet_metrics
from acprof.packet.sniff_parse_pcap import normalize_request_id
from acprof.pixel_metrics import per_megapixel
from acprof.results.csv import allow_large_csv_fields, require_current_fields
from acprof.results.latency import latency_distribution_metrics
from acprof.results.metrics import order_csv_fields

SNIFF_GROUP_FIELD = "sniff_group_id"
NETWORK_RECORD_TO_CSV_FIELD = {
    "request_wire_bytes": "packet_request_wire_bytes_per_request",
    "response_wire_bytes": "packet_response_wire_bytes_per_request",
    "total_wire_bytes": "packet_total_wire_bytes_per_request",
    "tcp_payload_bytes": "packet_tcp_payload_bytes_per_request",
    "protocol_overhead_bytes": "packet_protocol_overhead_bytes_per_request",
}


def _to_float(value: object) -> float:
    try:
        return float(cast(str | SupportsFloat | SupportsIndex, value))
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def _request_records(payload: object) -> dict[str, dict[str, Any]]:
    return require_packet_metrics(payload)["requests"]


@dataclass
class _PacketGroup:
    records: list[dict[str, Any]] = field(default_factory=list)

    @cached_property
    def request_ids(self) -> list[str | None]:
        """Only coverage checks and the optional request journal need HTTP IDs."""
        observed = (record.get("request_id") for record in self.records)
        return [normalize_request_id(value) if isinstance(value, str) else None for value in observed]


def _group_requests(requests: dict[str, dict[str, Any]]) -> dict[str, _PacketGroup]:
    groups: dict[str, _PacketGroup] = defaultdict(_PacketGroup)
    for frame, record in requests.items():
        # The only per-request value shared by the CSV and journal. Normalize it
        # in the private decoded record so both outputs use the same value.
        if "latency_s" in record:
            record["latency_s"] = _to_float(record["latency_s"])
        groups[frame.split(":", 1)[0]].records.append(record)
    return groups


def _input_units_per_request(row: dict, batch_size: float) -> float:
    explicit = _to_float(row.get("input_units_per_request", "nan"))
    if math.isfinite(explicit) and explicit > 0.0:
        return explicit
    input_scale = _to_float(row.get("input_scale", "nan"))
    if (
        math.isfinite(input_scale)
        and input_scale > 0.0
        and math.isfinite(batch_size)
        and batch_size > 0.0
    ):
        return input_scale * batch_size
    return float("nan")


def _set_packet_network_metrics(row: dict, records: list[dict[str, Any]]) -> None:
    totals = {}
    for record_field, csv_field in NETWORK_RECORD_TO_CSV_FIELD.items():
        numbers = (_to_float(record[record_field]) for record in records if record_field in record)
        finite = [value for value in numbers if math.isfinite(value)]
        if finite:
            total = sum(finite)
            totals[record_field] = total
            mean = total / len(finite)
            if math.isfinite(mean):
                row[csv_field] = _fmt_float(mean)

    wire = totals.get("total_wire_bytes", 0.0)
    if wire > 0 and "protocol_overhead_bytes" in totals:
        row["packet_protocol_overhead_ratio"] = _fmt_float(totals["protocol_overhead_bytes"] / wire)


def _mean(values: list[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    return sum(finite_values) / len(finite_values) if finite_values else float("nan")


def _distribution_metrics(
    values: list[float], *, slow_latency_threshold_s: float | None = None,
) -> dict[str, float]:
    return latency_distribution_metrics("latency", values, slow_latency_threshold_s=slow_latency_threshold_s)


def _fmt_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.6f}"


def _mflops_for_latency(work_mflop: object, latency_s: float) -> float:
    work = _to_float(work_mflop)
    if math.isfinite(work) and math.isfinite(latency_s) and latency_s > 0.0:
        return work / latency_s
    return float("nan")


def _recompute_packet_flop_rates(row: dict, latency_s: float) -> None:
    require_current_fields(row)
    for source, destination in (
        ("model_logical_mflop_per_request_torch_profiler_eager", "model_logical_mflops_packet_torch_profiler_eager"),
        ("gpu_executed_mflop_per_request_ncu", "gpu_executed_mflops_packet_ncu"),
    ):
        value = _mflops_for_latency(row.get(source), latency_s)
        if destination in row and math.isfinite(value):
            row[destination] = _fmt_float(value)


def _estimate_cpu_cycles(row: dict, latency_s: float) -> float:
    freq_hz = _to_float(row.get("cpu_freq_avg_hz", "nan"))
    cpu_cores = _to_float(row.get("cpu_cores", "nan"))
    cpu_util_pct = _to_float(row.get("container_cpu_util_avg_pct", "nan"))
    if (
        latency_s == latency_s
        and freq_hz == freq_hz
        and cpu_cores == cpu_cores
        and cpu_util_pct == cpu_util_pct
        and latency_s > 0.0
        and freq_hz > 0.0
        and cpu_cores > 0.0
        and cpu_util_pct >= 0.0
    ):
        return latency_s * freq_hz * cpu_cores * (cpu_util_pct / 100.0)
    return float("nan")


def _compute_cpu_mips(row: dict, latency_s: float) -> float:
    instructions_per_request = _to_float(row.get("cpu_instructions_per_request", "nan"))
    if (
        latency_s == latency_s
        and instructions_per_request == instructions_per_request
        and latency_s > 0.0
        and instructions_per_request >= 0.0
    ):
        return instructions_per_request / latency_s / 1_000_000.0
    return float("nan")


def _sidecar_groups(stream: TextIO) -> Iterator[str]:
    for line_number, line in enumerate(stream, start=1):
        raw = line.strip()
        if not raw:
            raise ValueError(
                f"invalid sniff group sidecar record at line {line_number}: blank line"
            )
        try:
            payload = loads_finite_json(raw)
        except (ValueError, RecursionError) as exc:
            raise ValueError(
                f"invalid sniff group sidecar record at line {line_number}: malformed JSON"
            ) from exc
        if not isinstance(payload, dict) or SNIFF_GROUP_FIELD not in payload:
            raise ValueError(
                f"invalid sniff group sidecar record at line {line_number}: "
                f"missing {SNIFF_GROUP_FIELD}"
            )
        group_id = payload[SNIFF_GROUP_FIELD]
        if not isinstance(group_id, str):
            raise ValueError(
                f"invalid sniff group sidecar record at line {line_number}: "
                f"{SNIFF_GROUP_FIELD} must be a string"
            )
        yield group_id


def _merge_request_samples(csv_path: str, groups: dict[str, _PacketGroup], *, allow_partial=False) -> None:
    """Retain packet samples alongside app samples before PCAP cleanup."""
    from acprof.artifacts import atomic_write
    from acprof.results.layout import case_sidecar
    path = case_sidecar(csv_path, "requests")
    if not os.path.exists(path):
        return  # Historical CSVs have no raw application samples.

    # Frame keys select CSV groups; only explicit HTTP IDs select journal slots.
    # None marks an ambiguous ID, while NaN denotes a missing numeric latency.
    latency_by_id: dict[str, float | None] = {}
    for group in groups.values():
        for request_id, record in zip(group.request_ids, group.records, strict=True):
            if request_id:
                latency_by_id[request_id] = None if request_id in latency_by_id else record.get("latency_s", math.nan)

    def write(stream):
        with open(path, encoding="utf-8") as source:
            for line in source:
                window = json.loads(line)
                if window.get("schema_version") != 1:
                    raise ValueError("unsupported request latency schema")
                gid = window["sniff_group_id"]
                latencies: list[float | None] = []
                for index in range(len(window["latency_app_s"])):
                    request_id = f"{gid}:{index}"
                    value = latency_by_id.get(request_id, math.nan)
                    if value is None:
                        if not allow_partial:
                            raise ValueError(f"duplicate packet request identity: {request_id}")
                        latencies.append(None)
                        continue
                    latencies.append(value if math.isfinite(value) and value >= 0 else None)
                window["latency_packet_s"] = latencies
                stream.write(json.dumps(window, separators=(",", ":"), allow_nan=False) + "\n")

    atomic_write(path, write)


def _set_throughput_metrics(row: dict, latency: float, static_batch_size: float) -> None:
    batch_size = static_batch_size if math.isfinite(static_batch_size) else _to_float(row.get("batch_size"))
    if not (math.isfinite(latency) and latency > 0 and math.isfinite(batch_size) and batch_size > 0):
        return
    throughput = batch_size / latency
    row["throughput_samples_per_s"] = _fmt_float(throughput)
    input_units = _input_units_per_request(row, batch_size)
    if math.isfinite(input_units) and input_units > 0:
        row["latency_s_per_input_unit"] = _fmt_float(latency / input_units)
    cpu_cores = _to_float(row.get("cpu_cores"))
    if math.isfinite(cpu_cores) and cpu_cores > 0:
        row["throughput_samples_per_s_per_cpu_core"] = _fmt_float(throughput / cpu_cores)


def _merge_window(row: dict, gid: str, group: _PacketGroup | None, *, best_effort: bool,
                  static_batch_size: float, slow_latency_threshold_s: float | None) -> None:
    if not gid or group is None:
        if "packet_status" in row:
            row["packet_status"] = "unavailable" if gid else "not_applicable"
        return

    if "packet_status" in row:
        row["packet_status"] = "ok"
        if best_effort or row.get("runtime_kind") == "process":
            count = _to_float(row.get("repeat_in_window"))
            ids = group.request_ids
            if len(ids) != count or set(ids) != {f"{gid}:{index}" for index in range(len(ids))}:
                row["packet_status"] = "partial"
    latencies = [record.get("latency_s", math.nan) for record in group.records]
    row["latency_s"] = _fmt_float(_mean(latencies))
    for direction in ("input", "output"):
        field = f"latency_s_per_{direction}_megapixel"
        if field in row:
            row[field] = _fmt_float(per_megapixel(row["latency_s"], row.get(f"{direction}_pixels_per_request")))
    for field, value in _distribution_metrics(latencies, slow_latency_threshold_s=slow_latency_threshold_s).items():
        row[field] = _fmt_float(value)
    # Derived rates historically use the rounded value stored in the CSV.
    latency = _to_float(row["latency_s"])
    _set_throughput_metrics(row, latency, static_batch_size)
    _set_packet_network_metrics(row, group.records)
    _recompute_packet_flop_rates(row, latency)
    cycles = _estimate_cpu_cycles(row, latency)
    if math.isfinite(cycles):
        row["cpu_cycles_est_packet"] = _fmt_float(cycles)
    mips = _compute_cpu_mips(row, latency)
    if math.isfinite(mips):
        row["cpu_mips_packet"] = _fmt_float(mips)


def _merged_rows(rows: csv.DictReader, sidecar: TextIO | None, packets: dict[str, _PacketGroup], *,
                 best_effort: bool, static_batch_size: float, slow_latency_threshold_s: float | None) -> Iterator[dict]:
    groups = _sidecar_groups(sidecar) if sidecar is not None else ()
    group_count = row_count = 0
    for row, gid in zip_longest(rows, groups):
        if gid is not None:
            group_count += 1
        if row is not None:
            row_count += 1
            group_id = row.pop(SNIFF_GROUP_FIELD, "") or gid or ""
            _merge_window(row, group_id, packets.get(group_id), best_effort=best_effort,
                          static_batch_size=static_batch_size, slow_latency_threshold_s=slow_latency_threshold_s)
            yield row
    if sidecar is not None and group_count != row_count:
        raise ValueError(f"sniff group sidecar row count mismatch: {group_count} sidecar rows for {row_count} CSV rows")


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        raise SystemExit(
            "usage: python -m acprof.packet.merge_packet_latency "
            "<in_csv> <lat_json> <out_csv>"
        )

    in_csv, lat_json, out_csv = args
    # A capture grows with request count; the small-metadata cap does not apply.
    groups = _group_requests(_request_records(read_json_object(lat_json, label="packet metrics", bounded=False)))
    from acprof.results.layout import ArtifactLayout, case_sidecar
    metadata = read_static_metadata(ArtifactLayout.from_csv(in_csv).root)
    static_batch_size = _to_float(metadata.get("batch_size", "nan"))
    slow_latency_threshold_s = latency_slo_threshold(metadata)

    best_effort = metadata.get("profiling_mode") == "best_effort"
    sidecar_path = case_sidecar(in_csv, "sniff_groups")

    def write_merged_csv(stream: TextIO) -> None:
        with open(in_csv, encoding="utf-8-sig", newline="") as source:
            allow_large_csv_fields()
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise SystemExit("empty csv")
            require_current_fields(reader.fieldnames)
            fields = order_csv_fields(field for field in reader.fieldnames if field != SNIFF_GROUP_FIELD)
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            sidecar_source = sidecar_path.open(encoding="utf-8") if sidecar_path.exists() else nullcontext(None)
            with sidecar_source as sidecar:
                writer.writerows(_merged_rows(reader, sidecar, groups, best_effort=best_effort,
                                              static_batch_size=static_batch_size,
                                              slow_latency_threshold_s=slow_latency_threshold_s))

    atomic_write(out_csv, write_merged_csv)
    _merge_request_samples(in_csv, groups, allow_partial=best_effort)


if __name__ == "__main__":
    main()
