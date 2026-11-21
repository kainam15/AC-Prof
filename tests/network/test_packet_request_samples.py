"""Join packet samples through HTTP request identity, never frame position."""
import csv
import json
import math

import pytest

from acprof.packet import merge_packet_latency, sniff_parse_pcap


@pytest.mark.parametrize("ending", ["\r\n", r"\r\n", r"\\r\\n"])
def test_request_identity_excludes_tshark_header_line_ending(ending):
    # Real TShark field values append an escaped line ending to each header.
    headers = f"Host: fixture,X-Req-Id: window:0{ending},Connection: close"
    assert sniff_parse_pcap.extract_request_id(headers) == "window:0"


@pytest.mark.parametrize("invalid", [None, "", "not-a-number", object()])
def test_invalid_packet_numeric_value_is_unknown(invalid):
    assert math.isnan(merge_packet_latency._to_float(invalid))


def test_packet_numeric_converter_fault_is_not_silently_hidden():
    class FaultyNumber:
        def __float__(self):
            raise RuntimeError("unexpected number converter failure")

    with pytest.raises(RuntimeError, match="unexpected number converter failure"):
        merge_packet_latency._to_float(FaultyNumber())


def _write_inputs(tmp_path, payload):
    csv_path = tmp_path / "result.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, ["status", "error", "sniff_group_id", "latency_s"])
        writer.writeheader()
        writer.writerow({"status": "ok", "error": "", "sniff_group_id": "window", "latency_s": "nan"})
    samples = tmp_path / "result.csv.requests.jsonl"
    samples.write_text(json.dumps({"schema_version": 1, "sniff_group_id": "window",
                                   "latency_app_s": [0.4, 0.8, 0.9], "status": "ok"}) + "\n")
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(payload))
    return csv_path, packet_path, samples


def test_parser_and_merge_align_samples_by_client_sequence_with_frame_collision(tmp_path):
    def packet(frame, time, *, request_id=None, request_in=None):
        request = request_id is not None
        return "\t".join(map(str, [
            frame // 10, 50000 if request else 8002, 8002 if request else 50000,
            100, 10, frame, time, "1" if request else "", "/predict" if request else "",
            f"X-Req-Id: {request_id}" if request else "", "" if request else "1",
            "" if request else request_in, "" if request else "200",
        ]))
    payload = sniff_parse_pcap.parse_packets([
        packet(1, 100.0, request_id="window:0"),
        packet(2, 100.25, request_in=1),
        packet(10, 101.0, request_id="window:1"),
        packet(11, 101.75, request_in=10),
    ], server_port=8002)
    assert payload["requests"]["window:1"]["request_id"] == "window:0"
    csv_path, packet_path, samples = _write_inputs(tmp_path, payload)
    output = tmp_path / "merged.csv"

    merge_packet_latency.main([str(csv_path), str(packet_path), str(output)])

    assert json.loads(samples.read_text())["latency_packet_s"] == [0.25, 0.75, None]
    assert json.loads(samples.read_text())["latency_app_s"] == [0.4, 0.8, 0.9]
    with output.open() as stream:
        assert float(next(csv.DictReader(stream))["latency_s"]) == 0.5


def test_missing_client_identity_never_falls_back_to_a_matching_frame_number(tmp_path):
    csv_path, packet_path, samples = _write_inputs(tmp_path, {
        "schema_version": 2, "requests": {"window:0": {"latency_s": 9.0}},
    })

    merge_packet_latency.main([str(csv_path), str(packet_path), str(tmp_path / "merged.csv")])

    assert json.loads(samples.read_text())["latency_packet_s"] == [None, None, None]


def test_existing_packet_metadata_with_escaped_header_ending_is_readable(tmp_path):
    csv_path, packet_path, samples = _write_inputs(tmp_path, {
        "schema_version": 2,
        "requests": {"window:20": {"latency_s": 0.25, "request_id": r"window:0\\r\\n"}},
    })

    merge_packet_latency.main([str(csv_path), str(packet_path), str(tmp_path / "merged.csv")])

    assert json.loads(samples.read_text())["latency_packet_s"] == [0.25, None, None]


def test_ambiguous_client_identity_rejects_sample_replacement(tmp_path):
    csv_path, packet_path, samples = _write_inputs(tmp_path, {
        "schema_version": 2, "requests": {
            "window:10": {"latency_s": 0.25, "request_id": "window:0"},
            "window:20": {"latency_s": 0.75, "request_id": "window:0"},
        },
    })
    original = samples.read_bytes()

    with pytest.raises(ValueError, match="duplicate.*request.*window:0"):
        merge_packet_latency.main([str(csv_path), str(packet_path), str(tmp_path / "merged.csv")])

    assert samples.read_bytes() == original


@pytest.mark.parametrize(("runtime", "journal", "expected_ids"),
                         [("docker", False, 0), ("docker", True, 2), ("process", False, 2), ("process", True, 2)])
def test_packet_identity_and_numeric_fields_are_consumed_only_as_needed(tmp_path, monkeypatch, runtime, journal, expected_ids):
    csv_path, packet_path, samples = _write_inputs(tmp_path, {
        "schema_version": 2, "requests": {
            "window:13": {"request_id": r"window:0\r\n", "latency_s": "0.25", "wire_bytes_status": "available",
                          "total_wire_bytes": "4.0", "protocol_overhead_bytes": "2"},
            "window:17": {"request_id": "window:1", "latency_s": 0.75, "wire_bytes_status": "unavailable",
                          "total_wire_bytes": "unknown", "protocol_overhead_bytes": None},
        }})
    csv_path.write_text("sniff_group_id,runtime_kind,repeat_in_window,packet_status,latency_s,packet_protocol_overhead_ratio\n"
                        f"window,{runtime},2,pending,nan,nan\n")
    if not journal:
        samples.unlink()
    normalized = []
    strings_converted = []
    normalize = merge_packet_latency.normalize_request_id
    convert = merge_packet_latency._to_float

    def count_id(value):
        normalized.append(value)
        return normalize(value)

    def count_numeric(value):
        if isinstance(value, str):
            strings_converted.append(value)
        return convert(value)

    monkeypatch.setattr(merge_packet_latency, "normalize_request_id", count_id)
    monkeypatch.setattr(merge_packet_latency, "_to_float", count_numeric)
    output = tmp_path / "merged.csv"
    merge_packet_latency.main([str(csv_path), str(packet_path), str(output)])

    assert len(normalized) == expected_ids
    assert not {"available", "unavailable", "window:1", r"window:0\r\n"}.intersection(strings_converted)
    assert strings_converted.count("0.25") == 1
    with output.open() as stream:
        row = next(csv.DictReader(stream))
    assert (row["packet_status"], row["latency_s"], row["packet_protocol_overhead_ratio"]) == ("ok", "0.500000", "0.500000")
    if journal:
        assert json.loads(samples.read_text())["latency_packet_s"] == [0.25, 0.75, None]


@pytest.mark.parametrize("partial", [False, True])
def test_global_duplicate_http_ids_only_affect_matching_samples(tmp_path, partial):
    # Frames can come from distinct capture groups. The journal uses the HTTP ID.
    duplicated = "window:0" if partial else "unobserved:0"
    csv_path, packets, samples = _write_inputs(tmp_path, {
        "schema_version": 2, "requests": {
            "window:9": {"request_id": duplicated, "latency_s": 0.25},
            "other:3": {"request_id": duplicated, "latency_s": 1.25},
            "window:31": {"request_id": "window:1", "latency_s": 0.75},
        }})
    if partial:
        (tmp_path / "static_meta.json").write_text(json.dumps({"schema_version": 7, "profiling_mode": "best_effort"}))
    merge_packet_latency.main([str(csv_path), str(packets), str(tmp_path / "merged.csv")])
    assert json.loads(samples.read_text())["latency_packet_s"] == [None, 0.75, None]
