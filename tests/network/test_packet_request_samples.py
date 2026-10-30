"""Join packet samples through HTTP request identity, never frame position."""
import csv
import json

import pytest

from acprof.packet import merge_packet_latency, sniff_parse_pcap


@pytest.mark.parametrize("ending", ["\r\n", r"\r\n", r"\\r\\n"])
def test_request_identity_excludes_tshark_header_line_ending(ending):
    # Real TShark field values append an escaped line ending to each header.
    headers = f"Host: fixture,X-Req-Id: window:0{ending},Connection: close"
    assert sniff_parse_pcap.extract_request_id(headers) == "window:0"


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
