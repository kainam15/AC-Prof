import contextlib
import io
import json
from unittest.mock import patch

import pytest

from acprof.packet import sniff_parse_pcap


def test_reused_connection_keeps_request_latencies_without_double_counting_stream_bytes():
    def run(command):
        if "http.request.line" in command:
            return "10\t100.0\tX-Req-Id: load:0\t7\n20\t101.0\tX-Req-Id: load:1\t7\n"
        if "http.request_in" in command:
            return "10\t100.2\t200\n20\t101.4\t200\n"
        return "7\t50000\t8002\t900\t700\n"
    output = io.StringIO()
    with patch.object(sniff_parse_pcap, "run", side_effect=run), contextlib.redirect_stdout(output):
        sniff_parse_pcap.main(["reuse.pcap", "8002"])
    requests = json.loads(output.getvalue())["requests"]
    assert (len(requests)) == (2)
    assert (requests["load:10"]["latency_s"]) == (0.2) or round(abs((requests["load:10"]["latency_s"]) - (0.2)), 7) == 0
    assert (requests["load:20"]["latency_s"]) == (0.4) or round(abs((requests["load:20"]["latency_s"]) - (0.4)), 7) == 0
    assert ("total_wire_bytes") not in (requests["load:10"])
    assert (requests["load:20"]["request_id"]) == ("load:1")

def test_emits_schema_v2_latency_and_wire_metrics() -> None:
    def fake_run(command):
        if "http.request.line" in command:
            return (
                "10\t100.000\tPOST /predict HTTP/1.1,"
                "X-Req-Id: case_seq1_r0:0\t7\n"
            )
        if "http.request_in" in command:
            return "10\t100.250\t200\n"
        if "frame.len" in command:
            return "\n".join(
                [
                    "7\t50000\t8002\t100\t0",
                    "7\t50000\t8002\t200\t150",
                    "7\t8002\t50000\t120\t0",
                    "7\t8002\t50000\t480\t400",
                ]
            )
        pytest.fail(f"unexpected tshark command: {command}")

    stdout = io.StringIO()
    with patch.object(sniff_parse_pcap, "run", side_effect=fake_run), contextlib.redirect_stdout(
        stdout
    ):
        sniff_parse_pcap.main(["capture.pcap", "8002"])

    payload = json.loads(stdout.getvalue())
    assert (payload["schema_version"]) == (2)
    record = payload["requests"]["case_seq1_r0:10"]
    assert (record["latency_s"]) == (0.25)
    assert (record["request_wire_bytes"]) == (300)
    assert (record["response_wire_bytes"]) == (600)
    assert (record["total_wire_bytes"]) == (900)
    assert (record["tcp_payload_bytes"]) == (550)
    assert (record["protocol_overhead_bytes"]) == (350)
    assert (record["protocol_overhead_ratio"]) == (350 / 900) or round(abs((record["protocol_overhead_ratio"]) - (350 / 900)), 7) == 0


def test_nonfinite_response_timestamp_is_not_emitted() -> None:
    def fake_run(command):
        if "http.request.line" in command:
            return "10\t100.0\tX-Req-Id: case:0\t7\n"
        if "http.request_in" in command:
            return "10\t1e309\t200\n"
        if "frame.len" in command:
            return "7\t50000\t8002\t100\t50\n"
        pytest.fail(f"unexpected tshark command: {command}")

    stdout = io.StringIO()
    with patch.object(sniff_parse_pcap, "run", side_effect=fake_run), contextlib.redirect_stdout(
        stdout
    ):
        sniff_parse_pcap.main(["capture.pcap", "8002"])

    assert "Infinity" not in stdout.getvalue()
    assert json.loads(stdout.getvalue())["requests"] == {}


def test_request_parser_surfaces_unexpected_header_errors() -> None:
    def fake_run(command):
        if "http.request.line" in command:
            return "10\t100.0\tX-Req-Id: case:0\t7\n"
        return ""

    with patch.object(sniff_parse_pcap, "run", side_effect=fake_run), patch.object(
        sniff_parse_pcap,
        "extract_request_id",
        side_effect=RuntimeError("parser bug"),
    ):
        with pytest.raises(RuntimeError, match="parser bug"):
            sniff_parse_pcap.main(["capture.pcap", "8002"])
