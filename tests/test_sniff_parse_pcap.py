import contextlib
import io
import json
import unittest
from unittest.mock import patch

from acprof.packet import sniff_parse_pcap


class SniffParsePcapTests(unittest.TestCase):
    def test_reused_connection_keeps_request_latencies_without_double_counting_stream_bytes(self):
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
        self.assertEqual(len(requests), 2)
        self.assertAlmostEqual(requests["load:10"]["latency_s"], 0.2)
        self.assertAlmostEqual(requests["load:20"]["latency_s"], 0.4)
        self.assertNotIn("total_wire_bytes", requests["load:10"])
        self.assertEqual(requests["load:20"]["request_id"], "load:1")

    def test_emits_schema_v2_latency_and_wire_metrics(self) -> None:
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
            self.fail(f"unexpected tshark command: {command}")

        stdout = io.StringIO()
        with patch.object(sniff_parse_pcap, "run", side_effect=fake_run), contextlib.redirect_stdout(
            stdout
        ):
            sniff_parse_pcap.main(["capture.pcap", "8002"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["schema_version"], 2)
        record = payload["requests"]["case_seq1_r0:10"]
        self.assertEqual(record["latency_s"], 0.25)
        self.assertEqual(record["request_wire_bytes"], 300)
        self.assertEqual(record["response_wire_bytes"], 600)
        self.assertEqual(record["total_wire_bytes"], 900)
        self.assertEqual(record["tcp_payload_bytes"], 550)
        self.assertEqual(record["protocol_overhead_bytes"], 350)
        self.assertAlmostEqual(record["protocol_overhead_ratio"], 350 / 900)


if __name__ == "__main__":
    unittest.main()
