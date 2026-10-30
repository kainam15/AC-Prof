# Packet parser golden fixtures

These synthetic Ethernet/IPv4/TCP captures contain no real traffic or credentials.
`tests/support/pcap_fixture_builder.py` generates deterministic frames with valid checksums;
`padding_frames` can add pre-HTTP ACKs for memory and runtime benchmarks.

The JSON files were recorded using the three-scan parser at `acc6345` with
TShark 4.2.2 (`v4.2.2-1.1build3`). Numeric and frame/stream fields retain that baseline;
`request_id` excludes the escaped HTTP header line ending so it matches the client ID.
The streaming parser must match every field.

| Fixture | Builder options | Covered behavior |
| --- | --- | --- |
| single | defaults | Handshake and request/response bytes |
| segmented | `segmented=True` | TCP and HTTP reassembly |
| retransmission | `segmented=True, retransmit=True` | Retransmitted wire and payload bytes |
| reuse | `reuse=True` | Shared connection counted once |
| missing-response | `reuse=True, missing_response=True` | Missing response still makes the stream shared |
| ready-prefix | `ready=True` | Only `/predict` emits latency; preceding frames remain in stream totals |

`invalid.pcap` contains the literal text `not a capture\n`; both parsers reject it
with tshark's diagnostic. No exception is converted into an empty successful result.
