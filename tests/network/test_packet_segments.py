"""Capture-local stream numbers must remain traceable after a resumed run."""
import json
from types import SimpleNamespace

import pytest

from acprof.host.measurement.capture import PacketLatencyError
from acprof.host.measurement.packet_segments import merge_capture_segments


def segment(request, size):
    return {"schema_version": 2, "requests": {request: {"latency_s": 0.02, "tcp_stream": 0,
                                                       "request_wire_bytes": size}},
            "streams": {"0": {"request_wire_bytes": size}}}


def test_capture_local_streams_are_preserved_without_collision(monkeypatch):
    previous = segment("first:0", 111)
    current = segment("second:0", 222)
    monkeypatch.setattr("acprof.host.measurement.packet_segments.run_packet_command",
                        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(previous)))
    result = merge_capture_segments(current, ["archived.pcap"], 8000)
    assert "streams" not in result
    for request, size in (("first:0", 111), ("second:0", 222)):
        record = result["requests"][request]
        streams = result["capture_segments"][record["capture_segment"]]["streams"]
        assert record["tcp_stream"] == 0
        assert streams[str(record["tcp_stream"])]["request_wire_bytes"] == size
        assert record["request_wire_bytes"] == size
    assert "capture_segment" not in current["requests"]["second:0"]


def test_duplicate_request_identity_across_captures_is_rejected(monkeypatch):
    payload = segment("same:0", 111)
    monkeypatch.setattr("acprof.host.measurement.packet_segments.run_packet_command",
                        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(payload)))
    with pytest.raises(PacketLatencyError, match="duplicate request"):
        merge_capture_segments(payload, ["archived.pcap"], 8000)
