"""Derive packet metrics from separately preserved runtime capture segments."""
from __future__ import annotations

from acprof.artifacts import loads_finite_json
from acprof.host.measurement.capture import PacketLatencyError
from acprof.host.measurement.packet_processing import run_packet_command
from acprof.installation import module_command


def merge_capture_segments(payload, paths, server_port):
    if not isinstance(payload, dict) or not isinstance(payload.get("requests"), dict):
        raise PacketLatencyError("segmented capture requires packet schema with request identities")
    combined: dict[str, dict] = {}
    segments: list[dict] = []
    def add_segment(segment):
        requests = segment.get("requests")
        streams = segment.get("streams", {})
        if (not isinstance(requests, dict) or not isinstance(streams, dict)
                or any(not isinstance(record, dict) for record in requests.values())):
            raise PacketLatencyError("invalid capture request or stream records")
        if combined.keys() & requests.keys():
            raise PacketLatencyError("duplicate request identity across capture segments")
        index = len(segments)
        combined.update({key: {**record, "capture_segment": index} for key, record in requests.items()})
        segments.append({"kind": "current" if index == 0 else "archived", "streams": streams})

    add_segment(payload)
    for path in paths:
        result = run_packet_command([*module_command("acprof.packet.sniff_parse_pcap"), str(path), str(server_port)],
                                    stage="archived pcap parsing")
        if result.returncode:
            raise PacketLatencyError(f"archived capture parsing failed: {path}: {result.stderr}")
        try:
            previous = loads_finite_json(result.stdout)
            if not isinstance(previous, dict) or previous.get("schema_version") != payload.get("schema_version"):
                raise ValueError("packet schema mismatch")
        except (ValueError, KeyError, TypeError) as exc:
            raise PacketLatencyError(f"invalid archived capture: {path}") from exc
        add_segment(previous)
    return {**{key: value for key, value in payload.items() if key != "streams"},
            "requests": combined, "capture_segments": segments}
