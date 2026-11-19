"""Derive packet metrics from separately preserved runtime capture segments."""
from __future__ import annotations

from acprof.host.measurement.capture import (
    CaptureSegment,
    PacketLatencyError,
    packet_parser_command,
)
from acprof.host.measurement.packet_processing import parse_packet_output
from acprof.packet.protocol import require_packet_metrics


def merge_capture_segments(payload, paths, server_port, *, errors: list[str] | None = None):
    combined: dict[str, dict] = {}
    segments: list[dict] = []
    def add_segment(segment):
        try:
            require_packet_metrics(segment)
        except ValueError as exc:
            raise PacketLatencyError(f"invalid capture request or stream records: {exc}") from exc
        requests = segment["requests"]
        streams = segment.get("streams", {})
        if combined.keys() & requests.keys():
            raise PacketLatencyError("duplicate request identity across capture segments")
        index = len(segments)
        combined.update({key: {**record, "capture_segment": index} for key, record in requests.items()})
        segments.append({"kind": "current" if index == 0 else "archived", "streams": streams})

    add_segment(payload)
    for path in paths:
        command = (packet_parser_command(path.path, path.server_port) if isinstance(path, CaptureSegment)
                   else packet_parser_command(path, server_port))
        try:
            previous = parse_packet_output(command, stage="archived pcap parsing")
        except (PacketLatencyError, FileNotFoundError) as exc:
            if errors is None:
                raise
            errors.append(f"{path}: {str(exc).split('Recovery steps:', 1)[0].strip()}")
            continue
        # Overlapping identities in otherwise readable segments are a consistency failure.
        add_segment(previous)
    return {**{key: value for key, value in payload.items() if key != "streams"},
            "requests": combined, "capture_segments": segments}
