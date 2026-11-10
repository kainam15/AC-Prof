"""The existing packet schema shared by parser output and both consumers."""
from typing import Any

from acprof.artifacts import require_schema_version

PACKET_SCHEMA_VERSION = 2


def require_packet_metrics(payload: object) -> dict[str, Any]:
    """Validate the envelope without changing metric normalization or missing values.

    Empty requests are valid parser output, not proof of measured packet latency.
    Streams are optional for existing v2 files and combined capture segments.
    """
    require_schema_version(payload, PACKET_SCHEMA_VERSION, "packet metrics")
    if not isinstance(payload, dict):
        raise ValueError("packet schema v2 must be an object")
    requests = payload.get("requests")
    if not isinstance(requests, dict) or any(not isinstance(row, dict) for row in requests.values()):
        raise ValueError("packet schema v2 requires a requests object containing record objects")
    streams = payload.get("streams", {})
    if not isinstance(streams, dict) or any(not isinstance(row, dict) for row in streams.values()):
        raise ValueError("packet schema v2 requires a streams object containing record objects")
    return payload
