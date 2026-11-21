#!/usr/bin/env python3
import json
import math
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from contextlib import closing
from dataclasses import dataclass
from itertools import chain

from acprof.host.measurement.packet_processing import postprocessing_timeout
from acprof.packet.protocol import PACKET_SCHEMA_VERSION
from acprof.packet.tshark import stream_tshark, tshark_fields_command

# 用法：
#   python3 -m acprof.packet.sniff_parse_pcap <pcap> <port>
#
# 输出 JSON schema v2：每个请求包含时延、线上帧字节、TCP payload 和
# L2/L3/L4 协议开销。共享连接仅提供逐请求 HTTP 时延和一次流级字节总量，
# 不把整个 tcp.stream 的字节重复归给每个请求。


def normalize_request_id(value: str) -> str:
    """Remove the line terminator escaped by HTTP dissection and TShark fields."""
    value = value.strip()
    for ending in (r"\\r\\n", r"\r\n"):
        if value.endswith(ending):
            return value.removesuffix(ending).strip()
    return value


def extract_request_id(req_lines: str) -> str:
    """
    req_lines 通常包含多条 request line（请求行+headers）
    例如： 'POST /predict HTTP/1.1,Host: ...,Connection: close,X-Req-Id: abc:123,...'
    返回完整 X-Req-Id，保留客户端请求编号。
    """
    if not req_lines:
        return "group"

    # tshark 对重复字段通常用逗号连接
    parts = req_lines.split(",")
    for p in parts:
        p = p.strip()
        if p.lower().startswith("x-req-id:"):
            v = p.split(":", 1)[1].strip()  # 取 header value
            return normalize_request_id(v) or "group"
    return "group"


def extract_group_id_from_request_lines(req_lines: str) -> str:
    return extract_request_id(req_lines).split(":", 1)[0]


def _parse_int_field(raw: str) -> int | None:
    if "," in raw:
        raw = raw.partition(",")[0]
    try:
        parsed = int(raw.strip())
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _parse_finite_float_field(raw: str) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _write_packet_json(payload: object) -> None:
    encoder = json.JSONEncoder(allow_nan=False, indent=2, sort_keys=True)
    # The encoder emits ASCII, often one token at a time. Batch writes even
    # when the supervisor runs this worker with unbuffered stdout.
    block_size = 64 * 1024
    chunks: list[str] = []
    size = 0
    for part in chain(encoder.iterencode(payload), ("\n",)):
        if chunks and size + len(part) > block_size:
            sys.stdout.write("".join(chunks))
            chunks.clear()
            size = 0
        if len(part) > block_size:
            for offset in range(0, len(part), block_size):
                sys.stdout.write(part[offset:offset + block_size])
        else:
            chunks.append(part)
            size += len(part)
    if chunks:
        sys.stdout.write("".join(chunks))


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit(
            "usage: python -m acprof.packet.sniff_parse_pcap <pcap> <port>"
        )

    pcap = args[0]
    port = args[1] if len(args) > 1 else "8002"
    # CLI inputs are a single TCP port, never a tshark filter expression.
    # Packet field parsing accepts comma-separated values; CLI inputs cannot.
    if not port.isascii() or not port.isdigit():
        raise SystemExit(f"invalid TCP port: {port!r}")
    server_port = int(port)
    if not 1 <= server_port <= 65535:
        raise SystemExit(f"invalid TCP port: {port!r}")
    command = tshark_fields_command(pcap, server_port)
    try:
        with closing(stream_tshark(command, timeout=postprocessing_timeout())) as lines:
            payload = parse_packets(lines, server_port=server_port)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.stderr or f"tshark exited with status {exc.returncode}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"tshark timed out after {exc.timeout:g}s") from exc
    _write_packet_json(payload)


@dataclass(slots=True)
class _RequestFrame:
    timestamp: float
    request_id: str
    stream_id: int | None


def parse_packets(lines: Iterable[str], *, server_port: int):
    frames: dict[int, _RequestFrame] = {}
    requests: dict[str, dict] = {}
    stream_stats: dict[int, dict[str, int]] = {}
    for line in lines:
        parts = line.rstrip("\r\n").split("\t")
        if len(parts) < 5:
            continue
        stream_id = _parse_int_field(parts[0])
        frame_len = _parse_int_field(parts[3])
        tcp_payload_len = _parse_int_field(parts[4])
        # Keep frames before HTTP recognition, including handshake, ACKs and
        # retransmissions. A later /predict request may use this stream.
        if stream_id is not None and frame_len is not None and tcp_payload_len is not None:
            stats = stream_stats.get(stream_id)
            if stats is None:
                stats = {"request_wire_bytes": 0, "response_wire_bytes": 0, "tcp_payload_bytes": 0}
                stream_stats[stream_id] = stats
            if _parse_int_field(parts[2]) == server_port:
                stats["request_wire_bytes"] += frame_len
                stats["tcp_payload_bytes"] += tcp_payload_len
            elif _parse_int_field(parts[1]) == server_port:
                stats["response_wire_bytes"] += frame_len
                stats["tcp_payload_bytes"] += tcp_payload_len
        if len(parts) < 12:
            parts.extend([""] * (12 - len(parts)))
        if not parts[7] and not parts[10]:
            continue
        timestamp = _parse_finite_float_field(parts[6].strip())
        if timestamp is None:
            continue
        if parts[7] and "/predict" in parts[8].split(",") and parts[5].strip().isdigit():
            req_fn = int(parts[5])
            if stream_id is None and req_fn in frames:
                stream_id = frames[req_fn].stream_id
            frames[req_fn] = _RequestFrame(timestamp, extract_request_id(parts[9]), stream_id)
        # http.request_in refers to a request already dissected by tshark.
        # Keep the legacy handling of multiple/invalid associations and times.
        if parts[10] and parts[11].strip().isdigit():
            req_fn = int(parts[11])
            frame = frames.get(req_fn)
            if frame is not None:
                dt = timestamp - frame.timestamp
                if math.isfinite(dt) and dt >= 0:
                    group = frame.request_id.partition(":")[0]
                    requests[f"{group}:{req_fn}"] = {"latency_s": dt}

    # Count unanswered requests too: their wire bytes still belong to the stream.
    stream_requests = Counter(frame.stream_id for frame in frames.values())
    for key, record in requests.items():
        frame = frames[int(key.rsplit(":", 1)[1])]
        record["request_id"] = frame.request_id
        record["tcp_stream"] = frame.stream_id
        if frame.stream_id is not None and stream_requests[frame.stream_id] > 1:
            record["wire_bytes_status"] = "unavailable_shared_stream"
            continue
        stats = stream_stats.get(frame.stream_id) if frame.stream_id is not None else None
        if stats is None:
            record["wire_bytes_status"] = "unavailable"
            continue
        _attach_wire_bytes(record, stats)

    return {"schema_version": PACKET_SCHEMA_VERSION, "requests": requests, "streams": stream_stats}


def _attach_wire_bytes(record: dict, stats: dict[str, int]) -> None:
    total_wire = stats["request_wire_bytes"] + stats["response_wire_bytes"]
    tcp_payload = stats["tcp_payload_bytes"]
    protocol_overhead = max(0, total_wire - tcp_payload)
    record.update({
        "wire_bytes_status": "available",
        "request_wire_bytes": stats["request_wire_bytes"],
        "response_wire_bytes": stats["response_wire_bytes"],
        "total_wire_bytes": total_wire,
        "tcp_payload_bytes": tcp_payload,
        "protocol_overhead_bytes": protocol_overhead,
        "protocol_overhead_ratio": protocol_overhead / total_wire if total_wire else None,
    })


if __name__ == "__main__":
    main()
