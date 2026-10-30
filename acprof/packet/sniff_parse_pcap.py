#!/usr/bin/env python3
import json
import math
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from contextlib import closing

from acprof.host.measurement.packet_processing import postprocessing_timeout
from acprof.packet.tshark import stream_tshark

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
    value = str(raw or "").split(",", 1)[0].strip()
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _parse_finite_float_field(raw: str) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


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
    normalized_port = str(server_port)

    command = [
        "tshark", "-l", "-r", pcap,
        "-o", "tcp.desegment_tcp_streams:TRUE",
        "-o", "http.desegment_body:TRUE",
        "-d", f"tcp.port=={normalized_port},http",
        "-Y", f"tcp.port=={normalized_port}",
        "-T", "fields", "-E", "separator=\t",
    ]
    for field in (
        "tcp.stream", "tcp.srcport", "tcp.dstport", "frame.len", "tcp.len",
        "frame.number", "frame.time_epoch", "http.request", "http.request.uri",
        "http.request.line", "http.response", "http.request_in", "http.response.code",
    ):
        command.extend(("-e", field))
    try:
        with closing(stream_tshark(command, timeout=postprocessing_timeout())) as lines:
            payload = parse_packets(lines, server_port=server_port)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.stderr or f"tshark exited with status {exc.returncode}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"tshark timed out after {exc.timeout:g}s") from exc
    print(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True))


def parse_packets(lines, *, server_port: int):
    req_time = {}
    req_gid = {}
    req_stream = {}
    req_ids = {}
    latency_by_request = {}
    stream_stats: dict[int, dict[str, int]] = {}
    for line in lines:
        parts = line.rstrip("\r\n").split("\t")
        if len(parts) < 5:
            continue
        parts.extend([""] * max(0, 13 - len(parts)))
        stream_id = _parse_int_field(parts[0])
        src_port = _parse_int_field(parts[1])
        dst_port = _parse_int_field(parts[2])
        frame_len = _parse_int_field(parts[3])
        tcp_payload_len = _parse_int_field(parts[4])
        # Keep frames before HTTP recognition, including handshake, ACKs and
        # retransmissions. A later /predict request may use this stream.
        if stream_id is not None and frame_len is not None and tcp_payload_len is not None:
            stats = stream_stats.setdefault(stream_id, {
                "request_wire_bytes": 0, "response_wire_bytes": 0, "tcp_payload_bytes": 0,
            })
            if dst_port == server_port:
                stats["request_wire_bytes"] += frame_len
                stats["tcp_payload_bytes"] += tcp_payload_len
            elif src_port == server_port:
                stats["response_wire_bytes"] += frame_len
                stats["tcp_payload_bytes"] += tcp_payload_len
        timestamp = _parse_finite_float_field(parts[6].strip())
        if timestamp is None:
            continue
        if parts[7] and "/predict" in parts[8].split(",") and parts[5].strip().isdigit():
            req_fn = int(parts[5])
            req_time[req_fn] = timestamp
            req_gid[req_fn] = extract_group_id_from_request_lines(parts[9])
            req_ids[req_fn] = extract_request_id(parts[9])
            if stream_id is not None:
                req_stream[req_fn] = stream_id
        # http.request_in refers to a request already dissected by tshark.
        # Keep the legacy handling of multiple/invalid associations and times.
        if parts[10] and parts[11].strip().isdigit():
            req_fn = int(parts[11])
            if req_fn in req_time:
                dt = timestamp - req_time[req_fn]
                if math.isfinite(dt) and dt >= 0:
                    latency_by_request[f"{req_gid[req_fn]}:{req_fn}"] = dt

    requests = {}
    stream_requests = Counter(req_stream.values())
    for request_id, latency_s in latency_by_request.items():
        req_fn = int(request_id.rsplit(":", 1)[1])
        stream_id = req_stream.get(req_fn)
        selected_stats = stream_stats.get(stream_id) if stream_id is not None else None
        record = {"latency_s": latency_s, "request_id": req_ids[req_fn], "tcp_stream": stream_id,
                  "wire_bytes_status": "unavailable_shared_stream" if stream_id is not None and stream_requests[stream_id] > 1 else
                                       "available" if selected_stats is not None else "unavailable"}
        if selected_stats is not None and stream_id is not None and stream_requests[stream_id] == 1:
            request_wire = int(selected_stats.get("request_wire_bytes", 0))
            response_wire = int(selected_stats.get("response_wire_bytes", 0))
            tcp_payload = int(selected_stats.get("tcp_payload_bytes", 0))
            total_wire = request_wire + response_wire
            protocol_overhead = max(0, total_wire - tcp_payload)
            record.update({
                "request_wire_bytes": request_wire,
                "response_wire_bytes": response_wire,
                "total_wire_bytes": total_wire,
                "tcp_payload_bytes": tcp_payload,
                "protocol_overhead_bytes": protocol_overhead,
                "protocol_overhead_ratio": (
                    protocol_overhead / total_wire if total_wire > 0 else None
                ),
            })
        requests[request_id] = record

    return {"schema_version": 2, "requests": requests, "streams": stream_stats}


if __name__ == "__main__":
    main()
