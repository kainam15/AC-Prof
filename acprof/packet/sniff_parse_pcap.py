#!/usr/bin/env python3
import json
import math
import sys
from collections import Counter
from typing import Sequence

from acprof.host.command import run_command

# 用法：
#   python3 -m acprof.packet.sniff_parse_pcap <pcap> <port>
#
# 输出 JSON schema v2：每个请求包含时延、线上帧字节、TCP payload 和
# L2/L3/L4 协议开销。共享连接仅提供逐请求 HTTP 时延和一次流级字节总量，
# 不把整个 tcp.stream 的字节重复归给每个请求。


def run(cmd):
    p = run_command(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(p.stderr.strip() or p.stdout.strip())
    return p.stdout


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
            return v.strip() or "group"
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
    server_port = _parse_int_field(port)
    if server_port is None:
        raise SystemExit(f"invalid TCP port: {port!r}")

    # 1) request: /predict 的 request frame -> time + group_id
    req_cmd = [
        "tshark", "-r", pcap,
        "-o", "tcp.desegment_tcp_streams:TRUE",
        "-o", "http.desegment_body:TRUE",
        "-d", f"tcp.port=={port},http",
        "-Y", f"tcp.port=={port} && http.request && http.request.uri==\"/predict\"",
        "-T", "fields", "-E", "separator=\t",
        "-e", "frame.number",
        "-e", "frame.time_epoch",
        "-e", "http.request.line",
        "-e", "tcp.stream",
    ]

    req_time = {}
    req_gid = {}
    req_stream = {}
    req_ids = {}
    for line in run(req_cmd).splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue

        fn = parts[0].strip()
        t = parts[1].strip()
        req_lines = parts[2] if len(parts) >= 3 else ""
        stream = parts[3].strip() if len(parts) >= 4 else ""

        if not fn.isdigit():
            continue
        req_fn = int(fn)
        request_time = _parse_finite_float_field(t)
        if request_time is None:
            continue
        group_id = extract_group_id_from_request_lines(req_lines)
        request_id = extract_request_id(req_lines)
        stream_id = _parse_int_field(stream)
        req_time[req_fn] = request_time
        req_gid[req_fn] = group_id
        req_ids[req_fn] = request_id
        if stream_id is not None:
            req_stream[req_fn] = stream_id

    # 2) response: http.request_in 指回对应 request frame -> response time
    resp_cmd = [
        "tshark", "-r", pcap,
        "-o", "tcp.desegment_tcp_streams:TRUE",
        "-o", "http.desegment_body:TRUE",
        "-d", f"tcp.port=={port},http",
        "-Y", f"tcp.port=={port} && http.response",
        "-T", "fields", "-E", "separator=\t",
        "-e", "http.request_in",
        "-e", "frame.time_epoch",
        "-e", "http.response.code",
    ]

    latency_by_request = {}
    for line in run(resp_cmd).splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue

        req_in, t = parts[0].strip(), parts[1].strip()
        if not req_in.isdigit():
            continue

        req_fn = int(req_in)
        response_time = _parse_finite_float_field(t)
        if req_fn in req_time and response_time is not None:
            dt = response_time - req_time[req_fn]
            if math.isfinite(dt) and dt >= 0:
                gid = req_gid.get(req_fn, "group")
                latency_by_request[f"{gid}:{req_fn}"] = dt

    # 3) Sum captured wire bytes and TCP payload for each /predict stream.
    # frame.len includes the captured link-layer frame; tcp.len excludes
    # Ethernet/IP/TCP headers. Retransmissions are intentionally retained.
    stream_cmd = [
        "tshark", "-r", pcap,
        "-Y", f"tcp.port=={port}",
        "-T", "fields", "-E", "separator=\t",
        "-e", "tcp.stream",
        "-e", "tcp.srcport",
        "-e", "tcp.dstport",
        "-e", "frame.len",
        "-e", "tcp.len",
    ]
    stream_stats = {}
    for line in run(stream_cmd).splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        stream_id = _parse_int_field(parts[0])
        src_port = _parse_int_field(parts[1])
        dst_port = _parse_int_field(parts[2])
        frame_len = _parse_int_field(parts[3])
        tcp_payload_len = _parse_int_field(parts[4])
        if stream_id is None or frame_len is None or tcp_payload_len is None:
            continue
        stats = stream_stats.setdefault(
            stream_id,
            {
                "request_wire_bytes": 0,
                "response_wire_bytes": 0,
                "tcp_payload_bytes": 0,
            },
        )
        if dst_port == server_port:
            stats["request_wire_bytes"] += frame_len
        elif src_port == server_port:
            stats["response_wire_bytes"] += frame_len
        else:
            continue
        stats["tcp_payload_bytes"] += tcp_payload_len

    requests = {}
    stream_requests = Counter(req_stream.values())
    for request_id, latency_s in latency_by_request.items():
        req_fn = int(request_id.rsplit(":", 1)[1])
        stream_id = req_stream.get(req_fn)
        stats = stream_stats.get(stream_id) if stream_id is not None else None
        record = {"latency_s": latency_s, "request_id": req_ids[req_fn], "tcp_stream": stream_id,
                  "wire_bytes_status": "unavailable_shared_stream" if stream_requests[stream_id] > 1 else
                                       "available" if stats is not None else "unavailable"}
        if stats is not None and stream_requests[stream_id] == 1:
            request_wire = int(stats.get("request_wire_bytes", 0))
            response_wire = int(stats.get("response_wire_bytes", 0))
            tcp_payload = int(stats.get("tcp_payload_bytes", 0))
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

    print(json.dumps(
        {"schema_version": 2, "requests": requests, "streams": stream_stats},
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
