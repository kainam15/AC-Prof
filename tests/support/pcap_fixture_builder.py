"""Deterministic Ethernet/IPv4/TCP captures; no sockets or Scapy dependency."""
import socket
import struct
from pathlib import Path


def _checksum(data):
    if len(data) % 2:
        data += b"\0"
    total = sum(struct.unpack(f"!{len(data) // 2}H", data))
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return ~total & 0xFFFF


def write_capture(path: Path, *, segmented=False, retransmit=False, reuse=False,
                  missing_response=False, ready=False, padding_frames=0):
    frames = []
    client_ip, server_ip = map(socket.inet_aton, ("192.0.2.10", "198.51.100.20"))
    client_seq, server_seq = 1000, 2000

    def packet(from_server, seq, ack, flags, payload=b""):
        src, dst = (server_ip, client_ip) if from_server else (client_ip, server_ip)
        sport, dport = (8002, 50000) if from_server else (50000, 8002)
        tcp = struct.pack("!HHIIBBHHH", sport, dport, seq, ack, 5 << 4, flags, 65535, 0, 0)
        pseudo = src + dst + struct.pack("!BBH", 0, 6, len(tcp) + len(payload))
        checksum = _checksum(pseudo + tcp + payload)
        tcp = tcp[:16] + struct.pack("!H", checksum) + tcp[18:]
        ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 40 + len(payload), len(frames) % 65536,
                         0x4000, 64, 6, 0, src, dst)
        ip = ip[:10] + struct.pack("!H", _checksum(ip)) + ip[12:]
        frames.append(b"\x02\0\0\0\0\x02\x02\0\0\0\0\x01\x08\0" + ip + tcp + payload)

    packet(False, client_seq, 0, 0x02)
    packet(True, server_seq, client_seq + 1, 0x12)
    client_seq += 1
    server_seq += 1
    packet(False, client_seq, server_seq, 0x10)
    for _ in range(padding_frames):
        packet(False, client_seq, server_seq, 0x10)
    routes = (["/ready"] if ready else []) + ["/predict"] * (2 if reuse else 1)
    for index, route in enumerate(routes):
        request = (f"POST {route} HTTP/1.1\r\nHost: fixture\r\nX-Req-Id: case:{index}\r\n"
                   "Content-Length: 2\r\nConnection: keep-alive\r\n\r\n{}").encode()
        chunks = [request[:35], request[35:]] if segmented else [request]
        for chunk in chunks:
            packet(False, client_seq, server_seq, 0x18, chunk)
            if retransmit:
                packet(False, client_seq, server_seq, 0x18, chunk)
            client_seq += len(chunk)
            packet(True, server_seq, client_seq, 0x10)
        if missing_response and index == len(routes) - 1:
            continue
        response = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: keep-alive\r\n\r\n{}"
        for chunk in ([response[:24], response[24:]] if segmented else [response]):
            packet(True, server_seq, client_seq, 0x18, chunk)
            server_seq += len(chunk)
            packet(False, client_seq, server_seq, 0x10)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        for index, frame in enumerate(frames):
            stream.write(struct.pack("<IIII", 1_700_000_000 + index // 1000,
                                     index % 1000 * 1000, len(frame), len(frame)))
            stream.write(frame)
