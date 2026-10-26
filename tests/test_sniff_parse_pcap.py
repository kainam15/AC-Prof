import contextlib
import io
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from acprof.packet import sniff_parse_pcap


def packet(*, frame=1, time=100.0, stream=7, source=50000, destination=8002,
           wire=100, payload=0, request='', uri='', headers='', response='', request_in=''):
    return '\t'.join(map(str, (stream, source, destination, wire, payload, frame, time,
                              request, uri, headers, response, request_in, '200' if response else '')))


def run_parser(rows, port='8002'):
    output = io.StringIO()
    with patch.object(sniff_parse_pcap, 'stream_tshark', return_value=(row for row in rows)), contextlib.redirect_stdout(output):
        sniff_parse_pcap.main(['capture.pcap', port])
    return json.loads(output.getvalue())


def test_reused_connection_keeps_request_latencies_without_double_counting_stream_bytes():
    rows = []
    for frame, time, index in [(10, 100.0, 0), (20, 101.0, 1)]:
        rows.extend([
            packet(frame=frame, time=time, request=1, uri='/predict', headers=f'X-Req-Id: load:{index}'),
            packet(frame=frame + 1, time=time + (0.2 if index == 0 else 0.4), response=1, request_in=frame),
        ])
    result = run_parser(rows)
    requests = result['requests']
    assert len(requests) == 2
    assert requests['load:10']['latency_s'] == pytest.approx(0.2)
    assert requests['load:20']['latency_s'] == pytest.approx(0.4)
    assert 'total_wire_bytes' not in requests['load:10']
    assert requests['load:20']['request_id'] == 'load:1'
    assert result['streams']['7']['request_wire_bytes'] == 400


def test_emits_schema_v2_latency_and_wire_metrics_including_pre_http_frames():
    result = run_parser([
        packet(frame=1, wire=100),
        packet(frame=10, wire=200, payload=150, request=1, uri='/predict', headers='X-Req-Id: case_seq1_r0:0'),
        packet(frame=11, source=8002, destination=50000, wire=120),
        packet(frame=12, time=100.25, source=8002, destination=50000, wire=480, payload=400, response=1, request_in=10),
    ])
    assert result['schema_version'] == 2
    record = result['requests']['case_seq1_r0:10']
    assert record['latency_s'] == 0.25
    assert record['request_wire_bytes'] == 300
    assert record['response_wire_bytes'] == 600
    assert record['total_wire_bytes'] == 900
    assert record['tcp_payload_bytes'] == 550
    assert record['protocol_overhead_bytes'] == 350
    assert record['protocol_overhead_ratio'] == pytest.approx(350 / 900)


def test_nonfinite_response_timestamp_is_not_emitted():
    result = run_parser([
        packet(frame=10, request=1, uri='/predict', headers='X-Req-Id: case:0'),
        packet(frame=11, time='1e309', response=1, request_in=10),
    ])
    assert result['requests'] == {}


@pytest.mark.parametrize('port', [
    '0', '65536', '-1', '8002 && ip.addr==1.2.3.4', '8002,8080',
    '8002 || true', 'abc', '８００２', '8002\n',
])
def test_rejects_invalid_port_before_launching_tshark(port):
    with patch.object(sniff_parse_pcap, 'stream_tshark') as run, pytest.raises(SystemExit, match='invalid TCP port'):
        sniff_parse_pcap.main(['capture.pcap', port])
    run.assert_not_called()


def test_port_is_canonicalized_in_single_scan_with_reassembly():
    with patch.object(sniff_parse_pcap, 'stream_tshark', return_value=(row for row in [])) as run, contextlib.redirect_stdout(io.StringIO()):
        sniff_parse_pcap.main(['capture.pcap', '08002'])
    run.assert_called_once()
    command = run.call_args.args[0]
    assert 'tcp.port==08002' not in ' '.join(command)
    assert command[command.index('-Y') + 1] == 'tcp.port==8002'
    assert 'tcp.port==8002,http' in command
    assert 'tcp.desegment_tcp_streams:TRUE' in command
    assert 'http.desegment_body:TRUE' in command


def test_request_parser_surfaces_unexpected_header_errors_and_closes_stream():
    closed = []

    def rows():
        try:
            yield packet(frame=10, request=1, uri='/predict', headers='X-Req-Id: case:0')
        finally:
            closed.append(True)

    with patch.object(sniff_parse_pcap, 'stream_tshark', return_value=rows()), patch.object(
        sniff_parse_pcap, 'extract_request_id', side_effect=RuntimeError('parser bug'),
    ), pytest.raises(RuntimeError, match='parser bug'):
        sniff_parse_pcap.main(['capture.pcap', '8002'])
    assert closed == [True]


@pytest.mark.skipif(shutil.which('tshark') is None, reason='tshark required for golden PCAP acceptance')
@pytest.mark.parametrize('name', ['single', 'segmented', 'retransmission', 'reuse', 'missing-response', 'ready-prefix'])
def test_golden_capture_matches_legacy_parser(name, capsys):
    capture = Path(__file__).parent / 'fixtures' / 'packet' / name
    sniff_parse_pcap.main([str(capture.with_suffix('.pcap')), '8002'])
    assert json.loads(capsys.readouterr().out) == json.loads(capture.with_suffix('.json').read_text())


@pytest.mark.skipif(shutil.which('tshark') is None, reason='tshark required for malformed PCAP acceptance')
def test_invalid_capture_surfaces_tshark_error():
    capture = Path(__file__).parent / 'fixtures' / 'packet' / 'invalid.pcap'
    with pytest.raises(SystemExit, match="isn't a capture file"):
        sniff_parse_pcap.main([str(capture), '8002'])
