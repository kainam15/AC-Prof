"""Real child processes exercise packet JSON and merge-file protocol failures."""
import csv
import json
import os
import shutil
import stat
import sys
from pathlib import Path

import pytest

from acprof.host.measurement import packet_processing
from acprof.host.measurement.capture import PacketLatencyError
from acprof.host.measurement.packet_processing import run_packet_command
from acprof.installation import module_command


@pytest.mark.parametrize("output", [
    "", "not json", '{"schema_version":2,', '[]', 'null', '{"case:4":0.002}',
    '{"requests":{"case:4":{}}}', '{"schema_version":1,"requests":{}}',
    '{"schema_version":2.0,"requests":{}}', '{"schema_version":true,"requests":{}}',
    '{"schema_version":2,"requests":[]}', '{"schema_version":2,"requests":{"case:4":0.002}}',
    '{"schema_version":2,"requests":{"case:4":{"latency_s":NaN}}}',
    '{"schema_version":2,"requests":{"case:4":{"latency_s":1e999}}}',
    '{"schema_version":2,"requests":{},"streams":[]}',
    '{"schema_version":2,"requests":{}}\n{"extra":"second document"}',
])
def test_parser_boundary_rejects_invalid_output(output):
    from acprof.host.measurement.packet_processing import parse_packet_output

    with pytest.raises(PacketLatencyError, match="pcap pars"):
        parse_packet_output([sys.executable, "-c", "import sys; sys.stdout.write(sys.argv[1])", output])


def test_empty_requests_and_stderr_diagnostics_are_distinct_from_protocol_failure():
    from acprof.host.measurement.packet_processing import parse_packet_output

    payload = {"schema_version": 2, "requests": {}, "streams": {}}
    command = [sys.executable, "-c",
               "import sys; print(sys.argv[1]); print('diagnostic only', file=sys.stderr)", json.dumps(payload)]
    assert parse_packet_output(command) == payload


def test_nonzero_parser_exit_is_not_overridden_by_valid_stdout():
    from acprof.host.measurement.packet_processing import parse_packet_output

    command = [sys.executable, "-c",
               "import sys; print(sys.argv[1]); print('worker failed', file=sys.stderr); sys.exit(7)",
               '{"schema_version":2,"requests":{}}']
    with pytest.raises(PacketLatencyError, match="worker failed"):
        parse_packet_output(command)


@pytest.mark.parametrize("payload", [
    [], {"case:4": 0.002}, {"schema_version": 2, "requests": []},
    {"schema_version": 2, "requests": {"case:4": []}},
    {"schema_version": 2, "requests": {}, "streams": []},
])
def test_merge_worker_rejects_the_same_invalid_envelopes_before_outputs(tmp_path, payload):
    source = tmp_path / "result.csv"
    source.write_text("retained input\n")
    packet = tmp_path / "packet.json"
    packet.write_text(json.dumps(payload))
    output = tmp_path / "merged.csv"
    result = run_packet_command([*module_command("acprof.packet.merge_packet_latency"),
                                 str(source), str(packet), str(output)], stage="packet latency merge")
    assert result.returncode != 0
    assert "packet" in result.stderr
    assert source.read_text() == "retained input\n"
    assert not output.exists()


def test_merge_worker_missing_csv_fails_without_publishing(tmp_path):
    packet = tmp_path / "packet.json"
    packet.write_text('{"schema_version":2,"requests":{"case:4":{"latency_s":0.002}}}')
    source, output = tmp_path / "missing.csv", tmp_path / "merged.csv"
    result = run_packet_command([*module_command("acprof.packet.merge_packet_latency"),
                                 str(source), str(packet), str(output)], stage="packet latency merge")
    assert result.returncode != 0
    assert str(source) in result.stderr
    assert not source.exists()
    assert not output.exists()


@pytest.mark.skipif(shutil.which("tshark") is None, reason="offline packet integration requires tshark")
@pytest.mark.parametrize("empty_current", [False, True])
def test_real_archived_capture_uses_the_same_protocol_without_losing_stream_identity(tmp_path, empty_current):
    from acprof.host.measurement.capture import packet_parser_command
    from acprof.host.measurement.packet_processing import parse_packet_output
    from acprof.host.measurement.packet_segments import merge_capture_segments
    from tests.support.pcap_fixture_builder import write_capture

    current, archived = tmp_path / "current.pcap", tmp_path / "archived.pcap"
    write_capture(current, request_group="newg", missing_response=empty_current)
    write_capture(archived, request_group="oldg")
    originals = {path: path.read_bytes() for path in (current, archived)}
    payload = parse_packet_output(packet_parser_command(current))
    result = merge_capture_segments(payload, [archived], 8002)
    expected = {"oldg:4"} if empty_current else {"newg:4", "oldg:4"}
    assert result["schema_version"] == 2
    assert set(result["requests"]) == expected
    assert "streams" not in result
    assert len(result["capture_segments"]) == 2
    for name, segment in (("oldg:4", 1), *([] if empty_current else [("newg:4", 0)])):
        record = result["requests"][name]
        assert record["capture_segment"] == segment
        assert record["tcp_stream"] == 0
        assert record["latency_s"] == pytest.approx(0.002, abs=1e-6)
        streams = result["capture_segments"][segment]["streams"]
        assert streams["0"]["request_wire_bytes"] == 320
    assert {path: path.read_bytes() for path in originals} == originals
    assert all("capture_segment" not in row for row in payload["requests"].values())


def test_parser_command_treats_paths_as_literal_arguments():
    from acprof.host.measurement.capture import packet_parser_command

    path = Path("captures") / "a path; not-a-shell-command.pcap"
    command = packet_parser_command(path, 8013)
    assert command == [*module_command("acprof.packet.sniff_parse_pcap"), str(path), "8013"]


@pytest.fixture
def merge_inputs(tmp_path):
    source = tmp_path / "result.csv"
    source.write_text("sniff_group_id,latency_s\ncase,nan\n")
    samples = tmp_path / "result.csv.requests.jsonl"
    samples.write_text(json.dumps({"schema_version": 1, "sniff_group_id": "case", "latency_app_s": [0.5]}) + "\n")
    source.chmod(0o640)
    samples.chmod(0o600)
    packets = tmp_path / "packets.json"
    packets.write_text(json.dumps({"schema_version": 2, "requests": {"case:19": {"request_id": "case:0", "latency_s": 0.25}}}))
    return source, samples, packets


def test_host_csv_replace_failure_leaves_request_journal_and_csv_intact(merge_inputs, monkeypatch):
    source, samples, packets = merge_inputs
    original = [path.read_bytes() for path in (source, samples)]
    replace = os.replace

    def block_csv(staged, destination):
        if Path(destination) == source:
            raise OSError("blocked CSV rename")
        return replace(staged, destination)

    monkeypatch.setattr(os, "replace", block_csv)
    with pytest.raises(OSError, match="blocked CSV rename"):
        packet_processing.merge_packet_csv(str(source), str(packets))
    assert [path.read_bytes() for path in (source, samples)] == original
    assert not list(source.parent.glob(".packet-merge-*"))


def test_host_journal_replace_failure_reports_error_after_csv_install(merge_inputs, monkeypatch):
    source, samples, packets = merge_inputs
    original_journal = samples.read_bytes()
    replace = os.replace

    def block_journal(staged, destination):
        if Path(destination) == samples:
            raise OSError("blocked journal rename")
        return replace(staged, destination)

    monkeypatch.setattr(os, "replace", block_journal)
    with pytest.raises(OSError, match="blocked journal rename"):
        packet_processing.merge_packet_csv(str(source), str(packets))
    with source.open() as stream:
        assert next(csv.DictReader(stream))["latency_s"] == "0.250000"
    assert samples.read_bytes() == original_journal
    assert not list(source.parent.glob(".packet-merge-*"))


@pytest.mark.parametrize("staged_sample", ["missing", "symlink"])
def test_host_requires_current_request_output_before_installing_csv(merge_inputs, monkeypatch, staged_sample):
    from subprocess import CompletedProcess

    source, samples, packets = merge_inputs
    original = [path.read_bytes() for path in (source, samples)]

    def omit_requests(command, *, stage):
        Path(command[-1]).write_text("new CSV")
        if staged_sample == "symlink":
            Path(command[command.index("--requests-output") + 1]).symlink_to(samples)
        return CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(packet_processing, "run_packet_command", omit_requests)
    with pytest.raises(PacketLatencyError, match="merge did not produce"):
        packet_processing.merge_packet_csv(str(source), str(packets))
    assert [path.read_bytes() for path in (source, samples)] == original


@pytest.mark.parametrize("with_samples", [False, True], ids=["historical", "requests"])
def test_host_installs_csv_then_optional_requests_and_preserves_modes(merge_inputs, monkeypatch, with_samples):
    source, samples, packets = merge_inputs
    if not with_samples:
        samples.unlink()
    installed = []
    replace = os.replace

    def observe(staged, destination):
        if Path(destination) in (source, samples):
            installed.append(Path(destination))
        return replace(staged, destination)

    monkeypatch.setattr(os, "replace", observe)
    packet_processing.merge_packet_csv(str(source), str(packets))

    assert installed == ([source, samples] if with_samples else [source])
    assert stat.S_IMODE(source.stat().st_mode) == 0o640
    with source.open() as stream:
        assert next(csv.DictReader(stream))["latency_s"] == "0.250000"
    if with_samples:
        assert stat.S_IMODE(samples.stat().st_mode) == 0o600
        assert json.loads(samples.read_bytes())["latency_packet_s"] == [0.25]
    else:
        assert not samples.exists()
