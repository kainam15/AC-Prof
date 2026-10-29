import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from acprof.host.measurement import packet_processing
from acprof.host.measurement.capture import PacketLatencyError


@pytest.mark.parametrize("value", ["0", "9", "3601", "nan", "inf", "invalid"])
def test_invalid_derivation_budget_is_rejected(monkeypatch, value):
    monkeypatch.setenv("ACPROF_POSTPROCESS_TIMEOUT_S", value)
    with pytest.raises(ValueError, match="ACPROF_POSTPROCESS_TIMEOUT_S"):
        packet_processing.postprocessing_timeout()


def test_real_parser_timeout_stops_tshark_descendant(tmp_path, monkeypatch):
    tool = tmp_path / "tshark"
    pid_file = tmp_path / "pid"
    tool.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "pathlib.Path(os.environ['TEST_TSHARK_PID']).write_text(str(os.getpid()))\n"
        "print('started', flush=True)\n"
        "time.sleep(30)\n"
    )
    tool.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("TEST_TSHARK_PID", str(pid_file))
    monkeypatch.setattr(packet_processing, "postprocessing_timeout", lambda: 0.8)
    with pytest.raises(PacketLatencyError, match="pcap parsing timed out") as caught:
        packet_processing.run_packet_command(
            [sys.executable, "-m", "acprof.packet.sniff_parse_pcap", "fixture.pcap", "8002"],
            stage="pcap parsing",
        )
    assert isinstance(caught.value.__cause__, subprocess.TimeoutExpired)
    stat = Path(f"/proc/{int(pid_file.read_text())}/stat")
    # A grandchild may await adoption/reaping; it must no longer execute.
    deadline = time.monotonic() + 2
    while stat.exists() and not stat.read_text().split(") ", 1)[1].startswith("Z "):
        assert time.monotonic() < deadline, "tshark descendant is still executing"
        time.sleep(0.01)
