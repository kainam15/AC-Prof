"""Runtime checks must not elevate privileges or install capabilities."""
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from acprof.host import env_utils, packet_capture
from acprof.monitors import perf_mips


def test_perf_permission_failure_never_retries_with_sudo():
    failed = subprocess.CompletedProcess([], 1, "", "Permission denied")
    with patch.object(perf_mips.shutil, "which", return_value="/usr/bin/perf"), patch.object(
        perf_mips, "run_command", return_value=failed,
    ) as run, pytest.raises(perf_mips.MIPSProfilingError):
        perf_mips.resolve_perf_command_prefix(env={"PATH": "/usr/bin"})
    assert (run.call_count) == (1)
    assert ("sudo") not in (run.call_args.args[0])
    assert (run.call_args.kwargs["input"]) == ("")

def test_pid_attach_failure_never_retries_with_sudo():
    failed = subprocess.CompletedProcess([], 1, "", "Permission denied")
    with patch.object(perf_mips.shutil, "which", return_value="/usr/bin/perf"), patch.object(
        perf_mips, "run_command", return_value=failed,
    ) as run, pytest.raises(perf_mips.MIPSProfilingError):
        perf_mips.resolve_perf_command_prefix_for_pid(1234)
    assert (run.call_count) == (1)
    assert ("sudo") not in (run.call_args.args[0])
    assert (run.call_args.kwargs["input"]) == ("")

def test_packet_permission_failure_does_not_grant_capability():
    failed = subprocess.CompletedProcess([], 0, "", "")
    with patch.object(packet_capture.shutil, "which", side_effect=lambda name: "/usr/bin/" + name), patch.object(
        packet_capture.os, "geteuid", return_value=1000,
    ), patch("acprof.host.command.run_command", return_value=failed) as run:
        with pytest.raises(packet_capture.PacketLatencyError):
            packet_capture._resolve_packet_latency_runtime(".", "/tmp/test.pcap", "docker0")
    assert (all(call.args[0][0] == "getcap" for call in run.call_args_list))

def test_tcpdump_needs_only_effective_net_raw_and_disables_promiscuous_mode():
    with patch.object(packet_capture.shutil, "which", side_effect=lambda name: "/usr/bin/" + name), patch.object(
        packet_capture.os, "geteuid", return_value=1000,
    ), patch("acprof.host.command.run_command", return_value=subprocess.CompletedProcess(
        [], 0, "/usr/bin/tcpdump cap_net_raw=ep\n", "",
    )):
        runtime = packet_capture._resolve_packet_latency_runtime(".", "/tmp/test.pcap", "docker0")
    assert (runtime.tcpdump_cmd[0]) == ("/usr/bin/tcpdump")
    assert ("-p") in (runtime.tcpdump_cmd)

def test_local_password_setting_is_rejected_without_exposing_value():
    with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
        Path(directory, ".env.local").write_text("ACPROF_SUDO_PASSWORD=test-retired-secret\n")
        with pytest.raises(ValueError, match="ACPROF_SUDO_PASSWORD") as raised:
            env_utils.load_project_env(directory)
        assert ("test-retired-secret") not in (str(raised.value))
        assert ("ACPROF_SUDO_PASSWORD") not in (os.environ)
