import io
import math
from contextlib import redirect_stderr
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from acprof.monitors import perf_mips


@pytest.mark.parametrize('running', [False, True])
def test_close_releases_perf_output_streams_even_after_process_exit(running):
    monitor = perf_mips.PerfMIPSMonitor('fixture')
    process = SimpleNamespace(poll=lambda: None if running else 0, kill=Mock(),
                              communicate=Mock(side_effect=OSError('pipe read failed')),
                              stdin=None, stdout=io.StringIO(), stderr=io.StringIO())
    monitor._proc = process
    if running:
        with pytest.raises(OSError, match='pipe read failed'):
            monitor.close()
    else:
        monitor.close()
    assert process.stdout.closed
    assert process.stderr.closed
    assert process.kill.call_count == int(running)


def test_control_pipe_failure_does_not_skip_other_pipes_or_process_cleanup(monkeypatch):
    monitor = perf_mips.PerfMIPSMonitor('fixture')
    monitor._control_write, monitor._ack_read = 10, 11
    process = SimpleNamespace(poll=lambda: None, kill=Mock(), communicate=Mock(),
                              stdin=None, stdout=io.StringIO(), stderr=io.StringIO())
    monitor._proc = process
    close = Mock(side_effect=[OSError('control close failed'), None])
    monkeypatch.setattr(perf_mips.os, 'close', close)
    with pytest.raises(OSError, match='control close failed'):
        monitor.close()
    assert {call.args[0] for call in close.call_args_list} == {10, 11}
    process.kill.assert_called_once_with()
    assert process.stdout.closed
    assert process.stderr.closed


class TestPerfMIPS:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        locator = patch.object(perf_mips.shutil, "which", return_value="/usr/bin/perf")
        locator.start()
        self._request.addfinalizer(partial(locator.stop))

    def test_start_enables_already_prepared_perf_without_spawning_inside_window(self):
        monitor = perf_mips.PerfMIPSMonitor("case")
        with patch.object(perf_mips.common, "docker_container_pid", return_value=1234), patch.object(
            perf_mips, "resolve_perf_command_prefix_for_pid", return_value=["perf"],
        ), patch.object(perf_mips.subprocess, "Popen") as launch, patch.object(
            monitor, "_control", return_value=None,
        ) as control:
            launch.return_value.poll.return_value = None
            monitor.prepare()
            assert launch.call_count == 1
            assert "1234" in launch.call_args.args[0]
            assert "--delay=-1" in launch.call_args.args[0]
            assert any(str(arg).startswith("--control=fd:") for arg in launch.call_args.args[0])
            with patch.object(perf_mips.common, "docker_container_pid", side_effect=AssertionError("window discovery")):
                monitor.start()
            assert launch.call_count == 1
            assert [call.args[0] for call in control.call_args_list] == ["ping", "enable"]
            monitor.close()

    def test_control_ack_timeout_is_reported_as_failure(self):
        monitor = perf_mips.PerfMIPSMonitor("case")
        monitor._proc = SimpleNamespace(poll=lambda: None)
        monitor._control_write = 10
        monitor._ack_read = 11
        with patch.object(perf_mips.os, "write", return_value=7) as send, patch.object(
            perf_mips.select, "select", return_value=([], [], [])
        ), pytest.raises(perf_mips.MIPSProfilingError, match="enable timed out before ack"):
            monitor._control("enable")
        send.assert_called_once_with(10, b"enable\n")
        monitor._control_write = None
        monitor._ack_read = None

    def test_prepare_ack_failure_kills_process_and_closes_channels(self):
        monitor = perf_mips.PerfMIPSMonitor("case", command_prefix=["perf"])
        process = SimpleNamespace(poll=lambda: None, kill=Mock(), communicate=Mock())
        with patch.object(perf_mips.common, "docker_container_pid", return_value=1234), patch.object(
            perf_mips.subprocess, "Popen", return_value=process,
        ), patch.object(monitor, "_control", side_effect=perf_mips.MIPSProfilingError("ping timeout")):
            with pytest.raises(perf_mips.MIPSProfilingError, match="ping timeout"):
                monitor.prepare()
        process.kill.assert_called_once()
        assert monitor._control_write is None
        assert monitor._ack_read is None

    def test_unprepared_start_is_rejected_without_discovery(self):
        with patch.object(perf_mips.common, "docker_container_pid") as discover:
            with pytest.raises(perf_mips.MIPSProfilingError, match="prepared"):
                perf_mips.PerfMIPSMonitor("case").start()
        discover.assert_not_called()

    def test_preflight_rejects_cross_user_attach_denial_after_self_probe_passes(self):
        results = [
            SimpleNamespace(returncode=0, stdout='', stderr='1000,,instructions,100,100.00,,\n'),
            SimpleNamespace(returncode=1, stdout='', stderr='Error:\nNo supported events found.\nAccess denied'),
        ]
        with patch.object(perf_mips.shutil, 'which', return_value='/usr/bin/perf'), patch.object(
            perf_mips, 'run_command', side_effect=results,
        ) as run:
            with pytest.raises(perf_mips.MIPSProfilingError, match='PID 1'):
                perf_mips.resolve_perf_command_prefix(env={'PATH': '/usr/bin'})
        assert ('-p') in (run.call_args.args[0])
        assert (run.call_args.kwargs['env']) == ({'PATH': '/usr/bin'})
        assert (all(call.args[0][0] == 'perf' for call in run.call_args_list))

    def test_real_cycles_and_ipc_use_scaled_pmu_counts(self):
        parsed = perf_mips.parse_perf_stat_output(
            "1200,,instructions,100000,50.00,,\n"
            "600,,cycles,100000,50.00,,\n"
            "400,,ref-cycles,200000,100.00,,\n",
            fallback_elapsed_s=0.2,
        )
        assert (getattr(parsed, "cycles_total", None)) == (600)
        assert (parsed.ref_cycles_total) == (400)
        assert (parsed.ipc) == (2.0)
        assert (parsed.running_pct) == (50.0)

    def test_partial_hybrid_cycles_must_not_produce_ipc(self):
        parsed = perf_mips.parse_perf_stat_output(
            "1200,,cpu_core/instructions/,100000,100.00,,\n"
            "600,,cpu_atom/instructions/,100000,100.00,,\n"
            "600,,cpu_core/cycles/,100000,100.00,,\n"
            "<not counted>,,cpu_atom/cycles/,0,0.00,,\n"
            "<not supported>,,ref-cycles/,0,0.00,,\n",
            fallback_elapsed_s=0.2,
        )
        assert (math.isnan(getattr(parsed, "ipc", 0.0)))
        assert (math.isnan(parsed.cycles_total))
        assert (math.isnan(parsed.ref_cycles_total))

    def test_missing_pmu_row_is_not_a_complete_cycle_total(self):
        parsed = perf_mips.parse_perf_stat_output(
            "100,,cpu_core/instructions/,100,100.00,,\n"
            "50,,cpu_atom/instructions/,100,100.00,,\n"
            "40,,cpu_core/cycles/,100,100.00,,\n",
            fallback_elapsed_s=0.2,
        )
        assert (math.isnan(parsed.cycles_total))
        assert (math.isnan(parsed.ipc))

    def test_parses_perf_stat_csv_output(self) -> None:
        parsed = perf_mips.parse_perf_stat_output(
            """
123456789,,instructions,100.00,,
200000,,cache-references,100.00,,
10000,,cache-misses,100.00,,
50000,,dTLB-loads,100.00,,
250,,dTLB-load-misses,100.00,,
1.250000000 seconds time elapsed
"""
        )

        assert (parsed.instructions_total) == (123_456_789)
        assert (parsed.perf_elapsed_s) == (1.25) or round(abs((parsed.perf_elapsed_s) - (1.25)), 7) == 0
        assert (parsed.cache_references_total) == (200_000)
        assert (parsed.cache_misses_total) == (10_000)
        assert (parsed.dtlb_loads_total) == (50_000)
        assert (parsed.dtlb_load_misses_total) == (250)

    def test_parses_hybrid_pmu_event_labels(self) -> None:
        parsed = perf_mips.parse_perf_stat_output(
            """
1000,,cpu_core/instructions/,100.00,,
500,,cpu_atom/instructions/,100.00,,
200,,cpu_core/cache-references/,100.00,,
100,,cpu_atom/cache-references/,100.00,,
20,,cpu_core/cache-misses/,100.00,,
10,,cpu_atom/cache-misses/,100.00,,
50,,cpu_core/dTLB-loads/,100.00,,
25,,cpu_atom/dTLB-loads/,100.00,,
5,,cpu_core/dTLB-load-misses/,100.00,,
2,,cpu_atom/dTLB-load-misses/,100.00,,
""",
            fallback_elapsed_s=0.25,
        )

        assert (parsed.instructions_total) == (1_500)
        assert (parsed.cache_references_total) == (300)
        assert (parsed.cache_misses_total) == (30)
        assert (parsed.dtlb_loads_total) == (75)
        assert (parsed.dtlb_load_misses_total) == (7)

    def test_optional_events_can_be_unsupported_or_zero(self) -> None:
        parsed = perf_mips.parse_perf_stat_output(
            """
1000,,instructions,100.00,,
<not supported>,,cache-references,0.00,,
<not counted>,,cache-misses,0.00,,
0,,dTLB-loads,100.00,,
0,,dTLB-load-misses,100.00,,
""",
            fallback_elapsed_s=0.25,
        )

        assert (math.isnan(parsed.cache_references_total))
        assert (math.isnan(parsed.cache_misses_total))
        assert (parsed.dtlb_loads_total) == (0)
        assert (parsed.dtlb_load_misses_total) == (0)
        assert (math.isnan(perf_mips._miss_rate_pct(0.0, 0.0)))

    def test_preflight_accepts_modern_perf_csv_without_elapsed_line(self) -> None:
        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["perf", "stat"]:
                return SimpleNamespace(
                    returncode=0,
                    stdout="",
                    stderr="1000,,instructions,633068,100.00,,\n",
                )
            raise AssertionError(f"unexpected command: {cmd}")

        with patch("acprof.monitors.perf_mips.shutil.which", return_value="/usr/bin/perf"), patch(
            "acprof.monitors.perf_mips.run_command",
            side_effect=fake_run,
        ):
            prefix = perf_mips.resolve_perf_command_prefix()

        assert (prefix) == (["perf"])


    def test_failed_pmu_probe_is_bounded_and_preserves_unknown_state(self):
        with patch.object(
            perf_mips, "run_command",
            side_effect=perf_mips.subprocess.TimeoutExpired(["perf", "stat"], 1),
        ) as execute:
            result = perf_mips.probe_event_set(["perf"])
        assert result["events"] == ["instructions"]
        assert result["event_statuses"]["dTLB-load-misses"] == "probe_error"
        assert result["optional_probe_error"]
        assert execute.call_count <= 14

    def test_grouped_event_spec_keeps_ratio_pairs_together(self):
        spec = perf_mips.perf_event_spec(perf_mips.PERF_EVENTS)
        assert spec == (
            "{instructions,cycles},ref-cycles,{cache-references,cache-misses},"
            "{dTLB-loads,dTLB-load-misses}"
        )
        assert perf_mips.perf_event_spec(("instructions", "dTLB-load-misses")) == (
            "instructions,dTLB-load-misses"
        )

    def test_grouped_full_probe_preserves_multiplexed_counters(self):
        calls = []

        def run(cmd, **kwargs):
            calls.append(cmd)
            events = cmd[cmd.index("-e") + 1].replace("{", "").replace("}", "").split(",")
            output = "".join(f"100,,{event},400000,50.00,,\n" for event in events)
            return SimpleNamespace(returncode=0, stderr=output, stdout="")

        with patch.object(perf_mips, "run_command", side_effect=run):
            result = perf_mips.probe_event_set(["perf"])
        assert result["events"] == list(perf_mips.PERF_EVENTS)
        assert not result["unavailable_events"]
        assert result["event_running_pct"]["dTLB-load-misses"] == 50.0
        assert "dTLB-load-misses" in result["multiplexed_events"]
        assert result["probe_attempts"] == len(calls) == 1

    def test_not_counted_in_full_set_is_not_hardware_unsupported(self):
        calls = []

        def run(cmd, **kwargs):
            events = cmd[cmd.index("-e") + 1].replace("{", "").replace("}", "").split(",")
            calls.append(events)
            output = ""
            for event in events:
                # The crowded full set cannot schedule this counter; the
                # pair and the event itself are fine without ref-cycles.
                missing = (
                    event == "dTLB-load-misses" and "ref-cycles" in events
                )
                count = "<not counted>" if missing else "100"
                percent = "0.00" if missing else "100.00"
                output += f"{count},,{event},400000,{percent},,\n"
            return SimpleNamespace(returncode=0, stderr=output, stdout="")

        with patch.object(perf_mips, "run_command", side_effect=run):
            result = perf_mips.probe_event_set(["perf"])
        assert "dTLB-load-misses" in result["events"]
        assert "dTLB-loads" in result["events"]
        assert "ref-cycles" not in result["events"]
        assert result["event_statuses"]["ref-cycles"] == "combination_limited"
        assert result["event_statuses"]["dTLB-load-misses"] == "available"
        assert calls == [
            list(perf_mips.PERF_EVENTS), ["dTLB-load-misses"],
            [event for event in perf_mips.PERF_EVENTS if event != "ref-cycles"],
        ]

    @pytest.mark.parametrize(
        ("raw", "single_code", "single_output", "expected"),
        [
            ("<not supported>", 0, "<not supported>", "unsupported"),
            ("<not counted>", 1, "Permission denied", "permission_denied"),
        ],
    )
    def test_single_event_probe_identifies_non_conflict_failures(
        self, raw, single_code, single_output, expected
    ):
        def run(cmd, **kwargs):
            events = cmd[cmd.index("-e") + 1].replace("{", "").replace("}", "").split(",")
            if events == ["cache-misses"]:
                if single_code:
                    return SimpleNamespace(
                        returncode=single_code, stderr=single_output, stdout=""
                    )
                return SimpleNamespace(
                    returncode=0, stderr=f"{single_output},,cache-misses,0,0.00,,\n",
                    stdout="",
                )
            output = "".join(
                f"{raw if event == 'cache-misses' else '100'},,{event},400000,"
                f"{'0.00' if event == 'cache-misses' else '100.00'},,\n"
                for event in events
            )
            return SimpleNamespace(returncode=0, stderr=output, stdout="")

        with patch.object(perf_mips, "run_command", side_effect=run):
            result = perf_mips.probe_event_set(["perf"])
        assert result["event_statuses"]["cache-misses"] == expected
        assert "cache-misses" in result["unavailable_events"]
        assert "dTLB-load-misses" in result["events"]

    def test_partial_hybrid_miss_counters_do_not_publish_partial_values(self):
        parsed = perf_mips.parse_perf_stat_output(
            "100,,cpu_core/instructions/,100000,100.00,,\n"
            "100,,cpu_atom/instructions/,100000,100.00,,\n"
            "10,,cpu_core/cache-references/,100000,100.00,,\n"
            "10,,cpu_atom/cache-references/,100000,100.00,,\n"
            "3,,cpu_core/cache-misses/,100000,100.00,,\n"
            "<not counted>,,cpu_atom/cache-misses/,0,0.00,,\n"
            "10,,cpu_core/dTLB-loads/,100000,100.00,,\n"
            "10,,cpu_atom/dTLB-loads/,100000,100.00,,\n"
            "2,,cpu_core/dTLB-load-misses/,100000,100.00,,\n"
            "<not counted>,,cpu_atom/dTLB-load-misses/,0,0.00,,\n",
            fallback_elapsed_s=0.1,
        )
        assert math.isnan(parsed.cache_misses_total)
        assert math.isnan(parsed.dtlb_load_misses_total)
        assert not parsed.cache_pair_comparable
        assert not parsed.dtlb_pair_comparable

    def test_differing_running_times_make_counter_ratios_unavailable(self):
        parsed = perf_mips.parse_perf_stat_output(
            "1000,,instructions,400000,100.00,,\n"
            "500,,cycles,300000,75.00,,\n"
            "200,,dTLB-loads,400000,100.00,,\n"
            "10,,dTLB-load-misses,200000,50.00,,\n",
            fallback_elapsed_s=0.1,
        )
        assert math.isnan(parsed.ipc)
        assert not parsed.dtlb_pair_comparable
        assert parsed.dtlb_load_misses_total == 10
        assert parsed.running_pct == 50.0

    def test_monitor_uses_direct_perf_when_preflight_allows_it(self) -> None:
        popen_cmds = []

        class FakeProcess:
            returncode = 0

            def __init__(self, cmd, **kwargs):
                popen_cmds.append(cmd)
                self.stderr = io.StringIO(
                    "500000,,instructions,100.00,,\n"
                    "20000,,cache-references,100.00,,\n"
                    "1000,,cache-misses,100.00,,\n"
                    "4000,,dTLB-loads,100.00,,\n"
                    "40,,dTLB-load-misses,100.00,,\n"
                    "0.250000000 seconds time elapsed\n"
                )

            def send_signal(self, sig):
                self.signal = sig

            def poll(self):
                return None

            def communicate(self, timeout=None):
                self.returncode = 0
                return "", self.stderr.getvalue()

            def kill(self):
                self.returncode = -9

        fake_pid = SimpleNamespace(returncode=0, stdout="1234\n", stderr="")
        with patch("acprof.monitors.common.run_command", return_value=fake_pid), patch(
            "acprof.monitors.perf_mips.subprocess.Popen",
            side_effect=lambda cmd, **kwargs: FakeProcess(cmd, **kwargs),
        ), patch.object(perf_mips.PerfMIPSMonitor, "_control", return_value=None):
            monitor = perf_mips.PerfMIPSMonitor(
                container_name="case_container",
                command_prefix=["perf"],
            )
            monitor.prepare()
            monitor.start()
            result = monitor.stop(repeat_in_window=2, latency_app_s=0.125)

        assert (popen_cmds[0][:5]) == (["/usr/bin/perf", "stat", "--no-big-num", "-x", ","])
        assert ("-p") in (popen_cmds[0])
        assert ("1234") in (popen_cmds[0])
        assert perf_mips.perf_event_spec(perf_mips.PERF_EVENTS) in popen_cmds[0]
        assert (result.instructions_total) == (500_000)
        assert (result.instructions_per_request) == (250_000.0)
        assert (result.cpu_mips_app) == (2.0) or round(abs((result.cpu_mips_app) - (2.0)), 7) == 0
        assert (result.cache_references_per_request) == (10_000.0)
        assert (result.cache_misses_per_request) == (500.0)
        assert (result.cache_miss_rate_pct) == (5.0) or round(abs((result.cache_miss_rate_pct) - (5.0)), 7) == 0
        assert (result.dtlb_loads_per_request) == (2_000.0)
        assert (result.dtlb_load_misses_per_request) == (20.0)
        assert (result.dtlb_load_miss_rate_pct) == (1.0) or round(abs((result.dtlb_load_miss_rate_pct) - (1.0)), 7) == 0

    def test_monitor_does_not_start_when_pid_attach_is_denied(self):
        with patch('acprof.monitors.common.docker_container_pid', return_value=1234), patch.object(
            perf_mips.shutil, 'which', return_value='/usr/bin/perf',
        ), patch.object(perf_mips, 'run_command', return_value=SimpleNamespace(
            returncode=1, stdout='', stderr='Permission denied',
        )), patch.object(perf_mips.subprocess, 'Popen') as popen:
            with pytest.raises(perf_mips.MIPSProfilingError, match='Permission denied'):
                perf_mips.PerfMIPSMonitor('case_container').prepare()
        popen.assert_not_called()

    def test_monitor_uses_wall_elapsed_when_perf_omits_elapsed_line(self) -> None:
        class FakeProcess:
            returncode = 0

            def poll(self):
                return None

            def send_signal(self, sig):
                self.signal = sig

            def communicate(self, timeout=None):
                self.returncode = 0
                return "", "500000,,instructions,633068,100.00,,\n"

            def kill(self):
                self.returncode = -9

        fake_pid = SimpleNamespace(returncode=0, stdout="1234\n", stderr="")
        with patch("acprof.monitors.common.run_command", return_value=fake_pid), patch(
            "acprof.monitors.perf_mips.subprocess.Popen",
            side_effect=lambda cmd, **kwargs: FakeProcess(),
        ), patch("acprof.monitors.perf_mips.time.perf_counter", side_effect=[10.0, 10.25]), patch.object(
            perf_mips.PerfMIPSMonitor, "_control", return_value=None,
        ):
            monitor = perf_mips.PerfMIPSMonitor(
                container_name="case_container",
                command_prefix=["perf"],
            )
            monitor.prepare()
            monitor.start()
            result = monitor.stop(repeat_in_window=2, latency_app_s=0.125)

        assert (result.instructions_total) == (500_000)
        assert (result.perf_elapsed_s) == (0.25) or round(abs((result.perf_elapsed_s) - (0.25)), 7) == 0

    def test_stop_recovers_after_first_sigint_is_consumed(self) -> None:
        class SlowStopProcess:
            returncode = None

            def __init__(self):
                self.signals = []
                self.calls = 0

            def poll(self):
                return None if self.calls < 2 else self.returncode

            def send_signal(self, sig):
                self.signals.append(sig)

            def communicate(self, timeout=None):
                self.calls += 1
                if self.calls == 1:
                    assert timeout == perf_mips.PERF_STOP_SIGNAL_GRACE_S
                    raise perf_mips.subprocess.TimeoutExpired("perf", timeout)
                assert timeout == perf_mips.PERF_STOP_TIMEOUT_S - perf_mips.PERF_STOP_SIGNAL_GRACE_S
                self.returncode = -2
                return "", "12345,,instructions,100.00,,\n"

            def kill(self):
                raise AssertionError("graceful retry must not kill perf")

        process = SlowStopProcess()
        monitor = perf_mips.PerfMIPSMonitor("case_container", command_prefix=["perf"])
        monitor._proc = process
        monitor._t_start = 1.0
        with patch.object(monitor, "_control", return_value=None):
            result = monitor.stop(repeat_in_window=5, latency_app_s=0.1)
        assert process.signals == [perf_mips.signal.SIGINT, perf_mips.signal.SIGINT]
        assert result.instructions_total == 12345.0
        assert result.instructions_per_request == 2469.0

    def test_stop_keeps_reap_after_kill_bounded(self) -> None:
        class HungProcess:
            returncode = None

            def __init__(self):
                self.communicate_timeouts = []
                self.killed = False

            def poll(self):
                return None

            def send_signal(self, _sig):
                pass

            def communicate(self, timeout=None):
                self.communicate_timeouts.append(timeout)
                raise perf_mips.subprocess.TimeoutExpired("perf", timeout)

            def kill(self):
                self.killed = True

        process = HungProcess()
        monitor = perf_mips.PerfMIPSMonitor("case_container", command_prefix=["perf"])
        monitor._proc = process

        with patch.object(monitor, "_control", return_value=None), pytest.raises(
            perf_mips.MIPSProfilingError, match="reaped after kill"
        ):
            monitor.stop(repeat_in_window=1, latency_app_s=0.1)

        assert process.killed
        assert process.communicate_timeouts == [
            perf_mips.PERF_STOP_SIGNAL_GRACE_S,
            perf_mips.PERF_STOP_TIMEOUT_S - perf_mips.PERF_STOP_SIGNAL_GRACE_S,
            perf_mips.PERF_STOP_TIMEOUT_S,
        ]

    @pytest.mark.parametrize('prefix', [['sudo', '-S', '-p', '', 'perf'], ['sudo', '-n', 'perf']])
    def test_monitor_rejects_legacy_sudo_command_prefix(self, prefix):
        with patch('acprof.monitors.common.docker_container_pid', return_value=1234), patch.object(
            perf_mips.subprocess, 'Popen',
        ) as popen:
            with pytest.raises(perf_mips.MIPSProfilingError, match='direct perf'):
                perf_mips.PerfMIPSMonitor('case_container', command_prefix=prefix).prepare()
        popen.assert_not_called()

    def test_preflight_cannot_gain_access_from_retired_password_setting(self):
        with patch.dict('os.environ', {'ACPROF_SUDO_PASSWORD': 'test-retired-secret'}), patch.object(
            perf_mips.shutil, 'which', return_value='/usr/bin/perf',
        ), patch.object(perf_mips, 'run_command', return_value=SimpleNamespace(
            returncode=1, stdout='', stderr='Permission denied',
        )) as run, pytest.raises(perf_mips.MIPSProfilingError):
            perf_mips.resolve_perf_command_prefix()
        assert (run.call_count) == (1)
        assert (run.call_args.kwargs['input']) == ('')

    def test_preflight_failure_prints_friendly_remediation(self) -> None:
        stderr = io.StringIO()

        def fake_run(cmd, **kwargs):
            return SimpleNamespace(returncode=255, stdout="", stderr="perf_event_paranoid setting is 4")

        with patch("acprof.monitors.perf_mips.shutil.which", return_value="/usr/bin/perf"), patch(
            "acprof.monitors.perf_mips.run_command",
            side_effect=fake_run,
        ), patch("acprof.monitors.perf_mips.read_perf_event_paranoid", return_value="4"), pytest.raises(
            SystemExit
        ) as raised, redirect_stderr(stderr):
            perf_mips.require_mips_prerequisites()

        assert (raised.value.code) == (1)
        message = stderr.getvalue()
        assert ("[mips][ERROR]") in (message)
        assert ("MIPS profiling requires Linux perf access") in (message)
        assert ("perf_event_paranoid=4") in (message)
        assert ("cap_perfmon=ep") in (message)
        assert ("ACPROF_SUDO_PASSWORD") not in (message)
        assert ("Avoid `sudo acprof run ...`") in (message)



def test_process_pmu_checks_membership_and_root_start_ticks(monkeypatch):
    root = {"pid": 41, "start_ticks": 1041}
    parent = SimpleNamespace(pid=41, children=lambda *, recursive: [SimpleNamespace(pid=42)])
    monkeypatch.setattr("psutil.Process", lambda _pid: parent)
    monkeypatch.setattr("acprof.process_identity.process_identity", lambda pid: {"pid": pid, "start_ticks": 1000 + pid})
    monkeypatch.setattr("acprof.process_identity.same_process", lambda identity: identity == root)
    assert perf_mips._process_pmu_targets(root) == {41: 1041, 42: 1042}
    monkeypatch.setattr("acprof.process_identity.same_process", Mock(side_effect=[True, False]))
    with pytest.raises(perf_mips.MIPSProfilingError, match="identity changed"):
        perf_mips._process_pmu_targets(root)
