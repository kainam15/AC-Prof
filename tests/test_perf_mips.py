import io
import math
from contextlib import redirect_stderr
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.monitors import perf_mips


class TestPerfMIPS:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        locator = patch.object(perf_mips.shutil, "which", return_value="/usr/bin/perf")
        locator.start()
        self._request.addfinalizer(partial(locator.stop))

    def test_start_uses_prepared_command_without_discovery(self):
        monitor = perf_mips.PerfMIPSMonitor("case")
        with patch.object(perf_mips.common, "docker_container_pid", return_value=1234), patch.object(
            perf_mips, "resolve_perf_command_prefix_for_pid", return_value=["perf"],
        ):
            monitor.prepare()
        with patch.object(perf_mips.common, "docker_container_pid", side_effect=AssertionError("window discovery")), patch.object(
            perf_mips, "resolve_perf_command_prefix_for_pid", side_effect=AssertionError("window probe"),
        ), patch.object(perf_mips.subprocess, "Popen") as launch:
            monitor.start()
        assert ("1234") in (launch.call_args.args[0])

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
        ):
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
        assert (",".join(perf_mips.PERF_EVENTS)) in (popen_cmds[0])
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
        ), patch("acprof.monitors.perf_mips.time.perf_counter", side_effect=[10.0, 10.25]):
            monitor = perf_mips.PerfMIPSMonitor(
                container_name="case_container",
                command_prefix=["perf"],
            )
            monitor.prepare()
            monitor.start()
            result = monitor.stop(repeat_in_window=2, latency_app_s=0.125)

        assert (result.instructions_total) == (500_000)
        assert (result.perf_elapsed_s) == (0.25) or round(abs((result.perf_elapsed_s) - (0.25)), 7) == 0

    @pytest.mark.parametrize('prefix', (['sudo', '-S', '-p', '', 'perf'], ['sudo', '-n', 'perf']))
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
