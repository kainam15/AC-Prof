"""Quick checks must exercise the same perf access paths as formal collection."""
import os
import subprocess
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest
from environment_fixtures import isolated_environment

from acprof.experiment import RunConfig
from acprof.monitors.perf_mips import PERF_PROBE_TIMEOUT_S
from acprof.platform import Environment
from acprof.tui.diagnostics import quick_preflight


class TestTuiPreflight:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        from platform_fixtures import native_policy
        native_policy(self._request)
        temporary = tmp_path
        self.project_dir = Path(str(temporary))
        environment = patch.dict(os.environ, isolated_environment({"PATH": "/usr/bin:/bin"}), clear=True)
        environment.start()
        self._request.addfinalizer(partial(environment.stop))
        which = patch(
            "acprof.tui.diagnostics.shutil.which",
            side_effect=lambda command, **kwargs: f"/usr/bin/{command}",
        )
        self.which = which.start()
        self._request.addfinalizer(partial(which.stop))
        host_platform = patch("acprof.tui.diagnostics.detect_environment", return_value=Environment("native_linux"))
        host_platform.start()
        self._request.addfinalizer(partial(host_platform.stop))
        rapl = patch("acprof.tui.diagnostics._readable_rapl_paths", return_value=["/fake/energy_uj"])
        rapl.start()
        self._request.addfinalizer(partial(rapl.stop))

    @staticmethod
    def result(returncode=0, stderr="1000,,instructions,100.00,,\n"):
        return subprocess.CompletedProcess([], returncode, stdout="", stderr=stderr)

    @staticmethod
    def host_command(command, **kwargs):
        tool = Path(command[0]).name
        if tool == "docker":
            if tuple(command[1:3]) == ("context", "show"):
                stdout = "default"
            elif tuple(command[1:3]) == ("context", "inspect"):
                stdout = "unix:///var/run/docker.sock"
            else:
                stdout = "test-host|Linux"
        elif tool == "ip":
            stdout = "docker0"
        elif tool == "nvidia-smi":
            stdout = "test GPU"
        elif tool == 'getcap':
            stdout = '/usr/bin/tcpdump cap_net_raw=ep'
        elif tool == "perf":
            raise AssertionError("quick checks must use the shared perf resolver")
        else:
            raise AssertionError(f"unexpected host command: {command}")
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    def run_check(self, responses):
        with patch("acprof.monitors.perf_mips.run_command", side_effect=responses) as run:
            checks = quick_preflight(
                RunConfig(model="", gpus="on"), project_dir=self.project_dir,
                command_runner=self.host_command,
            )
        # A perf failure must leave the remaining independent checks usable.
        assert (next(check for check in checks if check.label == "NVIDIA GPU").status) == ("ok")
        for call in run.call_args_list:
            assert (call.kwargs["timeout"]) == (PERF_PROBE_TIMEOUT_S)
        return next(check for check in checks if check.label == "perf instructions"), run

    def test_direct_perf_success_does_not_try_sudo(self):
        check, run = self.run_check([self.result(), self.result()])
        assert (check.status) == ("ok")
        assert ("普通用户 perf 可用") in (check.detail)
        assert (run.call_count) == (2)
        assert (run.call_args.args[0][0]) == ("perf")
        assert (run.call_args.kwargs["input"]) == ("")

    def test_installed_tcpdump_without_capture_capability_is_a_failure(self):
        def command_runner(command, **kwargs):
            if Path(command[0]).name == 'getcap':
                return subprocess.CompletedProcess(command, 0, stdout='', stderr='')
            return self.host_command(command, **kwargs)
        with patch('acprof.monitors.perf_mips.run_command', return_value=self.result()), patch(
            'acprof.tui.diagnostics.os.geteuid', return_value=1000,
        ):
            checks = quick_preflight(RunConfig(gpus='off'), project_dir=self.project_dir,
                                     command_runner=command_runner)
        access = next((check for check in checks if check.label == 'tcpdump permissions'), None)
        assert (access) is not None, 'Finding tcpdump must not imply capture access'
        assert (access.status) == ('fail')

    def test_permission_denial_is_reported_without_privilege_fallback(self):
        check, run = self.run_check([self.result(1, 'Permission denied')])
        assert (check.status) == ('fail')
        assert (run.call_count) == (1)
        assert ('Permission denied') in (check.detail)

    def test_local_password_configuration_has_a_migration_error(self):
        (self.project_dir / '.env.local').write_text('ACPROF_SUDO_PASSWORD=test-file-secret\n')
        check, run = self.run_check([])
        assert (check.status) == ('fail')
        assert ('ACPROF_SUDO_PASSWORD') in (check.detail)
        assert ('test-file-secret') not in (check.detail)
        run.assert_not_called()

    def test_process_password_configuration_has_a_migration_error(self):
        with patch.dict(os.environ, {'ACPROF_SUDO_PASSWORD': 'test-process-secret'}):
            check, run = self.run_check([])
        assert (check.status) == ('fail')
        assert ('test-process-secret') not in (check.detail)
        run.assert_not_called()

    def test_failure_keeps_perf_diagnostic_and_setup_guidance(self):
        check, run = self.run_check([self.result(1, 'Permission denied\nperf_event_paranoid setting is 4')])
        assert (check.status) == ('fail')
        assert (run.call_count) == (1)
        assert ('perf_event_paranoid setting is 4') in (check.detail)
        assert ('Getting_Started.md') in (check.detail)

    @pytest.mark.parametrize('detail', ('<not supported>', '<not counted>'))
    def test_zero_exit_without_an_instruction_count_is_failure(self, detail):
        check, _ = self.run_check([self.result(0, detail + ',,instructions,0.00,,\n')])
        assert (check.status) == ('fail')
        assert (detail) in (check.detail)

    def test_timeout_is_visible_without_stopping_gpu_check(self):
        check, run = self.run_check([subprocess.TimeoutExpired(['perf', 'stat'], PERF_PROBE_TIMEOUT_S)])
        assert (check.status) == ('fail')
        assert ('TimeoutExpired') in (check.detail)
        assert (run.call_count) == (1)

    def test_missing_perf_is_reported_without_running_a_probe(self):
        self.which.side_effect = lambda command, **kwargs: None if command == "perf" else f"/usr/bin/{command}"
        check, run = self.run_check([])
        assert (check.status) == ("fail")
        assert ("perf command was not found") in (check.detail)
        run.assert_not_called()
