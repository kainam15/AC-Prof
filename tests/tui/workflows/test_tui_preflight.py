"""Quick checks must exercise the same perf access paths as formal collection."""
import os
import subprocess
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest

from acprof.experiments.configuration import RunConfig
from acprof.monitors.perf_mips import PERF_PROBE_TIMEOUT_S
from acprof.platform import Environment
from acprof.tui.run.diagnostics import quick_preflight
from tests.support.environment_fixtures import isolated_environment


class TestTuiPreflight:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path):
        self._request = request
        from tests.support.platform_fixtures import native_policy
        native_policy(self._request)
        temporary = tmp_path
        self.project_dir = Path(str(temporary))
        environment = patch.dict(os.environ, isolated_environment({"PATH": "/usr/bin:/bin"}), clear=True)
        environment.start()
        self._request.addfinalizer(partial(environment.stop))
        which = patch(
            "acprof.tui.run.diagnostics.shutil.which",
            side_effect=lambda command, **kwargs: f"/usr/bin/{command}",
        )
        self.which = which.start()
        self._request.addfinalizer(partial(which.stop))
        host_platform = patch("acprof.tui.run.diagnostics.detect_environment", return_value=Environment("native_linux"))
        host_platform.start()
        self._request.addfinalizer(partial(host_platform.stop))
        rapl = patch("acprof.tui.run.diagnostics._readable_rapl_paths", return_value=["/fake/energy_uj"])
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
            assert "--id=0" in command
            stdout = "GPU-fixture, 0, 0000:01:00.0, test GPU, 1024, Disabled\n"
        elif tool == 'getcap':
            stdout = '/usr/bin/tcpdump cap_net_raw=ep'
        elif tool == "perf":
            raise AssertionError("quick checks must use the shared perf resolver")
        else:
            raise AssertionError(f"unexpected host command: {command}")
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    def run_check(self, responses, *, gpu_status="ok"):
        with patch("acprof.monitors.perf_mips.run_command", side_effect=responses) as run:
            checks = quick_preflight(
                RunConfig(model="", gpus="on", profiling_mode="full"), project_dir=self.project_dir,
                command_runner=self.host_command,
            )
        # A perf failure must leave the remaining independent checks usable.
        assert all(check.check_id for check in checks)
        assert len({check.check_id for check in checks}) == len(checks)
        gpu = next(check for check in checks if check.check_id == "gpu_device")
        assert gpu.status == gpu_status
        if gpu_status == "fail":
            assert gpu.source == "environment"
            assert "ACPROF_SUDO_PASSWORD" in gpu.detail
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
            'acprof.tui.run.diagnostics.os.geteuid', return_value=1000,
        ):
            checks = quick_preflight(RunConfig(gpus='off', profiling_mode='full'), project_dir=self.project_dir,
                                     command_runner=command_runner)
        access = next((check for check in checks if check.label == 'tcpdump permissions'), None)
        assert (access) is not None, 'Finding tcpdump must not imply capture access'
        assert (access.status) == ('fail')

    def test_permission_denial_is_reported_without_privilege_fallback(self):
        check, run = self.run_check([self.result(1, 'Permission denied')])
        assert (check.status) == ('fail')
        assert (run.call_count) == (1)
        assert ('Permission denied') in (check.detail)

    @pytest.mark.parametrize(('exit_code', 'reason'), [(None, 'dependency_unavailable'), (1, 'error')])
    def test_unreadable_file_capabilities_keep_the_blocker_and_explain_why(self, exit_code, reason):
        self.which.side_effect = lambda tool, **kw: None if tool == 'getcap' and exit_code is None else '/usr/bin/' + tool
        def command(argv, **kw):
            if Path(argv[0]).name == 'getcap':
                return subprocess.CompletedProcess(argv, exit_code, '', '')
            return self.host_command(argv, **kw)
        with patch('acprof.monitors.perf_mips.run_command', return_value=self.result()), patch(
                'acprof.tui.run.diagnostics.os.geteuid', return_value=1000):
            rows = quick_preflight(RunConfig(gpus='off', profiling_mode='full'), project_dir=self.project_dir,
                                   command_runner=command)
        row = next(row for row in rows if row.check_id == 'capture_permissions')
        assert (row.status, row.reason_code, row.source) == ('fail', reason, 'getcap')

    def test_local_password_configuration_has_a_migration_error(self):
        (self.project_dir / '.env.local').write_text('ACPROF_SUDO_PASSWORD=test-file-secret\n')
        check, run = self.run_check([], gpu_status='fail')
        assert (check.status) == ('fail')
        assert ('ACPROF_SUDO_PASSWORD') in (check.detail)
        assert ('test-file-secret') not in (check.detail)
        run.assert_not_called()

    def test_process_password_configuration_has_a_migration_error(self):
        with patch.dict(os.environ, {'ACPROF_SUDO_PASSWORD': 'test-process-secret'}):
            check, run = self.run_check([], gpu_status='fail')
        assert (check.status) == ('fail')
        assert ('test-process-secret') not in (check.detail)
        run.assert_not_called()

    def test_failure_keeps_perf_diagnostic_and_setup_guidance(self):
        check, run = self.run_check([self.result(1, 'Permission denied\nperf_event_paranoid setting is 4')])
        assert (check.status) == ('fail')
        assert (run.call_count) == (1)
        assert ('perf_event_paranoid setting is 4') in (check.detail)
        assert ('installation.md') in (check.detail)

    @pytest.mark.parametrize('detail', ['<not supported>', '<not counted>'])
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


@pytest.mark.parametrize(("missing", "low_pct"), [
    (True, False), (False, True), (True, True), (False, False),
])
def test_auto_pmu_preflight_distinguishes_missing_fields_and_low_duty_cycle(tmp_path, monkeypatch, missing, low_pct):
    from acprof.tui.run.diagnostics import _automatic_checks

    evidence = {
        "unavailable_events": ["dTLB-load-misses"] if missing else [],
        "event_statuses": {"dTLB-load-misses": "combination_limited"},
        "event_running_pct": {"instructions": 7.0} if low_pct else {"instructions": 100.0},
    }
    fields = ("cpu_dtlb_load_misses_per_request", "cpu_dtlb_load_miss_rate_pct")
    plan = {
        "probes": {"cpu_instructions": {"status": "available", "detail": "", "evidence": evidence}},
        "fields": {"cpu_instructions": fields},
        "field_probes": ({
            key: {"status": "unavailable", "reason_code": "combination_limited", "detail": "combination"}
            for key in fields
        } if missing else {}),
    }
    monkeypatch.setattr("acprof.host.runtimes.capabilities.probe_collection_plan", lambda *a, **kw: plan)
    rows = _automatic_checks(
        RunConfig(profiling_mode="best_effort", gpus="off"), "process", lambda _: {},
        project_dir=tmp_path,
    )
    check = next(row for row in rows if row.check_id == "cpu_instructions")
    assert check.status == ("warn" if missing or low_pct else "ok")
    assert check.reason_code == ("partial" if missing else "pmu_multiplexed" if low_pct else "")
    assert check.affected_fields == (fields if missing else ())
    if missing:
        assert "dTLB-load-misses" in check.detail
        assert "PMU" in check.detail
    else:
        assert not any(field in check.detail for field in fields)
    assert ("7.0%" in check.detail) == low_pct


def test_pmu_multiplexing_warning_has_no_false_missing_fields_or_permission_remedy():
    from acprof.tui.i18n import translate
    from acprof.tui.run.diagnostics import PreflightCheck
    from acprof.tui.run.environment_view import check_presentation, global_check_view
    from acprof.tui.run.global_status import StatusItem

    row = PreflightCheck(
        "cpu_instructions", "warn",
        "available · PMU multiplexed; minimum running time 27.0%",
        "available", check_id="cpu_instructions", source="perf_stat",
        reason_code="pmu_multiplexed",
    )
    view = global_check_view(
        StatusItem(row, requested=True, target="permissions:perf"),
        technical_detail=row.detail, tr=lambda value: translate(value, "zh"),
        check_allowed=True, navigate_allowed=True,
    )
    assert view.callout is not None
    assert translate(check_presentation(row).label, "zh") == "PMU 多路复用提醒"
    assert "不能据此判定" in translate(view.callout.impact, "zh")
    assert "无需调整 perf 权限" in translate(view.callout.next_step, "zh")
    assert "cpu_cycles_per_request" not in str(view.callout.detail)
    assert "cpu_instructions_per_request" not in str(view.callout.detail)


def test_auto_pmu_permission_failure_keeps_potentially_affected_fields(tmp_path, monkeypatch):
    from acprof.tui.run.diagnostics import _automatic_checks

    fields = ("cpu_instructions_per_request", "cpu_mips_app")
    plan = {
        "probes": {"cpu_instructions": {
            "status": "permission_denied", "detail": "perf attach denied", "evidence": {},
        }},
        "fields": {"cpu_instructions": fields},
        "field_probes": {},
    }
    monkeypatch.setattr("acprof.host.runtimes.capabilities.probe_collection_plan", lambda *a, **kw: plan)
    rows = _automatic_checks(
        RunConfig(profiling_mode="best_effort", gpus="off"), "process", lambda _: {},
        project_dir=tmp_path,
    )
    check = next(row for row in rows if row.check_id == "cpu_instructions")
    assert check.status == "warn"
    assert check.reason_code == ""
    assert check.affected_fields == fields
