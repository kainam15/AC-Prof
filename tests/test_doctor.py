"""Doctor reports independent failures without changing host configuration."""
import io
import json
import subprocess
import tempfile
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from acprof.cli import doctor as cli
from acprof.host import doctor
from acprof.installation import cli_command, module_command


class TestDoctor:
    def test_retired_environment_setting_keeps_json_error_contract(self):
        stream = io.StringIO()
        with patch.object(cli, "load_project_env", side_effect=ValueError(
            "ACPROF_SUDO_PASSWORD is no longer supported",
        )), patch.object(cli, "collect_checks", return_value=[]), redirect_stdout(stream):
            status = cli.main(["--json", "--profiling-mode", "basic"])
        report = json.loads(stream.getvalue())
        assert (status) == (1)
        assert not (report["ready"])
        assert (report["checks"][0]["name"]) == ("environment")
        assert ("ACPROF_SUDO_PASSWORD") in (report["checks"][0]["detail"])

    def collect(self, mode="basic", **overrides):
        with ExitStack() as stack:
            for target, value in {
                "preflight.require_collection_host": None,
                "preflight.require_cgroup_prerequisites": "v2",
                "preflight.require_native_docker": None,
                "_architecture": "x86_64", "_resources": "bundled", "_command": "buildx",
                **overrides,
            }.items():
                options = {"side_effect": value} if isinstance(value, BaseException) else {"return_value": value}
                stack.enter_context(patch(f"acprof.host.doctor.{target}", **options))
            directory = stack.enter_context(tempfile.TemporaryDirectory())
            checks = doctor.collect_checks(profiling_mode=mode, output_dir=Path(directory))
            assert (list(Path(directory).iterdir())) == ([])
            return {check.name: check for check in checks}

    def test_basic_never_probes_full_metrics(self):
        with patch.object(doctor.preflight, "probe_cpu_energy", side_effect=AssertionError), patch.object(
            doctor.preflight, "probe_perf_instructions", side_effect=AssertionError,
        ), patch.object(doctor, "_packet", side_effect=AssertionError):
            checks = self.collect()
        for name in ("rapl", "perf", "packet", "gpu"):
            assert (checks[name].status) == ("not_requested")
        assert (doctor.report_dict(list(checks.values()), profiling_mode="basic", gpus="off")["ready"])

    def test_failure_and_timeout_do_not_hide_later_checks(self):
        checks = self.collect(**{"preflight.require_collection_host": SystemExit(1),
                                 "_command": subprocess.TimeoutExpired(["docker"], 15)})
        assert (checks["environment"].status) == ("unavailable")
        assert (checks["buildx"].status) == ("unavailable")
        assert (checks["resources"].status) == ("available")
        assert (checks["workspace"].status) == ("available")
        assert not (doctor.report_dict(list(checks.values()), profiling_mode="basic", gpus="off")["ready"])

    def test_json_failure_has_remedy_and_nonzero_status(self):
        failed = doctor.DoctorCheck("docker", "unavailable", "socket denied", "Check Docker socket access")
        stream = io.StringIO()
        with patch.object(cli, "collect_checks", return_value=[failed]), redirect_stdout(stream):
            status = cli.main(["--json", "--profiling-mode", "basic"])
        report = json.loads(stream.getvalue())
        assert (status) == (1)
        assert not (report["ready"])
        assert (report["checks"][0]["remedy"]) == ("Check Docker socket access")

    def test_gpu_driver_alone_does_not_pass_container_readiness(self):
        with patch.object(doctor, "_command", side_effect=["Test GPU", "{}"]):
            check = doctor._check("gpu", doctor._gpu, "Configure NVIDIA runtime")
        assert (check.status) == ("unavailable")
        assert ("NVIDIA runtime") in (check.detail)

    def test_explicit_remote_context_overrides_local_host_variable(self):
        context = subprocess.CompletedProcess([], 0, "remote-lab\n", "")
        endpoint = subprocess.CompletedProcess([], 0, "tcp://192.0.2.10:2376\n", "")
        info = subprocess.CompletedProcess([], 0, "OperatingSystem=Ubuntu\n", "")
        with patch.dict("os.environ", {"DOCKER_CONTEXT": "remote-lab",
                                       "DOCKER_HOST": "unix:///var/run/docker.sock"}, clear=True), patch(
            "acprof.host.preflight.run_command", side_effect=[context, endpoint, info],
        ):
            check = doctor._check("docker", doctor.preflight.require_native_docker, "Use native Docker")
        assert (check.status) == ("unavailable")
        assert ("tcp://192.0.2.10:2376") in (check.detail)

    def test_unknown_docker_endpoint_is_not_reported_available(self):
        context = subprocess.CompletedProcess([], 1, "", "no context")
        info = subprocess.CompletedProcess([], 0, "OperatingSystem=Ubuntu\n", "")
        with patch.dict("os.environ", {}, clear=True), patch(
            "acprof.host.preflight.run_command", side_effect=[context, info],
        ):
            check = doctor._check("docker", doctor.preflight.require_native_docker, "Inspect Docker context")
        assert (check.status) == ("unavailable")

    def test_frozen_child_commands_reenter_binary(self):
        with patch("sys.frozen", True, create=True):
            assert (cli_command("run", python_executable="/opt/acprof")) == (["/opt/acprof", "run"])
            assert (module_command("acprof.host.client", python_executable="/opt/acprof")) == (["/opt/acprof", "_worker", "acprof.host.client"])
