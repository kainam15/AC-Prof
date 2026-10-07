import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stderr
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import acprof.host.preflight as host_preflight
from acprof.cli import run
from acprof.host import input_plan, orchestrator, runtime_images
from acprof.host.detect import TaskInfo


def test_terminal_log_is_not_started_outside_tmux() -> None:
    with patch.dict(
        "acprof.cli.run.os.environ",
        {},
        clear=True,
    ), patch("acprof.cli.terminal_log.run_command") as mock_run:
        terminal_log = run.start_terminal_log(
            "/tmp/acprof-results",
            ["acprof run", "--model", "dummy-model"],
        )

    assert (terminal_log) is None
    mock_run.assert_not_called()

def test_terminal_log_records_and_atomically_finalizes_tmux_output() -> None:
    completed = SimpleNamespace(returncode=0, stdout="", stderr="")
    pipe_status = SimpleNamespace(returncode=0, stdout="0\n", stderr="")
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        if command[1] == "display-message":
            return pipe_status
        return completed

    with tempfile.TemporaryDirectory() as tmp, patch.dict(
        "acprof.cli.run.os.environ",
        {
            "TMUX": "/tmp/tmux-1000/default,123,0",
            "TMUX_PANE": "%7",
        },
        clear=True,
    ), patch(
        "acprof.cli.terminal_log.run_command",
        side_effect=fake_run,
    ):
        output_dir = os.path.join(tmp, "results", "org--model")
        terminal_log = run.start_terminal_log(
            output_dir,
            ["acprof run", "--model", "org/model"],
        )
        assert (terminal_log) is not None
        pane_id, partial_path, log_path = terminal_log

        with open(partial_path, "a", encoding="utf-8") as f:
            f.write("experiment output\n")

        finalized = run.stop_terminal_log(terminal_log)

        assert (finalized)
        assert (pane_id) == ("%7")
        assert not (os.path.exists(partial_path))
        with open(log_path, "r", encoding="utf-8") as f:
            terminal_text = f.read()

    assert ("$ acprof run --model org/model") in (terminal_text)
    assert ("experiment output") in (terminal_text)
    assert (commands[0][1]) == ("display-message")
    assert (commands[1][1]) == ("pipe-pane")
    assert ("-O") in (commands[1])
    assert (commands[2]) == (["tmux", "pipe-pane", "-t", "%7"])


@pytest.mark.skipif(os.name != "posix", reason="directory fsync is a POSIX durability contract")
def test_terminal_log_finalization_syncs_file_and_directory() -> None:
    completed = SimpleNamespace(returncode=0, stdout="", stderr="")

    with tempfile.TemporaryDirectory() as tmp:
        partial_path = os.path.join(tmp, "tmux_all.log.part")
        log_path = os.path.join(tmp, "tmux_all.log")
        with open(partial_path, "w", encoding="utf-8") as stream:
            stream.write("completed output\n")

        with patch(
            "acprof.cli.terminal_log.run_command",
            return_value=completed,
        ), patch("acprof.artifacts.os.fsync", wraps=os.fsync) as fsync:
            finalized = run.stop_terminal_log(("%7", partial_path, log_path))

        assert finalized
        assert os.path.exists(log_path)
        assert not os.path.exists(partial_path)
        assert fsync.call_count >= 2

def test_main_finalizes_tmux_log_when_profiling_raises() -> None:
    terminal_log = ("%3", "/tmp/tmux_all.log.part", "/tmp/tmux_all.log")

    def fail_after_starting_log():
        run._ACTIVE_TMUX_TERMINAL_LOG = terminal_log
        raise RuntimeError("profiling failed")

    with patch(
        "acprof.cli.run._run_main",
        side_effect=fail_after_starting_log,
    ), patch(
        "acprof.cli.run.stop_terminal_log",
        return_value=True,
    ) as stop_log:
        with pytest.raises(RuntimeError, match="profiling failed"):
            run.main()

    stop_log.assert_called_once_with(terminal_log)
    assert (run._ACTIVE_TMUX_TERMINAL_LOG) is None


class TestNativeDockerGuard:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        monkeypatch.setattr("acprof.host.interface_probe.probe_interface", Mock(return_value={"status": "ok"}))
        self._request = request
        from platform_fixtures import native_policy
        native_policy(self._request)
        self.resolved_task = TaskInfo(
            model_id="dummy-model",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="1" * 40,
            detection_method="unit",
        )
        selection = patch('acprof.host.gpu_device.resolve_gpu_device', return_value={
            'uuid': 'GPU-fixture', 'index': 1, 'name': 'Fixture', 'memory_total_bytes': 8 * 1024 ** 3,
            'pci_bus_id': '00000000:02:00.0',
        })
        selection.start()
        self._request.addfinalizer(partial(selection.stop))
        # Local developer credentials must never make CLI tests send messages.
        notification_env = patch.dict(
            "acprof.cli.run.os.environ",
            {"ACPROF_WECOM_WEBHOOK_URL": ""},
        )
        notification_env.start()
        self._request.addfinalizer(partial(notification_env.stop))

    def test_native_linux_host_allows_ubuntu(self) -> None:
        with patch("acprof.platform.platform.system", return_value="Linux"), patch(
            "acprof.platform.platform.release",
            return_value="7.0.0-28-generic",
        ), patch.dict("acprof.host.preflight.os.environ", {}, clear=True):
            host_preflight.require_native_linux_host()

    def test_native_linux_host_rejects_wsl(self) -> None:
        stderr = io.StringIO()
        from acprof.platform import Environment

        with patch("acprof.host.preflight.detect_environment", return_value=Environment("wsl2")), patch(
            "acprof.platform.platform.release",
            return_value="6.6.87.2-microsoft-standard-WSL2",
        ), patch.dict(
            "acprof.host.preflight.os.environ",
            {"WSL_DISTRO_NAME": "Ubuntu"},
            clear=True,
        ), pytest.raises(SystemExit) as raised, redirect_stderr(stderr):
            host_preflight.require_native_linux_host()

        assert (raised.value.code) == (1)
        message = stderr.getvalue()
        assert ("native Linux host") in (message)
        assert ("WSL2 / PARTIAL") in (message)
        assert ("acprof run") in (message)

    def test_native_linux_host_rejects_windows(self) -> None:
        stderr = io.StringIO()
        from acprof.platform import Environment

        with patch("acprof.host.preflight.detect_environment", return_value=Environment()), patch.dict(
            "acprof.host.preflight.os.environ",
            {},
            clear=True,
        ), pytest.raises(SystemExit) as raised, redirect_stderr(stderr):
            host_preflight.require_native_linux_host()

        assert (raised.value.code) == (1)
        assert ("unknown / UNKNOWN") in (stderr.getvalue())

    def test_detect_cgroup_version_distinguishes_v2_and_v1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cgroup_root = os.path.join(tmp, "cgroup")
            proc_self_cgroup = os.path.join(tmp, "self.cgroup")
            os.makedirs(cgroup_root)
            with open(
                os.path.join(cgroup_root, "cgroup.controllers"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("cpu io memory\n")
            with open(proc_self_cgroup, "w", encoding="utf-8") as f:
                f.write("0::/user.slice/test.scope\n")

            assert (host_preflight.detect_cgroup_version(
                    cgroup_root=cgroup_root,
                    proc_self_cgroup_path=proc_self_cgroup,
                )) == ("v2")

            os.remove(os.path.join(cgroup_root, "cgroup.controllers"))
            with open(proc_self_cgroup, "w", encoding="utf-8") as f:
                f.write("2:cpu,cpuacct:/docker/test\n")
                f.write("3:memory:/docker/test\n")

            assert (host_preflight.detect_cgroup_version(
                    cgroup_root=cgroup_root,
                    proc_self_cgroup_path=proc_self_cgroup,
                )) == ("v1")

    def test_cgroup_preflight_requires_v2_by_default(self) -> None:
        stderr = io.StringIO()
        with patch(
            "acprof.host.preflight.detect_cgroup_version",
            return_value="v1",
        ), pytest.raises(SystemExit) as raised, redirect_stderr(stderr):
            run.require_cgroup_prerequisites()

        assert (raised.value.code) == (1)
        message = stderr.getvalue()
        assert ("requires unified cgroup v2") in (message)
        assert ("detected v1") in (message)
        assert ("--allow-cgroup-v1") not in (message)


    def test_partial_results_require_matching_cgroup_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            partial_path = os.path.join(tmp, "result_case_model_1c_2g_off.csv")
            static_meta_path = os.path.join(tmp, "static_meta.json")
            with open(partial_path, "w", encoding="utf-8") as f:
                f.write("status\nok\n")
            with open(static_meta_path, "w", encoding="utf-8") as f:
                json.dump({"cgroup_version": "v1"}, f)

            stderr = io.StringIO()
            with pytest.raises(SystemExit) as raised, redirect_stderr(stderr):
                run.require_result_cgroup_compatibility(
                    tmp,
                    cgroup_version="v2",
                )

            assert (raised.value.code) == (1)
            message = stderr.getvalue()
            assert ("Current cgroup_version:  v2") in (message)
            assert ("Existing cgroup_version: v1") in (message)
            assert ("will not mix") in (message)

            run.require_result_cgroup_compatibility(
                tmp,
                cgroup_version="v1",
            )

    @pytest.mark.parametrize(
        "static_metadata",
        [
            '{"cgroup_version":"v2","corrupt":NaN}',
            '{"cgroup_version":"v2","padding":"' + ("x" * (4 * 1024 * 1024)) + '"}',
        ],
    )
    def test_partial_results_reject_invalid_cgroup_metadata(
        self, static_metadata: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with open(
                os.path.join(tmp, "result_case_model_1c_2g_off.csv"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("status\nok\n")
            with open(
                os.path.join(tmp, "static_meta.json"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(static_metadata)

            with pytest.raises(SystemExit), redirect_stderr(io.StringIO()):
                run.require_result_cgroup_compatibility(
                    tmp,
                    cgroup_version="v2",
                )

    def test_partial_results_without_cgroup_provenance_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with open(
                os.path.join(tmp, "result_case_model_1c_2g_off.csv"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write("status\nok\n")

            with pytest.raises(SystemExit), redirect_stderr(io.StringIO()):
                run.require_result_cgroup_compatibility(
                    tmp,
                    cgroup_version="v2",
                )

    def test_main_rejects_legacy_cgroup_flag_before_preflight(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["acprof run", "--model", "dummy-model", "--allow-cgroup-v1"],
        ), patch(
            "acprof.cli.run.bootstrap_project_env",
            return_value=None,
        ), patch(
            "acprof.cli.run.require_collection_host",
        ), patch(
            "acprof.cli.run.require_native_docker",
        ), patch(
            "acprof.cli.run.require_cgroup_prerequisites",
            side_effect=SystemExit(7),
        ) as preflight, patch(
            "acprof.host.detect.detect_task",
            side_effect=AssertionError("task detection must follow cgroup preflight"),
        ):
            with pytest.raises(SystemExit) as raised:
                run.main()

        assert (raised.value.code) == (2)
        preflight.assert_not_called()

    def test_cpu_energy_preflight_exits_with_remediation_when_unavailable(self) -> None:
        stderr = io.StringIO()

        with patch("acprof.monitors.energy_cpu.detect_cpu_power_source", return_value="unavailable"), patch(
            "acprof.monitors.energy_cpu.detect_vcpu_power_method",
            return_value="unavailable",
        ), pytest.raises(SystemExit) as raised, redirect_stderr(stderr):
            run.require_cpu_energy_prerequisites()

        assert (raised.value.code) == (1)
        message = stderr.getvalue()
        assert ("[cpu-energy][ERROR]") in (message)
        assert ("CPU/vCPU energy profiling is required") in (message)
        assert ("cpu_power_source=unavailable") in (message)
        assert ("sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj") in (message)
        assert ("/etc/tmpfiles.d/acprof-rapl.conf") in (message)

    def test_cpu_energy_preflight_allows_rapl_cgroup_share(self) -> None:
        with patch("acprof.monitors.energy_cpu.detect_cpu_power_source", return_value="rapl"), patch(
            "acprof.monitors.energy_cpu.detect_vcpu_power_method",
            return_value="rapl_cgroup_cpu_share",
        ):
            run.require_cpu_energy_prerequisites()

    def test_main_runs_cpu_energy_preflight_before_runtime_preparation(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["acprof run", "--model", "dummy-model", "--notify", "none"],
        ), patch(
            "acprof.cli.run.bootstrap_project_env",
            return_value=None,
        ), patch("acprof.cli.run.require_collection_host"), patch(
            "acprof.cli.run.require_native_docker"
        ), patch(
            "acprof.cli.run.require_cgroup_prerequisites",
            return_value="v2",
        ), patch(
            "acprof.cli.run.require_packet_latency_prerequisites",
        ), patch(
            "acprof.cli.run.require_cpu_energy_prerequisites",
            side_effect=SystemExit(3),
        ) as preflight, patch(
            "acprof.cli.run.require_mips_prerequisites",
        ), patch(
            "acprof.host.detect.detect_task",
            return_value=self.resolved_task,
        ) as detect, patch(
            "acprof.cli.run._prepare_runtime",
            side_effect=AssertionError("runtime preparation must not run before CPU energy preflight"),
        ):
            with pytest.raises(SystemExit) as raised:
                run.main()

        assert (raised.value.code) == (3)
        detect.assert_called_once()
        preflight.assert_called_once_with()

    def test_main_runs_mips_preflight_before_runtime_preparation(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["acprof run", "--model", "dummy-model", "--notify", "none"],
        ), patch(
            "acprof.cli.run.bootstrap_project_env",
            return_value=None,
        ), patch("acprof.cli.run.require_collection_host"), patch(
            "acprof.cli.run.require_native_docker"
        ), patch(
            "acprof.cli.run.require_cgroup_prerequisites",
            return_value="v2",
        ), patch(
            "acprof.cli.run.require_packet_latency_prerequisites",
        ), patch(
            "acprof.cli.run.require_cpu_energy_prerequisites",
        ), patch(
            "acprof.cli.run.require_mips_prerequisites",
            side_effect=SystemExit(4),
        ) as preflight, patch(
            "acprof.host.detect.detect_task",
            return_value=self.resolved_task,
        ) as detect, patch(
            "acprof.cli.run._prepare_runtime",
            side_effect=AssertionError("runtime preparation must not run before MIPS preflight"),
        ):
            with pytest.raises(SystemExit) as raised:
                run.main()

        assert (raised.value.code) == (4)
        detect.assert_called_once()
        preflight.assert_called_once_with()

    def test_docker_desktop_context_exits_before_docker_info(self) -> None:
        context = SimpleNamespace(returncode=0, stdout="desktop-linux\n", stderr="")

        with patch.dict("acprof.host.preflight.os.environ", {}, clear=True), patch(
            "acprof.host.preflight.run_command",
            return_value=context,
        ) as mock_run, patch(
            "builtins.print"
        ) as mock_print:
            with pytest.raises(SystemExit) as raised:
                run.require_native_docker()

        assert (raised.value.code) == (1)
        assert (mock_run.call_count) == (1)
        message = "\n".join(str(call.args[0]) for call in mock_print.call_args_list)
        assert ("Docker Desktop") in (message)
        assert ("native Docker") in (message)

    def test_docker_desktop_info_exits_with_native_docker_hint(self) -> None:
        context = SimpleNamespace(returncode=0, stdout="default\n", stderr="")
        endpoint = SimpleNamespace(
            returncode=0,
            stdout="unix:///var/run/docker.sock\n",
            stderr="",
        )
        completed = SimpleNamespace(
            returncode=0,
            stdout=(
                "Name=docker-desktop\n"
                "OperatingSystem=Docker Desktop\n"
                "DockerRootDir=/var/lib/docker\n"
            ),
            stderr="",
        )

        with patch.dict("acprof.host.preflight.os.environ", {}, clear=True), patch(
            "acprof.host.preflight.run_command",
            side_effect=[context, endpoint, completed],
        ), patch(
            "builtins.print"
        ) as mock_print:
            with pytest.raises(SystemExit) as raised:
                run.require_native_docker()

        assert (raised.value.code) == (1)
        message = "\n".join(str(call.args[0]) for call in mock_print.call_args_list)
        assert ("Docker Desktop") in (message)
        assert ("native Docker") in (message)
        assert ("DOCKER_HOST=unix:///var/run/docker.sock") in (message)

    def test_native_linux_docker_info_is_allowed(self) -> None:
        context = SimpleNamespace(returncode=0, stdout="default\n", stderr="")
        endpoint = SimpleNamespace(
            returncode=0,
            stdout="unix:///var/run/docker.sock\n",
            stderr="",
        )
        completed = SimpleNamespace(
            returncode=0,
            stdout=(
                "Name=kainam-Jiaolong16S-Series-GM6HG0X\n"
                "OperatingSystem=Ubuntu 24.04.4 LTS\n"
                "DockerRootDir=/var/lib/docker\n"
            ),
            stderr="",
        )

        with patch.dict("acprof.host.preflight.os.environ", {}, clear=True), patch(
            "acprof.host.preflight.run_command",
            side_effect=[context, endpoint, completed],
        ):
            run.require_native_docker()

    def test_remote_docker_context_exits_before_docker_info(self) -> None:
        context = SimpleNamespace(returncode=0, stdout="remote-lab\n", stderr="")
        endpoint = SimpleNamespace(
            returncode=0,
            stdout="tcp://192.0.2.10:2376\n",
            stderr="",
        )
        stderr = io.StringIO()

        with patch.dict("acprof.host.preflight.os.environ", {}, clear=True), patch(
            "acprof.host.preflight.run_command",
            side_effect=[context, endpoint],
        ) as mock_run, pytest.raises(SystemExit) as raised, redirect_stderr(stderr):
            run.require_native_docker()

        assert (raised.value.code) == (1)
        assert (mock_run.call_count) == (2)
        assert ("tcp://192.0.2.10:2376") in (stderr.getvalue())
        assert ("/var/run/docker.sock") in (stderr.getvalue())

    def test_wsl_native_socket_override_is_rejected(self) -> None:
        context = SimpleNamespace(returncode=0, stdout="default\n", stderr="")
        stderr = io.StringIO()

        with patch.dict(
            "acprof.host.preflight.os.environ",
            {"DOCKER_HOST": "unix:///var/run/docker-native.sock"},
            clear=True,
        ), patch(
            "acprof.host.preflight.run_command",
            return_value=context,
        ) as mock_run, pytest.raises(SystemExit) as raised, redirect_stderr(stderr):
            run.require_native_docker()

        assert (raised.value.code) == (1)
        assert (mock_run.call_count) == (1)
        assert ("docker-native.sock") in (stderr.getvalue())

    def test_main_invokes_native_linux_guard_after_parsing_args(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["acprof run", "--model", "dummy-model", "--notify", "none"],
        ), patch(
            "acprof.cli.run.require_collection_host",
            side_effect=RuntimeError("host guard called"),
        ), patch(
            "acprof.cli.run.require_native_docker",
            side_effect=AssertionError("Docker guard must run after host guard"),
        ), patch(
            "acprof.host.detect.detect_task", return_value=self.resolved_task,
        ) as detect, patch("acprof.cli.run.bootstrap_project_env"):
            with pytest.raises(RuntimeError, match="host guard called"):
                run.main()
        detect.assert_not_called()

    def test_main_invokes_native_docker_guard_after_host_guard(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["acprof run", "--model", "dummy-model", "--notify", "none"],
        ), patch(
            "acprof.cli.run.require_collection_host",
        ), patch(
            "acprof.cli.run.require_native_docker",
            side_effect=RuntimeError("guard called"),
        ), patch(
            "acprof.host.detect.detect_task", return_value=self.resolved_task,
        ) as detect, patch("acprof.cli.run.bootstrap_project_env"):
            with pytest.raises(RuntimeError, match="guard called"):
                run.main()
        detect.assert_called_once()

    def test_main_runs_packet_latency_preflight_before_runtime_preparation(self) -> None:
        with patch.object(sys, "argv", ["acprof run", "--model", "dummy-model"]), patch(
            "acprof.cli.run.bootstrap_project_env",
            return_value=None,
        ), patch("acprof.cli.run.require_collection_host"), patch(
            "acprof.cli.run.require_native_docker"
        ), patch(
            "acprof.cli.run.require_cgroup_prerequisites",
            return_value="v2",
        ), patch(
            "acprof.cli.run.require_packet_latency_prerequisites",
            side_effect=SystemExit(2),
        ) as preflight, patch(
            "acprof.cli.run.require_cpu_energy_prerequisites",
        ), patch(
            "acprof.host.detect.detect_task",
            return_value=self.resolved_task,
        ) as detect, patch(
            "acprof.cli.run._prepare_runtime",
            side_effect=AssertionError("runtime preparation must not run before packet latency preflight"),
        ):
            with pytest.raises(SystemExit) as raised:
                run.main()

        assert (raised.value.code) == (2)
        detect.assert_called_once()
        preflight.assert_called_once_with(sniff_iface="docker0")

    def test_main_defaults_to_auto_window_and_compute_profiler_disabled(self) -> None:
        task_info = TaskInfo(
            model_id="dummy-model",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="1" * 40,
            detection_method="unit",
        )
        stderr = io.StringIO()
        built_image = runtime_images.ImageInfo(tag="acprof-nlp-dummy-model:latest")

        def collect_metadata(**kwargs):
            build_image.assert_called_once_with(task_info, run.PROJECT_DIR)
            assert (kwargs["image_info"]) is (built_image)
            return SimpleNamespace()

        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            sys,
            "argv",
            [
                "acprof run",
                "--model",
                "dummy-model",
                "--skip-build",
                "--cpus",
                "1",
                "--mems",
                "2",
                "--gpus",
                "on",
                "--output-dir",
                tmp_dir,
            ],
        ), patch(
            "acprof.cli.run.bootstrap_project_env",
            return_value=None,
        ), patch(
            "acprof.cli.run.require_collection_host"
        ), patch(
            "acprof.cli.run.require_native_docker"
        ), patch(
            "acprof.cli.run.require_cgroup_prerequisites",
            return_value="v2",
        ), patch(
            "acprof.cli.run.require_packet_latency_prerequisites"
        ), patch(
            "acprof.cli.run.require_cpu_energy_prerequisites"
        ), patch(
            "acprof.cli.run.require_mips_prerequisites"
        ), patch(
            "acprof.host.detect.detect_task",
            return_value=task_info,
        ), patch(
            "acprof.host.command.run_command",
            return_value=SimpleNamespace(returncode=1, stdout="", stderr="No such image"),
        ), patch(
            "acprof.host.runtime_images.build_runtime_image",
            return_value=built_image,
        ) as build_image, patch(
            "acprof.host.static_metadata.collect_static_meta",
            side_effect=collect_metadata,
        ) as collect_static_meta, patch.multiple(
            "acprof.host.static_metadata",
            enrich_static_meta_from_input_plan=Mock(side_effect=lambda meta, planned: meta),
            enrich_static_meta=Mock(side_effect=lambda meta, values: meta),
        ), patch(
            "acprof.host.runtime_validation.validate_runtime", return_value={"status": "ok"}
        ), patch(
            "acprof.host.static_metadata.write_static_meta_json"
        ), patch(
            "acprof.host.input_plan.plan_input_scales",
            return_value=input_plan.PlannedInputScales(
                scales=[1.0],
                source="unit",
                plan_file=None,
            ),
        ), patch(
            "acprof.host.orchestrator.run_matrix",
            side_effect=orchestrator.EnergyProfilingError("gpu_idle_power_w unstable"),
        ) as run_matrix, pytest.raises(SystemExit) as raised, redirect_stderr(stderr):
            run.main()

        assert (raised.value.code) == (1)
        assert ("[energy][ERROR] gpu_idle_power_w unstable") in (stderr.getvalue())
        _, kwargs = run_matrix.call_args
        assert (kwargs["image_info"]) is (built_image)
        assert (kwargs["repeat_in_window"]) == (0)
        assert (kwargs["repeat_window_seconds"]) == (10.0)
        assert (kwargs["request_timeout_seconds"]) == (300.0)
        assert (kwargs["idle_seconds"]) == (20.0)
        assert (kwargs["idle_cooldown_seconds"]) == (5.0)
        assert (kwargs["compute_profile_plan_file"]) == ("")
        assert (kwargs["prune_startup_oom"])
        assert not (collect_static_meta.call_args.kwargs["compute_profile_enabled"])

    def test_main_can_enable_dual_compute_profiles_and_keep_artifacts(self) -> None:
        task_info = TaskInfo(
            model_id="dummy-model",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="unit",
        )

        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            sys,
            "argv",
            [
                "acprof run",
                "--model",
                "dummy-model",
                "--skip-build",
                "--compute-profile-tool",
                "both",
                "--cpus",
                "1",
                "--mems",
                "2",
                "--gpus",
                "off,on",
                "--no-prune-startup-oom",
                "--request-timeout-seconds",
                "123.5",
                "--output-dir",
                tmp_dir,
            ],
        ), patch(
            "acprof.cli.run.bootstrap_project_env",
            return_value=None,
        ), patch(
            "acprof.cli.run.require_collection_host"
        ), patch(
            "acprof.cli.run.require_native_docker"
        ), patch(
            "acprof.cli.run.require_cgroup_prerequisites",
            return_value="v2",
        ), patch(
            "acprof.cli.run.require_packet_latency_prerequisites"
        ), patch(
            "acprof.cli.run.require_cpu_energy_prerequisites"
        ), patch(
            "acprof.cli.run.require_mips_prerequisites"
        ), patch(
            "acprof.host.detect.detect_task",
            return_value=task_info,
        ), patch(
            "acprof.host.runtime_images.prepare_image",
            return_value=runtime_images.ImageInfo(tag="acprof-nlp-dummy-model:latest"),
        ), patch(
            "acprof.host.static_metadata.collect_static_meta",
            return_value=SimpleNamespace(),
        ), patch.multiple(
            "acprof.host.static_metadata",
            enrich_static_meta_from_input_plan=Mock(side_effect=lambda meta, planned: meta),
            enrich_static_meta=Mock(side_effect=lambda meta, values: meta),
        ), patch(
            "acprof.host.runtime_validation.validate_runtime", return_value={"status": "ok"}
        ), patch(
            "acprof.host.static_metadata.write_static_meta_json"
        ), patch(
            "acprof.host.input_plan.plan_input_scales",
            return_value=input_plan.PlannedInputScales(
                scales=[1.0],
                source="unit",
                plan_file=None,
            ),
        ), patch(
            "acprof.host.compute_profile.collect_compute_profile_plan",
            return_value=f"{tmp_dir}/dummy-model/compute_profile_plan.json",
        ) as collect_compute_profile_plan, patch(
            "acprof.host.orchestrator.run_matrix",
            return_value=[],
        ) as run_matrix:
            run.main()

        _, kwargs = collect_compute_profile_plan.call_args
        assert (kwargs["compute_profile_tool"]) == ("both")
        assert (kwargs["torch_profiler_repeat"]) == (1)
        assert (kwargs["ncu_repeat"]) == (1)
        assert (kwargs["keep_profiles"])
        assert not (run_matrix.call_args.kwargs["prune_startup_oom"])
        assert (run_matrix.call_args.kwargs["request_timeout_seconds"]) == (123.5)

    def test_main_rejects_invalid_request_timeout(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "acprof run",
                "--model",
                "dummy-model",
                "--request-timeout-seconds",
                "0",
            ],
        ), patch(
            "acprof.cli.run.bootstrap_project_env",
            return_value=None,
        ), pytest.raises(SystemExit) as raised:
            run.main()

        assert (raised.value.code) == (2)

    def test_main_records_invocation_command_in_static_meta(self) -> None:
        task_info = TaskInfo(
            model_id="dummy-model",
            pipeline_tag="fill-mask",
            task_family="nlp",
            runtime_backend="transformers_pipeline",
            library_name="transformers",
            model_revision="main",
            detection_method="unit",
        )

        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            sys,
            "argv",
            [
                "acprof run",
                "--model",
                "dummy-model",
                "--skip-build",
                "--compute-profile-tool", "none",
                "--cpus",
                "1",
                "--mems",
                "2",
                "--gpus",
                "off",
                "--output-dir",
                tmp_dir,
            ],
        ), patch(
            "acprof.cli.run.bootstrap_project_env",
            return_value=None,
        ), patch(
            "acprof.cli.run.require_collection_host"
        ), patch(
            "acprof.cli.run.require_native_docker"
        ), patch(
            "acprof.cli.run.require_cgroup_prerequisites",
            return_value="v2",
        ), patch(
            "acprof.cli.run.require_packet_latency_prerequisites"
        ), patch(
            "acprof.cli.run.require_cpu_energy_prerequisites"
        ), patch(
            "acprof.cli.run.require_mips_prerequisites"
        ), patch(
            "acprof.host.detect.detect_task",
            return_value=task_info,
        ), patch(
            "acprof.host.runtime_images.prepare_image",
            return_value=runtime_images.ImageInfo(tag="acprof-nlp-dummy-model:latest"),
        ), patch(
            "acprof.host.static_metadata.collect_static_meta",
            return_value=SimpleNamespace(),
        ) as collect_static_meta, patch.multiple(
            "acprof.host.static_metadata",
            enrich_static_meta_from_input_plan=Mock(side_effect=lambda meta, planned: meta),
            enrich_static_meta=Mock(side_effect=lambda meta, values: meta),
        ), patch(
            "acprof.host.runtime_validation.validate_runtime", return_value={"status": "ok"}
        ), patch(
            "acprof.host.static_metadata.write_static_meta_json"
        ) as write_static_meta_json, patch(
            "acprof.cli.run.write_collection_history_json"
        ) as write_collection_history_json, patch(
            "acprof.host.input_plan.plan_input_scales",
            return_value=input_plan.PlannedInputScales(
                scales=[1.0],
                source="unit",
                plan_file=None,
            ),
        ), patch(
            "acprof.host.orchestrator.run_matrix",
            return_value=[],
        ):
            run.main()

        _, kwargs = collect_static_meta.call_args
        assert (kwargs["run_command"]) == ("acprof run --model dummy-model --skip-build --compute-profile-tool none "
            "--cpus 1 --mems 2 --gpus off --output-dir "
            + tmp_dir)
        assert (kwargs["cgroup_version"]) == ("v2")
        assert (kwargs["cgroup_collection_mode"]) == ("strict_v2")
        assert (write_static_meta_json.called)
        assert (write_static_meta_json.call_args_list[0].args[1]) == (os.path.join(tmp_dir, "dummy-model", "static_meta.json"))
        assert (write_collection_history_json.call_args.args[1]) == (os.path.join(tmp_dir, "dummy-model", "metadata", "collection_history.json"))
        assert (write_collection_history_json.call_args.args[0]["schema_version"]) == (1)
