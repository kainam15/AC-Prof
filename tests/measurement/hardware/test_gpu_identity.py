import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.host import docker_runtime, gpu_device
from acprof.host.detect import TaskInfo
from acprof.host.images import runtime as runtime_images

DEVICE = {"uuid": "GPU-second", "index": 1, "pci_bus_id": "00000000:02:00.0",
          "name": "Second GPU", "memory_total_bytes": 8 * 1024 ** 3}


def test_resolves_index_to_uuid_and_pins_it_only_for_one_run():
    result = SimpleNamespace(returncode=0, stderr="", stdout=
                             "GPU-second, 1, 00000000:02:00.0, Second GPU, 8192, N/A\n")
    with gpu_device.gpu_device_scope(), patch.object(gpu_device, "run_command", return_value=result) as run:
        assert (gpu_device.pin_gpu_device("1")) == (DEVICE)
        assert (gpu_device.gpu_docker_args()[1]) == ("device=GPU-second")
        run.assert_called_once()
        assert ("--id=1") in (run.call_args.args[0])
    assert (gpu_device.selected_gpu_device()) == ({})

@pytest.mark.parametrize('selector', ['all', '0,1', 'MIG-GPU-abc/0/1', '-1', 'GPU-one\nsecond'])
def test_rejects_ambiguous_and_mig_device_selection(selector):
    with patch.object(gpu_device, "run_command") as run:
        with pytest.raises(ValueError, match="GPU selector must be one physical GPU index or UUID"):
            gpu_device.resolve_gpu_device(selector)
        run.assert_not_called()
    with patch.object(gpu_device, "run_command", return_value=SimpleNamespace(
        returncode=0, stderr="", stdout="GPU-second, 1, bus, GPU, 8192, Enabled\n"
    )), pytest.raises(RuntimeError, match="MIG"):
        gpu_device.resolve_gpu_device("1")

def test_energy_and_resource_monitors_resolve_the_same_uuid():
    from acprof.monitors import energy_nvml, resource_usage
    fake_nvml = SimpleNamespace(nvmlInit=lambda: None, nvmlShutdown=lambda: None,
                                nvmlDeviceGetName=lambda handle: "Second GPU")
    def by_uuid(uuid):
        assert (uuid) == ("GPU-second")
        return "second-handle"
    fake_nvml.nvmlDeviceGetHandleByUUID = by_uuid
    with patch.object(energy_nvml, "pynvml", fake_nvml), patch.object(resource_usage, "pynvml", fake_nvml):
        energy = energy_nvml.GPUEnergyMonitor(device_uuid="GPU-second")
        resources = resource_usage.ResourceUsageMonitor(use_gpu=True, device_uuid="GPU-second")
        assert (energy.handle) == ("second-handle")
        assert (resources._gpu_handle) == ("second-handle")
        energy.close()
        resources.close()

def test_container_exposes_only_the_selected_uuid_and_records_identity():
    task = TaskInfo(model_id="fixture/model", pipeline_tag="fill-mask", task_family="nlp",
                    runtime_backend="transformers_pipeline", library_name="transformers",
                    model_revision="main", detection_method="fixture")
    commands = []
    response = SimpleNamespace(status_code=200, text="", json=lambda: {"status": "ok"})
    with patch.object(docker_runtime, "resolve_gpu_device", return_value=DEVICE, create=True), patch("acprof.system.command.run_command", side_effect=lambda cmd, **kw: (
            commands.append(cmd) or SimpleNamespace(returncode=0, stdout="b" * 64 if cmd[:2] == ["docker", "run"] else "", stderr="")
        )
    ), patch("requests.get", return_value=response):
        session = docker_runtime.start_container_session(
            task, 1, 2, "on", runtime_images.ImageInfo("fixture"), "fixture", "[test]", cpuset_cpus="3,1-2"
        )
    command = next(cmd for cmd in commands if cmd[:3] == ["docker", "run", "-d"])
    assert '--cpus=1' in command
    assert '--cpuset-cpus=1-3' in command
    assert '--cap-drop=ALL' in command
    assert '--security-opt=no-new-privileges' in command
    assert '--privileged' not in command
    assert (command[command.index("--gpus") + 1]) == ("device=GPU-second")
    assert ("CUDA_VISIBLE_DEVICES=0") in (command)
    assert (session.gpu_device) == (DEVICE)


def test_docker_gpu_probe_requires_local_image():
    from unittest.mock import Mock
    run = Mock(return_value=subprocess.CompletedProcess([], 1, "", "No such image"))
    with pytest.raises(gpu_device.DockerGpuError, match="docker pull .*@sha256:") as failure:
        gpu_device.require_docker_gpu(DEVICE, command_runner=run)
    assert failure.value.reason_code == "probe_image_missing"
    assert all(call.args[0][3:5] == ["image", "inspect"] for call in run.call_args_list)


@pytest.mark.parametrize("observation", ["", "GPU-different", "GPU-second\nGPU-another", "timeout"])
def test_docker_gpu_probe_rejects_bad_proof_and_cleans_only_owned_container(observation):
    calls = []
    owned = "b" * 64
    image = "sha256:" + "a" * 64

    def command(cmd, *, timeout):
        assert cmd[:3] == ["docker", "--host", "unix:///var/run/docker.sock"]
        cmd = ["docker", *cmd[3:]]
        calls.append(cmd)
        if cmd[1] == "image":
            return subprocess.CompletedProcess(cmd, 0, json.dumps({"image_id": image, "os": "linux", "architecture": "amd64"}), "")
        if cmd[1] == "run":
            Path(cmd[cmd.index("--cidfile") + 1]).write_text(owned)
            if observation == "timeout":
                raise subprocess.TimeoutExpired(cmd, timeout)
            return subprocess.CompletedProcess(cmd, 0, observation, "")
        assert cmd == ["docker", "rm", "-f", owned]
        return subprocess.CompletedProcess(cmd, 0, owned, "")

    with pytest.raises(gpu_device.DockerGpuError) as failure:
        gpu_device.require_docker_gpu(DEVICE, command_runner=command)
    assert ("timed out" if observation == "timeout" else "Expected only GPU-second") in failure.value.detail
    assert calls[-1] == ["docker", "rm", "-f", owned]


@pytest.mark.parametrize("valid", [True, False])
def test_cached_platform_must_match_project_tag_and_full_lock(valid):
    from acprof.host.images.environment import PLATFORM_LABEL, platform_fingerprint
    from acprof.runtime.profiles import PLATFORMS

    platform_key = platform_fingerprint(PLATFORMS["cu128"])
    expected = f"acprof-platform-cu128:{platform_key[:20]}"
    image = "sha256:" + "d" * 64
    runs = []
    def command(cmd, *, timeout):
        if cmd[3:5] == ["image", "inspect"]:
            if cmd[5] != expected:
                return subprocess.CompletedProcess(cmd, 1, "", "No such image")
            labels = {"org.acprof.image-kind": "platform", "org.acprof.platform": "cu128",
                      PLATFORM_LABEL: platform_key if valid else "counterfeit"}
            return subprocess.CompletedProcess(cmd, 0, json.dumps(
                {"image_id": image, "labels": labels, "os": "linux", "architecture": "amd64"}), "")
        runs.append(cmd)
        assert cmd[cmd.index("--entrypoint") + 2] == image
        return subprocess.CompletedProcess(cmd, 0, DEVICE["uuid"], "")

    if valid:
        assert image in gpu_device.require_docker_gpu(DEVICE, command_runner=command)
        assert len(runs) == 1
    else:
        with pytest.raises(gpu_device.DockerGpuError):
            gpu_device.require_docker_gpu(DEVICE, command_runner=command)
        assert not runs


def test_probe_timeout_before_cidfile_cleans_by_exact_owner_labels():
    owned = "b" * 64
    calls = []

    def command(cmd, *, timeout):
        cmd = cmd[3:]
        calls.append(cmd)
        if cmd[0] == "run":
            raise subprocess.TimeoutExpired(cmd, timeout)
        if cmd[0] == "ps":
            labels = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--filter"]
            assert any(label.startswith("label=org.acprof.gpu-probe=") for label in labels)
            assert any(label.startswith("label=org.acprof.owner.pid=") for label in labels)
            return subprocess.CompletedProcess(cmd, 0, owned, "")
        assert cmd == ["rm", "-f", owned]
        return subprocess.CompletedProcess(cmd, 0, owned, "")

    with pytest.raises(gpu_device.DockerGpuError, match="timed out"):
        gpu_device.require_docker_gpu(DEVICE, command_runner=command, image_id="sha256:" + "a" * 64)
    assert calls[-1] == ["rm", "-f", owned]
