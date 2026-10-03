from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.host import docker_runtime, gpu_device, runtime_images
from acprof.host.detect import TaskInfo

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

@pytest.mark.parametrize('selector', ('all', '0,1', 'MIG-GPU-abc/0/1', '-1', 'GPU-one\nsecond'))
def test_rejects_ambiguous_and_mig_device_selection(selector):
    with patch.object(gpu_device, "run_command") as run:
        with pytest.raises(ValueError):
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
    with patch.object(docker_runtime, "resolve_gpu_device", return_value=DEVICE, create=True), patch("acprof.host.command.run_command", side_effect=lambda cmd, **kw: (
            commands.append(cmd) or SimpleNamespace(returncode=0, stdout="b" * 64 if cmd[:2] == ["docker", "run"] else "", stderr="")
        )
    ), patch("requests.get", return_value=response):
        session = docker_runtime.start_container_session(
            task, 1, 2, "on", runtime_images.ImageInfo("fixture"), "fixture", "[test]"
        )
    command = next(cmd for cmd in commands if cmd[:3] == ["docker", "run", "-d"])
    assert (command[command.index("--gpus") + 1]) == ("device=GPU-second")
    assert ("CUDA_VISIBLE_DEVICES=0") in (command)
    assert (session.gpu_device) == (DEVICE)
