"""Docker port failures are configuration errors, not failed measurements."""
import subprocess
from dataclasses import replace
from unittest.mock import Mock

import pytest

from acprof.application.run import RunApplication
from acprof.experiments.configuration import RunConfig, RunConfigError
from acprof.host import docker_runtime
from acprof.host.detect import TaskInfo
from acprof.host.images.runtime import ImageInfo
from acprof.host.runtimes.selection import RuntimeSelection
from acprof.system.cleanup import format_exception_notes


@pytest.mark.parametrize(("cpu", "mem"), [(600, 1), (575, 34)])
def test_service_port_rejects_out_of_range_resources(cpu, mem):
    with pytest.raises(ValueError, match="65535"):
        docker_runtime.service_port(cpu, mem)


@pytest.mark.parametrize(("cpu", "mem", "expected"), [(1, 1, 8103), (575, 33, 65535)])
def test_service_port_keeps_legal_boundaries(cpu, mem, expected):
    assert docker_runtime.service_port(cpu, mem) == expected


@pytest.mark.parametrize(("cpus", "mems", "resolved", "port"), [
    ("1,600", "1", (600, 1), 68003),
    ("auto", "56734", (8, 56734), 65536),
    ("576", "auto", (576, 16), 65618),
])
def test_docker_configuration_reports_both_resource_fields(cpus, mems, resolved, port):
    config = replace(RunConfig.smoke("fixture/model"), runtime="docker", cpus=cpus, mems=mems)
    with pytest.raises(RunConfigError, match=str(port)) as caught:
        config.validate()
    assert [issue.field for issue in caught.value.issues] == ["cpus", "mems"]
    assert f"cpus={resolved[0]}, mems={resolved[1]}" in str(caught.value)


@pytest.mark.parametrize(("cpus", "mems"), [
    ("575", "33"), ("1,2", "200,100"), ("auto", "56733"), ("575", "auto"), ("auto", "auto"),
])
def test_legal_docker_matrix_and_sequential_port_reuse_remain_supported(cpus, mems):
    config = replace(RunConfig.smoke("fixture/model"), runtime="docker", cpus=cpus, mems=mems)
    assert config.validate() == config


@pytest.mark.parametrize(("cpus", "mems", "field"), [("invalid", "56734", "cpus"), ("576", "0", "mems")])
def test_malformed_resources_do_not_fall_back_to_docker_defaults(cpus, mems, field):
    config = replace(RunConfig.smoke("fixture/model"), runtime="docker", cpus=cpus, mems=mems)
    with pytest.raises(RunConfigError) as caught:
        config.validate()
    assert [issue.field for issue in caught.value.issues] == [field]


def test_duplicate_resources_preserve_port_and_axis_diagnostic_order():
    config = replace(RunConfig.smoke("fixture/model"), runtime="docker", cpus="576,576", mems="auto")
    with pytest.raises(RunConfigError) as caught:
        config.validate()
    assert [issue.field for issue in caught.value.issues] == ["cpus", "mems", "cpus"]
    assert all("65618" in issue.reason for issue in caught.value.issues[:2])


@pytest.mark.parametrize("runtime", ["process", "auto"])
@pytest.mark.parametrize(("cpus", "mems"), [("600", "1"), ("auto", "56734"), ("576", "auto")])
def test_unresolved_or_process_configuration_does_not_use_docker_port_limits(runtime, cpus, mems):
    config = replace(RunConfig.smoke("fixture/model"), runtime=runtime, cpus=cpus, mems=mems)
    assert config.validate() == config


@pytest.mark.parametrize("mems", ["1", "auto"])
def test_auto_docker_rejects_invalid_port_before_task_resolution(tmp_path, monkeypatch, mems):
    workflow = Mock()
    workflow.resolve.side_effect = AssertionError("task resolution must not begin")
    prepare, state = Mock(), Mock()
    application = RunApplication(
        workflow_factory=lambda _: workflow, runtime_preparer=prepare,
        state_factory=state, environment_setup=lambda _: None, notifications=Mock(),
    )
    monkeypatch.setattr("acprof.application.run.require_collection_host", lambda **_: None)
    monkeypatch.setattr("acprof.host.runtimes.selection.select_runtime", lambda *_, **__: RuntimeSelection(
        "auto", "docker", "fixture Docker selection",
    ))
    config = replace(RunConfig.smoke("fixture/model"), runtime="auto", cpus="600", mems=mems,
                     output_dir=str(tmp_path / "not-created"))
    with pytest.raises(RunConfigError, match="65535"):
        application.run(config)
    workflow.resolve.assert_not_called()
    prepare.assert_not_called()
    state.assert_not_called()
    assert not (tmp_path / "not-created").exists()


@pytest.mark.parametrize("detail", ["bind: address already in use", "port is already allocated"])
def test_docker_port_conflict_preserves_error_and_mount_ownership(monkeypatch, detail):
    from acprof.host.model_store import store

    mount = Mock(args=[])
    mount.begin_launch.side_effect = lambda path: path
    retain = Mock()
    monkeypatch.setattr(store, "acquire_mount", lambda _: mount)
    monkeypatch.setattr(store, "retain_mount_for_cleanup_debt", retain)
    monkeypatch.setattr(docker_runtime, "recover_abandoned_containers", lambda *_: None)
    command = Mock(return_value=subprocess.CompletedProcess(["docker", "run"], 125, "", detail))
    monkeypatch.setattr(docker_runtime.host_command, "run_command", command)
    task = TaskInfo("fixture/model", "tabular-classification", "structured", "onnxruntime", "onnx", "a" * 40, "manual")
    with pytest.raises(RuntimeError, match="docker run failed") as caught:
        docker_runtime.start_container_session(task, 1, 1, "off", ImageInfo("fixture"), "fixture", "[test]")
    diagnostic = format_exception_notes(caught.value)
    assert detail in diagnostic
    assert "127.0.0.1:8103" in diagnostic
    assert "docker ps" in diagnostic
    retain.assert_called_once_with(mount)
    mount.close.assert_not_called()
    command.assert_called_once()
