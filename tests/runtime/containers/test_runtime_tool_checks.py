"""Runtime availability requires execution evidence and settled containers."""
import json
import subprocess
from pathlib import Path

import pytest

from acprof.host.container_lifecycle import ContainerCleanupError
from acprof.host.profiling import runtime_checks
from acprof.host.profiling.runtime_checks import check_runtime_tools

IMAGE = "sha256:" + "b" * 64
OTHER = "sha256:" + "c" * 64
CONTAINER = "a" * 64


def evidence(tool):
    payload = {"tool": tool, "version": "fixture"}
    payload.update({"events": 1, "flops": 128} if tool == "torch" else
                   {"snapshots": 3, "peak_heap_bytes": 131072})
    return json.dumps(payload)


@pytest.fixture
def docker(monkeypatch):
    class Docker:
        images = (IMAGE, IMAGE)
        calls = []
        outcomes = {}
        cleanup_failure = False

        def run(self, command, **kwargs):
            command = list(command)
            self.calls.append((command, kwargs))
            assert 0 < kwargs["timeout"] <= 30
            output = ""
            if command[:3] == ["docker", "image", "ls"]:
                output = "\n".join(self.images)
            elif command[:2] == ["docker", "run"]:
                Path(command[command.index("--cidfile") + 1]).write_text(CONTAINER)
                tool = command[-1]
                image = command[command.index("-c") - 1]
                outcome = self.outcomes.get((image, tool), evidence(tool))
                if isinstance(outcome, Exception):
                    raise outcome
                if isinstance(outcome, subprocess.CompletedProcess):
                    return outcome
                output = outcome
            elif command[:2] == ["docker", "rm"] and self.cleanup_failure:
                return subprocess.CompletedProcess(command, 1, "", "cannot remove")
            elif command[:2] == ["docker", "inspect"]:
                output = json.dumps([{"Id": CONTAINER, "State": {"Running": True}}])
            return subprocess.CompletedProcess(command, 0, output, "")

        @property
        def launches(self):
            return [command for command, _ in self.calls if command[:2] == ["docker", "run"]]

    fake = Docker()
    monkeypatch.setattr("acprof.host.profiling.commands.run_command", fake.run)
    return fake


def test_each_tool_uses_immutable_local_image_and_cleans_up(docker):
    results = check_runtime_tools(docker.run)
    assert [result.status for result in results] == ["ok", "ok"]
    assert len(docker.launches) == 2, "duplicate tags must not repeat tool checks"
    for command in docker.launches:
        assert command[command.index("-c") - 1] == IMAGE
        for option in ("--pull=never", "--network=none", "--read-only", "--user=65532:65532",
                       "--cpus=1", "--memory=2g", "--pids-limit=128", "--entrypoint=python"):
            assert option in command
        assert not any(option in command for option in ("--gpus", "--privileged", "--volume"))
    assert sum(command[:3] == ["docker", "rm", "-f"] and command[-1] == CONTAINER
               for command, _ in docker.calls) == 2
    assert all(IMAGE in result.detail for result in results)


def test_self_check_supplies_user_identity_and_writable_torch_cache(docker):
    check_runtime_tools(docker.run)
    for command in docker.launches:
        assert "--env=USER=acprof" in command
        assert "--env=LOGNAME=acprof" in command
        assert "--env=TORCHINDUCTOR_CACHE_DIR=/tmp/acprof-torch" in command


@pytest.mark.parametrize("output", ["", "not json", "{}", "[]", '{"tool":"torch","version":"2.9","events":0,"flops":128}',
                                    '{"tool":"torch","version":"2.9","events":1,"flops":1}',
                                    '{"tool":"torch","version":"2.9","events":true,"flops":128}'])
def test_empty_or_invalid_evidence_never_passes_and_other_tool_is_checked(docker, output):
    docker.outcomes[(IMAGE, "torch")] = output
    results = check_runtime_tools(docker.run)
    assert [result.status for result in results] == ["fail", "ok"]


@pytest.mark.parametrize("tool", ["torch", "massif"])
def test_nonzero_exit_with_valid_json_is_still_unavailable(docker, tool):
    docker.outcomes[(IMAGE, tool)] = subprocess.CompletedProcess("docker run", 7, evidence(tool), "tool failed")
    results = {result.tool: result for result in check_runtime_tools(docker.run)}
    assert results[tool].status == "fail"
    assert "exit=7; tool failed" in results[tool].detail
    assert results["massif" if tool == "torch" else "torch"].status == "ok"


def test_timeout_is_cleaned_before_trying_another_runtime(docker):
    docker.images = (IMAGE, OTHER)
    docker.outcomes[(IMAGE, "torch")] = subprocess.TimeoutExpired("docker run", 15)
    results = check_runtime_tools(docker.run)
    assert [result.status for result in results] == ["ok", "ok"]
    assert "timed out" in results[0].detail
    assert OTHER in results[0].detail
    commands = [command for command, _ in docker.calls]
    launches = [index for index, command in enumerate(commands) if command[:2] == ["docker", "run"]]
    assert ["docker", "rm", "-f", CONTAINER] in commands[launches[0] + 1:launches[1]]


def test_cleanup_failure_aborts_all_remaining_runtime_checks(docker):
    docker.cleanup_failure = True
    with pytest.raises(ContainerCleanupError):
        check_runtime_tools(docker.run)
    assert len(docker.launches) == 1


def test_budget_exhaustion_is_explicit_and_does_not_skip_the_other_tool(docker, monkeypatch):
    docker.images = (IMAGE, OTHER)
    docker.outcomes[(IMAGE, "torch")] = subprocess.TimeoutExpired("docker run", 15)
    ticks = iter((0, 0, 31, 31, 31))
    monkeypatch.setattr(runtime_checks.time, "monotonic", lambda: next(ticks))
    results = check_runtime_tools(docker.run)
    assert [result.status for result in results] == ["pending", "ok"]
    assert "1 runtimes untested" in results[0].detail
    assert [command[-1] for command in docker.launches] == ["torch", "massif"]


def test_no_images_reports_unavailable_without_downloading(docker):
    docker.images = ()
    results = check_runtime_tools(docker.run)
    assert [result.status for result in results] == ["fail", "fail"]
    assert all("No local" in result.detail for result in results)
    assert not docker.launches
    assert all(command[:3] == ["docker", "image", "ls"] for command, _ in docker.calls)


def test_surviving_model_image_is_used_if_dependency_base_was_pruned(docker):
    def runner(command, **kwargs):
        if "label=org.acprof.image-kind=environment" in command:
            return subprocess.CompletedProcess(command, 0, "", "")
        return docker.run(command, **kwargs)

    assert all(result.status == "ok" for result in check_runtime_tools(runner))
    assert len(docker.launches) == 2


def test_nonimmutable_image_identity_is_rejected(docker):
    docker.images = ("acprof-runtime:latest",)
    with pytest.raises(ValueError, match="immutable"):
        check_runtime_tools(docker.run)
    assert not docker.launches
