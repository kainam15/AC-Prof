"""Offline Docker/containerd transcripts exercise content, recovery and progress contracts."""
import base64
import io
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from acprof.events.image import ImageProgress
from acprof.events.preparation import encode_event, parse_event
from acprof.host.images import content, environment, operations
from acprof.host.images.errors import DockerPreparationError
from acprof.host.images.progress import DockerOutput
from acprof.host.preparation.collection_workflow import PreparationWorkflow
from acprof.system.command import run_command
from acprof.tui.run.progress import RunProgressTracker

IMAGE_ID = "sha256:" + "a" * 64
TARGET = "sha256:" + "b" * 64
IMAGE = "acprof-platform-cpu:fixture"
REFERENCE = "registry.example/acprof@" + TARGET


@pytest.mark.parametrize(("message", "reason"), [
    ("dial tcp: i/o timeout", "timeout"), ("lookup mirror.example: no such host", "dns"),
    ("proxyconnect tcp: connection refused http://user:secret@proxy.example", "proxy"),
    ("manifest unknown", "not-found"), ("digest mismatch", "integrity"),
])
@pytest.mark.parametrize("format_kind", ["plain", "batch", "vertex"])
def test_frontend_failure_is_specific_redacted_and_never_retried(monkeypatch, message, reason, format_kind):
    frontend = "mirror.example/docker/dockerfile:1@" + TARGET
    step = "resolve image config for docker-image://" + frontend
    vertex = {"id": "frontend", "name": step, "error": message}
    transcript = ("ERROR: " + step + ": " + message if format_kind == "plain" else
                  json.dumps({"vertexes": [vertex]} if format_kind == "batch" else vertex))
    run = Mock(return_value=subprocess.CompletedProcess([], 1, "", transcript))
    monkeypatch.setattr(operations.host_command, "run_command", run)
    with pytest.raises(DockerPreparationError, match="BuildKit frontend") as caught:
        operations.run_docker(["docker", "build"], component="environment", image=IMAGE,
                              operation="build", frontend=frontend)
    error = caught.value.to_dict()
    assert error["reason_code"] == "docker_frontend_failed"
    assert error["registry"] == "mirror.example"
    assert error["network_context"] == "buildkit"
    assert error["network_reason"] == reason
    assert error["actions"] == ["retry"]
    assert "secret" not in json.dumps(error)
    run.assert_called_once()


def test_later_dependency_failure_does_not_become_frontend_failure(monkeypatch):
    frontend = "mirror.example/docker/dockerfile:1@" + TARGET
    transcript = ("#1 resolve image config for docker-image://" + frontend + "\n#1 DONE 0.1s\n"
                  "ERROR: failed to solve: process python environment_tools.py exited with code 1")
    monkeypatch.setattr(operations.host_command, "run_command",
                        Mock(return_value=subprocess.CompletedProcess([], 1, "", transcript)))
    with pytest.raises(DockerPreparationError, match="Docker 构建失败") as caught:
        operations.run_docker(["docker", "build"], component="environment", image=IMAGE,
                              operation="build", frontend=frontend)
    assert caught.value.reason_code == "docker_build_failed"


@pytest.fixture
def store(monkeypatch):
    state = SimpleNamespace(status="complete", present=9, required="9", endpoint="unix:///var/run/docker.sock",
                            ctr_code=0, pull_code=0, pull_error="", repaired=False, commands=[],
                            target=TARGET, version="29.4.0", security=[])
    monkeypatch.setattr(content.shutil, "which", lambda _name: "/usr/bin/ctr")
    monkeypatch.setattr(content.platform, "system", lambda: "Linux")
    monkeypatch.setattr(content.platform, "machine", lambda: "x86_64")
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    monkeypatch.delenv("ACPROF_MAX_DOWNLOAD", raising=False)
    monkeypatch.delenv("ACPROF_REQUIRE_PREPARED_IMAGE", raising=False)
    monkeypatch.setattr(operations, "_anonymous_socket", lambda: None)

    def execute(command, **kwargs):
        state.commands.append(command)
        code = 0
        if command[:3] == ["docker", "context", "inspect"]:
            output = json.dumps(state.endpoint)
        elif command[:2] == ["docker", "info"]:
            output = json.dumps({"ServerVersion": state.version, "SecurityOptions": state.security,
                                 "DriverStatus": [["driver-type", "io.containerd.snapshotter.v1"]]})
        elif command[:3] == ["docker", "image", "inspect"]:
            output = json.dumps([{"Id": IMAGE_ID, "Descriptor": {"digest": TARGET}, "RepoDigests": [REFERENCE]}])
        elif command[0] == "/usr/bin/ctr":
            assert "--namespace" in command
            assert "moby" in command
            assert command[-1] == "target.digest==" + TARGET
            code = state.ctr_code
            output = f"REF TYPE DIGEST STATUS SIZE UNPACKED\n{IMAGE} application/vnd.oci.image.manifest.v1+json {state.target} {state.status} ({state.present}/{state.required}) 8.0MiB/9.0MiB true\n"
        elif command[:2] == ["docker", "pull"]:
            code = state.pull_code
            output = state.pull_error if code else "Status: Image is up to date"
            if state.repaired:
                state.status, state.present = "complete", 9
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, code, output, "permission denied" if state.ctr_code else "")

    monkeypatch.setattr(content.host_command, "run_command", execute)
    return state


@pytest.mark.parametrize(("state", "present", "required"), [("complete", 9, "9"), ("incomplete", 8, "9"), ("unavailable", 0, "?")])
def test_content_distinguishes_snapshot_from_required_blobs(store, state, present, required):
    store.status, store.present, store.required = state, present, required
    result = content.check_content(IMAGE, IMAGE_ID)
    assert result.status == state
    assert result.present == present
    assert result.required == (None if required == "?" else int(required))
    assert result.repair_reference == REFERENCE
    assert "29.9.0" in result.detail


def test_missing_blob_blocks_manifest_run_and_cache_hit(store, monkeypatch):
    from acprof.host.images import runtime
    store.status, store.present = "incomplete", 8
    monkeypatch.setattr(runtime, "inspect_identity", lambda _name: {"image_id": IMAGE_ID, "labels": {}})
    manifest = Mock(side_effect=AssertionError("must reject before running image"))
    monkeypatch.setattr(environment, "read_image_manifest", manifest)
    with pytest.raises(DockerPreparationError) as caught:
        environment.checked_image(IMAGE, "label", "fingerprint", {}, {}, "platform")
    assert caught.value.reason_code == "docker_content_incomplete"
    assert caught.value.content["present"] == 8
    manifest.assert_not_called()


@pytest.mark.parametrize("endpoint", ["ssh://remote", "tcp://example:2376", "unix:///home/user/.docker/desktop/docker.sock", "unix:///run/user/1000/docker.sock"])
def test_unsupported_context_does_not_probe_host_containerd(store, endpoint):
    store.endpoint = endpoint
    assert content.check_content(IMAGE, IMAGE_ID).status == "unverified"
    assert not any(command[0] == "/usr/bin/ctr" for command in store.commands)


@pytest.mark.parametrize("fault", ["permission", "wrong_digest", "bad_count", "rootless"])
def test_unavailable_checker_never_reports_complete(store, fault):
    if fault == "permission":
        store.ctr_code = 1
    elif fault == "wrong_digest":
        store.target = "sha256:" + "c" * 64
    elif fault == "bad_count":
        store.present = 8
    else:
        store.security = ["name=rootless"]
    assert content.check_content(IMAGE, IMAGE_ID).status == "unverified"


def test_no_ctr_is_not_a_success(store, monkeypatch):
    monkeypatch.setattr(content.shutil, "which", lambda _: None)
    assert content.check_content(IMAGE).status == "unverified"
    assert store.commands == []


@pytest.mark.parametrize("repaired", [False, True])
def test_repair_reuses_digest_and_rechecks_even_when_pull_says_up_to_date(store, repaired):
    store.status, store.present, store.repaired = "incomplete", 8, repaired
    broken = content.check_content(IMAGE, IMAGE_ID).to_dict()
    if repaired:
        content.repair_content(broken)
    else:
        with pytest.raises(DockerPreparationError) as caught:
            content.repair_content(broken)
        assert caught.value.content["status"] == "incomplete"
    pulls = [cmd for cmd in store.commands if cmd[:2] == ["docker", "pull"]]
    assert pulls == [["docker", "pull", "--platform", "linux/amd64", REFERENCE]]
    assert not any(word in cmd for cmd in store.commands for word in ("prune", "rm", "ingest", "sudo"))


@pytest.mark.parametrize(("detail", "code"), [("unauthorized: authentication required", "docker_auth_failed"),
                                             ("TLS handshake timeout", "docker_network_failed")])
def test_registry_error_is_not_reported_as_missing_local_content(store, detail, code):
    store.pull_code, store.pull_error = 1, detail
    with pytest.raises(DockerPreparationError) as caught:
        operations.pull_image(REFERENCE, component="platform")
    assert caught.value.reason_code == code
    assert store.status == "complete"


def test_repair_then_retry_keeps_completed_stages(store, monkeypatch):
    store.status, store.present, store.repaired = "incomplete", 8, True
    workflow = PreparationWorkflow(interactive=True)
    monkeypatch.setattr("sys.stdin", io.StringIO('{"id":1,"action":"repair"}\n'))
    completed = Mock(return_value="retained")
    assert workflow.run("resolution", completed) == "retained"
    result = workflow.run("environment", content.require_content, IMAGE, IMAGE_ID)
    assert result.status == "complete"
    completed.assert_called_once()
    assert not workflow.rebuild_environment


def test_unoffered_repair_action_is_rejected(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO('{"id":1,"action":"repair"}\n'))
    with pytest.raises(ValueError, match="unexpected"):
        PreparationWorkflow(interactive=True).ask("environment", "error")


def test_known_missing_content_cannot_be_retried_with_unavailable_checker(store, monkeypatch):
    store.status, store.present = "incomplete", 8
    operation = Mock(side_effect=lambda: content.require_content(IMAGE, IMAGE_ID))
    replies = iter(["retry", "cancel"])

    def answer(_limit):
        store.ctr_code = 1
        return json.dumps({"id": workflow.request_id, "action": next(replies)}) + "\n"

    workflow = PreparationWorkflow(interactive=True)
    monkeypatch.setattr("sys.stdin", SimpleNamespace(readline=answer))
    with pytest.raises(KeyboardInterrupt):
        workflow.run("image", operation)
    operation.assert_called_once()


@pytest.mark.parametrize("policy", ["ACPROF_REQUIRE_PREPARED_IMAGE", "ACPROF_MAX_DOWNLOAD"])
def test_repair_does_not_bypass_download_policy(store, monkeypatch, policy):
    store.status, store.present = "incomplete", 8
    monkeypatch.setenv(policy, "1")
    broken = content.check_content(IMAGE, IMAGE_ID).to_dict()
    assert "repair" not in DockerPreparationError("docker_content_incomplete", "missing", content=broken).actions
    with pytest.raises(DockerPreparationError) as caught:
        content.repair_content(broken)
    assert caught.value.reason_code == "docker_repair_forbidden"
    assert not any(cmd[:2] == ["docker", "pull"] for cmd in store.commands)


def test_split_buildkit_log_secrets_are_redacted_before_display(capsys):
    output = DockerOutput("environment", IMAGE, "building")
    for part in (b"TOKEN=split", b"-secret\ninstall complete\n"):
        output.feed(json.dumps({"logs": [{"stream": 1, "data": base64.b64encode(part).decode()}]}))
    output.finish()
    text = capsys.readouterr().out
    assert "split" not in text
    assert "secret" not in text
    assert "[REDACTED]" in text
    assert "install complete" in text


def test_preparation_and_measurement_clocks_are_separate(monkeypatch):
    now = [10.0]
    monkeypatch.setattr("acprof.tui.run.progress.time.monotonic", lambda: now[0])
    tracker = RunProgressTracker()
    tracker.feed(encode_event("environment", "running"))
    now[0] = 110.0
    tracker.feed('ACPROF_EVENT {"version":1,"event":"case_started","case_id":"a"}')
    now[0] = 120.0
    tracker.feed('ACPROF_EVENT {"version":1,"event":"measurement_started","case_id":"a"}')
    now[0] = 123.0
    snapshot = tracker.feed('ACPROF_EVENT {"version":1,"event":"measurement_stopped","case_id":"a"}')
    assert snapshot.preparation_elapsed_s == 100.0
    assert snapshot.measurement_elapsed_s == 3.0
    assert snapshot.measurement_started_at == 0.0


def test_buildkit_and_engine_json_keep_real_byte_counts_and_error(monkeypatch, capsys):
    monkeypatch.setenv("ACPROF_TUI", "1")
    output = DockerOutput("platform", IMAGE, "pulling")
    output.feed(json.dumps({"id": "layer-a", "status": "Downloading", "progressDetail": {"current": 40, "total": 100}}))
    events = [event for line in capsys.readouterr().out.splitlines() if (event := parse_event(line))]
    assert events[0]["image_progress"]["current"] == 40
    assert events[0]["image_progress"]["total"] == 100
    output.feed(json.dumps({"vertexes": [{"name": "RUN install dependencies", "error": "content digest sha256:abc not found"}]}))
    assert output.step == "RUN install dependencies"
    assert "not found" in output.detail
    assert "40" not in output.detail


def test_progress_never_invents_totals_and_is_rejected_inside_measurement():
    progress = ImageProgress("platform", IMAGE, "pulling")
    line = encode_event("environment", "running", image_progress=progress.to_dict())
    assert parse_event(line)["image_progress"]["total"] is None
    tracker = RunProgressTracker()
    tracker.feed('ACPROF_EVENT {"version":1,"event":"case_started","case_id":"a"}')
    tracker.feed('ACPROF_EVENT {"version":1,"event":"measurement_started","case_id":"a"}')
    with pytest.raises(ValueError, match="inside measurement"):
        tracker.feed(line)


def test_streamed_command_retains_output_without_consuming_controller_stdin():
    lines = []
    result = run_command([sys.executable, "-c", "import sys; print('diagnostic'); print('error', file=sys.stderr); assert sys.stdin.read() == ''; sys.exit(2)"],
                         capture_output=False, output_line=lines.append)
    assert result.returncode == 2
    assert sorted(line.strip() for line in lines) == ["diagnostic", "error"]


def test_engine_error_in_successful_http_stream_still_fails(monkeypatch):
    monkeypatch.setattr(operations, "_anonymous_socket", lambda: "/fake.sock")
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b'{"errorDetail":{"message":"unauthorized: denied"}}\n'))
    monkeypatch.setattr(operations.httpx, "HTTPTransport", lambda **_kwargs: transport)
    with pytest.raises(DockerPreparationError) as caught:
        operations.pull_image(REFERENCE, component="platform")
    assert caught.value.reason_code == "docker_auth_failed"


def test_stream_callback_failure_reaps_child(tmp_path):
    import os
    pid_file = tmp_path / "pid"

    def fail(_line):
        raise RuntimeError("consumer stopped")

    with pytest.raises(RuntimeError, match="consumer stopped"):
        run_command([sys.executable, "-c",
                     "import os,sys,time; open(sys.argv[1], 'w').write(str(os.getpid())); print('ready', flush=True); time.sleep(60)",
                     str(pid_file)], capture_output=False, output_line=fail)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_docker_failure_is_persisted_with_original_output_and_log_path(tmp_path, monkeypatch):
    from acprof.host.preparation.preparation_evidence import PreparationEvidence
    from acprof.results.layout import ArtifactLayout

    def fail(command, **kwargs):
        kwargs["output_line"]("failed to read blob sha256:abc: not found\n")
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(operations.host_command, "run_command", fail)
    workflow = PreparationWorkflow()
    evidence = PreparationEvidence(ArtifactLayout.discover(tmp_path), "docker-test")
    def build():
        return operations.run_docker(["docker", "build"], component="environment", image=IMAGE,
                                     operation="build", step="runtime.Dockerfile")
    with pytest.raises(DockerPreparationError) as caught, evidence.capture(workflow):
        workflow.run("image", build)
    assert caught.value.log_path == str(evidence.log_path)
    assert "blob sha256:abc: not found" in evidence.log_path.read_text()
    saved = json.loads(evidence.summary_path.read_text())
    assert saved["reason_code"] == "docker_content_incomplete"
    assert saved["last_failure"]["docker_error"]["exit_code"] == 1


@pytest.mark.parametrize(("message", "code"), [
    ("manifest unknown", "docker_manifest_missing"),
    ("no matching manifest for linux/amd64 in the manifest list entries", "docker_platform_mismatch"),
    ("context deadline exceeded", "docker_network_failed"),
    ("denied: token request timeout", "docker_auth_failed"),
])
@pytest.mark.parametrize("status", [200, 500])
def test_engine_pull_errors_keep_diagnostics_without_premature_failure(monkeypatch, capsys, message, code, status):
    monkeypatch.setenv("ACPROF_TUI", "1")
    monkeypatch.setattr(operations, "_anonymous_socket", lambda: "/fake/docker.sock")
    def respond(request):
        assert request.url.params["platform"] == "linux/amd64"
        body = {"errorDetail": {"message": message}, "error": message} if status == 200 else {"message": message}
        return httpx.Response(status, content=(json.dumps(body) + "\n").encode())
    monkeypatch.setattr(operations.httpx, "HTTPTransport", lambda **kwargs: httpx.MockTransport(respond))
    with pytest.raises(DockerPreparationError) as caught:
        operations.pull_image(REFERENCE, component="platform")
    assert caught.value.reason_code == code
    assert message in caught.value.detail
    output = capsys.readouterr().out
    assert message in output
    events = [event for line in output.splitlines() if (event := parse_event(line))]
    assert all(event["image_progress"]["phase"] != "failed" for event in events)
    assert all(message not in event["image_progress"]["detail"] for event in events)
