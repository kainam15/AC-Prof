"""GHCR publishing must skip verified content without downloading image layers."""
from __future__ import annotations

import hashlib
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.host.images.environment import registry_reference
from acprof.runtime.profiles import ENVIRONMENTS, PLATFORMS
from scripts import publish_runtime_images as publisher


def _labels(expected: dict[str, str], kind: str) -> dict[str, str]:
    result = {"org.acprof.image-kind": kind,
              "org.acprof.platform": expected["platform_id"],
              "org.acprof.platform-build-fingerprint": expected["platform_build_fingerprint"]}
    if kind == "environment":
        result.update({"org.acprof.environment": expected["environment_id"],
                       "org.acprof.environment-build-fingerprint": expected["environment_build_fingerprint"]})
    return result


def _remote_payload(expected: dict[str, str], kind: str, *,
                    image_id: str = "sha256:" + "1" * 64,
                    diff_ids: tuple[str, ...] = ("sha256:" + "2" * 64,),
                    labels: dict[str, str] | None = None) -> tuple[str, str]:
    manifest = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "config": {"mediaType": "application/vnd.docker.container.image.v1+json", "digest": image_id},
        "layers": [{"digest": "sha256:" + "3" * 64} for _ in diff_ids],
    }
    raw = json.dumps(manifest, separators=(",", ":"))
    metadata = {
        "manifest": {
            "mediaType": manifest["mediaType"],
            "digest": "sha256:" + hashlib.sha256(raw.encode()).hexdigest(),
        },
        "image": {
            "os": "linux", "architecture": "amd64",
            "config": {"Labels": labels if labels is not None else _labels(expected, kind)},
            "rootfs": {"type": "layers", "diff_ids": list(diff_ids)},
        },
    }
    return json.dumps(metadata), raw


def _simulate_registry(monkeypatch, expected: dict[str, str], kind: str, **options):
    metadata, raw = _remote_payload(expected, kind, **options)
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert command[:4] == ["docker", "buildx", "imagetools", "inspect"]
        if command[-1] == "--raw":
            return subprocess.CompletedProcess(command, 0, raw, "")
        assert command[-2:] == ["--format", "{{json .}}"]
        return subprocess.CompletedProcess(command, 0, metadata, "")

    monkeypatch.setattr(publisher.subprocess, "run", fake_run)
    return calls


def test_existing_remote_platform_checks_oci_metadata_without_pulling_layers(monkeypatch):
    expected = publisher._platform_labels(PLATFORMS["python-cpu"])
    calls = _simulate_registry(monkeypatch, expected, "platform")
    reference = registry_reference("platform", expected["platform_build_fingerprint"])
    actual = publisher.inspect_remote_image(reference, expected, "platform")
    assert actual == publisher.RemoteImage(
        "sha256:" + "1" * 64,
        "sha256:" + hashlib.sha256(_remote_payload(expected, "platform")[1].encode()).hexdigest(),
        ("sha256:" + "2" * 64,),
    )
    assert len(calls) == 2
    assert all(command[:4] == ["docker", "buildx", "imagetools", "inspect"] for command in calls)


def test_remote_environment_parent_chain_is_verified(monkeypatch):
    expected = publisher._environment_labels(
        ENVIRONMENTS["onnxruntime-cpu"],
        publisher.RemoteImage("sha256:" + "1" * 64, "sha256:" + "4" * 64, ("sha256:" + "2" * 64,)),
    )
    ref = registry_reference("environment", expected["environment_build_fingerprint"])
    _simulate_registry(monkeypatch, expected, "environment",
                       diff_ids=("sha256:" + "2" * 64, "sha256:" + "5" * 64))
    assert publisher.inspect_remote_image(
        ref, expected, "environment",
        parent=publisher.RemoteImage("sha256:" + "1" * 64, "sha256:" + "4" * 64,
                                     ("sha256:" + "2" * 64,)),
    ) is not None
    _simulate_registry(monkeypatch, expected, "environment",
                       diff_ids=("sha256:" + "9" * 64,))
    with pytest.raises(RuntimeError, match="inherit"):
        publisher.inspect_remote_image(
            ref, expected, "environment",
            parent=publisher.RemoteImage("sha256:" + "1" * 64, "sha256:" + "4" * 64,
                                         ("sha256:" + "2" * 64,)),
        )


@pytest.mark.parametrize(("error", "missing"), [
    ("ERROR: ghcr.io/acprof/runtime:whatever: not found", True),
    ("manifest unknown", True),
    ("unexpected status: 404 Not Found", True),
    ("403 forbidden", False),
    ("failed to authorize: unauthorized: not found", False),
    ("connection timed out", False),
    ("docker: 'buildx' is not a docker command.", False),
])
def test_remote_lookup_only_builds_on_confirmed_missing_tag(monkeypatch, error, missing):
    command = Mock(return_value=subprocess.CompletedProcess([], 1, "", error))
    monkeypatch.setattr(publisher.subprocess, "run", command)
    expected = publisher._platform_labels(PLATFORMS["python-cpu"])
    if missing:
        assert publisher.inspect_remote_image("ghcr.io/acprof/runtime:whatever", expected, "platform") is None
    else:
        with pytest.raises(RuntimeError, match="GHCR lookup failed"):
            publisher.inspect_remote_image("ghcr.io/acprof/runtime:whatever", expected, "platform")
    command.assert_called_once()


@pytest.mark.parametrize(("mutate", "reason"), [
    (lambda meta, raw: (meta["image"]["config"]["Labels"].update({"org.acprof.platform": "evil"})), "标签"),
    (lambda meta, raw: meta["image"].update({"architecture": "arm64"}), "linux/amd64"),
    (lambda meta, raw: raw.update({"config": {"mediaType": "application/vnd.docker.container.image.v1+json",
                                            "digest": "invalid"}}), "changed"),
])
def test_remote_metadata_errors_never_fall_back_to_build(monkeypatch, mutate, reason):
    expected = publisher._platform_labels(PLATFORMS["python-cpu"])
    meta_text, raw_text = _remote_payload(expected, "platform")
    meta, raw = json.loads(meta_text), json.loads(raw_text)
    mutate(meta, raw)

    def fake_run(command, **kwargs):
        value = json.dumps(raw) if command[-1] == "--raw" else json.dumps(meta)
        return subprocess.CompletedProcess(command, 0, value, "")

    monkeypatch.setattr(publisher.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match=reason):
        publisher.inspect_remote_image("ghcr.io/example:tag", expected, "platform")


def test_remote_digest_mismatch_fails_closed(monkeypatch):
    expected = publisher._platform_labels(PLATFORMS["cpu"])
    meta_text, raw_text = _remote_payload(expected, "platform")
    meta = json.loads(meta_text)
    meta["manifest"]["digest"] = "sha256:" + "a" * 64

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, raw_text if command[-1] == "--raw"
                                           else json.dumps(meta), "")

    monkeypatch.setattr(publisher.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="manifest changed"):
        publisher.inspect_remote_image("ghcr.io/example:tag", expected, "platform")


def test_platform_present_skips_build_and_push(monkeypatch, tmp_path):
    remote = publisher.RemoteImage("sha256:" + "a" * 64, "sha256:" + "b" * 64,
                                   ("sha256:" + "c" * 64,))
    monkeypatch.setattr(publisher, "inspect_remote_image", Mock(return_value=remote))
    build = Mock(side_effect=AssertionError("existing images must not be built"))
    monkeypatch.setattr(publisher, "prepare_platform_image", build)
    cli = Mock(side_effect=AssertionError("existing images must not be pushed"))
    monkeypatch.setattr(publisher.subprocess, "run", cli)
    report = tmp_path / "platform.json"
    assert publisher.main(["--platform", "cpu", "--push", "--report", str(report)]) == 0
    result = json.loads(report.read_text())
    assert result["pushed"] is False
    assert result["reused"] is True
    assert result["image_id"] == remote.image_id
    assert result["manifest_digest"] == remote.manifest_digest
    build.assert_not_called()
    cli.assert_not_called()


def test_environment_present_skips_platform_pull_and_build(monkeypatch, tmp_path):
    parent = publisher.RemoteImage("sha256:" + "a" * 64, "sha256:" + "b" * 64,
                                   ("sha256:" + "c" * 64,))
    env_remote = publisher.RemoteImage("sha256:" + "d" * 64, "sha256:" + "e" * 64,
                                       ("sha256:" + "c" * 64, "sha256:" + "f" * 64))
    inspection = Mock(side_effect=[parent, env_remote])
    monkeypatch.setattr(publisher, "inspect_remote_image", inspection)
    build = Mock(side_effect=AssertionError("no build/pull on existing GHCR environment"))
    monkeypatch.setattr(publisher, "prepare_environment_image", build)
    monkeypatch.setattr(publisher, "prepare_platform_image", build)
    cli = Mock(side_effect=AssertionError("no docker push on existing GHCR environment"))
    monkeypatch.setattr(publisher.subprocess, "run", cli)
    report = tmp_path / "environment.json"
    assert publisher.main(["--environment", "onnxruntime-cpu", "--pull-platform",
                           "--push", "--report", str(report)]) == 0
    expected = publisher._environment_labels(ENVIRONMENTS["onnxruntime-cpu"], parent)
    assert inspection.call_args_list[1].args[:3] == (
        registry_reference("environment", expected["environment_build_fingerprint"]),
        expected, "environment",
    )
    assert json.loads(report.read_text())["reused"] is True
    build.assert_not_called()
    cli.assert_not_called()


def test_missing_environment_builds_once_with_verified_parent_and_pushes(monkeypatch, tmp_path):
    parent = publisher.RemoteImage("sha256:" + "a" * 64, "sha256:" + "b" * 64,
                                   ("sha256:" + "c" * 64,))
    expected = publisher._environment_labels(ENVIRONMENTS["onnxruntime-cpu"], parent)
    pushed = publisher.RemoteImage("sha256:" + "d" * 64, "sha256:" + "e" * 64,
                                   ("sha256:" + "c" * 64, "sha256:" + "f" * 64))
    monkeypatch.setattr(publisher, "inspect_remote_image",
                        Mock(side_effect=[parent, None, None, pushed]))
    platform_pull = Mock(return_value=SimpleNamespace(image_id=parent.image_id))
    build = Mock(return_value=SimpleNamespace(
        image_id=pushed.image_id, platform_image_id=parent.image_id,
        manifest={"environment_build_fingerprint": expected["environment_build_fingerprint"]},
    ))
    monkeypatch.setattr(publisher, "prepare_platform_image", platform_pull)
    monkeypatch.setattr(publisher, "prepare_environment_image", build)
    cli = Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(publisher.subprocess, "run", cli)
    report = tmp_path / "environment.json"
    assert publisher.main(["--environment", "onnxruntime-cpu", "--pull-platform",
                           "--push", "--report", str(report)]) == 0
    assert platform_pull.call_args.kwargs["image_source"] == "pull"
    assert build.call_args.kwargs["image_source"] == "build"
    assert platform_pull.call_count == build.call_count == 1
    assert [call.args[0][:2] for call in cli.call_args_list] == [
        ["docker", "tag"], ["docker", "push"],
    ]
    assert json.loads(report.read_text())["pushed"] is True


def test_missing_platform_builds_once_with_no_ghcr_pull(monkeypatch):
    expected = publisher._platform_labels(PLATFORMS["cpu"])
    parent = publisher.RemoteImage("sha256:" + "a" * 64, "sha256:" + "b" * 64,
                                   ("sha256:" + "c" * 64,))
    monkeypatch.setattr(publisher, "inspect_remote_image",
                        Mock(side_effect=[None, None, parent]))
    build = Mock(return_value=SimpleNamespace(
        image_id=parent.image_id,
        expected={"platform_build_fingerprint": expected["platform_build_fingerprint"]},
    ))
    monkeypatch.setattr(publisher, "prepare_platform_image", build)
    cli = Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(publisher.subprocess, "run", cli)
    assert publisher.main(["--platform", "cpu", "--push"]) == 0
    assert build.call_args.kwargs["image_source"] == "build"
    assert [call.args[0][:2] for call in cli.call_args_list] == [
        ["docker", "tag"], ["docker", "push"],
    ]


def test_platform_change_during_environment_pull_blocks_publication(monkeypatch):
    parent = publisher.RemoteImage("sha256:" + "a" * 64, "sha256:" + "b" * 64,
                                   ("sha256:" + "c" * 64,))
    monkeypatch.setattr(publisher, "inspect_remote_image", Mock(side_effect=[parent, None]))
    monkeypatch.setattr(publisher, "prepare_platform_image",
                        Mock(return_value=SimpleNamespace(image_id="sha256:" + "9" * 64)))
    build = Mock(side_effect=AssertionError("unexpected build"))
    monkeypatch.setattr(publisher, "prepare_environment_image", build)
    with pytest.raises(RuntimeError, match="platform changed"):
        publisher.main(["--environment", "onnxruntime-cpu", "--pull-platform", "--push"])
    build.assert_not_called()


def test_missing_parent_and_raced_tag_never_build_or_overwrite(monkeypatch):
    parent = publisher.RemoteImage("sha256:" + "a" * 64, "sha256:" + "b" * 64,
                                   ("sha256:" + "c" * 64,))
    expected = publisher._platform_labels(PLATFORMS["cpu"])
    cli = Mock(side_effect=AssertionError("must not overwrite existing fingerprint tag"))
    monkeypatch.setattr(publisher.subprocess, "run", cli)
    monkeypatch.setattr(publisher, "inspect_remote_image", Mock(return_value=None))
    with pytest.raises(RuntimeError, match="platform image is missing"):
        publisher.main(["--environment", "onnxruntime-cpu", "--pull-platform", "--push"])
    # The built image ID differs from a tag published after the first lookup.
    monkeypatch.setattr(publisher, "inspect_remote_image",
                        Mock(side_effect=[None, publisher.RemoteImage(
                            "sha256:" + "9" * 64, parent.manifest_digest, parent.diff_ids)]))
    monkeypatch.setattr(publisher, "prepare_platform_image", Mock(return_value=SimpleNamespace(
        image_id=parent.image_id,
        expected={"platform_build_fingerprint": expected["platform_build_fingerprint"]},
    )))
    with pytest.raises(RuntimeError, match="different image"):
        publisher.main(["--platform", "cpu", "--push"])
    cli.assert_not_called()


def test_publish_environment_requires_published_platform_option(monkeypatch):
    cli = Mock(side_effect=AssertionError("no commands expected"))
    monkeypatch.setattr(publisher.subprocess, "run", cli)
    with pytest.raises(SystemExit, match="2"):
        publisher.main(["--environment", "onnxruntime-cpu", "--push"])
    cli.assert_not_called()


def test_matrix_does_not_probe_registry_or_run_docker(monkeypatch, capsys):
    cli = Mock(side_effect=AssertionError("matrix must not use docker"))
    monkeypatch.setattr(publisher.subprocess, "run", cli)
    assert publisher.main(["--matrix"]) == 0
    assert len(json.loads(capsys.readouterr().out)["platforms"]) == 4
    cli.assert_not_called()
