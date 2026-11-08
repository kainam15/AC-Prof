"""OCI routing preserves locked content and never invents an automatic mirror."""
from unittest.mock import patch

import pytest

from acprof.host.images.environment import dockerfile_frontend, platform_fingerprint
from acprof.network_policy import DownloadPolicyError, configured_oci_sources, docker_hub_image
from acprof.runtime.profiles import PLATFORMS
from tests.support.runtime_fixture import copy_dependency_tree

DIGEST = "sha256:" + "a" * 64


@pytest.mark.parametrize("reference", ["python:3.12@" + DIGEST, "docker/dockerfile:1@" + DIGEST,
                                       "docker.io/library/python:3.12@" + DIGEST])
def test_official_selection_does_not_change_reference(reference):
    assert docker_hub_image(reference, {}) == reference
    assert configured_oci_sources({}) == {
        "docker_hub_mirror": "", "runtime_registry": "ghcr.io/kainam15/ac-prof/runtime", "source_mode": "auto",
    }


@pytest.mark.parametrize("reference", ["python:3.12", "python:3.12@sha256:abc", "python:3.12@sha512:" + "a" * 128])
def test_mirror_requires_locked_sha256(reference):
    with pytest.raises(DownloadPolicyError, match="locked SHA256"):
        docker_hub_image(reference, {"ACPROF_DOCKER_HUB_MIRROR": "mirror.example/docker.io"})


@pytest.mark.parametrize("value", ["https://user:secret@host.example", "user:secret@host.example",
                                   "mirror.example/../path", "mirror.example/path?token=secret",
                                   "mirror.example:70000", "mirror.example/path:latest", "mirror.example/a//b"])
def test_unsafe_mirror_is_rejected_without_echoing_credentials(value):
    with pytest.raises(DownloadPolicyError, match="ACPROF_DOCKER_HUB_MIRROR") as caught:
        configured_oci_sources({"ACPROF_DOCKER_HUB_MIRROR": value})
    assert "secret" not in str(caught.value)


def test_docker_hub_mirror_does_not_rewrite_other_registries():
    reference = "ghcr.io/example/image@" + DIGEST
    assert docker_hub_image(reference, {"ACPROF_DOCKER_HUB_MIRROR": "mirror.example"}) == reference


def test_frontend_digest_changes_build_identity_but_transport_does_not(tmp_path):
    copy_dependency_tree(tmp_path)
    with patch.dict("os.environ", {}, clear=True):
        before = platform_fingerprint(PLATFORMS["cpu"], tmp_path)
        frontend = dockerfile_frontend(tmp_path)
        assert frontend.startswith("docker/dockerfile:1@sha256:")
        with patch.dict("os.environ", {"ACPROF_DOCKER_HUB_MIRROR": "mirror.example/docker.io"}):
            assert platform_fingerprint(PLATFORMS["cpu"], tmp_path) == before
            assert dockerfile_frontend(tmp_path) == "mirror.example/docker.io/" + frontend
        recipe = tmp_path / "dockerfiles/platform.Dockerfile"
        recipe.write_text(recipe.read_text().replace(frontend, "docker/dockerfile:1@" + DIGEST))
        assert platform_fingerprint(PLATFORMS["cpu"], tmp_path) != before


def test_unpinned_frontend_fails_before_network(tmp_path):
    (tmp_path / "dockerfiles").mkdir()
    (tmp_path / "dockerfiles/platform.Dockerfile").write_text("# syntax=docker/dockerfile:1\nFROM scratch\n")
    with pytest.raises(ValueError, match="pinned by SHA256"):
        dockerfile_frontend(tmp_path)
