"""Image names are display hints; raw references and image IDs remain authoritative."""
from dataclasses import replace

import pytest

from acprof.host.images.inventory import DockerConnection, ImageInventory, ImageLayer, ManagedImage
from acprof.runtime.profiles import PYTHON_BASE_IMAGE
from acprof.tui.i18n import translate
from acprof.tui.images.presentation import (
    filtered_images,
    image_diagnostics,
    image_display_name,
    image_metadata,
    layer_image_detail,
)

IMAGE_ID = "sha256:a33c10a97f11005523814c6eee013986119fe41f3122f444518117536675c8e1"
PINNED_DIGEST = PYTHON_BASE_IMAGE.split("@", 1)[1]


def upstream(*, repository="python", digest=PINNED_DIGEST, tags=()):
    return ManagedImage(IMAGE_ID, tags, 119_000_000, "", "other" if tags else "untagged",
                        repo_digests=(f"{repository}@{digest}",))


@pytest.mark.parametrize("repository", [
    "python", "library/python", "docker.io/python", "docker.io/library/python",
    "index.docker.io/library/python", "registry-1.docker.io/library/python",
])
def test_python_digest_aliases_keep_full_patch_version(repository):
    item = upstream(repository=repository)
    assert image_display_name(item) == "Python 3.12.15 Slim（上游基础）"
    assert translate(image_display_name(item), "en") == "Python 3.12.15 Slim (upstream base)"
    inventory = ImageInventory(DockerConnection((), "test"), "daemon", (item,))
    diagnostics = translate(image_diagnostics(item, inventory), "en")
    assert "Name source:" in diagnostics
    assert item.repo_digests[0] in diagnostics
    assert PYTHON_BASE_IMAGE in diagnostics
    assert IMAGE_ID in diagnostics
    assert filtered_images(inventory, "Python 3.12.15", "all") == (item,)
    assert filtered_images(inventory, item.repo_digests[0], "all") == (item,)


def test_configured_mirror_is_an_explicit_alias(monkeypatch):
    item = upstream(repository="mirror.example:5443/docker.io/library/python")
    monkeypatch.delenv("ACPROF_DOCKER_HUB_MIRROR", raising=False)
    assert image_display_name(item) == "mirror.example:5443/docker.io/library/python（无标签）"
    monkeypatch.setenv("ACPROF_DOCKER_HUB_MIRROR", "mirror.example:5443/docker.io/")
    assert image_display_name(item) == "Python 3.12.15 Slim（上游基础）"
    assert image_display_name(upstream(repository="mirror.example:5443/docker.io.evil/library/python")) == (
        "mirror.example:5443/docker.io.evil/library/python（无标签）")


@pytest.mark.parametrize(("repository", "digest", "expected"), [
    ("python", "sha256:" + "0" * 64, "python（无标签）"),
    ("another.example/python", PINNED_DIGEST, "another.example/python（无标签）"),
    ("someone/python", PINNED_DIGEST, "someone/python（无标签）"),
    ("docker.io/library/ubuntu", "sha256:" + "1" * 64, "ubuntu（无标签）"),
    ("docker.io/docker/dockerfile", "sha256:" + "2" * 64, "docker/dockerfile（无标签）"),
    ("localhost:5000/library/python", PINNED_DIGEST, "localhost:5000/library/python（无标签）"),
])
def test_untagged_repository_names_do_not_guess_content(repository, digest, expected):
    item = upstream(repository=repository, digest=digest)
    assert image_display_name(item) == expected
    inventory = ImageInventory(DockerConnection((), "test"), "daemon", (item,))
    assert "version unverified" in translate(image_diagnostics(item, inventory), "en")


@pytest.mark.parametrize(("tag", "expected"), [
    ("python:3.12.15-slim-trixie", "Python 3.12.15 Slim (trixie)"),
    ("docker.io/library/python:3.13.9-slim-bookworm", "Python 3.13.9 Slim (bookworm)"),
    ("index.docker.io/library/python:3.13.9", "Python 3.13.9"),
    ("python:3.14.0-alpine3.22", "Python 3.14.0 alpine3.22"),
    ("docker.io/library/ubuntu:24.04", "ubuntu"),
    ("docker.io/docker/dockerfile:1", "docker/dockerfile"),
    ("example.com:5000/library/python:3.13.9", "example.com:5000/library/python"),
])
def test_tag_names_are_readable_without_changing_original_references(tag, expected):
    item = upstream(tags=(tag,))
    original = replace(item)
    inventory = ImageInventory(DockerConnection((), "test"), "daemon", (item,))
    assert image_display_name(item) == expected
    assert tag in image_metadata(item, inventory)
    diagnostics = translate(image_diagnostics(item, inventory), "en")
    assert "Name source:" in diagnostics
    assert tag in diagnostics
    assert "content unverified" in diagnostics
    assert item == original


def test_missing_references_fall_back_to_id_without_inferred_parent_names():
    item = replace(upstream(), repo_digests=())
    assert image_display_name(item, upstream()) == IMAGE_ID[7:19]
    assert image_display_name(replace(upstream(), acprof=True)) == IMAGE_ID[7:19]


def test_layer_references_translate_names_and_preserve_raw_digest():
    item = upstream()
    inventory = ImageInventory(DockerConnection((), "test"), "daemon", (item,))
    layer = ImageLayer("chain", "diff", 20, (IMAGE_ID,))
    detail = translate(layer_image_detail(layer, inventory), "en")
    assert "Python 3.12.15 Slim (upstream base)" in detail
    assert IMAGE_ID[7:19] in detail
    assert item.repo_digests[0] in detail
