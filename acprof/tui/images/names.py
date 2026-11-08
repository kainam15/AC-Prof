"""Presentation-only image names from existing inventory; never query or retag Docker."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from acprof.host.images.inventory import ManagedImage
from acprof.messages import join_messages
from acprof.runtime.profiles import PYTHON_BASE_IMAGE
from acprof.tui.i18n import message_id

IMAGE_PLATFORMS = {
    "python-cpu": "Python CPU", "cpu": "PyTorch CPU",
    "cu124": "PyTorch CUDA 12.4", "cu128": "PyTorch CUDA 12.8",
}
# Known aliases come from dependency locks; do not guess digits in other names.
IMAGE_RUNTIME_NAMES = {
    "moss-transformers5104": "moss-transformers5.10.4",
    "multimodal-transformers4576": "multimodal-transformers4.57.6",
}


@dataclass(frozen=True)
class _ImageName:
    name: str
    evidence: str


def split_image_reference(reference: str) -> tuple[str, str, str]:
    """Split an inspect-provided reference, preserving registry ports and raw values."""
    named, _, digest = reference.partition("@")
    repository, separator, tag = named.rpartition(":")
    if not separator or "/" in tag:
        return named, "", digest
    return repository, tag, digest


def image_repository_and_tag(item: ManagedImage) -> tuple[str, str]:
    reference = item.name if item.tags else next(iter(item.repo_digests), "")
    repository, tag, _ = split_image_reference(reference)
    return repository, tag


def _familiar_repository(repository: str) -> str:
    # Follow distribution/reference's Hub/library rules, preserving namespaces
    # and private registries. Only the user's exact mirror prefix is an alias.
    mirror = os.environ.get("ACPROF_DOCKER_HUB_MIRROR", "").strip().rstrip("/")
    if mirror and repository.startswith(mirror + "/"):
        repository = "docker.io/" + repository[len(mirror) + 1:]
    domain, separator, path = repository.partition("/")
    if separator and domain in {"docker.io", "index.docker.io", "registry-1.docker.io"}:
        repository = path
    elif separator and ("." in domain or ":" in domain or domain == "localhost" or domain != domain.lower()):
        return repository
    if repository.startswith("library/") and repository.count("/") == 1:
        return repository.removeprefix("library/")
    return repository


def _python_tag_name(tag: str) -> str:
    version, _, variant = tag.partition("-")
    if not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", version):
        return "python:" + tag
    if variant == "slim" or variant.startswith("slim-"):
        distro = variant.removeprefix("slim").removeprefix("-")
        variant = "Slim" + (f" ({distro})" if distro else "")
    return "Python " + version + (" " + variant if variant else "")


def _image_name(item: ManagedImage, parent: ManagedImage | None = None) -> _ImageName:
    platform = IMAGE_PLATFORMS.get(item.platform_id, item.platform_id)
    metadata = message_id('images.name.source_metadata', item.platform_id or "—",
                          item.environment_id or "—", ", ".join(item.profiles) or "—")
    if item.kind == "base" and item.platform_id in IMAGE_PLATFORMS:
        return _ImageName(platform, metadata)
    if item.kind == "runtime" and item.environment_id:
        profiles = (profile.removesuffix("-" + item.platform_id) if item.platform_id else profile
                    for profile in item.profiles)
        label = " / ".join(IMAGE_RUNTIME_NAMES.get(profile, profile) for profile in profiles)
        label = label or "env-" + item.environment_id[:12]
        inherited = (parent is not None and parent.kind in {"base", "runtime"}
                     and parent.platform_id == item.platform_id and item.platform_id in IMAGE_PLATFORMS)
        return _ImageName(label + (" · " + platform if platform and not inherited else ""), metadata)
    if item.model_id:
        return _ImageName(item.model_id, message_id('images.name.source_model', item.model_id))
    if item.tags:
        repository, tag, _ = split_image_reference(item.name)
        repository = _familiar_repository(repository)
        name = _python_tag_name(tag) if repository == "python" and tag else repository
        return _ImageName(name, message_id('images.name.source_tag', item.name))
    if not item.acprof and item.repo_digests:
        pinned_repository, pinned_tag, pinned_digest = split_image_reference(PYTHON_BASE_IMAGE)
        version = re.fullmatch(r"(\d+\.\d+(?:\.\d+)?)-slim(?:-[a-z0-9]+)?", pinned_tag)
        for reference in item.repo_digests:
            repository, _, digest = split_image_reference(reference)
            if (version and pinned_digest and digest == pinned_digest
                    and _familiar_repository(repository) == _familiar_repository(pinned_repository)):
                return _ImageName(message_id('images.name.python_base', version[1]),
                                  message_id('images.name.source_pinned', reference, PYTHON_BASE_IMAGE))
        reference = item.repo_digests[0]
        repository, _, _ = split_image_reference(reference)
        return _ImageName(message_id('images.name.untagged_repository', _familiar_repository(repository)),
                          message_id('images.name.source_repository', reference))
    return _ImageName(item.display_name, message_id('images.name.source_id'))


def image_display_name(item: ManagedImage, parent: ManagedImage | None = None) -> str:
    return _image_name(item, parent).name


def image_name_evidence(item: ManagedImage) -> str:
    return _image_name(item).evidence


def image_row_label(item: ManagedImage, parent: ManagedImage | None = None) -> str:
    name = image_display_name(item, parent)
    short_id = item.image_id[7:19]
    return name if name == short_id else join_messages(" · ", (name, short_id))
