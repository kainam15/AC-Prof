"""PEP 440 candidate selection from PyPI's per-file release metadata."""
from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Literal

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.tags import Tag, sys_tags
from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import InvalidVersion, Version

Channel = Literal["stable", "preview"]


def version_type(version: Version) -> str:
    if version.is_devrelease:
        return "Dev"
    if version.pre:
        return {"a": "Alpha", "b": "Beta", "rc": "RC"}[version.pre[0]]
    return "Stable"


@dataclass(frozen=True)
class ReleaseVersion:
    current: Version
    latest_stable: Version | None
    latest_preview: Version | None
    channel: Channel = "stable"

    @property
    def latest(self) -> Version | None:
        candidates = [self.latest_stable]
        if self.channel == "preview":
            candidates.append(self.latest_preview)
        return max((v for v in candidates if v is not None), default=None)

    @property
    def update_available(self) -> bool:
        return self.latest is not None and self.latest > self.current


def _compatible_wheel(file: dict, version: Version, supported: frozenset[Tag]) -> bool:
    filename = file.get("filename")
    if file.get("packagetype") != "bdist_wheel" or not isinstance(filename, str):
        return False
    try:
        name, wheel_version, _, tags = parse_wheel_filename(filename)
    except InvalidWheelFilename:
        return False
    return name == "acprof" and wheel_version == version and not supported.isdisjoint(tags)


def select_releases(data: object, current: str, channel: Channel = "stable", *,
                    python: str | None = None, wheel_only: bool = False) -> ReleaseVersion:
    if channel not in ("stable", "preview"):
        raise ValueError("Unknown update channel")
    if not isinstance(data, dict) or not isinstance(data.get("releases"), dict):
        raise ValueError("PyPI response is missing releases")
    runtime = Version(python or platform.python_version())
    supported = frozenset(sys_tags()) if wheel_only else None
    stable: list[Version] = []
    preview: list[Version] = []
    for name, files in data["releases"].items():
        try:
            version = Version(name)
        except (InvalidVersion, TypeError):
            continue  # Historical non-PEP-440 records are not candidates.
        if version.is_devrelease or version.local or not isinstance(files, list):
            continue
        for file in files:
            # Fail closed for incomplete or malformed file eligibility metadata.
            if not isinstance(file, dict) or file.get("yanked") is not False:
                continue
            if supported is not None and not _compatible_wheel(file, version, supported):
                continue
            requires = file.get("requires_python")
            if requires is not None and not isinstance(requires, str):
                continue
            try:
                if requires and not SpecifierSet(requires).contains(runtime, prereleases=True):
                    continue
            except InvalidSpecifier:
                continue
            (preview if version.is_prerelease else stable).append(version)
            break
    return ReleaseVersion(Version(current), max(stable, default=None), max(preview, default=None), channel)
