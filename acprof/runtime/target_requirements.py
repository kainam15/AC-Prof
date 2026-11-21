"""Select requirements and compatible wheels for a registered container, not the host."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from packaging.markers import Marker
    from packaging.tags import Tag

    from acprof.runtime.profiles import PlatformSpec


# pip requirements comments start at the beginning of a line or after whitespace.
# An attached #fragment belongs to the URL; its marker can follow the fragment.
_COMMENT = re.compile(r"(?<!\S)#")


def requirement_lines(source: str) -> Iterator[str]:
    for line in source.splitlines():
        if value := _COMMENT.split(line, maxsplit=1)[0].strip():
            yield value


def _require_supported_target(platform: PlatformSpec) -> None:
    if platform.architecture != "linux/amd64" or platform.python_target != "x86_64-manylinux_2_28":
        raise ValueError(f"Unregistered Python target: {platform.architecture}, {platform.python_target}")


def applies_to_target(marker: Marker | None, platform: PlatformSpec) -> bool:
    """Return applicability or raise ValueError if the marker needs an unbound field."""
    if marker is None:
        return True
    _require_supported_target(platform)
    # packaging 25/26's public Marker.evaluate *overlays* the host, even with an
    # explicit environment. Use its evaluator with ONLY the target's bound keys;
    # a container does not pin a kernel release/version. Do not reimplement PEP 508.
    from packaging.markers import _evaluate_markers

    version = platform.python_version
    try:
        return _evaluate_markers(marker._markers, {
            "python_version": ".".join(version.split(".")[:2]), "python_full_version": version,
            "implementation_name": "cpython", "implementation_version": version,
            "platform_python_implementation": "CPython", "os_name": "posix", "sys_platform": "linux",
            "platform_system": "Linux", "platform_machine": "x86_64", "extra": "",
        })
    except KeyError as exc:
        raise ValueError(f"Target does not define marker {exc.args[0]!r}: {marker}") from exc


def target_tags(platform: PlatformSpec) -> list[Tag]:
    from packaging.tags import compatible_tags, cpython_tags

    _require_supported_target(platform)
    version = tuple(int(value) for value in platform.python_version.split(".")[:2])
    interpreter = "cp" + "".join(str(value) for value in version)
    # Use the declared manylinux 2.28 floor, not the host Python/ABI/architecture.
    platforms = [f"manylinux_2_{v}_x86_64" for v in range(28, 4, -1)]
    platforms += ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64", "linux_x86_64"]
    tags = list(cpython_tags(version, abis=[interpreter], platforms=platforms))
    tags += list(compatible_tags(version, interpreter=interpreter, platforms=platforms))
    return tags
