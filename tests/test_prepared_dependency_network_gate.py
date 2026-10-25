"""Fail-closed offline hardware collection must not prepare dependency images."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.host.images import environment as dependency_images


def test_required_prepared_platform_does_not_pull_or_build(tmp_path: Path) -> None:
    platform = SimpleNamespace(platform_id="python-cpu", python_version="3.10",
                               python_base_image="python:3.10-slim", architecture="amd64")
    with (patch.dict("os.environ", {"ACPROF_REQUIRE_PREPARED_IMAGE": "1"}),
          patch.object(dependency_images, "_platform_fingerprint", return_value="a" * 64),
          patch.object(dependency_images, "checked_image", return_value=None),
          patch.object(dependency_images, "_pull_cached_image") as pull,
          patch.object(dependency_images, "build_dependency") as build):
        with pytest.raises(RuntimeError, match="禁止隐式构建或下载.*预备平台镜像不存在"):
            dependency_images._prepare_platform(platform, tmp_path, {}, "auto", None)
        pull.assert_not_called()
        build.assert_not_called()


def test_required_prepared_environment_does_not_pull_or_build(tmp_path: Path) -> None:
    environment = SimpleNamespace(platform=object())
    platform = dependency_images.PreparedPlatform("sha256:" + "a" * 64,
                                                  "acprof-platform-test",
                                                  {"platform_build_fingerprint": "a" * 64,
                                                   "platform_id": "python-cpu"})
    with (patch.dict("os.environ", {"ACPROF_REQUIRE_PREPARED_IMAGE": "1"}),
          patch.object(dependency_images, "environment_identity", return_value={"platform": {}, "packages": []}),
          patch.object(dependency_images, "_prepare_platform", return_value=platform),
          patch.object(dependency_images, "_runtime_fingerprint", return_value="b" * 64),
          patch.object(dependency_images, "runtime_fingerprint", return_value="b" * 64),
          patch.object(dependency_images, "checked_image", return_value=None),
          patch.object(dependency_images, "_pull_cached_image") as pull,
          patch.object(dependency_images, "build_dependency") as build):
        with pytest.raises(RuntimeError, match="禁止隐式构建或下载.*预备依赖环境镜像不存在"):
            dependency_images.prepare_environment_image(environment, tmp_path)
        pull.assert_not_called()
        build.assert_not_called()
