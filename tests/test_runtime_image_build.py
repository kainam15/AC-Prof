import dataclasses
import hashlib
import json
import os
import re
import subprocess
import tempfile
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest
from runtime_fixture import copy_dependency_tree

from acprof.dependency_locks import (
    content_digest,
    package_versions,
    read_python_lock,
    system_lock_identity,
)
from acprof.host import runtime_images
from acprof.host.detect import TaskInfo
from acprof.runtime_profiles import RuntimeProfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {
    "nlp": ("fill-mask", "transformers_pipeline"),
    "audio": ("automatic-speech-recognition", "transformers_pipeline"),
    "cv": ("image-classification", "transformers_pipeline"),
    "diffusion": ("text-to-image", "diffusers"),
    "multimodal": ("image-text-to-text", "transformers_model"),
    "structured": ("tabular-regression", "sklearn"),
    "timeseries": ("time-series-forecasting", "chronos"),
}


class TestRuntimeImageBuild:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.task = TaskInfo(
            model_id="example/model", pipeline_tag="fill-mask", task_family="nlp",
            runtime_backend="transformers_pipeline", library_name="transformers",
            model_revision="0123456789abcdef0123456789abcdef01234567",
            detection_method="test",
        )
        self.commands = []
        self.images = {}
        self.manifests = {}
        self.contexts = {}
        self.service_context_files = set()
        self.failed_dockerfile = None
        for mocked in (
            patch.dict(os.environ, {"ACPROF_RUNTIME_IMAGE_SOURCE": "build"}, clear=True),
            patch("acprof.host.command.run_command", side_effect=self.fake_run),
            patch.object(runtime_images, "inspect_identity", side_effect=self.images.get),
            patch.object(runtime_images, "verified_image", side_effect=self.verified_image),
            patch("acprof.host.model_store.plan_model", return_value={}),
            patch("acprof.host.network_preflight.preflight", return_value={"sources": []}),
            patch("acprof.host.model_store.prepare_model", return_value={"plan_sha256": "d" * 64}),
            patch("acprof.host.runtime_images.select_nlp_torch_index_url",
                         return_value="https://download.pytorch.org/whl/cu124"),
        ):
            mocked.start()
            self._request.addfinalizer(partial(mocked.stop))

    def fake_run(self, command, **kwargs):
        self.commands.append(command)
        if command[:2] == ["docker", "build"]:
            recipe = Path(command[command.index("-f") + 1]).name
            if recipe == self.failed_dockerfile:
                return subprocess.CompletedProcess(command, 1, "", "build failed")
            arguments = dict(command[index + 1].split("=", 1)
                             for index, value in enumerate(command) if value == "--build-arg")
            identifier = "sha256:" + f"{len(self.commands):064x}"
            Path(command[command.index("--iidfile") + 1]).write_text(identifier)
            labels = {"org.acprof.model-files-key": arguments.get("MODEL_FILES_KEY", ""),
                      "org.acprof.platform-build-fingerprint": arguments.get("PLATFORM_BUILD_FINGERPRINT", ""),
                      "org.acprof.environment-build-fingerprint": arguments.get("ENVIRONMENT_BUILD_FINGERPRINT", "")}
            self.images[identifier] = {"image_id": identifier, "labels": labels}
            if recipe == "runtime-final.Dockerfile":
                context = Path(command[-1])
                self.service_context_files = {p.relative_to(context).as_posix()
                                              for p in [*context.iterdir(), *(context / "acprof").rglob("*")]
                                              if p.is_file()}
            if recipe in {"platform.Dockerfile", "runtime.Dockerfile"}:
                context = Path(command[-1])
                expected = json.loads((context / "expectation.json").read_text())
                labels.update({"org.acprof.image-kind": "platform" if recipe == "platform.Dockerfile" else "environment",
                               "org.acprof.platform": expected["platform_id"],
                               "org.acprof.platform-build-fingerprint": expected["platform_build_fingerprint"]})
                if recipe == "runtime.Dockerfile":
                    labels["org.acprof.environment"] = expected["environment_id"]
                lock = read_python_lock(context / "requirements.lock")
                system = json.loads((context / "system.lock").read_text())
                self.manifests[identifier] = {
                    **expected, "schema_version": 1, "packages": package_versions(lock),
                    "system_packages": system["packages"],
                    "system_lock_sha256": content_digest(system_lock_identity(system)),
                    "dependency_lock_sha256": hashlib.sha256((context / "requirements.lock").read_bytes()).hexdigest(),
                }
                self.contexts[recipe] = {p.name: p.read_text() for p in context.iterdir() if p.is_file()}
        elif command[:2] == ["docker", "tag"]:
            self.images[command[3]] = self.images[command[2]]
        elif command[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(command, 0, json.dumps(self.manifests[command[-2]]), "")
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    def verified_image(self, task, name, fingerprint, project_dir=None):
        # 服务清单由 test_image_reuse 覆盖；依赖清单走实际核验函数。
        return runtime_images.ImageInfo(tag=self.images[name]["image_id"], name=name)

    def build_commands(self):
        return [command for command in self.commands if command[:2] == ["docker", "build"]]

    def test_service_build_context_contains_only_runtime_sources_and_licenses(self):
        runtime_images.build_runtime_image(self.task, str(PROJECT_ROOT))
        assert ("acprof/container/server.py") in (self.service_context_files)
        assert ("acprof/extensions/builtin/manifest.json") in (self.service_context_files)
        assert ("LICENSE") in (self.service_context_files)
        assert not (any(name.startswith(("acprof/tui/", "acprof/host/", ".env", "tests/", ".git/"))
                             for name in self.service_context_files))

    def test_budget_rejection_precedes_every_build_and_weight_download(self):
        from acprof.network_policy import DownloadPolicyError
        with patch("acprof.host.network_preflight.preflight", side_effect=DownloadPolicyError("over budget")), patch(
            "acprof.host.model_store.prepare_model",
        ) as download:
            with pytest.raises(DownloadPolicyError, match="over budget"):
                runtime_images.build_runtime_image(self.task, str(PROJECT_ROOT))
            download.assert_not_called()
        assert (self.commands) == ([])

    def test_hub_endpoint_is_not_model_identity_or_runtime_configuration(self):
        runtime_images.build_runtime_image(self.task, str(PROJECT_ROOT))
        self.commands.clear()
        with patch.dict(os.environ, {"HF_DOWNLOAD_MODE": "mirror-preferred", "HF_ENDPOINT": "https://mirror.example",
                                    "HF_FALLBACK_ENDPOINTS": "https://huggingface.co"}):
            runtime_images.build_runtime_image(self.task, str(PROJECT_ROOT))
        builds = self.build_commands()
        assert ([Path(cmd[cmd.index("-f") + 1]).name for cmd in builds]) == ([
            "runtime-final.Dockerfile",
        ])
        assert not any("HF_ENDPOINT=" in item or "HF_FALLBACK_ENDPOINTS=" in item for cmd in builds for item in cmd)

    def test_prebuilt_images_are_verified_and_used_without_dependency_builds(self):
        from acprof.host.dependency_images import prepare_environment_image
        from acprof.runtime_profiles import ENVIRONMENTS
        environment = ENVIRONMENTS["onnxruntime-cpu"]
        prepared = prepare_environment_image(environment)
        # Retain immutable images/manifests, then simulate an empty local tag cache.
        platforms = {key: value for key, value in self.images.items() if key.startswith("acprof-platform-")}
        runtime = self.images[prepared.name]
        self.images = {key: value for key, value in self.images.items() if key.startswith("sha256:")}
        self.commands.clear()

        def registry_run(command, **kwargs):
            if command[:2] == ["docker", "pull"]:
                self.commands.append(command)
                self.images[command[-1]] = next(iter(platforms.values())) if ":platform-" in command[-1] else runtime
                return subprocess.CompletedProcess(command, 0, "", "")
            return self.fake_run(command, **kwargs)

        with patch.dict(os.environ, {"ACPROF_RUNTIME_IMAGE_SOURCE": "pull"}), patch("acprof.host.command.run_command", side_effect=registry_run,
        ), patch.object(runtime_images, "inspect_identity", side_effect=self.images.get):
            reused = prepare_environment_image(environment)
        assert (reused.image_id) == (prepared.image_id)
        assert (self.build_commands()) == ([])
        pulls = [command for command in self.commands if command[:2] == ["docker", "pull"]]
        assert (len(pulls)) == (2)

    def test_pull_only_failure_never_builds_locally(self):
        from acprof.host.dependency_images import prepare_environment_image
        from acprof.runtime_profiles import ENVIRONMENTS
        with patch.dict(os.environ, {"ACPROF_RUNTIME_IMAGE_SOURCE": "pull"}), patch("acprof.host.command.run_command", return_value=subprocess.CompletedProcess([], 1, "", "manifest unknown"),
        ) as command:
            with pytest.raises(RuntimeError, match="拉取"):
                prepare_environment_image(ENVIRONMENTS["onnxruntime-cpu"])
        assert (all(call.args[0][:2] == ["docker", "pull"] for call in command.call_args_list))

    def test_auto_pull_failure_builds_the_locked_dependencies(self):
        from acprof.host.dependency_images import prepare_environment_image
        from acprof.runtime_profiles import ENVIRONMENTS

        def unavailable_registry(command, **kwargs):
            if command[:2] == ["docker", "pull"]:
                return subprocess.CompletedProcess(command, 1, "", "registry unavailable")
            return self.fake_run(command, **kwargs)

        with patch("acprof.host.command.run_command", side_effect=unavailable_registry):
            image = prepare_environment_image(ENVIRONMENTS["onnxruntime-cpu"], image_source="auto")
        assert (image.image_id.startswith("sha256:"))
        assert (len(self.build_commands())) == (2)

    def test_successful_pull_with_wrong_identity_does_not_fall_back_to_build(self):
        from acprof.host.dependency_images import prepare_environment_image
        from acprof.runtime_profiles import ENVIRONMENTS

        def corrupted_registry(command, **kwargs):
            assert (command[:2]) == (["docker", "pull"])
            self.images[command[-1]] = {"image_id": "sha256:" + "f" * 64, "labels": {}}
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch("acprof.host.command.run_command", side_effect=corrupted_registry):
            with pytest.raises(RuntimeError, match="标签内容不匹配"):
                prepare_environment_image(ENVIRONMENTS["onnxruntime-cpu"], image_source="auto")
        assert (self.build_commands()) == ([])

    @pytest.mark.parametrize('family_case', range(7))
    def test_unlocked_profiles_fail_before_any_docker_command(self, family_case):
        family = tuple(FAMILIES)[family_case]
        with pytest.raises(ValueError, match="lock"):
            RuntimeProfile("test-" + family, family)
        assert (self.commands) == ([])

    def test_locked_runtime_passes_fixed_model_revision_to_model_layer(self):
        runtime_images.build_runtime_image(self.task, str(PROJECT_ROOT))
        builds = self.build_commands()
        assert ([Path(cmd[cmd.index("-f") + 1]).name for cmd in builds]) == ([
            "platform.Dockerfile", "runtime.Dockerfile", "runtime-model.Dockerfile", "runtime-final.Dockerfile",
        ])
        assert ("# torch==2.11.0+cu128") in (self.contexts["runtime.Dockerfile"]["requirements.lock"])
        assert ("acprof") not in (self.contexts["runtime.Dockerfile"])
        assert ("MODEL_REVISION=0123456789abcdef0123456789abcdef01234567") in (builds[2])

    def test_equal_locks_share_one_runtime_build_across_task_families(self):
        audio = dataclasses.replace(self.task, task_family="audio",
                                    pipeline_tag="automatic-speech-recognition",
                                    runtime_profile_id="audio-cpu")
        multimodal = dataclasses.replace(self.task, task_family="multimodal",
                                         pipeline_tag="image-text-to-text",
                                         runtime_backend="transformers_model",
                                         runtime_profile_id="multimodal-transformers4576-cpu")
        runtime_images.build_runtime_image(audio, str(PROJECT_ROOT))
        runtime_images.build_runtime_image(multimodal, str(PROJECT_ROOT))
        runtime_builds = [cmd for cmd in self.build_commands()
                          if Path(cmd[cmd.index("-f") + 1]).name == "runtime.Dockerfile"]
        assert (len(runtime_builds)) == (1)

    def test_model_version_dots_survive_weights_and_service_image_names(self):
        self.task.model_id = "Qwen/Qwen2.5-0.5B"
        result = runtime_images.build_runtime_image(self.task, str(PROJECT_ROOT))
        builds = self.build_commands()
        weights_name = next(cmd[3] for cmd in self.commands if cmd[:2] == ["docker", "tag"] and cmd[3].startswith("acprof-model-plan-"))
        service_name = result.name
        assert re.search(r"^acprof-model-plan-nlp-qwen--qwen2\.5-0\.5b:[0-9a-f]{20}$", weights_name)
        assert re.search(r"^acprof-nlp-qwen--qwen2\.5-0\.5b:[0-9a-f]{20}$", service_name)
        assert (result.name) == (service_name)
        assert ("MODEL_ID=Qwen/Qwen2.5-0.5B") in (builds[2])
        assert (self.task.model_id) == ("Qwen/Qwen2.5-0.5B")

    def test_mutable_model_revision_is_rejected_before_docker(self):
        self.task.model_revision = "main"
        with pytest.raises(RuntimeError, match="固定 model revision"):
            runtime_images.build_runtime_image(self.task, str(PROJECT_ROOT))
        assert (self.commands) == ([])

    def test_host_model_store_token_never_enters_docker_build(self):
        with patch.dict(os.environ, {"HF_TOKEN": "test-secret-value"}):
            runtime_images.build_runtime_image(self.task, str(PROJECT_ROOT))
        for command in self.build_commands():
            assert ("--secret") not in (command)
        assert not (any("test-secret-value" in arg or arg.startswith("HF_TOKEN=")
                             for cmd in self.commands for arg in cmd))

    def test_wrong_dependency_cache_label_stops_before_building_more_layers(self):
        runtime_images.build_runtime_image(self.task, str(PROJECT_ROOT))
        name = next(name for name in self.images if name.startswith("acprof-runtime-env:"))
        self.images[name]["labels"]["org.acprof.environment-build-fingerprint"] = "wrong"
        count = len(self.build_commands())
        with pytest.raises(RuntimeError, match="缓存标签"):
            runtime_images.build_runtime_image(self.task, str(PROJECT_ROOT))
        assert (len(self.build_commands())) == (count)

    def test_environment_id_label_must_match_even_with_valid_build_fingerprint(self):
        runtime_images.build_runtime_image(self.task, str(PROJECT_ROOT))
        name = next(name for name in self.images if name.startswith("acprof-runtime-env:"))
        self.images[name]["labels"]["org.acprof.environment"] = "wrong"
        count = len(self.build_commands())
        with pytest.raises(RuntimeError, match="缓存标签"):
            runtime_images.build_runtime_image(self.task, str(PROJECT_ROOT))
        assert (len(self.build_commands())) == (count)

    def test_extra_installed_package_in_cached_environment_is_rejected(self):
        runtime_images.build_runtime_image(self.task, str(PROJECT_ROOT))
        name = next(name for name in self.images if name.startswith("acprof-runtime-env:"))
        self.manifests[self.images[name]["image_id"]]["packages"]["undeclared-package"] = "1.0"
        with pytest.raises(ValueError, match="extra=.*undeclared-package"):
            runtime_images.build_runtime_image(self.task, str(PROJECT_ROOT))

    def test_retagged_parent_during_build_does_not_publish_environment_cache(self):
        from acprof.host.dependency_images import prepare_environment_image
        from acprof.runtime_profiles import ENVIRONMENTS
        def changed(command, **kwargs):
            result = self.fake_run(command, **kwargs)
            if command[:2] == ["docker", "build"] and Path(command[command.index("-f") + 1]).name == "runtime.Dockerfile":
                source = next(arg.split("=", 1)[1] for arg in command if arg.startswith("PLATFORM_IMAGE="))
                self.images[source] = {"image_id": "sha256:" + "e" * 64, "labels": {}}
            return result
        with patch("acprof.host.command.run_command", side_effect=changed), pytest.raises(RuntimeError, match="父镜像引用发生变化"):
            prepare_environment_image(ENVIRONMENTS["audio-cpu"], PROJECT_ROOT)
        assert not (any(name.startswith("acprof-runtime-env:") for name in self.images))

    def test_changed_build_input_does_not_publish_dependency_cache(self):
        from acprof.host.dependency_images import prepare_environment_image
        from acprof.runtime_profiles import ENVIRONMENTS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copy_dependency_tree(root)
            def changed(command, **kwargs):
                result = self.fake_run(command, **kwargs)
                if command[:2] == ["docker", "build"]:
                    recipe = root / "dockerfiles/platform.Dockerfile"
                    recipe.write_text(recipe.read_text() + "\n# changed during build\n")
                return result
            with patch("acprof.host.command.run_command", side_effect=changed), pytest.raises(RuntimeError, match="发生变化"):
                prepare_environment_image(ENVIRONMENTS["audio-cpu"], root)
            assert not (any(name.startswith("acprof-platform-") for name in self.images))

    def test_input_change_between_declaration_read_and_fingerprinting_is_rejected(self):
        from acprof.host import dependency_images
        from acprof.runtime_profiles import ENVIRONMENTS, environment_identity
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copy_dependency_tree(root)
            changed = False
            def read_then_change(environment, project_dir):
                nonlocal changed
                identity = environment_identity(environment, project_dir)
                if not changed:
                    changed = True
                    path = root / environment.requirements_lock
                    path.write_text(path.read_text().replace("numpy-2.2.6-", "numpy-2.2.7-"))
                return identity
            with patch.object(dependency_images, "environment_identity", side_effect=read_then_change), pytest.raises(RuntimeError, match="输入发生变化"):
                dependency_images.prepare_environment_image(ENVIRONMENTS["audio-cpu"], root)
            assert not (any(name.startswith("acprof-runtime-env:") for name in self.images))

    @pytest.mark.parametrize('recipe', ('platform.Dockerfile', 'runtime.Dockerfile', 'runtime-model.Dockerfile', 'runtime-final.Dockerfile'))
    def test_build_failure_stops_before_later_layers(self, recipe):
        self.commands.clear()
        self.images.clear()
        self.failed_dockerfile = recipe
        with pytest.raises(RuntimeError, match="Docker 构建失败"):
            runtime_images.build_runtime_image(self.task, str(PROJECT_ROOT))
        last_build = self.build_commands()[-1]
        assert (Path(last_build[last_build.index("-f") + 1]).name) == (recipe)
