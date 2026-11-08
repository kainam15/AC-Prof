"""按依赖内容准备平台和环境缓存；主构建与离线验证共用此入口。"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from acprof.host.images.content import require_content
from acprof.host.images.errors import DockerPreparationError, rebuild_target
from acprof.host.images.operations import pull_image, run_docker
from acprof.host.images.progress import emit_image_progress
from acprof.installation import resource_root
from acprof.network_policy import (
    DEFAULT_RUNTIME_REGISTRY as DEFAULT_RUNTIME_REGISTRY,
    configured_artifact_mirrors,
    configured_oci_sources,
    debian_snapshot_fallback_url,
    dependency_artifact_url,
    docker_hub_image,
    oci_repository,
    oci_source_url,
    runtime_image_source,
)
from acprof.runtime.dependency_locks import (
    content_digest,
    package_versions,
    python_lock_text,
    require_exact_packages,
    system_lock_identity,
)
from acprof.runtime.profiles import (
    DependencyEnvironment,
    PlatformSpec,
    environment_identity,
    platform_identity,
)
from acprof.system import command as host_command

PROJECT_ROOT = resource_root()
PLATFORM_LABEL = "org.acprof.platform-build-fingerprint"
ENVIRONMENT_LABEL = "org.acprof.environment-build-fingerprint"
BUILD_HELPERS = ("dockerfiles/environment_tools.py", "acprof/runtime/dependency_locks.py", "acprof/network_policy.py")


def registry_reference(kind: str, fingerprint: str, registry: str | None = None) -> str:
    registry = oci_repository(registry, "ACPROF_RUNTIME_REGISTRY") if registry is not None else configured_oci_sources()["runtime_registry"]
    if kind not in {"platform", "environment"} or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("预构建镜像要求完整内容指纹")
    return f"{registry}:{kind}-{fingerprint}"


def dockerfile_frontend(root: Path, recipe: str = "platform.Dockerfile") -> str:
    declaration = (root / "dockerfiles" / recipe).read_text().splitlines()[0]
    match = re.fullmatch(r"# syntax=(\S+@sha256:[a-f0-9]{64})", declaration)
    if match is None:
        raise ValueError(f"{recipe}: Dockerfile frontend must be pinned by SHA256")
    return docker_hub_image(match[1])


def local_build_sources(identity: dict, root: Path, *, platform: bool) -> list[str]:
    """Disclose conditional auto fallback without probing or transferring payloads."""
    base = identity.get("platform", identity)
    sources = {oci_source_url(dockerfile_frontend(root, "runtime.Dockerfile"))}
    artifacts = [("python", entry) for entry in identity["packages"]]
    if platform:
        sources.add(oci_source_url(dockerfile_frontend(root)))
        sources.add(oci_source_url(docker_hub_image(base["python_base_image"])))
        artifacts.extend(("debian", entry) for entry in base["system"]["artifacts"])
    for category, entry in artifacts:
        url = dependency_artifact_url(category, entry["url"])
        parsed = urlsplit(url)
        sources.add(f"{parsed.scheme}://{parsed.netloc}")
        if category == "debian" and (fallback := debian_snapshot_fallback_url(url)):
            sources.add("https://" + str(urlsplit(fallback).netloc))
    return sorted(sources)


def _pull_cached_image(name: str, label: str, fingerprint: str, expected: dict,
                       identity: dict, kind: str, source: str, registry: str | None, root: Path = PROJECT_ROOT):
    if source == "build":
        return None
    reference = registry_reference(kind, fingerprint, registry)
    print(f"[runtime] 拉取预构建依赖：{reference}", flush=True)
    try:
        pull_image(reference, component=kind)
    except DockerPreparationError as exc:
        if source == "pull":
            raise
        if exc.reason_code in {"docker_auth_failed", "docker_content_incomplete", "docker_integrity_failed", "docker_tls_failed"}:
            raise  # Authentication and damaged content are not cache misses.
        if os.environ.get("ACPROF_MAX_DOWNLOAD", "").strip():
            raise RuntimeError("预构建镜像拉取失败；预算模式禁止未预检的 local build fallback。请显式选择 build 后重新预检。") from exc
        from acprof.network_policy import require_source_transition
        for origin in local_build_sources(identity, root, platform=kind == "platform"):
            require_source_transition("https://" + reference, origin)
        print("[runtime] 预构建镜像不可用，回退到本机锁定依赖构建。", flush=True)
        return None
    # A successful pull with bad identity is never silently replaced by a local build.
    verified = checked_image(reference, label, fingerprint, expected, identity, kind)
    if verified is None:
        raise RuntimeError(f"拉取后未找到预构建镜像：{reference}")
    host_command.run_command(['docker', 'tag', verified[0], name], check=True)
    return verified


@dataclass(frozen=True)
class PreparedEnvironment:
    image_id: str
    name: str
    platform_image_id: str
    manifest: dict


@dataclass(frozen=True)
class PreparedPlatform:
    image_id: str
    name: str
    expected: dict


def recipe_identity(root: Path, recipe: str) -> dict:
    return {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in ("dockerfiles/" + recipe, *BUILD_HELPERS)}


def platform_fingerprint(platform: PlatformSpec, project_dir=PROJECT_ROOT) -> str:
    root = Path(project_dir)
    return platform_identity_fingerprint(platform_identity(platform, root), root)


def platform_identity_fingerprint(identity: dict, root: Path) -> str:
    return content_digest({"platform": identity,
                           "recipe": recipe_identity(root, "platform.Dockerfile")})


def runtime_fingerprint(environment: DependencyEnvironment, project_dir=PROJECT_ROOT) -> str:
    """构建声明指纹；实际环境镜像的缓存键另加入固定平台 image ID。"""
    root = Path(project_dir)
    identity = environment_identity(environment, root)
    return _runtime_fingerprint(identity, platform_identity_fingerprint(identity["platform"], root), root)


def _runtime_fingerprint(identity: dict, platform_key: str, root: Path) -> str:
    return content_digest({"environment": identity, "platform_build": platform_key,
                           "recipe": recipe_identity(root, "runtime.Dockerfile")})


def read_image_manifest(image_id: str, path: str) -> dict:
    result = host_command.run_command(['docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'cat', image_id, path], check=False, timeout=host_command.DOCKER_RUN_TIMEOUT_S)
    if result.returncode:
        raise RuntimeError(f"镜像缺少依赖清单：{image_id}: {path}")
    try:
        manifest = json.loads(result.stdout)
        if not isinstance(manifest, dict):
            raise ValueError("manifest is not an object")
        return manifest
    except (ValueError, TypeError) as error:
        raise RuntimeError("镜像依赖清单无效") from error


def verify_manifest(manifest: dict, expected: dict, identity: dict) -> None:
    if manifest.get("schema_version") != 1 or any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError("镜像平台或依赖环境身份不匹配；拒绝复用")
    system = identity.get("system", identity.get("platform", {}).get("system"))
    if manifest.get("system_lock_sha256") != content_digest(system_lock_identity(system)):
        raise RuntimeError("镜像系统依赖 lock 不匹配")
    require_exact_packages(package_versions(identity["packages"]), manifest.get("packages", {}))
    require_exact_packages(system["packages"], manifest.get("system_packages", {}), kind="system")


def verify_labels(image: dict, expected: dict, kind: str) -> None:
    required = {"org.acprof.image-kind": kind, "org.acprof.platform": expected["platform_id"],
                PLATFORM_LABEL: expected["platform_build_fingerprint"]}
    if kind != "platform":
        required.update({"org.acprof.environment": expected["environment_id"],
                         ENVIRONMENT_LABEL: expected["environment_build_fingerprint"]})
    if any((image.get("labels") or {}).get(key) != value for key, value in required.items()):
        raise RuntimeError("依赖缓存标签内容不匹配")


def require_image_source(source: str, image_id: str) -> None:
    from acprof.host.images.runtime import inspect_identity
    actual = inspect_identity(source)
    if actual is None or actual["image_id"] != image_id:
        raise RuntimeError("构建父镜像引用发生变化，拒绝发布缓存")


def checked_image(name: str, label: str, fingerprint: str, expected: dict, identity: dict, kind: str):
    from acprof.host.images.runtime import inspect_identity
    image = inspect_identity(name)
    if image is None:
        return None
    if rebuild_target.get() == name:
        return None
    content = require_content(name, image["image_id"], component=kind)
    if (image.get("labels") or {}).get(label) != fingerprint:
        raise RuntimeError(f"依赖缓存标签内容不匹配：{name}")
    verify_labels(image, expected, kind)
    emit_image_progress(kind, name, "verifying", content=content.to_dict())
    manifest = read_image_manifest(image["image_id"], f"/opt/acprof/{kind}-manifest.json")
    verify_manifest(manifest, expected, identity)
    emit_image_progress(kind, name, "complete", content=content.to_dict())
    return image["image_id"], manifest


def build_dependency(root: Path, recipe: str, name: str, arguments: dict, expected: dict,
                     identity: dict, current_fingerprint, original_fingerprint: str) -> None:
    from acprof.host.images.runtime import inspect_identity
    frontend = dockerfile_frontend(root, recipe)
    mirrors = configured_artifact_mirrors()
    original_parent = arguments.get("PLATFORM_IMAGE") or arguments.get("PYTHON_BASE_IMAGE")
    arguments = {**arguments, "BUILDKIT_SYNTAX": frontend}
    if "PYTHON_BASE_IMAGE" in arguments:
        arguments["PYTHON_BASE_IMAGE"] = docker_hub_image(arguments["PYTHON_BASE_IMAGE"])
    parent = arguments.get("PLATFORM_IMAGE") or arguments.get("PYTHON_BASE_IMAGE")
    for reference in dict.fromkeys((original_parent, parent)):
        if reference and (parent_identity := inspect_identity(reference)) is not None:
            require_content(reference, parent_identity["image_id"], component="platform")
    with tempfile.TemporaryDirectory(prefix="acprof-dependency-build-") as directory:
        context = Path(directory)
        for relative in ("dockerfiles/" + recipe, *BUILD_HELPERS):
            (context / Path(relative).name).write_bytes((root / relative).read_bytes())
        (context / "requirements.lock").write_text(python_lock_text(identity["packages"]))
        system = identity.get("system", identity.get("platform", {}).get("system"))
        (context / "system.lock").write_text(json.dumps(system, sort_keys=True) + "\n")
        (context / "expectation.json").write_text(json.dumps(expected, sort_keys=True) + "\n")
        iidfile = context / "image-id"
        command = ["docker", "build", "--platform", "linux/amd64", "--iidfile", str(iidfile),
                   "-f", str(context / recipe)]
        if rebuild_target.get() == name:
            command.append("--no-cache")
        # Network mode controls only dependency downloads, never the measured
        # inference container. Host networking is explicit opt-in because it
        # relaxes the build's normal network namespace isolation.
        build_network = os.environ.get("ACPROF_DEPENDENCY_BUILD_NETWORK", "default").strip()
        if build_network not in {"default", "host"}:
            raise ValueError("ACPROF_DEPENDENCY_BUILD_NETWORK must be default or host")
        if build_network == "host":
            command.append("--network=host")
        for key, value in arguments.items():
            command += ["--build-arg", f"{key}={value}"]
        # Docker does not automatically inherit proxy environment variables
        # from the invoking shell. Pass standard proxy ARG names only so the
        # CLI resolves their values without exposing credentials in argv or
        # making the transport route part of the dependency image identity.
        for proxy_name in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY"):
            if os.environ.get(proxy_name):
                command += ["--build-arg", proxy_name]
        # Optional transport routing never becomes part of the locked artifact
        # identity. Validate all settings and pass ARG names, never proxy values.
        for mirror_name, selected in mirrors.items():
            if selected:
                command += ["--build-arg", mirror_name]
        command.append(str(context))
        if current_fingerprint() != original_fingerprint:
            raise RuntimeError("依赖构建输入发生变化，尚未构建镜像")
        run_docker(command, component="platform" if recipe == "platform.Dockerfile" else "environment",
                   image=name, operation="build", step=recipe, frontend=frontend)
        image_id = iidfile.read_text().strip()
        image = inspect_identity(image_id)
        if image is None or image["image_id"] != image_id:
            raise RuntimeError("构建没有返回不可变 image ID")
        if current_fingerprint() != original_fingerprint:
            raise RuntimeError("构建期间依赖或配方发生变化，未发布缓存标签")
        kind = "platform" if recipe == "platform.Dockerfile" else "environment"
        require_content(image_id, image_id, component=kind)
        verify_labels(image, expected, kind)
        verify_manifest(read_image_manifest(image_id, f"/opt/acprof/{kind}-manifest.json"), expected, identity)
        host_command.run_command(['docker', 'tag', image_id, name], check=True)


def _source_policy(image_source: str | None) -> str:
    configured_oci_sources()
    configured_artifact_mirrors()
    return runtime_image_source(image_source)


def _prepare_platform(platform: PlatformSpec, root: Path, platform_data: dict,
                      source_policy: str, registry: str | None) -> PreparedPlatform:
    # 构建键与输入目录必须来自同一份声明，避免读锁和计算键之间的变更被误标为新缓存。
    platform_key = platform_identity_fingerprint(platform_data, root)
    platform_name = f"acprof-platform-{platform.platform_id}:{platform_key[:20]}"
    expected_platform = {"platform_id": platform.platform_id, "platform_definition_id": content_digest(platform_data),
                         "platform_build_fingerprint": platform_key, "python_version": platform.python_version,
                         "python_base_image": platform.python_base_image, "architecture": platform.architecture}
    existing = checked_image(platform_name, PLATFORM_LABEL, platform_key, expected_platform, platform_data, "platform")
    print(f"[runtime-cache] platform={'hit' if existing else 'miss'} source={source_policy}", flush=True)
    if existing is None and os.environ.get("ACPROF_REQUIRE_PREPARED_IMAGE") == "1":
        raise RuntimeError(f"硬件验证禁止隐式构建或下载：预备平台镜像不存在：{platform_name}")
    if existing is None:
        existing = _pull_cached_image(platform_name, PLATFORM_LABEL, platform_key, expected_platform,
                                      platform_data, "platform", source_policy, registry, root)
    if existing is None:
        build_dependency(root, "platform.Dockerfile", platform_name,
                         {"PYTHON_BASE_IMAGE": platform.python_base_image, "PLATFORM_ID": platform.platform_id,
                          "PLATFORM_BUILD_FINGERPRINT": platform_key}, expected_platform, platform_data,
                         lambda: platform_fingerprint(platform, root), platform_key)
        token = rebuild_target.set("")
        try:
            existing = checked_image(platform_name, PLATFORM_LABEL, platform_key, expected_platform, platform_data, "platform")
        finally:
            rebuild_target.reset(token)
    if existing is None:
        raise RuntimeError("平台构建后未找到缓存镜像")
    if platform_fingerprint(platform, root) != platform_key:
        raise RuntimeError("平台依赖构建输入发生变化，拒绝使用缓存")
    return PreparedPlatform(existing[0], platform_name, expected_platform)


def prepare_platform_image(platform: PlatformSpec, project_dir=PROJECT_ROOT,
                           *, image_source: str | None = None,
                           registry: str | None = None) -> PreparedPlatform:
    root = Path(project_dir)
    return _prepare_platform(platform, root, platform_identity(platform, root),
                             _source_policy(image_source), registry)


def prepare_environment_image(environment: DependencyEnvironment, project_dir=PROJECT_ROOT,
                              *, image_source: str | None = None,
                              registry: str | None = None) -> PreparedEnvironment:
    from acprof.host.images.runtime import inspect_identity
    root = Path(project_dir)
    source_policy = _source_policy(image_source)
    identity = environment_identity(environment, root)
    platform = _prepare_platform(environment.platform, root, identity["platform"], source_policy, registry)
    expected_platform = platform.expected
    platform_key = expected_platform["platform_build_fingerprint"]
    platform_image_id = platform.image_id
    spec_key = _runtime_fingerprint(identity, platform_key, root)
    if runtime_fingerprint(environment, root) != spec_key:
        raise RuntimeError("依赖构建输入发生变化，尚未构建环境镜像")
    environment_key = content_digest({"definition": spec_key, "platform_image_id": platform_image_id})
    name = f"acprof-runtime-env:{environment_key[:20]}"
    expected = {**expected_platform, "platform_image_id": platform_image_id,
                "environment_id": content_digest(identity), "environment_build_fingerprint": environment_key}
    existing = checked_image(name, ENVIRONMENT_LABEL, environment_key, expected, identity, "environment")
    print(f"[runtime-cache] environment={'hit' if existing else 'miss'} source={source_policy}", flush=True)
    if existing is None and os.environ.get("ACPROF_REQUIRE_PREPARED_IMAGE") == "1":
        raise RuntimeError(f"硬件验证禁止隐式构建或下载：预备依赖环境镜像不存在：{name}")
    if existing is None:
        existing = _pull_cached_image(name, ENVIRONMENT_LABEL, environment_key, expected, identity,
                                      "environment", source_policy, registry, root)
    if existing is None:
        source = "acprof-build-source:" + platform_image_id.split(":", 1)[1]
        host_command.run_command(['docker', 'tag', platform_image_id, source], check=True)
        def current_environment_fingerprint():
            require_image_source(source, platform_image_id)
            return runtime_fingerprint(environment, root)
        build_dependency(root, "runtime.Dockerfile", name,
                         {"PLATFORM_IMAGE": source, "ENVIRONMENT_ID": expected["environment_id"],
                          "ENVIRONMENT_BUILD_FINGERPRINT": environment_key}, expected, identity,
                         current_environment_fingerprint, spec_key)
        token = rebuild_target.set("")
        try:
            existing = checked_image(name, ENVIRONMENT_LABEL, environment_key, expected, identity, "environment")
        finally:
            rebuild_target.reset(token)
    if existing is None:
        raise RuntimeError("依赖环境构建后未找到缓存镜像")
    if runtime_fingerprint(environment, root) != spec_key:
        raise RuntimeError("依赖构建输入发生变化，拒绝使用缓存")
    current = inspect_identity(name)
    if current is None or current["image_id"] != existing[0]:
        raise RuntimeError("依赖缓存标签在核验期间发生变化")
    return PreparedEnvironment(existing[0], name, platform_image_id, existing[1])
