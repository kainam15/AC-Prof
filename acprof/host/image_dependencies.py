"""Match current locks; resolve historical image snapshots only on demand."""

from __future__ import annotations

from dataclasses import replace

from acprof.host.image_management import ImageInventory, ManagedImage


def dependency_stage(rows: list[dict]) -> str:
    """核对最近的文件操作，拒绝继承已知标签后又安装包的自定义镜像。

    history 只在读取时检查，返回阶段标识；不保留可能含凭据的原始命令。
    """
    commands = []
    for row in rows:
        command = row["command"].split("#(nop)")[-1].strip().removesuffix(" # buildkit")
        if command.startswith("RUN ") and "/bin/sh -c " in command:
            command = "RUN " + command.partition("/bin/sh -c ")[2]
        command = " ".join(command.split())
        if command.partition(" ")[0] not in {
            "ARG", "ENV", "LABEL", "CMD", "ENTRYPOINT", "EXPOSE", "USER", "WORKDIR",
            "STOPSIGNAL", "HEALTHCHECK", "SHELL", "VOLUME", "ONBUILD",
        }:
            commands.append(command)
        if len(commands) == 4:
            break
    stages = {
        "base": ["RUN python /build/environment_tools.py platform",
                 "COPY requirements.lock expectation.json /opt/acprof/",
                 "RUN python /build/environment_tools.py system /opt/acprof/system.lock",
                 "COPY system.lock /opt/acprof/system.lock"],
        "runtime": ["RUN python /build/environment_tools.py environment",
                    "COPY requirements.lock expectation.json /opt/acprof/"],
        "weights": ['RUN if [ -s /run/secrets/hf_token ]; then export HF_TOKEN="$(cat /run/secrets/hf_token)"; fi; python /opt/acprof/download_model.py',
                    "COPY acprof/container/download_model.py acprof/container/model_files.py /opt/acprof/"],
        "model": ["RUN python -m acprof.container.runtime_manifest", "COPY acprof/ /app/acprof/"],
        "model-plan": ["COPY model-store.json /models/model-store.json"],
    }
    return next((kind for kind, expected in stages.items() if commands[:len(expected)] == expected), "")


def describe_dependencies(inventory: ImageInventory) -> ImageInventory:
    from acprof.dependency_locks import content_digest, package_versions
    from acprof.host.dependency_images import _platform_fingerprint
    from acprof.installation import resource_root
    from acprof.runtime_profiles import (
        ENVIRONMENTS,
        PLATFORMS,
        environment_identity,
        platform_identity,
    )
    root = resource_root()
    platforms, environments = {}, {}
    for spec in PLATFORMS.values():
        try:
            identity = platform_identity(spec, root)
            platforms[(spec.platform_id, _platform_fingerprint(identity, root))] = identity
        except (OSError, ValueError, KeyError):
            continue
    for spec in ENVIRONMENTS.values():
        try:
            identity = environment_identity(spec, root)
            environments[(spec.platform.platform_id, content_digest(identity))] = identity
        except (OSError, ValueError, KeyError):
            continue

    indexed: dict[str, ManagedImage] = {}
    for item in sorted(inventory.images, key=lambda image: len(image.ancestor_ids)):
        item = replace(item, python_dependencies=(), system_dependencies=(), dependency_source="unknown")
        platform = platforms.get((item.platform_id, item.platform_key))
        if item.dependency_stage == item.kind and item.parent_source not in {"ambiguous", "conflict"}:
            if item.kind == "base" and platform:
                # 系统完整包表含上游镜像已有包；artifacts 才是本层实际安装的制品。
                system = tuple(sorted((f'{entry["name"]}:{entry["architecture"]}', entry["version"])
                                      for entry in platform["system"]["artifacts"]))
                item = replace(item, python_dependencies=tuple(sorted(package_versions(platform["packages"]).items())),
                               system_dependencies=system, dependency_source="platform-lock")
            elif item.kind == "runtime" and platform:
                environment = environments.get((item.platform_id, item.environment_id))
                if environment and environment["platform"] == platform:
                    inherited = package_versions(platform["packages"])
                    delta = tuple(sorted((name, version) for name, version in package_versions(environment["packages"]).items()
                                         if inherited.get(name) != version))
                    item = replace(item, python_dependencies=delta, dependency_source="environment-lock")
            elif item.kind in {"weights", "model-plan", "model"}:
                parent = indexed.get(item.parent_id)
                expected = {"runtime"} if item.kind in {"weights", "model-plan"} else {"weights", "model-plan"}
                if (parent and parent.kind in expected and parent.dependency_source != "unknown"
                        and item.environment_id and item.environment_key and item.platform_key
                        and (item.environment_id, item.environment_key, item.platform_key)
                        == (parent.environment_id, parent.environment_key, parent.platform_key)):
                    item = replace(item, dependency_source="inherited")
        indexed[item.image_id] = item
    result = replace(inventory, images=tuple(indexed[item.image_id] for item in inventory.images))
    # A periodic inventory refresh is metadata-only: reuse persisted evidence
    # without starting a container or querying manifests for all images.
    for item in result.images:
        if item.dependency_source == "unknown":
            result = resolve_image_dependencies(result, item.image_id, cached_only=True)
    return result



def resolve_image_dependencies(inventory: ImageInventory, image_id: str, *, cached_only: bool = False) -> ImageInventory:
    """Resolve one selected image and its ancestors without scanning the full inventory.

    A historical snapshot is evidence of the *whole* image. Only call a list
    'new in this layer' when a verified parent snapshot is available.
    """
    from acprof.host.image_dependency_scan import DependencySnapshot, cached_snapshot, read_snapshot

    original = {item.image_id: item for item in inventory.images}
    target = original.get(image_id)
    if target is None:
        return inventory
    ids = (*target.ancestor_ids, image_id)
    updated = dict(original)
    snapshots: dict[str, DependencySnapshot] = {}
    for key in ids:
        item = updated[key]
        if item.kind in {"base", "runtime"}:
            snapshot, reason = (
                (cached_snapshot(inventory.daemon_id, item.image_id), "")
                if cached_only and item.acprof and item.dependency_stage == item.kind
                else (None, "构建步骤无法核验，不能复用历史依赖缓存。") if cached_only
                else read_snapshot(inventory.connection, inventory.daemon_id, item)
            )
            if snapshot is None:
                if item.dependency_source == "unknown":
                    updated[key] = replace(item, dependency_note=reason)
                continue
            snapshots[key] = snapshot
            if item.dependency_source != "unknown":
                continue  # Keep current-lock evidence and its established display semantics.
            parent = updated.get(item.parent_id)
            previous = snapshots.get(item.parent_id)
            compatible_parent = (
                item.kind == "runtime" and parent is not None and parent.kind == "base"
                and item.parent_source not in {"ambiguous", "conflict", "missing"}
                and parent.platform_id == item.platform_id
                and parent.platform_key == item.platform_key and previous is not None
            )
            if compatible_parent and previous is not None:
                py_parent, system_parent = dict(previous.python), dict(previous.system)
                python = tuple((name, version) for name, version in snapshot.python
                               if py_parent.get(name) != version)
                system = tuple((name, version) for name, version in snapshot.system
                               if system_parent.get(name) != version)
                scope = "delta"
                note = ""
            else:
                python, system, scope = snapshot.python, snapshot.system, "full"
                note = "无法验证父镜像依赖，显示完整安装集合而非本层新增。"
            updated[key] = replace(item, python_dependencies=python, system_dependencies=system,
                                   dependency_source="historical-manifest" if snapshot.source == "manifest"
                                   else "actual-scan", dependency_scope=scope, dependency_note=note)
        elif item.kind in {"weights", "model-plan", "model"} and item.dependency_source == "unknown":
            parent = updated.get(item.parent_id)
            expected = {"runtime"} if item.kind in {"weights", "model-plan"} else {"weights", "model-plan"}
            if (item.dependency_stage == item.kind and parent is not None and parent.kind in expected
                    and parent.dependency_source != "unknown"
                    and item.parent_source not in {"ambiguous", "conflict", "missing"}
                    and item.environment_id and item.environment_key and item.platform_key
                    and (item.environment_id, item.environment_key, item.platform_key)
                    == (parent.environment_id, parent.environment_key, parent.platform_key)):
                updated[key] = replace(item, dependency_source="inherited-historical",
                                       dependency_note="已核对父镜像身份和构建步骤。")
    return replace(inventory, images=tuple(updated[item.image_id] for item in inventory.images))
