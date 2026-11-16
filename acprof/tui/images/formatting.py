"""Pure image inventory formatting, filtering, and deletion message helpers."""
from __future__ import annotations

import re
from datetime import datetime

from acprof.host.images.graph import reclaimable_image_bytes
from acprof.host.images.inventory import (
    ImageInventory,
    ImageLayer,
    ImageManagementError,
    ManagedImage,
)
from acprof.messages import Message as SourceMessage, join_messages, message
from acprof.tui.i18n import message_id
from acprof.tui.images.names import image_display_name, image_name_evidence, image_row_label
from acprof.tui.ui.presentation import NOT_APPLICABLE, STATUS_LEGEND, UNKNOWN, format_bytes

IMAGE_KINDS = {
    "base": "公共基础", "runtime": "运行依赖", "weights": "模型文件",
    "model-plan": "模型清单",
    "model": "推理服务", "debug": "调试镜像", "other": "其它镜像", "untagged": "无标签",
}
IMAGE_HINT = "勾选含全部下层；◩ 保留本镜像；空格切换，←→ 折叠；清单自动刷新。"


def dependency_packages(item: ManagedImage) -> list[str]:
    # 先显示辨识环境最有用的包；版本一律取已匹配的锁，不从 profile 名称猜测。
    priority = ("torch", "diffusers", "transformers", "sentence-transformers", "torchvision",
                "torchaudio", "librosa", "scikit-learn", "pandas", "numpy", "triton")
    order = {name: index for index, name in enumerate(priority)}
    return [f"{name}=={version}" for name, version in sorted(
        item.python_dependencies, key=lambda pair: (order.get(pair[0], len(order)), pair[0]))]


def dependency_title(item: ManagedImage) -> str:
    if item.dependency_source == "unknown":
        return message_id("images.dependencies.unknown_title")
    if item.dependency_source in {"inherited", "inherited-historical"}:
        return message_id("images.dependencies.no_additions_title")
    if item.dependency_source in {"historical-manifest", "actual-scan"}:
        title = "历史构建" if item.dependency_source == "historical-manifest" else "实际扫描"
        return message_id(
            "images.dependencies.source_and_counts",
            message(title), len(item.python_dependencies), len(item.system_dependencies),
        )
    return message_id(
        "images.dependencies.package_counts",
        len(item.python_dependencies), len(item.system_dependencies),
    )


def dependency_detail(item: ManagedImage) -> str:
    if item.dependency_source == "unknown":
        return join_messages("\n", (message_id("images.dependencies.layer_unverified"), item.dependency_note))
    if item.dependency_source in {"inherited", "inherited-historical"}:
        return join_messages("\n", (
            message_id("images.dependencies.inherited_layer"),
            message("本层添加模型文件；包依赖由运行环境提供。" if item.kind == "weights" else
                    "本层仅保存模型清单；权重通过只读 Model Store 挂载。" if item.kind == "model-plan" else
                    "本层添加推理服务代码与运行清单；包依赖由运行环境提供。"),
        ))
    packages = dependency_packages(item)
    historical = item.dependency_source in {"historical-manifest", "actual-scan"}
    full = historical and item.dependency_scope == "full"
    python_label = ("镜像已安装 Python 包（{0}，含继承）：\n{1}" if full else
                    "平台 Python 依赖（{0}，含基础镜像已有包）：\n{1}" if item.dependency_source == "platform-lock" else
                    "本层新增 Python 依赖（{0}）：\n{1}")
    parts = [message(python_label, len(packages), "\n".join(packages) or message_id("images.dependencies.no_new_packages"))]
    if item.system_dependencies:
        system_label = "镜像已安装系统包（{0}，含继承）：\n{1}" if full else "本层系统安装制品（{0}）：\n{1}"
        parts.append(message(system_label, len(item.system_dependencies),
                             "\n".join(f"{name}={version}" for name, version in item.system_dependencies)))
    else:
        parts.append(message_id("images.dependencies.inherited_system"))
    if historical:
        parts.append(dependency_source_detail(item))
    if item.dependency_note:
        parts.append(message(item.dependency_note))
    return join_messages("\n\n", parts)


def dependency_source_detail(item: ManagedImage) -> SourceMessage:
    if item.dependency_source == "unknown":
        return message_id("images.dependencies.layer_unverified")
    if item.dependency_source == "historical-manifest":
        return message_id("images.dependencies.build_manifest_source")
    if item.dependency_source == "actual-scan":
        return message_id("images.dependencies.isolated_scan_source")
    if item.dependency_source in {"inherited", "inherited-historical"}:
        return message_id("images.dependencies.build_steps_source")
    return message_id("images.dependencies.lockfile_source")


def filtered_images(inventory: ImageInventory, query: str, scope: str) -> tuple[ManagedImage, ...]:
    terms = query.casefold().replace("/", "--").split()
    items = []
    for item in inventory.images:
        if scope == "acprof" and not item.acprof:
            continue
        if scope == "models" and (not item.acprof or not item.model_key):
            continue
        if scope == "runtime" and item.kind not in {"base", "runtime"}:
            continue
        if scope == "untagged" and item.tags:
            continue
        if scope in {"cpu", "cu124", "cu128"} and item.platform_id != scope:
            continue
        if scope == "base" and item.kind != "base":
            continue
        haystack = " ".join((
            *item.tags, *item.repo_digests, item.image_id, item.model_id, item.display_name,
            image_display_name(item), item.environment_id, item.platform_id, *item.profiles,
            *dependency_packages(item), *(f"{name}={version}" for name, version in item.system_dependencies),
        )).casefold().replace("/", "--")
        if all(term in haystack for term in terms):
            items.append(item)
    return tuple(items)


def image_path(item: ManagedImage, inventory: ImageInventory) -> str:
    indexed = {image.image_id: image for image in inventory.images}

    def path_name(image):
        name = image_display_name(image, indexed.get(image.parent_id))
        if image.model_id:
            name = join_messages(" · ", (message(IMAGE_KINDS[image.kind]), name))
        return join_messages("", (name, " ≈" if image.parent_source == "layer-prefix" else ""))

    path = [path_name(indexed[key]) for key in item.ancestor_ids if key in indexed]
    if item.parent_source == "missing":
        path.append(message_id("images.inheritance.parent_missing"))
    path.append(path_name(item))
    return message_id("images.inheritance.path", join_messages(" › ", path))


def image_created_text(created: str) -> str:
    """Render Docker image inspect's timezone-aware creation instant in local time."""
    try:
        # Docker uses nanosecond fractions; Python 3.10 only parses up to microseconds.
        normalized = re.sub(r"(\.\d{6})\d+(?=Z$|[+-]\d{2}:\d{2}$)", r"\1", created)
        timestamp = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        if timestamp.tzinfo is not None:
            return timestamp.astimezone().strftime("%Y-%m-%d %H:%M")
    except (ValueError, OverflowError):
        pass
    return UNKNOWN


def image_summary(item: ManagedImage, inventory: ImageInventory) -> str:
    return join_messages("\n", (
        message_id(
            "images.summary.size_breakdown",
            format_bytes(item.size_bytes), format_bytes(item.inherited_bytes), format_bytes(item.added_bytes),
        ),
        message_id("images.summary.reclaim_and_references", reclaimable_text(inventory, (item.image_id,)), len(item.containers)),
    ))


def image_metadata(item: ManagedImage, inventory: ImageInventory) -> str:
    return join_messages("\n\n", (
        image_path(item, inventory),
        message_id("images.metadata.shared_and_unique", format_bytes(item.shared_bytes), format_bytes(item.unique_bytes)),
        message_id(
            "images.metadata.model_and_created",
            item.model_id or (NOT_APPLICABLE if item.kind in {"base", "runtime"} else UNKNOWN), item.created or UNKNOWN,
        ),
        message_id("images.metadata.container_references", ", ".join(item.containers) if item.containers else message_id("common.value.none")),
        message_id("images.metadata.all_tags", "\n".join(item.tags) if item.tags else message_id("images.kind.untagged")),
    ))


def image_diagnostics(item: ManagedImage, inventory: ImageInventory) -> str:
    sources = {
        "recorded": "构建记录", "metadata": "构建指纹与层前缀核验", "layer-prefix": "层前缀推断，未确认 FROM",
        "missing": "父镜像不在本地", "ambiguous": "存在多个候选父镜像", "conflict": "父镜像记录与层链冲突",
        "unknown": "本地父镜像未知",
    }
    identity = [message_id("images.diagnostics.image_id", item.image_id)]
    if item.repo_digests:
        identity.append(message_id("images.diagnostics.repository_digest", ", ".join(item.repo_digests)))
    if item.parent_id:
        identity.append(message_id("images.diagnostics.parent_id", item.parent_id))
    evidence = (
        image_name_evidence(item),
        dependency_source_detail(item),
        message_id("images.diagnostics.parent_evidence", message(sources[item.parent_source]), len(item.descendant_ids)),
        message_id("images.diagnostics.size_source", message({
            "layers": "层链与已核验 history 字节数", "docker-df": "Docker df 近似值", "": "未知",
        }[item.space_source])),
    )
    space = [message_id("images.diagnostics.full_size", format_bytes(item.size_bytes))]
    if item.inherited_bytes is not None and item.size_bytes:
        inherited_cells = min(30, round(30 * item.inherited_bytes / item.size_bytes))
        space.append(message_id("images.diagnostics.size_composition", "█" * inherited_cells + "░" * (30 - inherited_cells)))
    space.extend((STATUS_LEGEND, message_id("images.diagnostics.reclaim_limitations")))
    parts = [join_messages("\n", identity), join_messages("\n", evidence), join_messages("\n", space)]
    parts.extend(message(warning) for warning in inventory.warnings)
    return join_messages("\n\n", parts)


def reclaimable_text(inventory: ImageInventory, image_ids: tuple[str, ...]) -> str:
    value = reclaimable_image_bytes(inventory, image_ids)
    return UNKNOWN if value is None else "0 B" if value == 0 else message_id("common.value.approximate", format_bytes(value))


def layer_image_detail(layer: ImageLayer, inventory: ImageInventory) -> str:
    indexed = {item.image_id: item for item in inventory.images}
    return message_id("images.layers.image_list", join_messages("\n\n", (
        message(
            "{0}\n  {1}", image_row_label(indexed[key]),
            "\n  ".join((*indexed[key].tags, *indexed[key].repo_digests)) or indexed[key].image_id,
        )
        for key in layer.image_ids
    )))


def layer_diagnostics(layer: ImageLayer) -> str:
    return join_messages("\n", (
        message_id("images.layers.diff_id", layer.diff_id),
        message_id("images.layers.chain_id", layer.chain_id),
        message_id("images.layers.accounting_notice"),
    ))


def image_selection_marker(item: ManagedImage, selected: set[str]) -> str:
    if item.image_id in selected:
        return "☑"
    if any(key in selected for key in item.descendant_ids):
        return "◩"
    return "—" if item.containers else "□"


def deletion_message(inventory: ImageInventory, image_ids: tuple[str, ...], *, hidden_count: int = 0) -> str:
    items = [item for item in inventory.images if item.image_id in image_ids]
    parts = [message_id("images.delete.environment_summary", inventory.connection.name, len(items))]
    if hidden_count:
        parts.append(message_id("images.delete.hidden_selection", hidden_count))
    for item in items:
        parts.append(message_id(
            "images.delete.image_detail",
            "\n".join(item.tags) if item.tags else message_id("images.kind.untagged"),
            item.image_id, format_bytes(item.size_bytes),
        ))
        if item.repo_digests:
            parts.append(message_id("images.diagnostics.repository_digest", "\n".join(item.repo_digests)))
        if item.parent_source == "layer-prefix":
            parts.append(message_id("images.delete.inferred_parent", item.parent_id))
    return join_messages("\n\n", parts)


def image_error(error: BaseException) -> str:
    if isinstance(error, ImageManagementError):
        return message("{0}：{1}", message(error.reason), error.detail) if error.detail else message(error.reason)
    return str(error)
