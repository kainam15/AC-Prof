"""Pure structured projections of already resolved task-selection data."""
from __future__ import annotations

import json
from dataclasses import dataclass

from acprof.tasks.presentation import ConfigField, ConfigPath, SelectionGroup, TaskSelection
from acprof.tui.i18n import message_id
from acprof.tui.models.task_labels import parameter_fact
from acprof.tui.ui.view_models import FactRow, StatusSummary, ViewSection

_COMMON_PRIORITY: dict[ConfigPath, int] = {("interface",): 0, ("backend",): 1, ("operation",): 2}


@dataclass(frozen=True, slots=True)
class TaskGroupView:
    group_id: str
    differences: ViewSection


@dataclass(frozen=True, slots=True)
class TaskSelectionView:
    common: ViewSection
    groups: tuple[TaskGroupView, ...]
    semantic_count: int


def _common_order(field: ConfigField) -> int:
    return _COMMON_PRIORITY.get(field.path, len(_COMMON_PRIORITY))


def _group_view(group: SelectionGroup) -> TaskGroupView:
    differences = tuple(parameter_fact(field) for field in group.differences)
    semantic_ids = [task_id for purpose in group.purposes for task_id in purpose.task_ids]
    identities = (
        FactRow(message_id("models.tasks.group_id"), group.task_id, copyable=True),
        FactRow(message_id("models.tasks.representative_label"), group.representative["task_id"], copyable=True),
        FactRow(message_id("models.tasks.semantic_tasks"),
                json.dumps(semantic_ids, ensure_ascii=False), copyable=True),
    )
    rows = differences[:4] or (FactRow(message_id("common.field.note"),
                                       message_id("models.tasks.same_configuration")),)
    return TaskGroupView(group.task_id, ViewSection(
        message_id("models.tasks.differences_title"), rows, (*differences[4:], *identities),
    ))


def task_selection_view(presentation: TaskSelection) -> TaskSelectionView:
    """Keep comparison, grouping and IDs owned by the normalized presentation."""
    common = tuple(parameter_fact(field) for field in sorted(presentation.common, key=_common_order))
    rows = common[:3] or (FactRow(message_id("common.field.note"),
                                 message_id("models.tasks.no_common_parameters")),)
    note = FactRow(message_id("common.field.note"), message_id("models.tasks.merged_paths_note"))
    return TaskSelectionView(
        ViewSection(message_id("models.tasks.common_title"), rows, (*common[3:], note)),
        tuple(_group_view(group) for group in presentation.groups),
        presentation.semantic_count,
    )


def task_context_view(fields: dict[str, object]) -> ViewSection:
    """Keep caller-owned model identity available without crowding task choices."""
    rows = []
    for name, value in fields.items():
        rendered = (message_id("common.value.unknown") if value is None
                    else value if isinstance(value, str) else str(value))
        rows.append(FactRow(name, rendered, copyable=True))
    return ViewSection(message_id("models.tasks.model_context"), tuple(rows), collapsed=True)


def selection_summary(view: TaskSelectionView, selected: tuple[str, ...]) -> StatusSummary:
    """Describe the pending decision without certifying runtime readiness."""
    group_ids = {group.group_id for group in view.groups}
    if not group_ids:
        headline = message_id("models.tasks.summary_empty")
    elif count := len(group_ids.intersection(selected)):
        headline = message_id("models.tasks.summary_selected", count, len(group_ids), view.semantic_count)
    else:
        headline = message_id("models.tasks.summary_pending", len(group_ids), view.semantic_count)
    return StatusSummary("neutral", headline)
