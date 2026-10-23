"""Stable TUI process stage identities, independent of translated labels."""
from __future__ import annotations

from enum import Enum

from acprof.messages import Message, message


class Stage(Enum):
    WAITING = "waiting"
    STARTING = "starting"
    CLEANUP_INCOMPLETE = "cleanup_incomplete"
    VERIFYING_ARTIFACT = "verifying_artifact"
    TASK_UNSUPPORTED = "task_unsupported"
    RUNTIME_VALIDATION = "runtime_validation"
    INTERFACE_RESOLUTION = "interface_resolution"
    DEPENDENCY_RESOLUTION = "dependency_resolution"
    ENVIRONMENT_PREFLIGHT = "environment_preflight"
    PREPARING_IMAGE = "preparing_image"
    PREPARING_INPUT = "preparing_input"
    CONFIRMING_INTERFACE = "confirming_interface"
    PREPARING_ENVIRONMENT = "preparing_environment"
    PREPARING_MODEL = "preparing_model"
    STARTING_PROBE_CONTAINER = "starting_probe_container"
    PREPARING_MEMORY_PROBE = "preparing_memory_probe"
    SCANNING_MEMORY = "scanning_memory"
    MEMORY_FOUND = "memory_found"
    MEMORY_FEASIBILITY = "memory_feasibility"
    PROBE_COMPLETED = "probe_completed"
    PROBE_FAILED = "probe_failed"
    PREPARING_MATRIX = "preparing_matrix"
    STARTING_CONTAINER = "starting_container"
    MAX_SCALE_PROBE = "max_scale_probe"
    BUILDING_IMAGE = "building_image"
    PLANNING_INPUT = "planning_input"
    COMPUTE_PROFILING = "compute_profiling"
    EXECUTION_PROFILING = "execution_profiling"
    SERVER_READY = "server_ready"
    MEASURING = "measuring"
    CLEANING_CASE = "cleaning_case"
    CASE_COMPLETED = "case_completed"
    OOM_PRUNED = "oom_pruned"
    MERGING_RESULTS = "merging_results"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    PARTIAL = "partial"
    TERMINATED = "terminated"

    @property
    def label(self) -> Message:
        """Localized only by the presentation layer; safe for control flow."""
        return message(_LABELS[self])


_LABELS: dict[Stage, str] = {
    Stage.WAITING: "等待",
    Stage.STARTING: "启动中",
    Stage.CLEANUP_INCOMPLETE: "清理未完成",
    Stage.VERIFYING_ARTIFACT: "核验产物",
    Stage.TASK_UNSUPPORTED: "任务不支持",
    Stage.RUNTIME_VALIDATION: "运行验证",
    Stage.INTERFACE_RESOLUTION: "接口解析",
    Stage.DEPENDENCY_RESOLUTION: "解析依赖",
    Stage.ENVIRONMENT_PREFLIGHT: "环境预检",
    Stage.PREPARING_IMAGE: "准备镜像",
    Stage.PREPARING_INPUT: "准备输入",
    Stage.CONFIRMING_INTERFACE: "正在确认模型接口…",
    Stage.PREPARING_ENVIRONMENT: "正在准备运行环境…",
    Stage.PREPARING_MODEL: "正在准备模型…",
    Stage.STARTING_PROBE_CONTAINER: "启动探测容器",
    Stage.PREPARING_MEMORY_PROBE: "准备内存探测",
    Stage.SCANNING_MEMORY: "探测最低可用内存",
    Stage.MEMORY_FOUND: "找到最低可用内存",
    Stage.MEMORY_FEASIBILITY: "内存可行性探测",
    Stage.PROBE_COMPLETED: "探测完成",
    Stage.PROBE_FAILED: "探测失败",
    Stage.PREPARING_MATRIX: "准备矩阵",
    Stage.STARTING_CONTAINER: "启动容器",
    Stage.MAX_SCALE_PROBE: "最大尺度探测",
    Stage.BUILDING_IMAGE: "构建镜像",
    Stage.PLANNING_INPUT: "规划输入",
    Stage.COMPUTE_PROFILING: "计算分析",
    Stage.EXECUTION_PROFILING: "执行分析",
    Stage.SERVER_READY: "服务就绪",
    Stage.MEASURING: "正式测量",
    Stage.CLEANING_CASE: "清理 case",
    Stage.CASE_COMPLETED: "case 完成",
    Stage.OOM_PRUNED: "OOM 剪枝",
    Stage.MERGING_RESULTS: "合并结果",
    Stage.COMPLETED: "已完成",
    Stage.FAILED: "失败",
    Stage.STOPPED: "已停止",
    Stage.PARTIAL: "部分完成",
    Stage.TERMINATED: "已终止",
}
