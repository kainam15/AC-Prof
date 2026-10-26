"""Shared experiment configuration, validation and CLI command construction."""

from __future__ import annotations

import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field as dataclass_field, fields, replace
from pathlib import Path

from acprof.extensions import CATALOG
from acprof.gpu_mode import normalize_gpu_mode
from acprof.installation import cli_command
from acprof.messages import message
from acprof.run_options import RunOptions, configuration_arguments

TASK_FAMILIES = tuple(sorted(set(CATALOG.task_families.values())))
COMPUTE_PROFILE_TOOLS = ("none", "both", "torch", "ncu")
EXECUTION_PROFILE_TOOLS = ("none", "both", "massif", "nsys")
NOTIFY_MODES = ("auto", "none", "wecom")
PRESET_FIELDS = (
    "cpus", "mems", "gpus", "input_scales", "input_scale_policy", "batch_size", "warmup",
    "repeat", "repeat_in_window", "repeat_window_seconds", "request_timeout_seconds", "sample_hz",
    "idle_seconds", "idle_cooldown_seconds", "compute_profile_tool", "profiling_mode",
    "execution_profile_tool", "prune_startup_oom",
)


@dataclass(frozen=True)
class ConfigIssue:
    """A user-facing reason attached to a stable configuration field."""

    field: str | None
    reason: str


class RunConfigError(ValueError):
    """Invalid experiment options shared by CLI and TUI consumers."""

    def __init__(self, issues: Iterable[ConfigIssue]):
        self.issues = tuple(issues)
        super().__init__("；".join(f"[{item.field}] {item.reason}" if item.field else str(item.reason)
                                 for item in self.issues))


def _csv_values(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _positive_int_csv(value: str, label: str, field: str) -> list[int]:
    raw_values = _csv_values(value)
    if not raw_values:
        raise RunConfigError([ConfigIssue(field, message('{0}不能为空', label))])
    try:
        values = [int(item) for item in raw_values]
    except ValueError as exc:
        raise RunConfigError([ConfigIssue(field, message('{0}必须是逗号分隔的整数', label))]) from exc
    if any(item <= 0 for item in values):
        raise RunConfigError([ConfigIssue(field, message('{0}必须全部大于 0', label))])
    return values


def _positive_float_csv(value: str, label: str, field: str) -> list[float]:
    raw_values = _csv_values(value)
    if not raw_values:
        return []
    try:
        values = [float(item) for item in raw_values]
    except ValueError as exc:
        raise RunConfigError([ConfigIssue(field, message('{0}必须是逗号分隔的数字', label))]) from exc
    if any(not math.isfinite(item) or item <= 0.0 for item in values):
        raise RunConfigError([ConfigIssue(field, message('{0}必须全部大于 0', label))])
    return values


def _number(value: str | int | float, label: str, field: str, *, integer: bool) -> int | float:
    try:
        return int(value) if integer else float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        kind = message('整数') if integer else message('数字')
        raise RunConfigError([ConfigIssue(field, message('{0}必须是{1}', label, kind))]) from exc


@dataclass(frozen=True)
class RunConfig(RunOptions):
    """Common experiment options independent of presentation."""

    extra_options: dict = dataclass_field(default_factory=dict)

    def experiment_parameters(self) -> tuple[tuple[str, object], ...]:
        """Preset identity excludes transport, storage and execution controls."""
        execution = {"model", "output_dir", "download_mode", "max_download", "model_store",
                     "model_store_max", "notify", "rebuild_image", "resume", "idle_debug", "sniff_iface"}
        return tuple((item.name, getattr(self, item.name)) for item in fields(self)
                     if item.name not in execution)

    def with_preset(self, preset: str) -> RunConfig:
        """Change experimental parameters while retaining identity and user constraints."""
        if preset not in {"smoke", "main"}:
            raise RunConfigError([ConfigIssue(None, message("未知预设：{0}", preset))])
        template = self.smoke(self.model) if preset == "smoke" else self.main_matrix(self.model)
        updated = replace(self, **{name: getattr(template, name) for name in PRESET_FIELDS})
        if self.resume and self.experiment_parameters() != updated.experiment_parameters():
            raise RunConfigError([ConfigIssue("resume", message("续跑参数不能更改；请取消续跑并选择新的输出目录创建新实验。"))])
        return updated

    @classmethod
    def from_namespace(cls, args) -> RunConfig:
        from acprof.run_args import build_parser

        defaults = cls()
        names = {item.name for item in fields(cls)}
        public = {action.dest for action in build_parser()._actions} - {"help"}
        extra = dict(getattr(args, "extra_options", {}))
        extra.update({name: value for name, value in vars(args).items()
                      if name in public and name not in names})
        values = {field.name: getattr(args, field.name, getattr(defaults, field.name))
                      if getattr(args, field.name, None) is not None else getattr(defaults, field.name)
                      for field in fields(cls)}
        values["extra_options"] = extra
        return cls(**values)

    def to_namespace(self):
        """Adapt the shared configuration to existing host APIs, without a second schema."""
        from acprof.run_args import arguments_from_options, build_parser

        names = {item.name for item in fields(self)} - {"extra_options"}
        if names.intersection(self.extra_options):
            raise ValueError("extra_options must not override RunConfig fields")
        parser = build_parser()
        options = parser.parse_args(["--model", self.model,
                                     *arguments_from_options(self.extra_options)])
        for name in names:
            setattr(options, name, getattr(self, name))
        return options

    @classmethod
    def smoke(cls, model: str = "") -> RunConfig:
        """Return a basic CPU validation run without optional collectors or notifications."""
        return cls(
            model=model,
            cpus="1",
            mems="4",
            gpus="off",
            input_scale_policy="minimal",
            output_dir="results/smoke",
            warmup=0,
            repeat=1,
            repeat_in_window=1,
            idle_seconds=0,
            idle_cooldown_seconds=0,
            profiling_mode="basic",
            compute_profile_tool="none",
            execution_profile_tool="none",
            notify="none",
        )

    @classmethod
    def main_matrix(cls, model: str = "") -> RunConfig:
        """Return the normal matrix with isolated profilers deferred."""
        return cls(
            model=model,
            compute_profile_tool="none",
            execution_profile_tool="none",
        )

    def validate(self, *, project_dir: Path | None = None) -> RunConfig:
        """Normalize form values and reject invalid or misleading runs."""
        errors: list[ConfigIssue] = []
        if self.resume and self.rebuild_image:
            errors.append(ConfigIssue("rebuild_image", message("恢复实验必须使用原镜像；如需重建，请创建新实验。")))
        try:
            if not isinstance(self.extra_options, dict) or set(self.extra_options) & {item.name for item in fields(self)}:
                raise ValueError('extra options must contain only unmapped public CLI fields')
            from acprof.run_args import arguments_from_options
            arguments_from_options(self.extra_options)
        except (ValueError, TypeError) as exc:
            errors.append(ConfigIssue(None, message("保留的实验参数无效：{0}", exc)))
        if not isinstance(self.revision, str) or any(character.isspace() for character in self.revision):
            errors.append(ConfigIssue("revision", message("模型 revision 必须是无空白的 branch、tag 或 commit SHA")))
        from acprof.hf_endpoints import HF_DOWNLOAD_MODES
        from acprof.model_repository import MODEL_SOURCES
        if self.model_source not in MODEL_SOURCES:
            errors.append(ConfigIssue("model_source", message("模型来源必须是 huggingface 或 modelscope")))
        from acprof.network_policy import parse_bytes
        if self.download_mode not in HF_DOWNLOAD_MODES:
            errors.append(ConfigIssue("download_mode", message("下载源模式必须是 auto、mirror-only、mirror-preferred 或 official")))
        for field in ("max_download", "model_store_max"):
            try:
                parse_bytes(getattr(self, field))
            except ValueError as exc:
                errors.append(ConfigIssue(field, str(exc)))
        from acprof.cpu_affinity import normalize_cpu_set
        try:
            cpuset_cpus = normalize_cpu_set(self.cpuset_cpus)
        except ValueError:
            cpuset_cpus = ""
            errors.append(ConfigIssue("cpuset_cpus", message("CPU 集合格式无效；示例：0-3,8")))
        model = self.model.strip()
        if not model:
            errors.append(ConfigIssue("model", message('模型 ID 不能为空')))

        try:
            cpus = [] if self.cpus == "auto" else _positive_int_csv(self.cpus, message('CPU 列表'), "cpus")
        except RunConfigError as exc:
            errors.extend(exc.issues)
            cpus = []
        try:
            mems = [] if self.mems == "auto" else _positive_int_csv(self.mems, message('内存列表'), "mems")
        except RunConfigError as exc:
            errors.extend(exc.issues)
            mems = []

        gpus = _csv_values(self.gpus)
        if self.runtime not in {"auto", "docker", "process"}:
            errors.append(ConfigIssue("runtime", "Execution runtime must be auto, docker or process"))
        if self.runtime == "process" and (self.profiling_mode != "basic"
                or self.compute_profile_tool != "none" or self.execution_profile_tool != "none"
                or self.extra_options.get("dram_energy") == "required"):
            errors.append(ConfigIssue("runtime", "Process Runtime currently supports basic collection without independent profilers"))
        if not gpus:
            errors.append(ConfigIssue("gpus", message('GPU 模式不能为空')))
        else:
            try:
                gpus = [normalize_gpu_mode(item) for item in gpus]
            except ValueError:
                errors.append(ConfigIssue("gpus", message('GPU 模式只能包含 off 或 on')))

        if self.prune_startup_oom:
            if len(cpus) != len(set(cpus)):
                errors.append(ConfigIssue("cpus", message('启用启动 OOM 剪枝时 CPU 列表不能重复')))
            if len(mems) != len(set(mems)):
                errors.append(ConfigIssue("mems", message('启用启动 OOM 剪枝时内存列表不能重复')))
            if len(gpus) != len(set(gpus)):
                errors.append(ConfigIssue("gpus", message('启用启动 OOM 剪枝时 GPU 模式不能重复')))

        try:
            _positive_float_csv(self.input_scales, message('输入规模'), "input_scales")
        except RunConfigError as exc:
            errors.extend(exc.issues)
        if self.input_scale_policy not in {"auto", "minimal"}:
            errors.append(ConfigIssue("input_scale_policy", message("输入规划必须是 auto 或 minimal")))

        numbers: dict[str, int | float] = {}
        numeric_fields = (
            ("batch_size", "Batch size", True, True, 'Batch size 必须大于 0'),
            ("warmup", "Warmup", True, False, 'Warmup 不能小于 0'),
            ("repeat", "Repeat", True, True, 'Repeat 必须大于 0'),
            ("repeat_in_window", '每窗口请求数', True, False, '每窗口请求数不能小于 0'),
            ("repeat_window_seconds", '自动窗口秒数', False, True, '自动窗口秒数必须大于 0'),
            ("request_timeout_seconds", '单请求超时秒数', False, True, '单请求超时秒数必须是大于 0 的有限数字'),
            ("sample_hz", '采样频率', False, True, '采样频率必须大于 0'),
            ("raw_sample_limit", 'Raw sample limit', True, True, 'Raw sample limit must be positive'),
            ("idle_seconds", 'Idle 秒数', False, False, 'Idle 秒数不能小于 0'),
            ("idle_cooldown_seconds", 'Idle cooldown 秒数', False, False, 'Idle cooldown 秒数不能小于 0'),
        )
        for field, label, integer, positive, reason in numeric_fields:
            try:
                value = _number(getattr(self, field), message(label), field, integer=integer)
            except RunConfigError as exc:
                errors.extend(exc.issues)
                continue
            numbers[field] = value
            if not math.isfinite(float(value)) or (value <= 0 if positive else value < 0):
                if field == "repeat_window_seconds" and not math.isfinite(float(value)):
                    reason = '自动窗口秒数必须是有限数字'
                errors.append(ConfigIssue(field, message(reason)))

        task_family = self.task_family.strip().lower()
        if task_family and task_family not in TASK_FAMILIES:
            errors.append(ConfigIssue("task_family", message('任务族必须是 nlp/cv/audio/timeseries/diffusion/multimodal/structured')))
        if self.compute_profile_tool not in COMPUTE_PROFILE_TOOLS:
            errors.append(ConfigIssue("compute_profile_tool", message('无效的计算分析器')))
        if self.profiling_mode not in {"full", "basic"}:
            errors.append(ConfigIssue("profiling_mode", message('无效的画像模式')))
        if self.sampling_policy not in {"burst", "skip"}:
            errors.append(ConfigIssue("sampling_policy", "sampling_policy must be burst or skip"))
        if ("raw_sample_limit" in numbers and numbers["raw_sample_limit"] < 2
                or isinstance(self.raw_sample_limit, bool)
                or isinstance(self.raw_sample_limit, float) and not self.raw_sample_limit.is_integer()):
            errors.append(ConfigIssue("raw_sample_limit", "raw_sample_limit must be an integer of at least 2"))
        if self.execution_profile_tool not in EXECUTION_PROFILE_TOOLS:
            errors.append(ConfigIssue("execution_profile_tool", message('无效的执行分析器')))
        if self.notify not in NOTIFY_MODES:
            errors.append(ConfigIssue("notify", message('无效的通知模式')))
        if not self.output_dir.strip():
            errors.append(ConfigIssue("output_dir", message('输出目录不能为空')))
        if not self.sniff_iface.strip():
            errors.append(ConfigIssue("sniff_iface", message('抓包网卡不能为空')))

        workload_spec = self.workload_spec.strip()
        if workload_spec and project_dir is not None:
            workload_path = Path(workload_spec).expanduser()
            if not workload_path.is_absolute():
                workload_path = project_dir / workload_path
            if not workload_path.is_file():
                errors.append(ConfigIssue("workload_spec", message('Workload manifest 不存在：{0}', workload_spec)))

        model_spec = self.model_spec.strip()
        if model_spec and project_dir is not None:
            model_path = Path(model_spec).expanduser()
            if not model_path.is_absolute():
                model_path = project_dir / model_path
            try:
                from acprof.model_spec import read_model_spec
                read_model_spec(model_path)
            except (OSError, ValueError) as exc:
                errors.append(ConfigIssue("model_spec", message('模型接口声明无效：{0}；{1}', model_spec, str(exc))))

        if errors:
            raise RunConfigError(errors)

        return replace(
            self,
            model=model,
            task=self.task.strip(),
            task_family=task_family,
            backend=self.backend.strip(),
            cpus="auto" if self.cpus == "auto" else ",".join(str(value) for value in cpus),
            cpuset_cpus=cpuset_cpus,
            mems="auto" if self.mems == "auto" else ",".join(str(value) for value in mems),
            gpus=",".join(gpus),
            input_scales=",".join(_csv_values(self.input_scales)),
            workload_spec=workload_spec,
            model_spec=model_spec,
            output_dir=self.output_dir.strip(),
            batch_size=int(numbers["batch_size"]),
            warmup=int(numbers["warmup"]),
            repeat=int(numbers["repeat"]),
            repeat_in_window=int(numbers["repeat_in_window"]),
            repeat_window_seconds=float(numbers["repeat_window_seconds"]),
            request_timeout_seconds=float(numbers["request_timeout_seconds"]),
            sample_hz=float(numbers["sample_hz"]),
            raw_sample_limit=int(numbers["raw_sample_limit"]),
            idle_seconds=float(numbers["idle_seconds"]),
            idle_cooldown_seconds=float(numbers["idle_cooldown_seconds"]),
            sniff_iface=self.sniff_iface.strip(),
        )

    def result_dir(self, project_dir: Path) -> Path:
        output_root = Path(self.output_dir).expanduser()
        if not output_root.is_absolute():
            output_root = project_dir / output_root
        return output_root / self.model.replace("/", "--")

    def result_csv(self, project_dir: Path) -> Path:
        from acprof.artifact_layout import ArtifactLayout
        return ArtifactLayout.discover(self.result_dir(project_dir)).result_csv


def build_run_command(
    config: RunConfig,
    *,
    project_dir: Path,
    python_executable: str | Path = sys.executable,
) -> list[str]:
    """Build the installed CLI command without duplicating its work."""
    config = config.validate(project_dir=project_dir)
    from acprof.run_args import arguments_from_options

    return [
        *cli_command("run", python_executable=python_executable),
        *configuration_arguments(config, purpose="run"),
        *arguments_from_options(config.extra_options),
    ]
