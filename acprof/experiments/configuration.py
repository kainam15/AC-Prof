"""Shared experiment configuration, validation and CLI command construction."""

from __future__ import annotations

import math
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field as dataclass_field, fields, replace
from pathlib import Path

from acprof.config import docker_resource_limits
from acprof.experiments.constraints import PROFILER_NUMBERS, RUN_NUMBERS, ConstraintViolation
from acprof.experiments.options import RunOptions, configuration_arguments
from acprof.extensions import CATALOG
from acprof.gpu_mode import normalize_gpu_mode
from acprof.installation import cli_command
from acprof.messages import message

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


def csv_values(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _positive_int_csv(value: str, label: str, field: str) -> list[int]:
    raw_values = csv_values(value)
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
    raw_values = csv_values(value)
    if not raw_values:
        return []
    try:
        values = [float(item) for item in raw_values]
    except ValueError as exc:
        raise RunConfigError([ConfigIssue(field, message('{0}必须是逗号分隔的数字', label))]) from exc
    if any(not math.isfinite(item) or item <= 0.0 for item in values):
        raise RunConfigError([ConfigIssue(field, message('{0}必须全部大于 0', label))])
    return values


def _normalize_extra_options(options: object, mapped: set[str], errors: list[ConfigIssue]) -> dict:
    from acprof.experiments.arguments import arguments_from_options, build_parser

    if not isinstance(options, dict):
        errors.append(ConfigIssue("extra_options", message("保留的实验参数必须是对象")))
        return {}
    parser = build_parser()
    normalized = {}
    for name, value in options.items():
        try:
            if name in mapped:
                raise ValueError("extra options must contain only unmapped public CLI fields")
            if name in PROFILER_NUMBERS and value is not None:
                try:
                    value = PROFILER_NUMBERS[name].parse(value, allow_text=True)
                except ConstraintViolation:
                    errors.append(ConfigIssue(name, message("参数 {0} 必须是正整数", name)))
                    continue
            arguments_from_options({name: value}, parser=parser)
        except (ValueError, TypeError) as exc:
            errors.append(ConfigIssue(str(name), message("参数 {0} 无效：{1}", name, str(exc))))
        else:
            normalized[name] = value
    return normalized


@dataclass(frozen=True)
class RunConfig(RunOptions):
    """Common experiment options independent of presentation."""

    extra_options: dict = dataclass_field(default_factory=dict)

    @property
    def inference_review_requested(self) -> bool:
        return bool(self.inference_config or self.confirm_inference)

    def experiment_parameters(self) -> tuple[tuple[str, object], ...]:
        """Preset identity excludes transport, storage and execution controls."""
        execution = {"model", "output_dir", "download_mode", "max_download", "model_store",
                     "model_store_max", "notify", "rebuild_image", "resume", "idle_debug", "sniff_iface",
                     "inference_config", "confirm_inference"}
        return tuple((item.name, getattr(self, item.name)) for item in fields(self)
                     if item.name not in execution)

    def with_preset(self, preset: str) -> RunConfig:
        """Change experimental parameters while retaining identity and user constraints."""
        template = self.from_preset(preset, model=self.model, runtime=self.runtime)
        values = {name: getattr(template, name) for name in PRESET_FIELDS}
        if self.resume and preset == "main":
            # Older runs may have recorded either default axis as the literal "auto".
            for name in ("cpus", "mems"):
                if getattr(self, name) == "auto":
                    values.pop(name)
        updated = replace(self, **values)
        if self.resume and self.experiment_parameters() != updated.experiment_parameters():
            raise RunConfigError([ConfigIssue("resume", message("续跑参数不能更改；请取消续跑并选择新的输出目录创建新实验。"))])
        return updated

    @classmethod
    def from_namespace(cls, args) -> RunConfig:
        from acprof.experiments.arguments import build_parser

        defaults = cls()
        names = {item.name for item in fields(cls)}
        public = {action.dest for action in build_parser()._actions} - {"help"}
        extra = dict(getattr(args, "extra_options", {}))
        extra.update({name: value for name, value in vars(args).items()
                      if name in public and name not in names})
        values = {field.name: getattr(args, field.name, getattr(defaults, field.name))
                      if getattr(args, field.name, None) is not None else getattr(defaults, field.name)
                      for field in fields(cls)}
        if not values["resume"] and values["profiling_mode"] in {"basic", "full", "auto"}:
            values["profiling_mode"] = "best_effort"
        values["extra_options"] = extra
        return cls(**values)

    def to_namespace(self):
        """Adapt the shared configuration to existing host APIs, without a second schema."""
        from acprof.experiments.arguments import arguments_from_options, build_parser

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
    def from_preset(cls, preset: str, *, model: str = "", runtime: str = "auto") -> RunConfig:
        """Start a new experiment with all defaults of the selected preset."""
        if preset == "main":
            return cls.main_matrix(model, runtime=runtime)
        if preset == "smoke":
            return cls.smoke(model, runtime=runtime)
        raise RunConfigError([ConfigIssue(None, message("未知预设：{0}", preset))])

    @classmethod
    def smoke(cls, model: str = "", *, runtime: str = "auto") -> RunConfig:
        """Return a short CPU workload with a single request window."""
        return cls(
            model=model,
            runtime=runtime,
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
            profiling_mode="best_effort",
            compute_profile_tool="none",
            execution_profile_tool="none",
            notify="none",
        )

    @classmethod
    def main_matrix(cls, model: str = "", *, runtime: str = "auto") -> RunConfig:
        """Expose Docker's editable axes and keep unresolved or Process limits automatic."""
        if runtime == "docker":
            cpus, mems = docker_resource_limits()
            return cls(model=model, runtime=runtime, cpus=cpus, mems=mems)
        return cls(model=model, runtime=runtime)

    def validate(self, *, project_dir: Path | None = None, allow_auto_profiling: bool = False) -> RunConfig:
        """Validate static inputs; automatic preflight may retain a pending profiling mode."""
        pending_profiling = allow_auto_profiling and self.profiling_mode == "auto"
        errors: list[ConfigIssue] = []
        task_ids = [item.strip() for item in self.tasks.split(",")]
        for name, identifier in (("task_run_id", self.task_run_id), *[("tasks", item) for item in task_ids]):
            if identifier and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identifier):
                errors.append(ConfigIssue(name, message("参数 {0} 必须是安全标识符", name)))
        if self.resume and self.rebuild_image:
            errors.append(ConfigIssue("rebuild_image", message("恢复实验必须使用原镜像；如需重建，请创建新实验。")))
        extra_options = _normalize_extra_options(self.extra_options, {item.name for item in fields(self)}, errors)
        self._validate_source(errors)
        from acprof.cpu_affinity import normalize_cpu_set
        try:
            cpuset_cpus = normalize_cpu_set(self.cpuset_cpus)
        except ValueError:
            cpuset_cpus = ""
            errors.append(ConfigIssue("cpuset_cpus", message("CPU 集合格式无效；示例：0-3,8")))
        model = self.model.strip()
        if not model:
            errors.append(ConfigIssue("model", message('模型 ID 不能为空')))

        # Keep issue collection in presentation order; only publish normalized
        # values after every domain has been checked without errors.
        cpus, mems = self._validate_resource_limits(errors)
        gpus = self._validate_runtime(errors, cpus, mems, extra_options, pending_profiling)
        self._validate_input_scales(errors)
        numbers = self._normalize_numbers(errors)
        task_family = self._validate_modes(errors, pending_profiling)
        self._validate_policies(errors, extra_options)
        workload_spec, model_spec, inference_config, confirm_inference = self._validate_paths(errors, project_dir)
        if errors:
            raise RunConfigError(errors)

        return replace(
            self,
            model=model,
            extra_options=extra_options,
            task=self.task.strip(),
            tasks=",".join(task_ids),
            task_family=task_family,
            backend=self.backend.strip(),
            cpus="auto" if self.cpus == "auto" else ",".join(str(value) for value in cpus),
            cpuset_cpus=cpuset_cpus,
            mems="auto" if self.mems == "auto" else ",".join(str(value) for value in mems),
            gpus=",".join(gpus),
            input_scales=",".join(csv_values(self.input_scales)),
            workload_spec=workload_spec,
            model_spec=model_spec,
            inference_config=inference_config,
            confirm_inference=confirm_inference,
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

    def _validate_source(self, errors: list[ConfigIssue]) -> None:
        if not isinstance(self.revision, str) or any(character.isspace() for character in self.revision):
            errors.append(ConfigIssue("revision", message("模型 revision 必须是无空白的 branch、tag 或 commit SHA")))
        from acprof.hub.huggingface.endpoints import HF_DOWNLOAD_MODES
        from acprof.hub.repository import MODEL_SOURCES
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

    def _validate_resource_limits(self, errors: list[ConfigIssue]) -> tuple[list[int], list[int]]:
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

        if self.runtime == "docker" and cpus and mems:
            from acprof.runtime.ports import service_port
            try:
                # Positive resources make the mapping monotonic; checking the largest
                # pair covers the full matrix without forbidding sequential port reuse.
                service_port(max(cpus), max(mems))
            except ValueError as exc:
                errors.extend(ConfigIssue(field, str(exc)) for field in ("cpus", "mems"))
        return cpus, mems

    def _validate_runtime(
        self, errors: list[ConfigIssue], cpus: list[int], mems: list[int],
        extra_options: dict, pending_profiling: bool,
    ) -> list[str]:
        gpus = csv_values(self.gpus)
        if self.runtime not in {"auto", "docker", "process"}:
            errors.append(ConfigIssue("runtime", message("执行运行时必须是 auto、docker 或 process")))
        if self.runtime == "process" and ((self.profiling_mode not in {"basic", "full", "best_effort"} and not pending_profiling)
                or self.compute_profile_tool != "none" or self.execution_profile_tool != "none"
                or (extra_options.get("dram_energy") == "required" and self.profiling_mode == "basic")):
            errors.append(ConfigIssue("runtime", message("Process 不支持所选的独立分析器或旧采集策略")))
        if not gpus:
            errors.append(ConfigIssue("gpus", message('GPU 模式不能为空')))
        else:
            try:
                gpus = [normalize_gpu_mode(item) for item in gpus]
            except ValueError:
                errors.append(ConfigIssue("gpus", message('GPU 模式只能包含 off 或 on')))
                gpus = []

        if len(cpus) != len(set(cpus)):
            errors.append(ConfigIssue("cpus", message('CPU 列表不能重复')))
        if len(mems) != len(set(mems)):
            errors.append(ConfigIssue("mems", message('内存列表不能重复')))
        if len(gpus) != len(set(gpus)):
            errors.append(ConfigIssue("gpus", message('GPU 模式不能重复')))
        return gpus

    def _validate_input_scales(self, errors: list[ConfigIssue]) -> None:
        try:
            _positive_float_csv(self.input_scales, message('输入规模'), "input_scales")
        except RunConfigError as exc:
            errors.extend(exc.issues)
        if self.input_scale_policy not in {"auto", "minimal"}:
            errors.append(ConfigIssue("input_scale_policy", message("输入规划必须是 auto 或 minimal")))

    def _normalize_numbers(self, errors: list[ConfigIssue]) -> dict[str, int | float]:
        numbers: dict[str, int | float] = {}
        numeric_fields = (
            ("batch_size", "Batch size", "Batch size 必须大于 0"),
            ("warmup", "Warmup", "Warmup 不能小于 0"),
            ("repeat", "Repeat", "Repeat 必须大于 0"),
            ("repeat_in_window", "每窗口请求数", "每窗口请求数不能小于 0"),
            ("repeat_window_seconds", "自动窗口秒数", "自动窗口秒数必须大于 0"),
            ("request_timeout_seconds", "单请求超时秒数", "单请求超时秒数必须是大于 0 的有限数字"),
            ("sample_hz", "采样频率", "采样频率必须大于 0"),
            ("raw_sample_limit", "raw_sample_limit", "原始采样上限必须是至少 2 的整数"),
            ("idle_seconds", "Idle 秒数", "Idle 秒数不能小于 0"),
            ("idle_cooldown_seconds", "Idle cooldown 秒数", "Idle cooldown 秒数不能小于 0"),
        )
        for field, label, reason in numeric_fields:
            rule = RUN_NUMBERS[field]
            try:
                numbers[field] = rule.parse(getattr(self, field), allow_text=True)
            except ConstraintViolation as exc:
                if exc.code == "minimum":
                    detail = message(reason)
                elif exc.code == "finite":
                    detail = message("{0}必须是有限数字", message(label))
                else:
                    detail = message("{0}必须是{1}", message(label), message("整数" if rule.whole else "数字"))
                errors.append(ConfigIssue(field, detail))
        return numbers

    def _validate_modes(self, errors: list[ConfigIssue], pending_profiling: bool) -> str:
        task_family = self.task_family.strip().lower()
        if task_family and task_family not in TASK_FAMILIES:
            errors.append(ConfigIssue("task_family", message('任务族必须是 nlp/cv/audio/timeseries/diffusion/multimodal/structured')))
        if self.compute_profile_tool not in COMPUTE_PROFILE_TOOLS:
            errors.append(ConfigIssue("compute_profile_tool", message('无效的计算分析器')))
        if self.profiling_mode not in {"full", "basic", "best_effort"} and not pending_profiling:
            errors.append(ConfigIssue("profiling_mode", message('无效的画像模式')))
        if self.sampling_policy not in {"burst", "skip"}:
            errors.append(ConfigIssue("sampling_policy", message("采样策略必须是 burst 或 skip")))
        if self.execution_profile_tool not in EXECUTION_PROFILE_TOOLS:
            errors.append(ConfigIssue("execution_profile_tool", message('无效的执行分析器')))
        if self.notify not in NOTIFY_MODES:
            errors.append(ConfigIssue("notify", message('无效的通知模式')))
        if not self.output_dir.strip():
            errors.append(ConfigIssue("output_dir", message('输出目录不能为空')))
        if not self.sniff_iface.strip():
            errors.append(ConfigIssue("sniff_iface", message('抓包网卡不能为空')))
        return task_family

    def _validate_policies(self, errors: list[ConfigIssue], extra_options: dict) -> None:
        from acprof.latency_slo import parse_latency_slo_rules
        from acprof.monitors.rapl_topology import dram_policy

        try:
            parse_latency_slo_rules([str(item) for item in (extra_options.get("latency_slo") or [])])
        except ValueError as exc:
            errors.append(ConfigIssue("latency_slo", message("延迟 SLO 规则无效：{0}", str(exc))))
        if not self.resume and self.profiling_mode == "best_effort" and extra_options.get("dram_energy") == "required" and self.idle_seconds == 0:
            errors.append(ConfigIssue("idle_seconds", message("必需 DRAM 有效能耗需要正的 Idle 窗口")))
        if self.profiling_mode in {"basic", "full", "best_effort"}:
            try:
                dram_policy(self.profiling_mode, extra_options.get("dram_energy") or "auto")
            except ValueError:
                errors.append(ConfigIssue("dram_energy", message("DRAM 能耗策略与画像模式不兼容")))

    def _validate_paths(
        self, errors: list[ConfigIssue], project_dir: Path | None,
    ) -> tuple[str, str, str, str]:
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
                from acprof.models.spec import read_model_spec
                read_model_spec(model_path)
            except (OSError, ValueError) as exc:
                errors.append(ConfigIssue("model_spec", message('模型接口声明无效：{0}；{1}', model_spec, str(exc))))

        inference_config = self.inference_config.strip()
        if inference_config and project_dir is not None:
            inference_path = Path(inference_config).expanduser()
            if not inference_path.is_absolute():
                inference_path = project_dir / inference_path
            if not inference_path.is_file():
                errors.append(ConfigIssue("inference_config", message(
                    "推理请求 JSON 不存在：{0}", inference_config,
                )))
        confirm_inference = self.confirm_inference.strip()
        if confirm_inference and re.fullmatch(r"[0-9a-f]{64}", confirm_inference) is None:
            errors.append(ConfigIssue("confirm_inference", message(
                "推理确认必须为审阅时显示的完整小写 SHA-256",
            )))
        return workload_spec, model_spec, inference_config, confirm_inference

    def result_dir(self, project_dir: Path) -> Path:
        output_root = Path(self.output_dir).expanduser()
        if not output_root.is_absolute():
            output_root = project_dir / output_root
        root = output_root / self.model.replace("/", "--")
        if self.task_run_id and self.tasks and self.tasks != "all" and "," not in self.tasks:
            return root / "tasks" / self.tasks / self.task_run_id
        return root

    def result_csv(self, project_dir: Path) -> Path:
        from acprof.results.layout import ArtifactLayout
        return ArtifactLayout.discover(self.result_dir(project_dir)).result_csv


def build_run_command(
    config: RunConfig,
    *,
    project_dir: Path,
    python_executable: str | Path = sys.executable,
) -> list[str]:
    """Build the installed CLI command without duplicating its work."""
    config = config.validate(project_dir=project_dir)
    from acprof.experiments.arguments import arguments_from_options

    return [
        *cli_command("run", python_executable=python_executable),
        *configuration_arguments(config, purpose="run"),
        *arguments_from_options(config.extra_options),
    ]
