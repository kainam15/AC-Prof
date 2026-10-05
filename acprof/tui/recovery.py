"""Read-only recovery review and explicit new experiment destinations."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from acprof.artifact_layout import ArtifactLayout
from acprof.experiment import RunConfig, build_run_command
from acprof.host.run_state import (
    check_result_directory_idle,
    recovery_phase,
    run_options,
    validate_resume,
)
from acprof.messages import join_messages, message
from acprof.model_evidence import pinned_revision
from acprof.run_args import build_parser
from acprof.tui.commands import PendingLaunch
from acprof.tui.experiment_catalog import (
    ExperimentRecord,
    config_from_record,
    new_experiment_config,
    read_artifact,
    resume_command,
    scan_experiments,
)
from acprof.tui.i18n import error_message


@dataclass(frozen=True)
class RecoveryReview:
    directory: Path
    phase: str
    detail: str
    resume: PendingLaunch | None = None
    restart: PendingLaunch | None = None
    record: ExperimentRecord | None = None


def _check_writer(layout: ArtifactLayout) -> None:
    check_result_directory_idle(layout.root)


def review_recovery(pending: PendingLaunch, *, project_dir: Path, python_executable: Path,
                    record: ExperimentRecord | None = None) -> RecoveryReview | None:
    """Return None only for a new, writable-by-protocol experiment destination."""
    config = pending.config
    if config is None:
        raise ValueError("recovery requires an experiment configuration")
    directory = config.result_dir(project_dir).resolve()
    issues = []
    try:
        layout = ArtifactLayout.discover(directory)
        if record is None and not config.resume:
            try:
                ArtifactLayout.for_new_run(directory).check_new_run()
                return None
            except ValueError:
                pass  # Existing evidence needs a user-visible recovery decision.
        if record is None:
            catalog = scan_experiments([directory], max_depth=0)
            record = catalog.records[0] if catalog.records else None
            issues.extend(catalog.warnings)
        elif read_artifact(layout.path("run_state.json")) != record.state:
            raise ValueError(message("实验记录已变化，请重新选择后检查恢复条件。"))
    except (OSError, ValueError) as exc:
        issues.append(error_message(exc))
        record = None

    phase = "unknown"
    resume = None
    if record is not None:
        phase = recovery_phase(directory, record.state)
        if record.status == "complete":
            issues.append(message("原实验记录为已完成；可以查看结果或创建新实验。"))
        else:
            try:
                _check_writer(ArtifactLayout.discover(directory))
                command = resume_command(record, python_executable=python_executable)
                args = build_parser().parse_args(command[command.index("--model"):])
                normalized = RunConfig.from_namespace(args).validate(project_dir=project_dir)
                for name, value in asdict(normalized).items():
                    setattr(args, name, value)
                phase = validate_resume(directory, record.state, project_dir=project_dir, options=run_options(args))
                restored = config_from_record(record, reuse=False).validate(project_dir=project_dir)
                # Notification preferences do not belong to the frozen measurement identity.
                restored = replace(restored, notify=config.notify)
                command = (*command, "--notify", config.notify)
                resume = PendingLaunch(command, "run", restored, result_dir=str(directory))
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                issues.append(error_message(exc))
    else:
        issues.append(message("没有可还原的实验状态；原目录保留，重新运行会创建新实验。"))

    fresh = new_experiment_config(config, directory.parent)
    if (record is not None and pinned_revision(record.revision)
            and config.model == record.options.get("model")
            and config.revision == (record.options.get("revision") or "")):
        fresh = replace(fresh, revision=record.revision)
    restart = None
    try:
        command = build_run_command(fresh, project_dir=project_dir, python_executable=python_executable)
        restart = PendingLaunch(tuple(command), "run", fresh, result_dir=str(fresh.result_dir(project_dir)))
    except ValueError as exc:
        issues.append(error_message(exc))
    summary = {"preparation": "上次准备未完成，尚未进入正式采集。",
               "measurement": "原目录已有运行环境或测量记录；续跑须核对已有 case。",
               "complete": "原实验记录为已完成。", "unknown": "无法确认原实验的恢复状态。"}[phase]
    details = [message(summary), *issues]
    if resume is not None:
        details.append(message("恢复使用原实验的完整参数；启动时仍会核对镜像与实际运行条件。"))
    if restart is not None:
        details.append(message("新实验使用当前配置，原目录与历史记录保留。\n新输出目录：{0}", fresh.output_dir))
    return RecoveryReview(directory, phase, join_messages("\n\n", details), resume, restart, record)
