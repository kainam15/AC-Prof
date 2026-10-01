"""Presentation commands for probe, plot, statistics and shortcuts."""
from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from acprof.experiment import RunConfig, RunConfigError, _csv_values
from acprof.installation import cli_command
from acprof.messages import message


@dataclass(frozen=True)
class PendingLaunch:
    command: tuple[str, ...]
    kind: str
    config: RunConfig | None = None
    result_dir: str = ""
    result_csv: str = ""


def resolve_result_path(source: str, project_dir: Path) -> Path:
    path = Path(source).expanduser()
    return path if path.is_absolute() else project_dir / path


def prepare_plot(source: str, *, project_dir: Path, python_executable: Path) -> PendingLaunch:
    path = resolve_result_path(source, project_dir)
    if not path.is_file():
        raise RunConfigError([message("结果 CSV 不存在：{0}", path)])
    command = build_plot_command(path, project_dir=project_dir, python_executable=python_executable)
    return PendingLaunch(tuple(command), "plot", result_csv=str(path))


def prepare_stats(source: str, *, project_dir: Path, python_executable: Path) -> PendingLaunch:
    from acprof.artifact_layout import ArtifactLayout
    path = resolve_result_path(source, project_dir)
    if path.is_dir():
        path /= "result_all.csv"
    if path.suffix.lower() != ".csv" or not path.is_file():
        raise RunConfigError([message("请选择已有结果 CSV 或包含 result_all.csv 的实验目录。")])
    output_dir = ArtifactLayout.discover(path.parent).path("analysis")
    command = build_stats_command(path, output_dir, project_dir=project_dir, python_executable=python_executable)
    return PendingLaunch(tuple(command), "stats", result_csv=str(path))


def prepare_profile(source: str, *, tools: str, dry_run: bool,
                    project_dir: Path, python_executable: Path) -> PendingLaunch:
    path = resolve_result_path(source, project_dir)
    if not path.is_dir():
        raise FileNotFoundError(message("结果目录不存在：{0}", path))
    command = build_profile_command(path, tools=tools, dry_run=dry_run,
                                    project_dir=project_dir, python_executable=python_executable)
    return PendingLaunch(tuple(command), "profile-dry-run" if dry_run else "profile", result_dir=str(path))


def build_probe_command(
    config: RunConfig,
    *,
    project_dir: Path,
    python_executable: str | Path = sys.executable,
) -> list[str]:
    """Build a one-request largest-scale diagnostic probe command."""
    config = config.validate(project_dir=project_dir)
    command = [
        *cli_command("probe", python_executable=python_executable),
        "--model",
        config.model,
        "--cpus",
        config.cpus,
        "--mems",
        config.mems,
        "--gpus",
        config.gpus,
        "--batch-size",
        str(config.batch_size),
        "--output-dir",
        config.output_dir,
    ]
    for option, value in (
        ("--task", config.task),
        ("--task-family", config.task_family),
        ("--backend", config.backend),
        ("--input-scales", config.input_scales),
        ("--workload-spec", config.workload_spec),
        ("--model-spec", config.model_spec),
    ):
        if value:
            command.extend((option, value))
    if config.skip_build:
        command.append("--skip-build")
    return command


def build_plot_command(
    result_csv: str | Path,
    *,
    project_dir: Path,
    python_executable: str | Path = sys.executable,
) -> list[str]:
    return [
        *cli_command("plot", python_executable=python_executable),
        str(Path(result_csv).expanduser()),
    ]


def build_stats_command(
    result_csv: str | Path,
    output_dir: str | Path,
    *,
    project_dir: Path,
    python_executable: str | Path = sys.executable,
) -> list[str]:
    return [*cli_command("stats", python_executable=python_executable),
            str(Path(result_csv).expanduser()), "--output-dir", str(output_dir)]


def build_profile_command(
    result_dir: str | Path,
    *,
    tools: str = "torch,ncu",
    dry_run: bool = True,
    project_dir: Path,
    python_executable: str | Path = sys.executable,
) -> list[str]:
    normalized_tools = ",".join(_csv_values(tools))
    if not normalized_tools:
        raise RunConfigError([message('补采工具不能为空')])
    command = [
        *cli_command("profile", python_executable=python_executable),
        str(Path(result_dir).expanduser()),
        "--tools",
        normalized_tools,
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def format_command(command: Sequence[str], *, project_dir: Path | None = None) -> str:
    """Return a shell-safe, readable command preview."""
    display = list(command)
    if project_dir is not None:
        project_dir = project_dir.resolve()
        for index, item in enumerate(display):
            # Only entry-point names can be shortened. Resolving every flag,
            # numeric value and model ID needlessly touches the filesystem on
            # each form edit (and may be slow for paths on remote storage).
            if Path(item).name not in {"run.py", "probe.py", "plot.py", "profile.py", "stats.py"}:
                continue
            try:
                item_path = Path(item).resolve()
            except (OSError, RuntimeError, ValueError):
                continue
            if item_path.parent == project_dir and item_path.name in {
                "run.py",
                "probe.py",
                "plot.py",
                "profile.py",
                "stats.py",
            }:
                display[index] = item_path.name
    return shlex.join(display)


def parse_slash_command(value: str) -> tuple[str, list[str]]:
    """Parse a slash command using shell quoting rules, without executing it."""
    try:
        parts = shlex.split(value.strip())
    except ValueError as exc:
        raise RunConfigError([message('命令格式错误：{0}', exc)]) from exc
    if not parts or not parts[0].startswith("/"):
        raise RunConfigError([message('快捷命令必须以 / 开头')])
    return parts[0][1:].lower(), parts[1:]
