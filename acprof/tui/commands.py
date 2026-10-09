"""Presentation commands for probe, plot, statistics and shortcuts."""
from __future__ import annotations

import csv
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from acprof.experiment import ConfigIssue, RunConfig, RunConfigError, _csv_values
from acprof.installation import cli_command
from acprof.messages import message


@dataclass(frozen=True)
class OperationState:
    """One policy for buttons, bindings, commands and confirmation callbacks."""

    process: bool = False
    stoppable: bool = False
    checking: bool = False
    reading: bool = False
    maintenance: bool = False
    configuring: bool = False
    measuring: bool = False
    closing: bool = False

    @property
    def busy(self) -> bool:
        return any((self.process, self.reading, self.maintenance,
                    self.configuring, self.measuring, self.closing))

    def allows(self, operation: str) -> bool:
        if self.closing:
            return False
        if operation == "stop":
            return self.stoppable
        if operation == "quit":
            return not (self.process or self.maintenance)
        if operation == "configure":
            return not self.busy
        if operation in {"summary", "report"}:
            # Reading another selection is safe, but the old thread remains
            # owned until it returns, keeping all measurement starts locked.
            return not any((self.process, self.checking, self.maintenance,
                            self.configuring, self.measuring))
        return not (self.busy or self.checking)


@dataclass(frozen=True)
class PendingLaunch:
    command: tuple[str, ...]
    kind: str
    config: RunConfig | None = None
    result_dir: str = ""
    result_csv: str = ""
    report_path: str = ""


def resolve_result_path(source: str, project_dir: Path) -> Path:
    path = Path(source).expanduser()
    return path if path.is_absolute() else project_dir / path


def prepare_plot(source: str, *, project_dir: Path, python_executable: Path) -> PendingLaunch:
    path = resolve_result_path(source, project_dir)
    if path.is_dir():
        path /= "result_layers.json"
    if not path.is_file() or (path.name != "result_layers.json" and path.suffix.lower() != ".csv"):
        raise RunConfigError([ConfigIssue(None, message("结果文件不存在：{0}", path))])
    command = build_plot_command(path, project_dir=project_dir, python_executable=python_executable)
    return PendingLaunch(tuple(command), "plot", result_csv=str(path))


def prepare_stats(source: str, *, project_dir: Path, python_executable: Path) -> PendingLaunch:
    from acprof.artifact_layout import ArtifactLayout
    path = resolve_result_path(source, project_dir)
    if path.is_dir():
        path /= "result_layers.json"
    if not path.is_file() or (path.name != "result_layers.json" and path.suffix.lower() != ".csv"):
        raise RunConfigError([ConfigIssue(None, message("请选择有效的分层结果目录、清单或独立 CSV。"))])
    output_dir = ArtifactLayout.discover(path.parent).path("analysis")
    command = build_stats_command(path, output_dir, project_dir=project_dir, python_executable=python_executable)
    return PendingLaunch(tuple(command), "stats", result_csv=str(path))


def prepare_comparison(left: str, right: str, *, baseline: str, purpose: str,
                       project_dir: Path, python_executable: Path) -> PendingLaunch:
    """Reuse the public comparison command and preserve its baseline direction."""
    from acprof.artifact_layout import ArtifactLayout
    if baseline not in {"left", "right"} or purpose not in {"same-hardware", "cross-hardware", "resource-scaling"}:
        raise RunConfigError([ConfigIssue(None, message("请选择有效的比较基线和用途。"))])
    groups = []
    for text in (left, right):
        paths = [resolve_result_path(value.strip(), project_dir).resolve()
                 for value in next(csv.reader([text], delimiter=";", skipinitialspace=True)) if value.strip()]
        if not paths:
            raise RunConfigError([ConfigIssue(None, message("请选择左右两组实验目录。"))])
        for path in paths:
            source = path / "result_layers.json" if path.is_dir() else path
            if not source.is_file() or (source.name != "result_layers.json" and source.suffix.lower() != ".csv"):
                raise RunConfigError([ConfigIssue(None, message("请选择有效的分层结果目录、清单或独立 CSV。"))])
        groups.append(paths)
    if baseline == "right":
        groups.reverse()
    first = groups[0][0]
    layout = ArtifactLayout.discover(first) if first.is_dir() else ArtifactLayout.from_csv(first)
    output = layout.path("analysis") / f"experiment-comparison-{datetime.now(timezone.utc):%Y%m%d-%H%M%S-%f}.json"
    command = [*cli_command("compare", python_executable=python_executable)]
    for name, paths in zip(("--left", "--right"), groups):
        for path in paths:
            command.extend((name, str(path)))
    command.extend(("--purpose", purpose, "--output", str(output)))
    return PendingLaunch(tuple(command), "compare", report_path=str(output))


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
        ("--revision", config.revision),
        ("--model-source", config.model_source),
        ("--task", config.task),
        ("--task-family", config.task_family),
        ("--backend", config.backend),
        ("--input-scales", config.input_scales),
        ("--workload-spec", config.workload_spec),
        ("--model-spec", config.model_spec),
        ("--download-mode", config.download_mode),
        ("--max-download", config.max_download),
        ("--model-store", config.model_store),
        ("--model-store-max", config.model_store_max),
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
        raise RunConfigError([ConfigIssue(None, message('补采工具不能为空'))])
    command = [
        *cli_command("profile", python_executable=python_executable),
        str(Path(result_dir).expanduser()),
        "--tools",
        normalized_tools,
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def format_command(command: Sequence[str]) -> str:
    """Return a shell-safe, readable command preview."""
    display = list(command)
    # Child processes use this interpreter; users copy the public console command.
    if display[1:4] == ["-u", "-m", "acprof"]:
        display = ["acprof", *display[4:]]
    elif getattr(sys, "frozen", False) and display and display[0] == sys.executable:
        display = ["acprof", *display[1:]]
    return shlex.join(display)


def parse_slash_command(value: str) -> tuple[str, list[str]]:
    """Parse a slash command using shell quoting rules, without executing it."""
    try:
        parts = shlex.split(value.strip())
    except ValueError as exc:
        raise RunConfigError([ConfigIssue(None, message('命令格式错误：{0}', exc))]) from exc
    if not parts or not parts[0].startswith("/"):
        raise RunConfigError([ConfigIssue(None, message('快捷命令必须以 / 开头'))])
    return parts[0][1:].lower(), parts[1:]
