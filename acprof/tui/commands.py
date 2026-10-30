"""Presentation commands for probe, plot, statistics and shortcuts."""
from __future__ import annotations

import csv
import shlex
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from acprof.experiments.configuration import ConfigIssue, RunConfig, RunConfigError, _csv_values
from acprof.experiments.options import configuration_arguments
from acprof.installation import cli_command
from acprof.tui.i18n import message_id


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
        raise RunConfigError([ConfigIssue(None, message_id("results.input.missing_file", path))])
    command = build_plot_command(path, project_dir=project_dir, python_executable=python_executable)
    return PendingLaunch(tuple(command), "plot", result_csv=str(path))


def prepare_stats(source: str, *, project_dir: Path, python_executable: Path) -> PendingLaunch:
    path = resolve_result_path(source, project_dir)
    if path.is_dir():
        path /= "result_layers.json"
    if not path.is_file() or (path.name != "result_layers.json" and path.suffix.lower() != ".csv"):
        raise RunConfigError([ConfigIssue(None, message_id("results.input.invalid_source"))])
    command = build_stats_command(path, project_dir=project_dir, python_executable=python_executable)
    return PendingLaunch(tuple(command), "stats", result_csv=str(path))


def html_report_inputs(source: str, *, project_dir: Path) -> tuple[Path, ...]:
    """Normalize the TUI's quoted-semicolon list to report's positional inputs.

    The experiment picker returns a layer manifest; the public report command
    documents directories and CSVs. Preserve all layers by passing its directory.
    This only validates paths; analysis belongs to the report subprocess.
    """
    try:
        values = next(csv.reader([source], delimiter=";", skipinitialspace=True, strict=True))
    except csv.Error as exc:
        raise RunConfigError([ConfigIssue(None, message_id('results.html.inputs_hint'))]) from exc
    paths = []
    for value in values:
        if not value.strip():
            continue
        path = resolve_result_path(value.strip(), project_dir).resolve()
        reference = path / "result_layers.json" if path.is_dir() else path
        if not reference.is_file() or (reference.name != "result_layers.json" and reference.suffix.lower() != ".csv"):
            raise RunConfigError([ConfigIssue(None, message_id('results.html.invalid_input', path))])
        normalized = reference.parent if reference.name == "result_layers.json" else reference
        if normalized in paths:
            raise RunConfigError([ConfigIssue(None, message_id('results.html.duplicate_input', normalized))])
        paths.append(normalized)
    if not paths:
        raise RunConfigError([ConfigIssue(None, message_id('results.html.input_required'))])
    return tuple(paths)


def prepare_html_report(source: str, *, project_dir: Path, python_executable: Path) -> PendingLaunch:
    """Prepare a new offline HTML artifact without importing analysis or Plotly."""
    from acprof.artifact_layout import ArtifactLayout

    paths = html_report_inputs(source, project_dir=project_dir)
    first = paths[0]
    layout = ArtifactLayout.discover(first) if first.is_dir() else ArtifactLayout.from_csv(first)
    output = layout.path("analysis") / f"report-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid4().hex[:12]}.html"
    command = (*cli_command("report", python_executable=python_executable),
               *(str(path) for path in paths), "--output", str(output))
    return PendingLaunch(command, "html-report", report_path=str(output))


def prepare_comparison(left: str, right: str, *, baseline: str, purpose: str,
                       project_dir: Path, python_executable: Path) -> PendingLaunch:
    """Read the public comparison command's stdout without a mandatory artifact."""
    if baseline not in {"left", "right"} or purpose not in {"same-hardware", "cross-hardware", "resource-scaling"}:
        raise RunConfigError([ConfigIssue(None, message_id('results.comparison.invalid_options'))])
    groups = []
    for text in (left, right):
        paths = [resolve_result_path(value.strip(), project_dir).resolve()
                 for value in next(csv.reader([text], delimiter=";", skipinitialspace=True)) if value.strip()]
        if not paths:
            raise RunConfigError([ConfigIssue(None, message_id('results.comparison.groups_required'))])
        for path in paths:
            source = path / "result_layers.json" if path.is_dir() else path
            if not source.is_file() or (source.name != "result_layers.json" and source.suffix.lower() != ".csv"):
                raise RunConfigError([ConfigIssue(None, message_id("results.input.invalid_source"))])
        groups.append(paths)
    if baseline == "right":
        groups.reverse()
    command = [*cli_command("compare", python_executable=python_executable)]
    for name, paths in zip(("--left", "--right"), groups, strict=True):
        for path in paths:
            command.extend((name, str(path)))
    command.extend(("--purpose", purpose))
    return PendingLaunch(tuple(command), "compare")


def prepare_profile(source: str, *, tools: str, dry_run: bool,
                    project_dir: Path, python_executable: Path) -> PendingLaunch:
    path = resolve_result_path(source, project_dir)
    if not path.is_dir():
        raise FileNotFoundError(message_id('results.input.directory_missing', path))
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
    if config.runtime == "process":
        raise RunConfigError([ConfigIssue("runtime", "Minimum-memory probe requires Docker; use a Process input-scale run")])
    config = config.validate(project_dir=project_dir)
    return [
        *cli_command("probe", python_executable=python_executable),
        *configuration_arguments(config, purpose="probe"),
    ]


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
    *,
    project_dir: Path,
    python_executable: str | Path = sys.executable,
) -> list[str]:
    return [*cli_command("stats", python_executable=python_executable),
            str(Path(result_csv).expanduser())]


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
        raise RunConfigError([ConfigIssue(None, message_id('validation.profiling.empty_tools'))])
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
        raise RunConfigError([ConfigIssue(None, message_id('validation.commands.invalid_syntax', exc))]) from exc
    if not parts or not parts[0].startswith("/"):
        raise RunConfigError([ConfigIssue(None, message_id('validation.commands.missing_slash'))])
    return parts[0][1:].lower(), parts[1:]
