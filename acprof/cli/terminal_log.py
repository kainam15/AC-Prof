"""tmux pane capture and atomic terminal-log publication, owned by the CLI."""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from acprof.artifact_layout import ArtifactLayout
from acprof.host.command import run_command

TMUX_TERMINAL_LOG_FILENAME = "tmux_all.log"


def format_run_command(argv: list[str]) -> str:
    """Record the public CLI with shell-safe arguments for logs and metadata."""
    return shlex.join(["acprof", "run", *argv[1:]])


def start_terminal_log(
    output_dir: str,
    argv: list[str],
) -> tuple[str, str, str] | None:
    """Pipe all future output from the current tmux pane to a temporary log."""
    pane_id = os.environ.get("TMUX_PANE", "").strip()
    if not os.environ.get("TMUX") or not pane_id:
        return None

    try:
        pipe_status = run_command(
            [
                "tmux",
                "display-message",
                "-p",
                "-t",
                pane_id,
                "#{pane_pipe}",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f'[terminal-log][WARN] Cannot inspect tmux pane {pane_id}: {exc}', file=sys.stderr)
        return None

    if pipe_status.returncode != 0:
        detail = (pipe_status.stderr or pipe_status.stdout or "").strip()
        print(f"[terminal-log][WARN] Cannot inspect tmux pane {pane_id}: {detail or f'exit {pipe_status.returncode}'}", file=sys.stderr)
        return None
    if pipe_status.stdout.strip().lower() in {"1", "on", "true", "yes"}:
        print(f'[terminal-log][WARN] tmux pane {pane_id} already has an active pipe; leaving it unchanged and skipping automatic tmux_all.log', file=sys.stderr)
        return None

    os.makedirs(output_dir, exist_ok=True)
    log_path = str(ArtifactLayout.discover(output_dir).path(TMUX_TERMINAL_LOG_FILENAME))
    partial_path = f"{log_path}.part"
    try:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(partial_path, "w", encoding="utf-8") as f:
            f.write(f"$ {format_run_command(argv)}\n")
    except OSError as exc:
        print(f'[terminal-log][WARN] Cannot initialize {partial_path}: {exc}', file=sys.stderr)
        return None

    pipe_command = f"cat >> {shlex.quote(partial_path)}"
    try:
        pipe_result = run_command(
            [
                "tmux",
                "pipe-pane",
                "-O",
                "-t",
                pane_id,
                pipe_command,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f'[terminal-log][WARN] Cannot start tmux pane logging: {exc}', file=sys.stderr)
        return None

    if pipe_result.returncode != 0:
        detail = (pipe_result.stderr or pipe_result.stdout or "").strip()
        print(f"[terminal-log][WARN] Cannot start tmux pane logging: {detail or f'exit {pipe_result.returncode}'}", file=sys.stderr)
        return None

    print(f"[terminal-log] Recording tmux pane {pane_id}: {log_path}")
    return pane_id, partial_path, log_path


def stop_terminal_log(
    terminal_log: tuple[str, str, str],
) -> bool:
    """Stop the pane pipe and atomically publish the completed terminal log."""
    pane_id, partial_path, log_path = terminal_log
    try:
        close_result = run_command(
            ["tmux", "pipe-pane", "-t", pane_id],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f'[terminal-log][WARN] Cannot stop tmux pane logging; partial log remains at {partial_path}: {exc}', file=sys.stderr)
        return False

    if close_result.returncode != 0:
        detail = (close_result.stderr or close_result.stdout or "").strip()
        print(f"[terminal-log][WARN] Cannot stop tmux pane logging; partial log remains at {partial_path}: {detail or f'exit {close_result.returncode}'}", file=sys.stderr)
        return False

    try:
        os.replace(partial_path, log_path)
    except OSError as exc:
        print(f'[terminal-log][WARN] Cannot finalize {log_path}; partial log remains at {partial_path}: {exc}', file=sys.stderr)
        return False

    print(f"[terminal-log] Saved terminal display: {log_path}")
    return True
