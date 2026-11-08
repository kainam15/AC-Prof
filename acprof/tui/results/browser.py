"""Bounded local browser dispatch; no servers, downloads or desktop configuration."""
from __future__ import annotations

import base64
import re
import shutil
import subprocess
import webbrowser
from pathlib import Path, PureWindowsPath

from acprof import platform as host_platform
from acprof.tui.i18n import message_id

_OPEN_TIMEOUT = 5


def _run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            arguments, stdin=subprocess.DEVNULL, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=_OPEN_TIMEOUT,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        # The bridge may already have submitted the request. Do not retry and
        # risk opening duplicate windows, or call a timeout confirmed failure.
        raise RuntimeError(message_id("results.html.browser_timeout", Path(arguments[0]).name)) from exc


def _failure(name: str, result: subprocess.CompletedProcess[str]) -> str:
    detail = " ".join(result.stderr.split())[:500]
    return f"{name}: exit {result.returncode}" + (f" ({detail})" if detail else "")


def _powershell_open(path: Path, wslpath: str, powershell: str) -> None:
    converted = _run([wslpath, "-w", str(path)])
    if converted.returncode:
        raise OSError(_failure("wslpath", converted))
    target = converted.stdout.rstrip("\r\n")
    windows_path = PureWindowsPath(target)
    if (not windows_path.is_absolute() or windows_path.suffix.lower() != ".html"
            or any(character in target for character in "\x00\r\n")):
        raise OSError("wslpath returned an invalid Windows HTML path")
    # A filename is data, never PowerShell source. This also avoids the second
    # shell parsing layer of cmd.exe /c start and handles Unicode and quotes.
    payload = base64.b64encode(target.encode("utf-8")).decode("ascii")
    script = (
        "$ErrorActionPreference = 'Stop'; "
        "$ProgressPreference = 'SilentlyContinue'; "
        "$target = [Text.Encoding]::UTF8.GetString("
        f"[Convert]::FromBase64String('{payload}')); "
        "Start-Process -FilePath $target -ErrorAction Stop | Out-Null"
    )
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    opened = _run([powershell, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded])
    if opened.returncode:
        raise OSError(_failure("PowerShell", opened))


def _open_wsl(path: Path) -> str:
    errors: list[str] = []
    viewer = shutil.which("wslview")
    if viewer:
        # Some wslview versions interpolate paths into a shell command. Restrict
        # that route to simple paths; use data-only PowerShell for metacharacters.
        if re.fullmatch(r"[\w /.-]+", str(path)):
            try:
                result = _run([viewer, "--skip-validation-check", str(path)])
            except OSError as exc:
                errors.append(f"wslview: {exc}")
            else:
                if result.returncode == 0:
                    return "wslview"
                errors.append(_failure("wslview", result))
        else:
            errors.append("wslview: path requires literal-safe PowerShell dispatch")
    else:
        errors.append("wslview: not found")
    wslpath, powershell = shutil.which("wslpath"), shutil.which("powershell.exe")
    if wslpath and powershell:
        try:
            _powershell_open(path, wslpath, powershell)
            return "PowerShell"
        except OSError as exc:
            errors.append(str(exc))
    else:
        errors.extend(f"{name}: not found" for name, executable in (
            ("wslpath", wslpath), ("powershell.exe", powershell),
        ) if not executable)
    # Do not hand file:///home/... to Windows or silently fall back to a Linux
    # terminal browser. Keep the file intact and provide a manual access route.
    raise RuntimeError(message_id("results.html.wsl_browser_unavailable", "; ".join(errors)))


def open_report_in_browser(path: Path) -> str:
    """Submit a request and return its backend, not a claim of page visibility.

    The caller validates the resolved local HTML artifact before dispatch.
    Recheck policy here so non-UI callers cannot open an SSH server's desktop.
    """
    if not host_platform.can_open_local_browser():
        raise RuntimeError(message_id("results.html.remote_open_instructions", path))
    if host_platform.is_wsl():
        return _open_wsl(path)
    if not webbrowser.open(path.as_uri(), new=2):
        raise RuntimeError(message_id("results.html.browser_unavailable"))
    return "webbrowser"
