"""Browser dispatch tests never start a real desktop or Windows process."""
import base64
import re
import subprocess
from unittest.mock import Mock

import pytest

from acprof import platform as host_platform
from acprof.tui.i18n import error_message
from acprof.tui.results import browser
from acprof.tui.results.report_files import open_html_file


@pytest.fixture
def report(tmp_path):
    path = tmp_path / "中文 report.html"
    path.write_text("<!doctype html><p>offline report</p>", encoding="utf-8")
    return path


@pytest.fixture
def dispatch(monkeypatch):
    for key in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY", "WSL_DISTRO_NAME", "WSL_INTEROP",
                "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(host_platform.platform, "system", lambda: "Linux")
    monkeypatch.setattr(host_platform, "is_wsl", lambda **_: True)
    tools = {name: "/tools/" + name for name in ("wslview", "wslpath", "powershell.exe")}
    monkeypatch.setattr(browser.shutil, "which", tools.get)
    run = Mock()
    monkeypatch.setattr(browser.subprocess, "run", run)
    native = Mock(return_value=True)
    monkeypatch.setattr(browser.webbrowser, "open", native)
    return tools, run, native


def completed(code=0, stdout="", stderr=""):
    return subprocess.CompletedProcess([], code, stdout, stderr)


def test_wsl_can_request_host_browser_without_linux_gui():
    assert host_platform.can_open_local_browser(system="Linux", environ={"WSL_DISTRO_NAME": "Ubuntu"})


@pytest.mark.parametrize(("release", "environment", "files", "expected"), [
    ("4.4.0-Microsoft", {}, (), True),
    ("6.6.87.2-microsoft-standard-WSL2", {}, (), True),
    ("custom", {"WSL_DISTRO_NAME": "Ubuntu"}, (), True),
    ("custom", {"WSL_INTEROP": "/run/WSL/1_interop"}, (), True),
    ("custom", {}, ("/run/WSL",), True),
    ("custom", {}, ("/proc/sys/fs/binfmt_misc/WSLInterop",), True),
    ("6.14.0-generic", {}, (), False),
])
def test_wsl_detection_is_shared_but_not_a_collection_tier(release, environment, files, expected):
    assert host_platform.is_wsl(system="Linux", release=release, environ=environment,
                                exists=lambda path: path in files) is expected


@pytest.mark.parametrize(("system", "environment", "allowed"), [
    ("Windows", {}, True), ("Darwin", {}, True),
    ("Linux", {"DISPLAY": ":0"}, True),
    ("Linux", {"WAYLAND_DISPLAY": "wayland-0"}, True),
    ("Linux", {}, False), ("Other", {}, False),
])
def test_local_browser_policy(monkeypatch, system, environment, allowed):
    monkeypatch.setattr(host_platform, "is_wsl", lambda **_: False)
    assert host_platform.can_open_local_browser(system=system, environ=environment) is allowed


@pytest.mark.parametrize("key", ["SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"])
def test_ssh_takes_precedence_over_wsl_and_display(monkeypatch, key):
    probe = Mock(side_effect=AssertionError("SSH must short circuit local detection"))
    monkeypatch.setattr(host_platform, "is_wsl", probe)
    assert not host_platform.can_open_local_browser(
        system="Linux", environ={key: "remote", "WSL_DISTRO_NAME": "Ubuntu", "DISPLAY": ":0"},
    )
    probe.assert_not_called()


def test_wslview_is_preferred_and_offline(report, dispatch):
    _, run, native = dispatch
    run.return_value = completed()
    before = report.read_bytes()
    assert open_html_file(report) == "wslview"
    run.assert_called_once_with(
        ["/tools/wslview", "--skip-validation-check", str(report)],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=5, shell=False,
    )
    native.assert_not_called()
    assert report.read_bytes() == before


@pytest.mark.parametrize("first_error", [None, OSError("interop disabled"), completed(3, stderr="cannot open")])
@pytest.mark.parametrize("target", [r"C:\Users\name\中文 report.html", r"\\wsl.localhost\Ubuntu\home\name\report.html"])
def test_powershell_fallback_converts_wsl_paths(report, dispatch, first_error, target):
    tools, run, native = dispatch
    if first_error is None:
        del tools["wslview"]
    run.side_effect = ([] if first_error is None else [first_error]) + [completed(stdout=target + "\n"), completed()]
    assert open_html_file(report) == "PowerShell"
    conversion, launch = run.call_args_list[-2:]
    assert conversion.args[0] == ["/tools/wslpath", "-w", str(report)]
    assert launch.args[0][:4] == ["/tools/powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand"]
    script = base64.b64decode(launch.args[0][4]).decode("utf-16-le")
    payload = re.search(r"FromBase64String\('([^']+)'\)", script).group(1)
    assert base64.b64decode(payload).decode("utf-8") == target
    assert "Start-Process -FilePath $target -ErrorAction Stop" in script
    native.assert_not_called()


@pytest.mark.parametrize("name", ["quote' report.html", 'double"quote.html', "dollar$(touch pwned).html",
                                  "percent%20report.html", "semi;& report.html", "back`tick.html"])
def test_metacharacters_are_data_never_shell_source(tmp_path, dispatch, name):
    _, run, native = dispatch
    path = tmp_path / name
    path.write_text("report")
    target = "C:\\reports\\" + name
    run.side_effect = [completed(stdout=target + "\n"), completed()]
    assert open_html_file(path) == "PowerShell"
    assert run.call_args_list[0].args[0][0] == "/tools/wslpath"
    script = base64.b64decode(run.call_args_list[-1].args[0][4]).decode("utf-16-le")
    assert name not in script
    payload = re.search(r"FromBase64String\('([^']+)'\)", script).group(1)
    assert base64.b64decode(payload).decode("utf-8") == target
    assert all(call.kwargs["shell"] is False for call in run.call_args_list)
    native.assert_not_called()


@pytest.mark.parametrize("failed_stage", ["wslview", "wslpath", "powershell.exe"])
def test_timeout_is_unknown_and_never_retried(report, dispatch, failed_stage):
    tools, run, native = dispatch
    prior = []
    if failed_stage != "wslview":
        del tools["wslview"]
    if failed_stage == "powershell.exe":
        prior.append(completed(stdout="C:\\report.html\n"))
    run.side_effect = [*prior, subprocess.TimeoutExpired([failed_stage], 5)]
    with pytest.raises(RuntimeError) as exc:
        open_html_file(report)
    assert "超时" in error_message(exc.value)
    assert run.call_count == len(prior) + 1
    native.assert_not_called()


@pytest.mark.parametrize("present", [(), ("wslpath",), ("powershell.exe",)])
def test_missing_bridges_have_actionable_error(report, dispatch, present):
    tools, run, native = dispatch
    for name in tuple(tools):
        if name not in present:
            del tools[name]
    with pytest.raises(RuntimeError) as exc:
        open_html_file(report)
    text = error_message(exc.value)
    assert "WSL" in text and "PATH" in text and "not found" in text
    run.assert_not_called()
    native.assert_not_called()
    assert report.is_file()


@pytest.mark.parametrize("converted", [completed(1, stderr="conversion failed"), completed(stdout=""),
                                        completed(stdout="/home/report.html\n"), completed(stdout="C:\\report.exe\n"),
                                        completed(stdout="C:\\first.html\nC:\\other.html\n")])
def test_invalid_conversion_does_not_launch_windows(report, dispatch, converted):
    tools, run, native = dispatch
    del tools["wslview"]
    run.return_value = converted
    with pytest.raises(RuntimeError) as exc:
        open_html_file(report)
    assert "wslpath" in error_message(exc.value)
    assert run.call_count == 1
    native.assert_not_called()


def test_all_failed_backends_keep_diagnostics(report, dispatch):
    _, run, native = dispatch
    run.side_effect = [completed(7, stderr="wslview rejected"), completed(stdout="C:\\report.html\n"),
                       completed(9, stderr="PowerShell rejected")]
    with pytest.raises(RuntimeError) as exc:
        open_html_file(report)
    text = error_message(exc.value)
    assert "wslview: exit 7" in text and "PowerShell: exit 9" in text
    assert "PowerShell rejected" in text
    native.assert_not_called()


@pytest.mark.parametrize("environment", [{}, {"SSH_CONNECTION": "remote", "DISPLAY": ":0"},
                                         {"SSH_TTY": "pts/1", "WSL_DISTRO_NAME": "Ubuntu"}])
def test_direct_file_api_never_opens_headless_or_ssh(report, dispatch, monkeypatch, environment):
    _, run, native = dispatch
    monkeypatch.setattr(host_platform, "is_wsl", lambda **_: bool(environment.get("WSL_DISTRO_NAME")))
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(RuntimeError) as exc:
        open_html_file(report)
    assert "SFTP" in error_message(exc.value)
    run.assert_not_called()
    native.assert_not_called()


@pytest.mark.parametrize("system", ["Windows", "Darwin", "Linux"])
@pytest.mark.parametrize("outcome", [True, False, OSError("missing browser")])
def test_native_desktops_keep_webbrowser(report, dispatch, monkeypatch, system, outcome):
    _, run, native = dispatch
    monkeypatch.setattr(host_platform.platform, "system", lambda: system)
    monkeypatch.setattr(host_platform, "is_wsl", lambda **_: False)
    monkeypatch.setenv("DISPLAY", ":0")
    if isinstance(outcome, Exception):
        native.side_effect = outcome
    else:
        native.return_value = outcome
    if outcome is True:
        assert open_html_file(report) == "webbrowser"
    else:
        with pytest.raises((RuntimeError, OSError)):
            open_html_file(report)
    native.assert_called_once_with(report.as_uri(), new=2)
    run.assert_not_called()


@pytest.mark.parametrize("kind", ["missing", "directory", "extension", "symlink"])
def test_invalid_report_never_dispatches(tmp_path, dispatch, kind):
    _, run, native = dispatch
    path = tmp_path / "invalid.html"
    if kind == "directory":
        path.mkdir()
    elif kind in {"extension", "symlink"}:
        target = tmp_path / "program.exe"
        target.write_text("not HTML")
        if kind == "symlink":
            path.symlink_to(target)
        else:
            path = target
    with pytest.raises(ValueError):
        open_html_file(path)
    run.assert_not_called()
    native.assert_not_called()
