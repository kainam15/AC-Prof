"""Offline checks for the independent VHS/Chrome CI smoke test."""

from __future__ import annotations

import io
import signal
import subprocess
from pathlib import Path

import pytest

from scripts import smoke_tui_png_environment as smoke

pytestmark = pytest.mark.visual_png


class _FakeChrome:
    pid = 4201

    def __init__(self, returncode: int | None = None):
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, *, timeout: float) -> int:
        return self.returncode or 0


def test_browser_probe_keeps_sandbox_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[str] = []
    signals: list[int] = []

    def fake_popen(argv: list[str], **kwargs: object) -> _FakeChrome:
        captured.extend(argv)
        profile = tmp_path / "chrome-profile"
        profile.mkdir()
        (profile / "DevToolsActivePort").write_text("45123\n/devtools/browser/fixture\n")
        assert kwargs["start_new_session"] is True
        return _FakeChrome()

    monkeypatch.setattr(smoke.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(smoke.os, "killpg", lambda pid, sig: signals.append(sig))
    monkeypatch.setattr(
        smoke.urllib.request,
        "urlopen",
        lambda url, **kwargs: io.BytesIO(
            b'{"webSocketDebuggerUrl":"ws://127.0.0.1:45123/devtools/browser/test"}'
        ),
    )
    smoke.check_chrome("/chrome", tmp_path)
    assert captured[0] == "/chrome"
    assert "--headless" in captured
    assert "--disable-dev-shm-usage" in captured
    assert "--ozone-platform=headless" in captured
    assert "--no-sandbox" not in captured
    assert "--remote-debugging-port=0" in captured
    assert "about:blank" in captured
    assert signals == [signal.SIGTERM]


def test_browser_probe_reports_actual_launch_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    signals: list[int] = []

    def fake_popen(argv: list[str], **kwargs: object) -> _FakeChrome:
        stderr = kwargs["stderr"]
        stderr.write(b"No usable sandbox")
        stderr.flush()
        return _FakeChrome(returncode=1)

    monkeypatch.setattr(smoke.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(smoke.os, "killpg", lambda pid, sig: signals.append(sig))
    with pytest.raises(RuntimeError, match="No usable sandbox"):
        smoke.check_chrome("/chrome", tmp_path)
    assert signals == [signal.SIGTERM]


def test_browser_probe_timeout_cleans_child_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    signals: list[int] = []
    monkeypatch.setattr(smoke, "CHROME_STARTUP_TIMEOUT_SECONDS", 0)
    monkeypatch.setattr(smoke.subprocess, "Popen", lambda *args, **kwargs: _FakeChrome())
    monkeypatch.setattr(smoke.os, "killpg", lambda pid, sig: signals.append(sig))
    with pytest.raises(RuntimeError, match="DevTools endpoint unavailable"):
        smoke.check_chrome("/chrome", tmp_path)
    assert signals == [signal.SIGTERM]


def test_vhs_smoke_fails_when_frames_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        smoke.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, stdout="", stderr=""),
    )
    with pytest.raises(RuntimeError, match="produced no terminal PNG"):
        smoke.check_vhs("/vhs", tmp_path)


def test_vhs_smoke_checks_actual_png(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        tape = (tmp_path / "smoke.tape").read_text()
        assert "Set Columns 80" in tape
        assert 'Set FontFamily "Noto Sans Mono CJK SC"' in tape
        frames = tmp_path / "frames"
        frames.mkdir()
        (frames / "frame-text-000001.png").write_bytes(smoke.PNG_SIGNATURE + b"X" * 1200)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)
    assert smoke.check_vhs("/vhs", tmp_path).is_file()


def test_vhs_smoke_rejects_invalid_png(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        frames = tmp_path / "frames"
        frames.mkdir()
        (frames / "frame-text-000001.png").write_bytes(b"not-png" * 200)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="invalid or empty PNG"):
        smoke.check_vhs("/vhs", tmp_path)
