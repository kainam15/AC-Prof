"""Offline checks for the independent VHS/Chrome CI smoke test."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import smoke_tui_png_environment as smoke

pytestmark = pytest.mark.visual_png


def test_browser_probe_keeps_sandbox_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[str] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.extend(argv)
        return subprocess.CompletedProcess(
            argv, 0, stdout="<!DOCTYPE html><html></html>", stderr=""
        )

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)
    smoke.check_chrome("/chrome", tmp_path)
    assert captured[0] == "/chrome"
    assert "--headless" in captured
    assert "--no-sandbox" not in captured
    assert "about:blank" in captured


def test_browser_probe_reports_actual_launch_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        smoke.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            argv, 1, stdout="", stderr="No usable sandbox"
        ),
    )
    with pytest.raises(RuntimeError, match="No usable sandbox"):
        smoke.check_chrome("/chrome", tmp_path)


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
