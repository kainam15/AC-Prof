"""Check Chrome and VHS screenshot startup without launching the AC-Prof TUI."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from contextlib import suppress
from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
CHROME_STARTUP_TIMEOUT_SECONDS = 30


def _run_checked(argv: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Cannot start {argv[0]}: {exc}") from exc
    if result.returncode:
        raise RuntimeError(
            f"{argv[0]} exited {result.returncode}.\n"
            f"stderr:\n{result.stderr[-4000:]}\nstdout:\n{result.stdout[-1000:]}"
        )
    return result


def _stop_browser_group(browser: subprocess.Popen[bytes]) -> None:
    """Terminate Chrome and zygotes before the temporary profile is deleted."""
    with suppress(ProcessLookupError):
        os.killpg(browser.pid, signal.SIGTERM)
    try:
        browser.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(browser.pid, signal.SIGKILL)
        browser.wait(timeout=5)


def check_chrome(chrome: str, temp: Path) -> None:
    """Probe the DevTools endpoint VHS needs, without disabling Chrome's sandbox."""
    profile = temp / "chrome-profile"
    endpoint = profile / "DevToolsActivePort"
    stderr_path = temp / "chrome-stderr.log"
    argv = [
        chrome,
        "--headless",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--ozone-platform=headless",
        "--no-first-run",
        "--no-default-browser-check",
        "--remote-debugging-address=127.0.0.1",
        "--remote-debugging-port=0",
        f"--user-data-dir={profile}",
        "about:blank",
    ]
    with (temp / "chrome-stdout.log").open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            browser = subprocess.Popen(argv, stdout=stdout, stderr=stderr, start_new_session=True)
        except OSError as exc:
            raise RuntimeError(f"Unable to launch Chrome: {exc}") from exc
        reason = f"not ready within {CHROME_STARTUP_TIMEOUT_SECONDS}s"
        try:
            deadline = time.monotonic() + CHROME_STARTUP_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                if browser.poll() is not None:
                    reason = f"Chrome exited {browser.returncode}"
                    break
                if endpoint.is_file():
                    try:
                        port = int(endpoint.read_text(encoding="utf-8").splitlines()[0])
                        with urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/json/version", timeout=1
                        ) as response:
                            version = json.load(response)
                        if version.get("webSocketDebuggerUrl"):
                            return
                    except (OSError, ValueError, IndexError) as exc:
                        reason = f"DevTools not responding yet: {exc}"
                time.sleep(0.1)
        finally:
            _stop_browser_group(browser)
    raise RuntimeError(
        f"Chrome DevTools endpoint unavailable ({reason}).\n"
        f"Chrome stderr:\n{stderr_path.read_text(encoding='utf-8', errors='replace')[-4000:]}"
    )


def check_vhs(vhs: str, temp: Path) -> Path:
    """Independently check VHS/ttyd/Chrome before running TUI screenshot tests."""
    frames = temp / "frames"
    tape = temp / "smoke.tape"
    tape.write_text(
        "\n".join(
            [
                f"Output {json.dumps(str(frames) + '/')}",
                "Require bash",
                'Set Shell "bash"',
                "Set Columns 80",
                "Set Rows 24",
                'Set FontFamily "Noto Sans Mono CJK SC"',
                "Set FontSize 16",
                "Set Framerate 1",
                'Type "printf VHS_SMOKE_OK"',
                "Enter",
                "Sleep 2s",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _run_checked([vhs, str(tape)], timeout=60)
    generated = sorted(frames.glob("frame-text-*.png"))
    if not generated:
        raise RuntimeError("VHS exited successfully but produced no terminal PNG text frames")
    frame = generated[-1]
    with frame.open("rb") as stream:
        header = stream.read(8)
    if header != PNG_SIGNATURE or frame.stat().st_size < 1024:
        raise RuntimeError(f"VHS produced an invalid or empty PNG: {frame}")
    return frame


def main() -> int:
    chrome = next(
        (
            found
            for binary in ("google-chrome", "chromium", "chromium-browser")
            if (found := shutil.which(binary))
        ),
        None,
    )
    vhs = shutil.which("vhs")
    if not chrome or not vhs:
        print(f"Missing screenshot tools: Chrome={chrome!r}, VHS={vhs!r}", file=sys.stderr)
        return 1
    try:
        with tempfile.TemporaryDirectory(prefix="acprof-vhs-smoke-") as directory:
            temp = Path(directory)
            check_chrome(chrome, temp)
            print(f"Headless Chrome startup: OK ({chrome})", flush=True)
            frame = check_vhs(vhs, temp)
            print(f"VHS rendered PNG: OK ({frame.stat().st_size} bytes)", flush=True)
    except RuntimeError as exc:
        print(f"VHS environment smoke test failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
