"""One bounded PyPI check per TUI session, outside all measurement windows."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx
from packaging.version import Version
from textual.widgets import Static

from acprof import __version__
from acprof.host.run_state import RunStateError, offline_analysis_lock
from acprof.messages import join_messages
from acprof.tui.i18n import message_id

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui

PYPI_URL = "https://pypi.org/pypi/acprof/json"
CHECK_TIMEOUT = 3.0
MAX_RESPONSE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ReleaseVersion:
    current: Version
    latest: Version
    stable: bool

    @property
    def update_available(self) -> bool:
        return self.stable and self.latest > self.current


async def check_pypi_version(current: str) -> ReleaseVersion:
    """Read published metadata with a total deadline and bounded response size."""
    local = Version(current)
    async with asyncio.timeout(CHECK_TIMEOUT), httpx.AsyncClient(
        timeout=CHECK_TIMEOUT,
        headers={"Accept": "application/json", "User-Agent": f"acprof/{local}"},
    ) as client:
        async with client.stream("GET", PYPI_URL) as response:
            response.raise_for_status()
            body = bytearray()
            async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                body.extend(chunk)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise ValueError("PyPI response exceeds size limit")
    data = json.loads(body)
    info = data.get("info") if isinstance(data, dict) else None
    if not isinstance(info, dict) or not isinstance(info.get("version"), str):
        raise ValueError("PyPI response is missing info.version")
    latest = Version(info["version"])
    yanked = info.get("yanked")
    if not isinstance(yanked, bool):
        raise ValueError("PyPI response is missing info.yanked")
    return ReleaseVersion(local, latest, not (yanked or latest.is_prerelease or latest.is_devrelease))


class UpdateController:
    """Keep the form editable; release network/lock ownership before collection."""

    def __init__(self, app: AcprofTui) -> None:
        self.app = app
        self.started = False
        self.checking = False

    def start_check(self) -> None:
        # Startup preflight owns the same lock, so schedule only after it ends.
        if self.started or self.app.ui_closing or not self.app.is_running:
            return
        self.started = True
        if self.app.execution.session.kind or self.app.execution.session.snapshot.measurement_active:
            return
        self.checking = True
        self.app.run_worker(self.check(), name="check_pypi_version", group="updates")

    async def check(self) -> None:
        status: str = message_id("common.updates.checking", __version__)
        try:
            # Exclude Linux collection, while allowing non-Linux TUI sessions.
            with offline_analysis_lock():
                result = await check_pypi_version(__version__)
            versions = message_id("common.updates.versions", str(result.current), str(result.latest))
            if result.update_available:
                status = join_messages("\n", (versions, message_id("common.updates.available")))
            elif not result.stable:
                status = join_messages("\n", (versions, message_id("common.updates.unstable")))
            else:
                key = "common.updates.current" if result.current == result.latest else "common.updates.ahead"
                status = join_messages("\n", (versions, message_id(key)))
            if self.can_present() and result.update_available:
                self.app.notify(status, title=message_id("common.updates.title"), timeout=10)
        except (TimeoutError, httpx.TimeoutException):
            status = message_id("common.updates.failed", __version__, message_id("common.error.request_timeout"))
        except (httpx.HTTPError, ValueError, OSError, RunStateError) as exc:
            status = message_id("common.updates.failed", __version__, f"{type(exc).__name__}: {exc}")
        finally:
            self.checking = False
            if self.can_present():
                self.app.localization.set_text(self.app.query_one("#version-status", Static), status)
                self.app.set_busy(False)

    def can_present(self) -> bool:
        return (self.app.is_running and self.app.form_ready and not self.app.ui_closing
                and not self.app.execution.session.snapshot.measurement_active)
