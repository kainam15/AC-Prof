"""One bounded PyPI check per TUI session, outside all measurement windows."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from packaging.version import Version
from textual.widgets import Button, Select, Static

from acprof import __version__
from acprof.host.run_state import RunStateError, offline_analysis_lock
from acprof.messages import join_messages
from acprof.tui.i18n import message_id
from acprof.tui.ui.dialogs import ScrollConfirmActionScreen
from acprof.updates import service
from acprof.updates.installation import Installation, detect_installation
from acprof.updates.releases import Channel, ReleaseVersion, select_releases, version_type
from acprof.updates.worker import UpdatePlan, failure_kind

if TYPE_CHECKING:
    from acprof.tui.app import AcprofTui

PYPI_URL = "https://pypi.org/pypi/acprof/json"
CHECK_TIMEOUT = 3.0
MAX_RESPONSE_BYTES = 1024 * 1024


async def check_pypi_version(current: str, channel: Channel = "stable") -> ReleaseVersion:
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
    return select_releases(json.loads(body), current, channel)


class UpdateController:
    """App-owned state survives settings navigation; the worker survives TUI exit."""

    def __init__(self, app: AcprofTui, *, state_root: Path | None = None) -> None:
        self.app = app
        self.started = False
        self.checking = False
        self.installing = False
        self.needs_restart = False
        self.confirming = False
        self.restart_command: list[str] | None = None
        self.current = Version(__version__)
        self.channel: Channel = "preview" if self.current.is_prerelease else "stable"
        self.effective_channel = self.channel
        self.directory = service.state_directory(Path(sys.prefix).resolve(), state_root)
        self.installation: Installation | None = None
        self.release: ReleaseVersion | None = None
        self.status = "checking"
        self.detail = ""
        self.check_failed = False
        self.checked_at = ""
        self.install_state: dict = {}
        self.process: subprocess.Popen | None = None

    def start_check(self, *, manual: bool = False) -> None:
        if self.app.ui_closing or not self.app.is_running or self.checking or self.installing:
            return
        if self.started and not manual:
            return
        if manual and (not self.app.allow_operation("update") or self.needs_restart):
            return
        self.started = True
        if self.app.execution.session.kind or self.app.execution.session.snapshot.measurement_active:
            return
        self.checking = True
        self.check_failed = False
        self.status, self.detail = "checking", ""
        self.app.run_worker(self.check(), name="check_pypi_version", group="updates", exclusive=True)
        self.render()

    async def check(self) -> None:
        try:
            # A surviving worker owns the locks and gets observed without HTTP.
            if service.update_active(self.directory):
                self.installation = None
                self.installing = True
                self.app.set_busy(True)
                self.app.run_worker(self.observe_install(), group="installation")
                return
            with offline_analysis_lock():
                inspection = asyncio.create_task(asyncio.to_thread(detect_installation))
                try:
                    self.installation = await asyncio.shield(inspection)
                except asyncio.CancelledError:
                    await inspection  # Do not release measurement exclusion early.
                    raise
                fallback = self.installation.channel if self.installation.supported else self.channel
                self.effective_channel = service.saved_channel(self.directory, fallback)
                if self.release is None:
                    self.channel = self.effective_channel
                self.checked_at = str(service.read_state(self.directory, "check.json").get("checked_at", ""))
                self.install_state = service.read_state(self.directory)
                if self.install_state.get("status") == "running":
                    self.status, self.detail = "failed", message_id("updates.ui.interrupted", self.install_state.get("job_dir", ""))
                    self.needs_restart = True
                    return
                self.release = await check_pypi_version(str(self.current), self.channel)
                self.checked_at = datetime.now().astimezone().isoformat(timespec="seconds")
                self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                service.atomic_json(self.directory / "check.json", {"checked_at": self.checked_at})
            self.status = "available" if self.can_apply else "current" if self.release.latest else "failed"
            if self.can_present() and self.release.update_available:
                self.app.notify(message_id("updates.ui.notice", self.current, self.release.latest),
                                title=message_id("common.updates.title"), timeout=8)
        except (TimeoutError, httpx.TimeoutException):
            self.check_failed = True
            self.status, self.detail = "failed", message_id("common.error.request_timeout")
        except (httpx.HTTPError, ValueError, OSError, RunStateError) as exc:
            self.check_failed = True
            self.status, self.detail = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            self.checking = False
            self.render()
            if self.can_present():
                self.app.set_busy(False)

    @property
    def can_apply(self) -> bool:
        if not self.release or self.release.latest is None:
            return False
        return (self.release.update_available or self.channel != self.effective_channel
                or (self.channel == "stable" and self.current.is_prerelease))

    def select_channel(self, value: str) -> None:
        if value not in ("stable", "preview") or self.installing or self.checking or self.needs_restart:
            return
        if not self.app.operation_state().allows("update"):
            return
        self.channel = "preview" if value == "preview" else "stable"
        if self.release:
            self.release = replace(self.release, channel=self.channel)
            self.status = "available" if self.can_apply else "current" if self.release.latest else "failed"
        self.render()

    def request_install(self) -> None:
        if (self.confirming or self.needs_restart or self.status == "failed" or not self.can_apply or not self.installation
                or not self.installation.supported or not self.app.allow_operation("update")):
            return
        assert self.release is not None
        try:
            plan = service.make_plan(self.installation, self.release, self.directory)
        except (ValueError, OSError) as exc:
            self.status, self.detail = "failed", str(exc)
            self.render()
            return
        downgrade = Version(plan.target) < self.current
        action = message_id("updates.ui.downgrade" if downgrade else "updates.ui.apply")
        summary: str = message_id("updates.ui.confirm", action, plan.current, plan.target,
                             self.channel.title(), plan.index_url, plan.prefix)
        if self.channel == "preview":
            summary = join_messages("\n\n", (summary, message_id("updates.ui.preview_risk")))
        self.confirming = True
        self.app.push_screen(ScrollConfirmActionScreen(
            message_id("common.updates.title"), summary, action,
            variant="warning" if downgrade else "primary",
        ), lambda accepted: self.confirm_install(accepted, plan))

    def confirm_install(self, accepted: bool, plan: UpdatePlan) -> None:
        self.confirming = False
        # Re-check admission and the immutable selection after the modal closes.
        if (not accepted or self.installing or self.needs_restart or not self.app.allow_operation("update")
                or not self.release or str(self.release.latest) != plan.target or self.channel != plan.channel):
            return
        self.installing = True
        self.status, self.detail = "installing", ""
        self.render()
        self.app.set_busy(True)
        self.app.run_worker(self.install(plan), name="install_acprof", group="installation", exclusive=True)

    async def install(self, plan: UpdatePlan) -> None:
        try:
            # Only small copies and spawn occur here; installation runs externally.
            self.process = service.launch_update(plan)
        except (OSError, ValueError, RunStateError) as exc:
            self.installing = False
            self.status = "failed"
            self.detail = join_messages("\n", (message_id("updates.error." + failure_kind(exc)), str(exc)))
            self.render()
            self.app.set_busy(False)
            return
        await self.observe_install()

    async def observe_install(self) -> None:
        try:
            while True:
                self.install_state = service.read_state(self.directory)
                active = service.update_active(self.directory)
                # The launcher acquires the lock before publishing its plan.
                # Observe that interval without assuming status.json exists yet.
                if self.installation is None and self.install_state.get("job_dir"):
                    attached = UpdatePlan(**service.read_state(Path(self.install_state["job_dir"]), "plan.json"))
                    self.installation = Installation("uv tool", Path(attached.prefix),
                        Path(attached.prefix) / "bin/python", Path(attached.entrypoint), Path(attached.manager))
                    self.channel = "preview" if attached.channel == "preview" else "stable"
                    self.effective_channel = service.saved_channel(self.directory, self.effective_channel)
                if self.process is not None:
                    self.process.poll()  # Reap our child; never terminate it on UI cancellation.
                state = self.install_state.get("status")
                self.status = "installing" if active else "success" if state == "success" else "failed"
                self.render()
                if not active:
                    self.installing = False
                    self.needs_restart = state == "success" or self.install_state.get("rollback") != "not_needed"
                    if state == "success":
                        self.effective_channel = service.saved_channel(self.directory, self.effective_channel)
                    elif state == "running":
                        self.detail = message_id("updates.ui.interrupted", self.install_state.get("job_dir", ""))
                    else:
                        kind = self.install_state.get("error_kind", "install")
                        self.detail = join_messages("\n", (
                            message_id(f"updates.error.{kind}"), self.install_state.get("error", ""),
                            message_id("updates.rollback." + self.install_state.get("rollback", "not_needed")),
                        ))
                    break
                await asyncio.sleep(0.25)  # Only while the installer owns the measurement lock.
        except (OSError, ValueError, TypeError) as exc:
            self.status, self.detail = "failed", str(exc)
            # Unknown ownership remains blocked; do not start a measurement.
            self.needs_restart = True
        finally:
            if self.can_present():
                self.render()
                self.app.set_busy(False)

    def restart(self) -> None:
        if self.installing or not self.needs_restart or not self.installation:
            return
        command = [str(self.installation.entrypoint), "tui"]
        # Textual's stdout capture pretends to be a TTY even when redirected.
        if not os.isatty(0) or not os.isatty(1):
            self.app.notify(message_id("updates.ui.restart_manual", shlex.join(command)), severity="warning")
            return
        if self.install_state.get("status") != "success" and self.install_state.get("rollback") != "restored":
            self.app.notify(message_id("updates.ui.interrupted", self.install_state.get("job_dir", "")), severity="warning")
            return
        self.restart_command = command
        self.app.ui_closing = True
        self.app.exit()

    def show_log(self) -> None:
        if self.log_blocked():
            self.app.notify(message_id("common.task.wait_until_finished"), severity="warning")
            return
        path = Path(self.install_state.get("log", self.directory / "status.json"))
        try:
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 65536))
                content = stream.read().decode("utf-8", errors="replace")
        except OSError as exc:
            content = str(exc)
        self.app.push_screen(ScrollConfirmActionScreen(
            message_id("updates.ui.logs"), f"{path}\n\n{content}", message_id("updates.ui.copy_path"),
        ), lambda accepted: self.app.copy_to_clipboard(str(path)) if accepted else None)

    def copy_manual(self) -> None:
        if self.installation:
            self.app.copy_to_clipboard(self.installation.manual_command(self.channel))

    def can_present(self) -> bool:
        return (self.app.is_running and self.app.form_ready and not self.app.ui_closing
                and not self.app.execution.session.snapshot.measurement_active)

    def render(self) -> None:
        if not self.can_present():
            return
        missing = message_id("updates.ui.none" if self.release else "updates.ui.unknown")
        identity = "updates.ui.runtime_identity" if self.needs_restart else "updates.ui.identity"
        lines: list[str] = [message_id(identity, self.current, version_type(self.current), self.effective_channel.title()),
                 message_id("updates.ui.candidates", self.release.latest_stable or missing if self.release else missing,
                            self.release.latest_preview or missing if self.release else missing),
                 message_id("updates.ui.checked", self.checked_at or missing),
                 message_id("updates.state." + self.status)]
        if self.release and self.release.latest is None:
            lines.append(message_id("updates.ui.no_candidate"))
        elif self.release and self.release.latest is not None and self.release.latest < self.current:
            lines.append(message_id("updates.ui.lower", self.release.latest))
        if self.installing:
            lines.append(message_id("updates.phase." + self.install_state.get("phase", "backup")))
        if self.needs_restart:
            if self.install_state.get("status") == "success":
                lines.append(message_id("updates.ui.installed", self.install_state.get("target", missing)))
            lines.append(message_id("updates.ui.restart_needed"))
        if self.install_state.get("status") == "failed" and not self.detail:
            lines.append(message_id("updates.ui.last_failure"))
        if self.detail:
            lines.append(message_id("common.updates.failed", self.current, self.detail) if self.check_failed else self.detail)
        self.app.localization.set_text(self.app.query_one("#version-status", Static), join_messages("\n", lines))
        if self.installation:
            info: str = message_id("updates.ui.installation", self.installation.kind, self.installation.prefix)
            if not self.installation.supported:
                reason = self.installation.reason
                if reason in {"source", "standalone", "venv", "pipx", "system"}:
                    reason = message_id("updates.unsupported." + reason)
                info = join_messages("\n", (info, reason, self.installation.manual_command(self.channel)))
            self.app.localization.set_text(self.app.query_one("#version-installation", Static), info)
        with self.app.prevent(Select.Changed):
            self.app.query_one("#update-channel", Select).value = self.channel
        self.refresh_controls()

    def refresh_controls(self) -> None:
        if not self.app.is_running or not self.app.form_ready or self.app.ui_closing:
            return
        blocked = not self.app.operation_state().allows("update") or self.needs_restart
        self.app.query_one("#update-channel", Select).disabled = blocked
        self.app.query_one("#version-check", Button).disabled = blocked
        apply = self.app.query_one("#version-apply", Button)
        apply.disabled = blocked or self.status == "failed" or not self.can_apply or not self.installation or not self.installation.supported
        self.app.query_one("#version-restart", Button).display = self.needs_restart
        self.app.query_one("#version-restart", Button).disabled = self.installing
        self.app.query_one("#version-log", Button).disabled = not self.install_state or self.log_blocked()
        self.app.query_one("#version-manual", Button).display = bool(self.installation and not self.installation.supported)

    def log_blocked(self) -> bool:
        state = self.app.operation_state()
        return (state.process or state.measuring or state.checking or state.reading or state.closing
                or (state.maintenance and not self.installing))
