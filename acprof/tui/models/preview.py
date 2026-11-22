"""Read the existing CLI's static model inspection for the configuration form."""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
from contextlib import suppress
from pathlib import Path

from acprof.installation import cli_command
from acprof.messages import join_messages
from acprof.system.redaction import redact_text
from acprof.tui.i18n import message_id
from acprof.tui.ui.view_models import brief_error

MODEL_OPTIONS = ("revision", "model-variant", "task", "tasks", "backend", "model-spec", "model-store", "batch-size")


async def inspect_summary(arguments: tuple[str, ...], python: Path, cwd: Path) -> dict:
    command = [*cli_command("inspect", python_executable=python), "--summary-json", *arguments[1:], "--", arguments[0]]
    with subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, start_new_session=os.name == "posix") as process:
        output = asyncio.create_task(asyncio.to_thread(process.communicate))
        try:
            stdout, stderr = await asyncio.wait_for(asyncio.shield(output), timeout=45)
        finally:
            if not output.done():
                if os.name == "posix":
                    # ModelScope resolves mutable revisions through git ls-remote.
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            await output
    if stdout.strip() and process.returncode in (0, 1, 2):
        with suppress(ValueError):
            summary = json.loads(stdout)
            if isinstance(summary, dict):
                return summary
    diagnostic = stderr.decode("utf-8", "replace")
    detail = next((line for line in diagnostic.splitlines() if ": error: " in line), "") or brief_error(diagnostic)
    raise ValueError(redact_text(detail, limit=360) if detail else f"acprof inspect exited {process.returncode}")


def preview_text(summary: dict) -> str:
    if error := summary.get("lookup_error"):
        from acprof.host.preparation.model_errors import ModelLookupError
        from acprof.hub.repository import model_source_scope
        with model_source_scope(summary["source"]):
            failure = ModelLookupError(**error)
        return join_messages("\n", [message_id("models.preview.failure", failure.summary), failure.hint])
    unknown = message_id("common.value.unknown")
    platform = summary.get("platform")
    backend = summary.get("backend")
    if platform and backend and summary.get("compatibility") == "supported":
        platform = f"{platform} / {backend}"
    lines = [message_id(
        "models.preview.facts", summary.get("source") or unknown, summary.get("revision") or unknown,
        summary.get("architecture") or unknown, ", ".join(summary.get("formats") or []) or unknown,
        platform or backend or unknown,
        (summary.get("status") if summary.get("reason") else summary.get("compatibility")) or summary.get("status") or unknown,
    )]
    if summary.get("reason"):
        lines.append(summary["reason"])
    if summary.get("next_action"):
        lines.append(summary["next_action"])
    if variants := summary.get("variants"):
        lines.append(message_id("models.preview.variants", ", ".join(variants[:8])))
    return join_messages("\n", lines)
