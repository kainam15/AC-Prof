"""Run profiler commands with explicit ownership of short-lived Docker containers."""
from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from acprof.host.container_lifecycle import (
    ContainerCleanupError,
    container_owner_labels,
    owned_container_id,
    remove_owned_container,
)
from acprof.system.cleanup import cleanup_on_exit
from acprof.system.command import CommandResult, run_command


def run_profiler_command(command: Sequence[str], **kwargs: Any) -> CommandResult:
    """Settle each Docker run before returning; host tool commands pass through.

    A failed or interrupted client does not prove that its container exited.
    Cleanup failures supersede tool errors and keep the original as evidence,
    so callers can retain model leases and abort the remaining profiler plan.
    """
    if list(command[:2]) != ["docker", "run"]:
        return run_command(command, **kwargs)
    owner = container_owner_labels()
    labels = [part for key, value in owner.items() for part in ("--label", f"{key}={value}")]
    temporary = tempfile.TemporaryDirectory(prefix="acprof-profiler-")
    with cleanup_on_exit(temporary.cleanup, description="profiler container identity directory"):
        cidfile = Path(temporary.name) / "container.cid"
        argv = [*command[:2], "--cidfile", str(cidfile), *labels, *command[2:]]
        failure: BaseException | None = None
        identifier = ""
        try:
            result = run_command(argv, **kwargs)
            if result.returncode:
                failure = subprocess.CalledProcessError(
                    result.returncode, result.args, output=result.stdout, stderr=result.stderr,
                )
            return result
        except BaseException as exc:
            failure = exc
            raise
        finally:
            try:
                identifier = owned_container_id(cidfile)
                if identifier:
                    remove_owned_container(identifier, run_command)
                elif not isinstance(failure, (FileNotFoundError, PermissionError)):
                    raise ContainerCleanupError("", [{
                        "operation": "read_cidfile",
                        "detail": "no immutable ID after profiler dispatch; container absence is unconfirmed",
                    }])
            except ContainerCleanupError as exc:
                exc.run_error = failure
                if failure is not None:
                    raise exc from failure
                raise
            except BaseException as exc:
                raise ContainerCleanupError(identifier, [{
                    "operation": "profiler_cleanup", "exception_type": type(exc).__name__,
                    "detail": str(exc),
                }], run_error=failure) from exc
