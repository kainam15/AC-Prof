"""Keep load phase evidence before outer resource cleanup and report publication."""
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from acprof.artifacts import atomic_write_json
from acprof.system.cleanup import cleanup_on_exit, format_exception_notes


@contextmanager
def reserve_load_output(directory: Path) -> Iterator[None]:
    """Keep competing load commands out of this directory through final publication."""
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    with cleanup_on_exit(lambda: os.close(descriptor), description=f"load output directory {directory}"):
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"load output directory is in use: {directory}") from error
        if any(directory.iterdir()):
            raise ValueError(f"load output directory must be empty: {directory}")
        yield


@contextmanager
def load_phase(report: dict[str, Any], phase: Literal["warmup", "result"]) -> Iterator[None]:
    """Record partial requests and their failure before enclosing finalizers run."""
    try:
        yield
    except BaseException as error:
        if hasattr(error, "load_report"):
            report[phase] = error.load_report
        report.setdefault("phase_errors", {})[phase] = {
            "exception_type": type(error).__name__, "detail": format_exception_notes(error),
        }
        raise


@contextmanager
def publish_load_report(path: Path, report: dict[str, Any]) -> Iterator[None]:
    """Publish the final state without hiding an active failure if writing fails."""
    atomic_write_json(path, report, overwrite=False)
    with cleanup_on_exit(lambda: atomic_write_json(path, report), description=f"load report {path}"):
        try:
            yield
        except BaseException as error:
            report.update(
                successful=False,
                status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
                error=f"{type(error).__name__}: {format_exception_notes(error)}",
            )
            raise
