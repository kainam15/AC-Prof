"""Preserve operation failures while reporting secondary resource cleanup errors."""
from collections.abc import Callable, Iterator
from contextlib import contextmanager


def format_exception_notes(error: BaseException) -> str:
    """Keep exception notes visible at boundaries that do not print tracebacks."""
    return "\n".join((str(error), *getattr(error, "__notes__", ())))


@contextmanager
def cleanup_on_exit(cleanup: Callable[[], object], *, description: str,
                    on_error_only: bool = False) -> Iterator[None]:
    """Run cleanup without replacing an active failure with an ordinary error.

    An existing cancellation retains its identity. Cancellation during cleanup
    takes precedence over an ordinary operation failure, which remains its cause.
    Without an operation failure, cleanup errors still propagate normally.
    """
    try:
        yield
    except BaseException as primary:
        try:
            cleanup()
        except BaseException as secondary:
            if isinstance(primary, Exception) and not isinstance(secondary, Exception):
                # An enclosing cleanup may replace the explicit cause again.
                # Keep earlier failures in notes as well as in the chain.
                secondary.add_note(
                    f"Cleanup interrupted for {description}; preceding failure: "
                    f"{type(primary).__name__}: {format_exception_notes(primary)}"
                )
                raise secondary from primary
            primary.add_note(
                f"Cleanup failed for {description}: {type(secondary).__name__}: {format_exception_notes(secondary)}"
            )
        raise
    else:
        if not on_error_only:
            try:
                cleanup()
            except BaseException as error:
                error.add_note(f"Cleanup failed for {description}")
                raise
