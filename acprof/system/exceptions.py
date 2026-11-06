"""Lightweight exception-chain inspection shared by host and isolated runtimes."""
from collections.abc import Iterator


def exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Inspect explicit causes first, otherwise context, without revisiting cycles.

    Context suppression controls traceback display, not diagnostic classification.
    """
    seen: set[int] = set()
    cause: BaseException | None = exc
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        yield cause
        cause = cause.__cause__ if cause.__cause__ is not None else cause.__context__
