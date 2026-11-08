"""Bounded waits for an explicit observable TUI condition."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from textual.pilot import Pilot


async def wait_for_condition(
    pilot: Pilot[Any],
    predicate: Callable[[], bool],
    *,
    description: str,
    timeout: float,
    diagnostics: Callable[[], str] | None = None,
) -> None:
    """Process messages until a nonblocking predicate holds, without fixed delays.

    The caller identifies the relevant operation and retains any additional
    layout pause it needs. This does not wait for all workers or future timers.
    Predicate errors and external cancellation propagate unchanged.
    """
    deadline = asyncio.timeout(timeout)
    try:
        async with deadline:
            while not predicate():
                await pilot.pause()
    except TimeoutError as exc:
        if not deadline.expired():
            raise
        detail = diagnostics() if diagnostics is not None else ""
        message = f"Timed out after {timeout:g}s waiting for {description}"
        if detail:
            message += f"\n{detail}"
        raise AssertionError(message) from exc
