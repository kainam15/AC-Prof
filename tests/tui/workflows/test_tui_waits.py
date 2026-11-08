"""Failure and cancellation semantics of the shared condition wait."""

import asyncio
from types import SimpleNamespace

import pytest

from tests.support.tui_waits import wait_for_condition


def yielding_pilot():
    return SimpleNamespace(pause=lambda: asyncio.sleep(0))


async def test_condition_wait_observes_a_deferred_callback():
    completed = asyncio.Event()
    asyncio.get_running_loop().call_soon(completed.set)
    await wait_for_condition(
        yielding_pilot(), completed.is_set,
        description="the deferred callback", timeout=1,
    )
    assert completed.is_set()


async def test_condition_wait_fails_with_expected_condition_and_current_state():
    with pytest.raises(AssertionError, match="waiting for download progress from the new child\nstate=starting"):
        await wait_for_condition(
            yielding_pilot(), lambda: False,
            description="download progress from the new child", timeout=0,
            diagnostics=lambda: "state=starting",
        )


@pytest.mark.parametrize("error", [ValueError("broken predicate"), TimeoutError("predicate deadline")])
async def test_condition_wait_preserves_predicate_errors(error):
    def broken_predicate():
        raise error

    with pytest.raises(type(error)) as raised:
        await wait_for_condition(
            yielding_pilot(), broken_predicate,
            description="an invalid predicate", timeout=1,
        )
    assert raised.value is error


async def test_condition_wait_preserves_external_cancellation():
    entered = asyncio.Event()

    async def pause():
        entered.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(wait_for_condition(
        SimpleNamespace(pause=pause), lambda: False,
        description="a cancelled wait", timeout=1,
    ))
    try:
        await entered.wait()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
