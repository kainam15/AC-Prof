"""Safety checks for SVG fixtures; no renderer baseline is updated here."""

import os
import signal
import socket
import subprocess
from unittest.mock import Mock

import httpx
import pytest

from acprof.experiments.configuration import RunConfig
from acprof.tui.app import AcprofTui
from acprof.tui.run.process import ProcessLifecycle

pytestmark = pytest.mark.visual


def test_app_run_startup_and_unmount_are_isolated(svg_operations, tmp_path):
    app = AcprofTui(RunConfig.smoke("fixture/model"), settings_path=tmp_path / "settings.json")

    async def finish(pilot):
        await pilot.pause()
        await app.workers.wait_for_complete()
        assert app.preflight.check_completed
        assert app.preflight.lock_acquired
        assert app.preflight.environment_checks
        process = Mock(pid=12345, returncode=None)
        process.poll.return_value = None
        svg_operations.own_process(app.execution.lifecycle, process)
        app.exit()

    app.run(headless=True, auto_pilot=finish, size=(80, 24))
    assert app.execution.lifecycle.process is None
    events = svg_operations.events
    assert any(event["event"] == "fixture-preflight" for event in events)
    assert any(event["event"] == "fixture-version-check" for event in events)
    assert {"event": "fixture-stop", "pid": 12345, "closing": True} in events


def test_unregistered_process_cannot_reach_signal_cleanup(svg_operations):
    lifecycle = ProcessLifecycle()
    lifecycle.process = Mock(pid=12345, returncode=None)
    lifecycle.process.poll.return_value = None
    operation = "ProcessLifecycle.stop (unregistered process)"
    svg_operations.expect_blocked(operation)
    with pytest.raises(AssertionError, match="unregistered process"):
        lifecycle.stop(closing=True)
    assert lifecycle.process.poll.call_count == 0


def test_owned_process_rejects_stale_stop(svg_operations):
    lifecycle = ProcessLifecycle()
    process = Mock(pid=12345, returncode=None)
    process.poll.return_value = None
    svg_operations.own_process(lifecycle, process)
    result = lifecycle.stop(expected_process=Mock(pid=54321))
    assert result.pid is None
    assert lifecycle.process is process
    assert process.poll.call_count == 0
    assert lifecycle.stop(closing=True).complete
    assert lifecycle.process is None


@pytest.mark.parametrize("operation", [
    "ProcessLifecycle.start", "subprocess.Popen", "os.killpg", "httpx.Client.send",
])
def test_external_operations_fail_closed(svg_operations, operation):
    svg_operations.expect_blocked(operation)
    invoke = {
        "ProcessLifecycle.start": lambda: ProcessLifecycle().start(
            ["forbidden-command"], cwd="/workspace", env={},
        ),
        "subprocess.Popen": lambda: subprocess.Popen(["forbidden-command"]),
        "os.killpg": lambda: os.killpg(12345, signal.SIGTERM),
        "httpx.Client.send": lambda: httpx.get("https://example.invalid/"),
    }[operation]
    with pytest.raises(AssertionError, match="SVG fixture blocked real operation"):
        invoke()


@pytest.mark.parametrize("method", ["connect", "connect_ex"])
def test_socket_connections_fail_closed(svg_operations, method):
    operation = f"socket.socket.{method}"
    svg_operations.expect_blocked(operation)
    with socket.socket() as connection:
        connect = getattr(connection, method)
        with pytest.raises(AssertionError, match="SVG fixture blocked real operation"):
            connect(("127.0.0.1", 9))


def test_swallowed_block_still_fails_teardown(svg_operations):
    policy = type(svg_operations)()
    with pytest.raises(AssertionError, match="SVG fixture blocked real operation"):
        policy.deny("swallowed worker operation")
    with pytest.raises(AssertionError, match="Unexpected SVG operations"):
        policy.assert_clean()
    assert policy.events == [{"event": "blocked", "operation": "swallowed worker operation"}]


async def test_async_http_fails_closed(svg_operations):
    svg_operations.expect_blocked("httpx.AsyncClient.send")
    async with httpx.AsyncClient() as client:
        with pytest.raises(AssertionError, match="httpx.AsyncClient.send"):
            await client.send(httpx.Request("GET", "https://example.invalid/"))
