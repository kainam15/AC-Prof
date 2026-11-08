"""Startup version checks use the real UI and a controlled HTTP boundary."""

import asyncio
from dataclasses import replace

import httpx
import pytest
from textual.widgets import Button, Input, Static

from acprof import __version__
from acprof.experiments.configuration import RunConfig
from acprof.host.run_state import MeasurementLock, RunStateError
from acprof.tui.app import AcprofTui
from acprof.tui.commands import PendingLaunch
from acprof.tui.updates import check_pypi_version


@pytest.fixture
def ready_host(monkeypatch):
    monkeypatch.setattr("acprof.tui.run.preflight_controller.quick_preflight", lambda *a, **k: [])


def make_app(tmp_path):
    return AcprofTui(RunConfig.smoke(""), settings_path=tmp_path / "settings.json")


@pytest.mark.parametrize("language", ["zh", "en"])
async def test_startup_shows_new_release_without_blocking_input(tmp_path, monkeypatch, ready_host, language):
    started = asyncio.Event()
    release = asyncio.Event()
    requests = []

    async def respond(_transport, request):
        requests.append(request)
        started.set()
        await release.wait()
        return httpx.Response(200, json={"info": {"version": "9.0.0", "yanked": False}})

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", respond)
    app = make_app(tmp_path)
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=(80, 24), notifications=True) as pilot:
        try:
            await pilot.pause()
            assert app.title == f"AC-Prof {__version__}"
            assert app.query_one("HeaderTitle", Static).render_line(0).text.strip() == app.title
            assert started.is_set(), "startup must request the latest PyPI version"
            await pilot.press("a", "b", "c")
            assert app.query_one("#model", Input).value == "abc"
            assert app.query_one("#start-run", Button).disabled
            with pytest.raises(RunStateError), MeasurementLock():
                pytest.fail("version HTTP must exclude measurement in another process")
        finally:
            release.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        notice = app.query_one("Toast")
        text = notice.render().plain
        assert (f"当前版本：{__version__}" if language == "zh" else
                f"Current version: {__version__}") in text
        assert ("最新版本：9.0.0" if language == "zh" else "Latest version: 9.0.0") in text
        assert "uv tool upgrade acprof" in text
        assert notice.region.intersection(app.screen.region).height == notice.region.height
        app.save_screenshot(f"update-{language}.svg", path=str(tmp_path))
        with MeasurementLock():
            assert not app.query_one("#start-run", Button).disabled
        app.clear_notifications()
        app.activate_tab("settings-tab")
        await pilot.pause()
        status = app.query_one("#version-status", Static)
        status.scroll_visible(immediate=True)
        await pilot.pause()
        assert "uv tool upgrade acprof" in str(status.content)
        assert status.region.intersection(app.screen.region).height == status.region.height
        app.save_screenshot(f"version-settings-{language}.svg", path=str(tmp_path))
        await pilot.resize_terminal(120, 30)
        await pilot.pause()
        app.preflight.quick_check()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert len(requests) == 1
        assert str(requests[0].url) == "https://pypi.org/pypi/acprof/json"


@pytest.mark.parametrize(("current", "latest", "yanked", "available"), [
    ("0.2.0", "0.3.0", False, True),
    ("0.9.0", "0.10.0", False, True),
    ("0.3.0", "0.3.0", False, False),
    ("0.4.0", "0.3.0", False, False),
    ("0.3.0+local", "0.3.0", False, False),
    ("0.3.0rc1", "0.3.0", False, True),
    ("0.2.0", "0.3.0rc1", False, False),
    ("0.2.0", "0.3.0.dev1", False, False),
    ("0.2.0", "0.3.0", True, False),
])
async def test_pep440_comparison_and_release_eligibility(monkeypatch, current, latest, yanked, available):
    async def respond(_transport, request):
        return httpx.Response(200, json={"info": {"version": latest, "yanked": yanked}})

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", respond)
    result = await check_pypi_version(current)
    assert str(result.current) == current
    assert str(result.latest) == latest
    assert result.update_available is available


@pytest.mark.parametrize("payload", [
    {}, [], {"info": None}, {"info": {"version": 3}},
    {"info": {"version": "invalid", "yanked": False}},
    {"info": {"version": "0.3.0"}},
])
async def test_invalid_metadata_is_an_error(monkeypatch, payload):
    async def respond(_transport, request):
        return httpx.Response(200, json=payload)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", respond)
    with pytest.raises(ValueError, match="PyPI response|Invalid version"):
        await check_pypi_version("0.2.0")


@pytest.mark.parametrize("failure", ["offline", "http", "invalid-json", "timeout", "oversized"])
async def test_failed_check_is_visible_and_releases_collection(tmp_path, monkeypatch, ready_host, failure):
    cancelled = asyncio.Event()
    requests = []

    async def respond(_transport, request):
        requests.append(request)
        if failure == "offline":
            raise httpx.ConnectError("offline", request=request)
        if failure == "http":
            return httpx.Response(503)
        if failure == "invalid-json":
            return httpx.Response(200, content=b"not JSON")
        if failure == "oversized":
            return httpx.Response(200, content=b" " * 65)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", respond)
    if failure == "timeout":
        monkeypatch.setattr("acprof.tui.updates.CHECK_TIMEOUT", 0.1)
    if failure == "oversized":
        monkeypatch.setattr("acprof.tui.updates.MAX_RESPONSE_BYTES", 64)
    app = make_app(tmp_path)
    async with app.run_test(size=(80, 24), notifications=True) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = str(app.query_one("#version-status", Static).content)
        assert "无法检查 PyPI 最新版本" in text
        assert "仍可正常使用 AC-Prof" in text
        assert "uv tool upgrade" not in text
        assert not app.query("Toast")
        assert not app.query_one("#start-run", Button).disabled
        assert len(requests) == 1
        if failure == "timeout":
            assert cancelled.is_set()
            assert "请求超时" in text
        with MeasurementLock():
            pass


async def test_quit_cancels_http_and_releases_measurement_lock(tmp_path, monkeypatch, ready_host):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def respond(_transport, request):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", respond)
    app = make_app(tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        assert started.is_set()
        await pilot.press("ctrl+q")
    assert cancelled.is_set()
    with MeasurementLock():
        pass


async def test_other_measurement_prevents_any_pypi_request(tmp_path, monkeypatch, ready_host):
    requests = []

    async def respond(_transport, request):
        requests.append(request)
        raise AssertionError("must not send HTTP while another measurement owns the lock")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", respond)
    app = make_app(tmp_path)
    with MeasurementLock():
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            text = str(app.query_one("#version-status", Static).content)
            assert "无法检查 PyPI 最新版本" in text
            assert "测量锁" in text
    assert not requests


async def test_starting_run_clears_update_notification(tmp_path, monkeypatch, ready_host):
    async def respond(_transport, request):
        return httpx.Response(200, json={"info": {"version": "9.0.0", "yanked": False}})

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", respond)
    app = make_app(tmp_path)
    launches = []
    monkeypatch.setattr(app.process, "execute_command", lambda *a, **k: launches.append(a))
    async with app.run_test(size=(80, 24), notifications=True) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.query("Toast")
        app.launch(PendingLaunch(("fixture",), "run", RunConfig.smoke("fixture/model")))
        await pilot.pause()
        assert len(launches) == 1
        assert not app.query("Toast")


@pytest.mark.parametrize(("current", "latest", "yanked", "expected"), [
    ("0.3.0", "0.3.0", False, "当前已是最新版本"),
    ("0.3.0", "0.0.1", False, "当前版本高于 PyPI 最新版本"),
    ("0.3.0", "9.0.0rc1", False, "预发布或已撤回"),
    ("0.3.0", "9.0.0", True, "预发布或已撤回"),
    ("0.4.0a1", "0.4.0a1", False, "预发布或已撤回"),
])
async def test_no_upgrade_keeps_status_without_notification(
    tmp_path, monkeypatch, ready_host, current, latest, yanked, expected,
):
    async def respond(_transport, request):
        return httpx.Response(200, json={"info": {"version": latest, "yanked": yanked}})

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", respond)
    monkeypatch.setattr("acprof.tui.updates.__version__", current)
    app = make_app(tmp_path)
    async with app.run_test(size=(80, 24), notifications=True) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = str(app.query_one("#version-status", Static).content)
        assert f"当前版本：{current}" in text
        assert f"最新版本：{latest}" in text
        assert expected in text
        assert "uv tool upgrade" not in text
        assert not app.query("Toast")


@pytest.mark.parametrize("cancel", [False, True], ids=["deadline", "cancelled"])
async def test_slow_body_is_closed_on_timeout_or_cancellation(monkeypatch, cancel):
    started = asyncio.Event()
    closed = asyncio.Event()

    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"info":'
            started.set()
            await asyncio.Event().wait()

        async def aclose(self):
            closed.set()

    async def respond(_transport, request):
        return httpx.Response(200, stream=SlowBody())

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", respond)
    monkeypatch.setattr("acprof.tui.updates.CHECK_TIMEOUT", 0.1)
    task = asyncio.create_task(check_pypi_version("0.2.0"))
    await asyncio.wait_for(started.wait(), 1)
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
        await task
    assert closed.is_set()
