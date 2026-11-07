"""Shared plugin registration; domain fixtures live beside their tests."""

import pytest

pytest_plugins = ["acprof.testing.plugin"]


@pytest.fixture(autouse=True)
def offline_machine_environment(request, monkeypatch):
    """UI tests must not probe additional real hardware on application startup."""
    if not any(request.node.get_closest_marker(marker) for marker in ("tui", "visual")):
        return
    from acprof.tui.run.diagnostics import PreflightCheck
    from acprof.tui.run.environment_checks import ENVIRONMENT_CHECK_IDS

    monkeypatch.setattr("acprof.tui.run.preflight_controller.scan_environment", lambda **kwargs: [
        PreflightCheck(name, "ok", "fixture machine check", check_id=name, source="fixture")
        for name in ENVIRONMENT_CHECK_IDS
    ])


@pytest.fixture(autouse=True)
def offline_tui_version_check(request, monkeypatch):
    """Keep startup HTTP deterministic without bypassing its UI/lifecycle logic."""
    if not any(request.node.get_closest_marker(marker) for marker in ("tui", "visual")):
        return
    import httpx

    from acprof import __version__

    original = httpx.AsyncHTTPTransport.handle_async_request

    async def respond(transport, request):
        if str(request.url) == "https://pypi.org/pypi/acprof/json":
            return httpx.Response(200, json={"info": {"version": __version__, "yanked": False}})
        return await original(transport, request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", respond)
