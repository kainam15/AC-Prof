"""The new executable replaces the process only after Textual returns."""
from types import SimpleNamespace

from acprof.cli import tui
from acprof.experiments.configuration import RunConfig


def test_restart_preserves_arguments_after_terminal_restoration(monkeypatch):
    events = []
    app = SimpleNamespace(
        configuration=SimpleNamespace(initial_config=RunConfig.smoke("fixture")),
        updates=SimpleNamespace(restart_command=["/tool bin/acprof", "tui"]),
        run=lambda: events.append("terminal restored"),
    )
    monkeypatch.setattr("acprof.tui.app.AcprofTui", lambda **kwargs: app)
    monkeypatch.setattr(tui.os, "execv", lambda executable, argv: events.append((executable, argv)))
    tui.main(["--model", "fixture/model", "--color-system", "256"])
    assert events == ["terminal restored", (
        "/tool bin/acprof", ["/tool bin/acprof", "tui", "--model", "fixture/model", "--color-system", "256"],
    )]


def test_cli_main_restart_preserves_arguments_when_argv_is_none(monkeypatch):
    from acprof.cli.main import main as cli_main

    events = []
    app = SimpleNamespace(
        configuration=SimpleNamespace(initial_config=RunConfig.smoke("fixture")),
        updates=SimpleNamespace(restart_command=["/tool bin/acprof", "tui"]),
        run=lambda: events.append("terminal restored"),
    )
    monkeypatch.setattr("acprof.tui.app.AcprofTui", lambda **kwargs: app)
    monkeypatch.setattr(tui.os, "execv", lambda executable, argv: events.append((executable, argv)))

    assert cli_main(["tui", "--model", "fixture/model", "--preset", "smoke", "--color-system", "256"]) == 0
    assert events == [
        "terminal restored",
        (
            "/tool bin/acprof",
            ["/tool bin/acprof", "tui", "--model", "fixture/model", "--preset", "smoke", "--color-system", "256"],
        ),
    ]
