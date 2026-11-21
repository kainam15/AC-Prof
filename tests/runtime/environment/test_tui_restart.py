"""The new executable replaces the process only after Textual returns."""
import pytest

from acprof.cli import tui


@pytest.fixture
def restart_events(tmp_path, monkeypatch):
    events = []
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    def run(app):
        app.updates.restart_command = ["/tool bin/acprof", "tui"]
        events.append(("terminal restored", app.configuration.initial_config.model_source))

    monkeypatch.setattr("acprof.tui.app.AcprofTui.run", run)
    monkeypatch.setattr(tui.os, "execv", lambda executable, argv: events.append((executable, argv)))
    return events


def test_restart_preserves_arguments_after_terminal_restoration(restart_events):
    args = ["--model", "fixture/model", "--model-source", "modelscope", "--color-system", "256"]
    tui.main(args)
    assert restart_events == [("terminal restored", "modelscope"), (
        "/tool bin/acprof", ["/tool bin/acprof", "tui", *args],
    )]


def test_cli_main_restart_preserves_arguments_when_argv_is_none(restart_events):
    from acprof.cli.main import main as cli_main

    args = ["tui", "--model", "fixture/model", "--preset", "smoke", "--color-system", "256"]
    assert cli_main(args) == 0
    assert restart_events == [
        ("terminal restored", "huggingface"),
        ("/tool bin/acprof", ["/tool bin/acprof", *args]),
    ]
