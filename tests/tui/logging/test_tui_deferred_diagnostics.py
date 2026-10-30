"""Diagnostics survive abrupt exit and ownership handoff without live rendering."""
import io
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from acprof.tui.commands import PendingLaunch
from acprof.tui.diagnostic_buffer import BoundedLogBuffer
from acprof.tui.log import SelectableLog
from acprof.tui.process_controller import ProcessController
from acprof.tui.stages import Stage
from tests.support.tui_fixtures import AcprofTui


@pytest.mark.parametrize("stale", [False, True])
def test_late_reaper_preserves_bounded_diagnostics_and_respects_process_token(stale):
    token = object()
    process = Mock(pid=1234, stdout=io.StringIO("[WARN] late\n"), returncode=1)
    app = SimpleNamespace(
        is_running=True, form_ready=True, ui_closing=False, process_token=object() if stale else token,
        lifecycle=Mock(), show_deferred_lines=Mock(), show_suppressed_count=Mock(), process_finished=Mock(),
    )
    app.call_from_thread = lambda callback, *args: callback(*args)
    controller = ProcessController(app)
    buffer = BoundedLogBuffer(max_entries=4)
    for line in ("[WARN] first", "[WARN] a", "[WARN] b", "[WARN] c", "[WARN] last"):
        buffer.append(line)
    controller._watch_failed_process_worker(process, "run", None, "cleanup delayed", None, token, buffer, 7)
    process.wait.assert_called_once()
    app.lifecycle.release.assert_called_once_with(process)
    if stale:
        app.show_deferred_lines.assert_not_called()
        app.show_suppressed_count.assert_not_called()
        app.process_finished.assert_not_called()
    else:
        app.show_deferred_lines.assert_called_once_with(
            ("[WARN] first", "[WARN] c", "[WARN] last", "[WARN] late"), 2, 0,
        )
        app.show_suppressed_count.assert_called_once_with(7)
        app.process_finished.assert_called_once_with("run", 1, None, "cleanup delayed")
    assert not buffer


def test_failed_replay_keeps_ownership_for_late_cleanup(tmp_path):
    from acprof.events.progress import encode_event
    from acprof.tui.process import StopResult

    app = AcprofTui(settings_path=tmp_path / "settings.json")
    lines = [
        encode_event("case_started", "fixture"),
        encode_event("measurement_started", "fixture"),
        "[WARN] preserved after failed callback",
        encode_event("measurement_stopped", "fixture"),
    ]
    process = Mock(pid=1234, stdout=io.StringIO("\n".join(lines) + "\n"))
    process.poll.return_value = None
    app.process_started = Mock()
    app.consume_process_line = Mock()
    app.show_deferred_lines = Mock(side_effect=RuntimeError("render callback failed"))
    app.call_from_thread = lambda callback, *args: callback(*args)
    app.process.deliver_process_callback = lambda token, callback, *args: callback(*args)
    with (
        patch.object(app.lifecycle, "start", return_value=process),
        patch.object(app.lifecycle, "stop", return_value=StopResult(1234, None, "still exiting")),
        patch.object(app.process, "safe_process_callback"),
        patch.object(app.process, "watch_failed_process") as watch,
    ):
        app.process._execute_command_worker(["fixture"], "run", config=None)
    watch.assert_called_once()
    buffer = watch.call_args.args[-2]
    assert buffer.snapshot().entries == ("[WARN] preserved after failed callback",)


async def test_abrupt_child_failure_releases_suspension_and_shows_omission_notice(tmp_path):
    script = (
        "from acprof.events.progress import emit_event\n"
        "emit_event('case_started', 'fixture')\n"
        "emit_event('measurement_started', 'fixture')\n"
        "for line in ('[WARN] first', '[WARN] a', '[WARN] b', '[WARN] c', '[WARN] last', 'x' * 65):\n"
        "    print(line, flush=True)\n"
        "raise SystemExit(1)\n"
    )
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)) as pilot:
        with patch("acprof.tui.process_controller.BoundedLogBuffer",
                   side_effect=lambda: BoundedLogBuffer(max_entries=4, max_bytes=512, max_entry_bytes=64)):
            app.launch(PendingLaunch((sys.executable, "-u", "-c", script), "run"))
            for _ in range(160):
                await pilot.pause(0.05)
                if not app.is_busy():
                    break
        assert not app.is_busy()
        log = app.query_one("#run-log", SelectableLog)
        assert not log.suspended
        assert not app.latest_snapshot.measurement_active
        assert app.latest_snapshot.stage is Stage.FAILED
        assert "[WARN] first" in log.raw_text
        assert "[WARN] last" in log.raw_text
        assert log.raw_text.count("省略 2 条") == 1
        assert "其中 1 条" in log.raw_text
