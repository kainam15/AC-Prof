"""One UI journey through an interrupted CLI fixture and audited resumed results."""
import asyncio
import os
import sys
from dataclasses import replace
from unittest.mock import patch

import pytest
from textual.widgets import Button, Input, Static

from acprof.analysis.audit import audit_result
from acprof.experiments.configuration import RunConfig
from acprof.host import run_state
from acprof.results.csv import read_result_csv
from acprof.tui.logging.widget import SelectableLog
from acprof.tui.run.process import ProcessLifecycle
from acprof.tui.run.stages import Stage
from tests.support.platform_fixtures import native_policy
from tests.support.tui_fixtures import AcprofTui


@pytest.mark.parametrize(("language", "size"), [("zh", (80, 24)), ("en", (120, 30))])
async def test_measurement_stop_resume_preserves_completed_case_and_displays_results(
        tmp_path, request, language, size):
    native_policy(request)
    root = tmp_path / "results"
    config = replace(RunConfig.smoke("org/model"), runtime="docker", cpus="1,2", input_scales="64",
                     prune_startup_oom=False, extra_options={"matrix_order": "declared"},
                     output_dir=str(root))
    app = AcprofTui(config, settings_path=tmp_path / "settings.json")
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    original_start = ProcessLifecycle.start
    launches = []

    def launch(owner, command, **kwargs):
        arguments = command[command.index("--model"):]
        launches.append(arguments)
        return original_start(owner, [
            sys.executable, "-m", "tests.support.tui_recovery_process",
            str(run_state.MEASUREMENT_LOCK_ROOT), run_state.MEASUREMENT_LOCK_NAMESPACE.hex(),
            *arguments,
        ], **kwargs)

    with patch.object(ProcessLifecycle, "start", autospec=True, side_effect=launch):
        async with app.run_test(size=size) as pilot:
            async def wait_until(predicate):
                async def wait():
                    while not predicate():
                        await pilot.pause()
                try:
                    await asyncio.wait_for(wait(), timeout=15)
                except TimeoutError:
                    app.logs.flush()
                    pytest.fail(f"Timed out in {app.execution.session.snapshot}:\n"
                                f"{app.query_one('#run-log', SelectableLog).raw_text}")
                await pilot.pause()

            assert await pilot.click("#start-run")
            await wait_until(lambda: app.execution.session.snapshot.measurement_active
                             and app.execution.session.snapshot.cpu == "2")
            child = app.execution.lifecycle.process
            assert child is not None
            assert app.execution.session.snapshot.cpu == "2"
            assert app.query_one("#start-run", Button).disabled
            assert str(app.query_one("#status-stage", Static).content) == app.tr(Stage.MEASURING.label)

            # Declining the confirmation keeps this same running attempt alive.
            await pilot.press("ctrl+x")
            await pilot.pause()
            assert await pilot.click("#confirm-no")
            await pilot.pause()
            assert child.poll() is None
            assert app.execution.session.snapshot.measurement_active
            assert len(launches) == 1

            assert await pilot.click("#stop-run")
            await pilot.pause()
            assert await pilot.click("#confirm-yes")
            await wait_until(lambda: not app.is_busy())
            app.logs.flush()
            assert child.poll() == 130
            with pytest.raises(ProcessLookupError):
                os.kill(child.pid, 0)
            assert app.execution.lifecycle.process is None
            assert app.execution.session.snapshot.stage is Stage.STOPPED
            assert not app.execution.session.snapshot.measurement_active
            assert app.query_one("#stop-run", Button).disabled
            assert not app.query_one("#start-run", Button).disabled
            assert "fixture diagnostic retained across cancellation" in app.query_one("#run-log", SelectableLog).raw_text

            directory = config.result_dir(app.project_dir)
            stopped = run_state.load_run_state(directory)
            completed = {name: record for name, record in stopped["cases"].items()
                         if record["status"] == "complete"}
            assert len(completed) == 1
            assert stopped["attempts"][-1]["pid"] == child.pid
            assert stopped["attempts"][-1]["ended_at"]
            with run_state.MeasurementLock(), run_state.ResultDirectoryLock(directory):
                pass
            original_plan = (directory / "metadata/matrix_plan.json").read_bytes()

            app.activate_tab("run-tab")
            await pilot.pause()
            assert await pilot.click("#start-run")
            await wait_until(lambda: bool(app.screen.query("#recovery-resume")))
            assert not app.screen.query_one("#recovery-resume", Button).disabled
            assert str(app.screen.query_one("#recovery-resume", Button).label) == (
                "继续实验" if language == "zh" else "Resume")
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            assert await pilot.click("#recovery-resume")
            await wait_until(lambda: len(launches) == 2 and not app.is_busy())
            assert "--resume" in launches[1]
            assert app.execution.session.snapshot.stage is Stage.COMPLETED
            assert not app.execution.session.snapshot.measurement_active
            assert (root / "case_calls.txt").read_text().splitlines() == ["1", "2", "2"]
            resumed = run_state.load_run_state(directory)
            assert resumed["status"] == "complete"
            assert resumed["run_id"] == stopped["run_id"]
            assert len(resumed["attempts"]) == 2
            assert resumed["attempts"][-1]["resume"]
            assert resumed["attempts"][-1]["ended_at"]
            assert {name: resumed["cases"][name] for name in completed} == completed
            assert (directory / "metadata/matrix_plan.json").read_bytes() == original_plan
            _, rows = read_result_csv(directory / "result_layers.json")
            assert [row["cpu_cores"] for row in rows] == ["1", "2"]
            audit = audit_result(directory)
            assert audit["valid"], audit["issues"]
            assert audit["completion"] == "complete"
            assert audit["coverage"]["missing"] == 0
            assert list((directory / ".acprof/recovery/interrupted_cases").rglob("*.csv"))
            assert app.results.formal_ok_rows == 2
            assert ('有效正式窗口：2' if language == 'zh' else 'Valid formal windows: 2') in str(
                app.query_one('#result-data-status', Static).content)
            assert app.query_one("#experiment-source", Input).value == str(directory)
            assert str(app.query_one("#status-stage", Static).content) == app.tr(Stage.COMPLETED.label)
            assert not app.query_one("#start-run", Button).disabled
            assert app.query_one("#stop-run", Button).disabled
            app.activate_tab("results-tab")
            await pilot.pause()
            assert app.query_one("#result-summary", Static).region.height > 0
