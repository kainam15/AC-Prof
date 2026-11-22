"""One continuous synthetic TUI journey using actual Pilot clicks and UI callbacks.

This module deliberately avoids all real model preparation, downloads, subprocesses,
or GPU sampling. Production ownership checks read only synthetic fixture artifacts.
"""

from __future__ import annotations

import asyncio
import io
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from rich.console import Console
from textual.widgets import Button, Input, Static, TabbedContent

from acprof.events.download import DownloadProgress
from acprof.events.preparation import encode_event, parse_event
from acprof.experiments.configuration import RunConfig, build_run_command
from acprof.tui.commands import PendingLaunch
from acprof.tui.run.progress import ProgressSnapshot, RunProgressTracker
from acprof.tui.run.recovery_controller import RecoveryScreen
from acprof.tui.run.results import RunArtifacts
from acprof.tui.run.stages import Stage
from acprof.tui.ui.progress_widgets import DownloadProgressPanel
from acprof.tui.ui.structured import FactGrid
from scripts.tui_pilot_font import use_cjk_svg_font
from tests.support.tui_fixtures import AcprofTui, fact_rows, write_interrupted_preparation
from tests.support.tui_run_artifacts import publish_synthetic_window


async def finish_attempt(app, pilot, *, pid: int, published_pid: int,
                         metrics_complete: bool, expected: Stage) -> Path:
    """Drive the worker's post-exit attribution and the visible monitor from fixture files."""
    session = app.execution.session
    run_dir = session.config.result_dir(app.project_dir)
    before = RunArtifacts.read(run_dir)
    previous_selection = app.query_one("#experiment-source", Input).value
    verifying = replace(session.snapshot, stage=Stage.VERIFYING_ARTIFACT, measurement_active=False)
    app.consume_process_line("", verifying, True)
    await pilot.pause()
    stage = app.query_one("#status-stage", Static)
    assert app.tr(Stage.VERIFYING_ARTIFACT.label) in str(stage.content)
    source = publish_synthetic_window(
        run_dir, model=session.config.model, pid=published_pid, metrics_complete=metrics_complete,
    )
    app.execution.lifecycle.process = None
    await asyncio.to_thread(app.process.inspect_finished_run, before, pid, session.token)
    app.process_finished("run", 0, verifying, "")
    await app.workers.wait_for_complete()
    await pilot.pause()
    attributed = expected is not Stage.FAILED
    assert session.result.belongs_to_attempt is attributed
    assert session.result.result_csv == (str(source) if attributed else "")
    assert session.result.completed_cases == int(attributed)
    assert app.query_one("#experiment-source", Input).value == (
        str(source.parent) if attributed else previous_selection
    )
    assert session.snapshot.stage is expected
    assert app.tr(expected.label) in str(stage.content)
    assert app.query_one("#stop-run", Button).disabled
    assert app.query_one("#main-tabs", TabbedContent).active == "monitor-tab"
    return source


async def exercise_journey(
    *,
    workspace: Path,
    size: tuple[int, int] = (120, 36),
    language: str = "zh",
    capture: Path | None = None,
) -> list[dict]:
    """Exercise recovery -> monitor -> confirmation -> download -> measurement -> result.

    Snapshot metadata records *observable* UI state, not synthetic assertions of
    real collection. Synthetic artifacts go through the normal result inspection. Test-only SVGs are
    review evidence and not long-lived golden baselines.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    old_dir = workspace / "previous-experiment"
    config = replace(RunConfig.smoke("fixture/model"), output_dir=str(old_dir))
    app = AcprofTui(config, settings_path=workspace / "settings.json")
    app.configuration.preferences = replace(
        app.configuration.preferences,
        language=language,
        theme="acprof-graphite",
        show_command_bar=False,
    )
    records: list[dict] = []

    with (
        patch("acprof.host.run_state.host_identity", return_value={"source_sha256": "synthetic"}),
        patch.object(app.process, "execute_command") as execute,
    ):
        destination = write_interrupted_preparation(config)
        old_state = destination / ".acprof/run_state.json"
        saved_state = old_state.read_bytes()
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            tabs = app.query_one("#main-tabs", TabbedContent)

            async def record(step: str, action: str, *, screen: str = "_default") -> None:
                await pilot.pause()
                assert app.screen.id == screen or app.screen.__class__.__name__ == screen, step
                filename = f"{step}.svg"
                selected = app.screen.query("Button")
                buttons = [
                    button.id
                    for button in selected
                    if button.display and button.region.width and not button.disabled
                ]
                controls = {}
                for target in (
                    "#run-plan-panel",
                    "#status-grid",
                    "#download-progress",
                    "#run-log",
                    "#result-summary",
                    "#recovery-actions",
                ):
                    found = app.screen.query(target)
                    if found and found.first().display and found.first().region.height:
                        region = found.first().region
                        controls[target[1:]] = [region.x, region.y, region.width, region.height]
                focus = app.focused.id if app.focused is not None else None
                entry = {
                    "id": step,
                    "action": action,
                    "page": tabs.active,
                    "screen": app.screen.__class__.__name__,
                    "focus": focus,
                    "buttons": buttons[:12],
                    "regions": controls,
                    "terminal": [pilot.app.size.width, pilot.app.size.height],
                    "image": filename if capture is not None else None,
                }
                records.append(entry)
                if capture is not None:
                    capture.mkdir(parents=True, exist_ok=True)
                    svg_path = Path(app.save_screenshot(filename=filename, path=str(capture)))
                    svg_path.write_text(
                        use_cjk_svg_font(svg_path.read_text(encoding="utf-8")), encoding="utf-8"
                    )
                    # Keep a faithful ANSI rendering of this same Pilot state.
                    # VHS/xterm.js renders it with the fixed Noto font; rasterizing
                    # Rich SVG directly still suffers from SVG textLength CJK drift.
                    terminal = io.StringIO()
                    console = Console(
                        width=pilot.app.size.width,
                        height=pilot.app.size.height,
                        file=terminal,
                        force_terminal=True,
                        color_system="truecolor",
                        legacy_windows=False,
                        safe_box=False,
                    )
                    console.print(
                        app.screen._compositor.render_update(
                            full=True, screen_stack=app._background_screens
                        )
                    )
                    svg_path.with_suffix(".ansi").write_bytes(
                        ("\x1b[?25l" + terminal.getvalue().rstrip("\n")).encode("utf-8")
                    )

            assert tabs.active == "run-tab"
            await record("configuration", "entered experiment configuration")
            start = app.query_one("#start-run", Button)
            assert not start.disabled
            assert await pilot.click(start)
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert isinstance(app.screen, RecoveryScreen)
            assert tabs.active == "run-tab"  # Recovery is a modal, not a tab.
            execute.assert_not_called()
            retry = app.screen.query_one("#recovery-resume", Button)
            assert not retry.disabled
            assert str(retry.label) == ("重试准备" if language == "zh" else "Retry preparation")
            assert not app.screen.query_one("#recovery-new", Button).disabled
            await record(
                "launch-review",
                "clicked Start with an existing destination",
                screen="RecoveryScreen",
            )
            assert await pilot.click("#recovery-new")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.screen.id == "_default"
            assert tabs.active == "monitor-tab"
            assert app.execution.session.kind == "run"
            execute.assert_called_once()  # No actual subprocess was started.
            fresh = execute.call_args.kwargs["config"]
            assert not fresh.resume
            assert fresh.result_dir(Path.cwd()) != destination
            assert old_state.read_bytes() == saved_state
            await record("monitor-started", "chose New Experiment; monitor opened")

            fake_process = Mock(pid=12345, stdin=io.StringIO())
            fake_process.poll.return_value = None
            app.execution.lifecycle.process = fake_process
            app.consume_process_line(
                "",
                ProgressSnapshot(
                    stage=Stage.PREPARING_MODEL,
                    interface_status="running",
                    detail="Synthetic model preparation",
                ),
                True,
            )
            report = {
                "expected_download_bytes": 123_000_000,
                "max_download_bytes": 500_000_000,
                "model": {
                    "total_bytes": 250_000_000,
                    "cached_bytes": 127_000_000,
                    "endpoint": "https://huggingface.co/fixture/model",
                },
                "disk": {
                    "total_bytes": 1_000_000_000,
                    "free_bytes": 800_000_000,
                    "reclaimable_bytes": 10_000_000,
                    "remaining_bytes": 700_000_000,
                },
                "model_store_path": "/fixture/model-store",
                "runtime": {"kind": "process", "environment_local": True},
                "sources": [{"category": "model", "url": "https://huggingface.co/fixture/model"}],
                "probe_payload_bytes": 1024,
                "probe_byte_limit": 262144,
            }
            app.preflight.preparation_event(
                {
                    "stage": "model",
                    "status": "waiting",
                    "request": {
                        "id": 1,
                        "kind": "review",
                        "resolved": True,
                        "download_report": report,
                        "questions": [],
                    },
                }
            )
            await record(
                "download-review", "synthetic download review request", screen="PreparationScreen"
            )
            assert not app.screen.query_one("#preparation-continue", Button).disabled
            assert await pilot.click("#preparation-continue")
            await pilot.pause()
            assert '"action": "confirm"' in fake_process.stdin.getvalue()
            assert app.screen.id == "_default"
            app.preflight.download_progress(
                DownloadProgress(
                    "fixture/model",
                    "huggingface",
                    "downloading",
                    250_000_000,
                    25_000_000,
                    123_000_000,
                    100_000_000,
                    12_000_000,
                    1,
                    2,
                )
            )
            await pilot.pause()
            assert app.query_one("#download-progress", DownloadProgressPanel).display
            assert tabs.active == "monitor-tab"
            await record("download-progress", "confirmed download; synthetic progress event")

            # Deliver the same tracker update and UI callback as a preparation event from stdout.
            tracker = RunProgressTracker()
            tracker.feed(encode_event("resolution", "passed"))

            def runtime_event(status, **fields):
                line = encode_event("runtime", status, **fields)
                app.consume_process_line("", tracker.feed(line), True)
                app.preflight.preparation_event(parse_event(line))

            runtime_event("running")
            await pilot.pause()
            assert not app.query_one("#download-progress", DownloadProgressPanel).display
            checks = app.query_one("#status-preparation", FactGrid)
            assert checks.display
            runtime_label = app.tr(Stage.RUNTIME_VALIDATION.label)
            running = "进行中" if language == "zh" else "Running"
            assert any(label == runtime_label and running in value for label, value in fact_rows(checks))
            extended = language == "zh" and size == (120, 36)
            if extended:
                await record("validation-running", "runtime check started; download panel cleared")
                error = "Synthetic runtime error: missing fixture weights"
                runtime_event("failed", request={"id": 2, "kind": "error", "detail": error})
                await pilot.pause()
                assert error in app.screen.query_one("#preparation-traceback").text
                assert app.execution.session.snapshot.runtime_status == "failed"
                assert not app.query_one("#download-progress").display
                await record("validation-failed", "runtime error displayed; retry available",
                             screen="PreparationScreen")
                assert await pilot.click("#preparation-continue")
                await pilot.pause()
                assert '"action": "retry"' in fake_process.stdin.getvalue()
                runtime_event("running")
            runtime_event("passed")
            await pilot.pause()
            passed = "通过" if language == "zh" else "Passed"
            assert any(label == runtime_label and passed in value for label, value in fact_rows(checks))
            assert not app.query_one("#download-progress").display
            if extended:
                await record("validation-passed", "retry passed in the same simulated subprocess")

            measuring = ProgressSnapshot(
                stage=Stage.MEASURING,
                detail="Synthetic measurement; no real samples",
                current_case=1,
                total_cases=1,
                cpu="1",
                mem="4",
                gpu="off",
                measurement_active=True,
                interface_status="passed",
                runtime_status="passed",
                measurement_status="running",
            )
            app.consume_process_line("", measuring, True)
            await pilot.pause()
            assert not app.query_one("#download-progress", DownloadProgressPanel).display
            assert app.query_one("#run-log").suspended
            assert app.query_one("#monitor-cases", FactGrid).display
            await record("measurement", "synthetic measurement boundary")
            assert await pilot.click("#nav-run-tab")
            await pilot.pause()
            assert tabs.active == "run-tab"
            assert app.query_one("#start-run", Button).disabled
            await record(
                "measurement-config",
                "clicked Configuration during measurement; start remains disabled",
            )
            assert await pilot.click("#nav-monitor-tab")
            await pilot.pause()
            assert tabs.active == "monitor-tab"
            assert app.execution.session.snapshot.measurement_active

            result_csv = await finish_attempt(
                app, pilot, pid=fake_process.pid, published_pid=fake_process.pid,
                metrics_complete=True, expected=Stage.COMPLETED,
            )
            await record("finished", "matching synthetic attempt, window and capability evidence")
            assert await pilot.click("#nav-results-tab")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert tabs.active == "results-tab"
            assert app.query_one("#experiment-source", Input).value == str(result_csv.parent)
            data_status = str(app.query_one("#result-data-status", Static).content)
            assert ("有效正式窗口：1" if language == "zh" else "Valid formal windows: 1") in data_status, (
                "Unexpected data status after Pilot selection: " + repr(data_status)
            )
            await record("result-overview", "clicked Results tab; read synthetic CSV")
            if size[0] > 80:
                await pilot.resize_terminal(80, 24)
                await pilot.pause()
                action = app.query_one("#report-calculate", Button)
                assert action.region.width > 0
                assert action.region.bottom <= 24
                await record("results-small", "resized results page to 80x24")
            execute.assert_called_once()
            if extended:
                await pilot.resize_terminal(*size)
                for step, expected, published_pid, metrics_complete, action in (
                    ("finish-partial", Stage.PARTIAL, fake_process.pid, False,
                     "new synthetic attempt; persisted capability report marks requested metrics incomplete"),
                    ("finish-failed", Stage.FAILED, fake_process.pid + 1, True,
                     "new synthetic attempt; existing results belong to a different process"),
                ):
                    next_config = replace(config, output_dir=str(workspace / step))
                    command = build_run_command(next_config, project_dir=app.project_dir)
                    app.launch(PendingLaunch(tuple(command), "run", next_config))
                    await pilot.pause()
                    assert await pilot.click("#clear-log")
                    await finish_attempt(
                        app, pilot, pid=fake_process.pid, published_pid=published_pid,
                        metrics_complete=metrics_complete, expected=expected,
                    )
                    await record(step, action)
                assert execute.call_count == 3
    return records
