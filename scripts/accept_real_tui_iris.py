"""Manually accept one bounded *real* TUI -> Docker CPU -> results workflow.

Unlike tests/tui/workflows/visual_journey.py, this launches the real CLI,
resolves actual remote metadata, and runs a real model/backend. Only the test
driver observes modal/progress/completion events; no fake progress is injected.
All captures are outside formal measurement windows. Not a CI test.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, fields, replace
from datetime import UTC, datetime
from pathlib import Path

from textual.screen import ModalScreen
from textual.widgets import Button, Static, TabbedContent

from acprof.experiments.configuration import RunConfig, build_run_command
from acprof.results.layers import read_result_layers
from acprof.tui.app import AcprofTui
from acprof.tui.logging.widget import SelectableLog
from acprof.tui.run.preparation import PreparationScreen
from acprof.tui.run.recovery_controller import RecoveryScreen
from acprof.tui.run.stages import Stage
from acprof.tui.ui.dialogs import ConfirmActionScreen

REVISION = "273d867c65e8e5fa5c64c23dcf19738282452beb"
MODEL = "Ritual-Net/iris-classification"
MAX_DOWNLOAD_BYTES = 5_000_000


def _utc() -> str:
    return datetime.now(UTC).isoformat()


def _write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def check_closed_attempt(attempt: dict) -> None:
    """Require observed process exit and production ownership release."""
    if (attempt["child_returncode"] is None
            or attempt["child_returncode"] != attempt["returncode"]
            or not attempt["lifecycle_released"]
            or not attempt["coordinator_idle"]
            or not attempt["fresh_token"]):
        raise RuntimeError("attempt did not exit and release its execution ownership")


def check_cancelled_attempt(attempt: dict, stages: list[dict]) -> None:
    check_closed_attempt(attempt)
    result = attempt["run_result"]
    if (not attempt["stop_requested"]
            or attempt["snapshot"]["stage"] is not Stage.STOPPED
            or any(item["measurement_active"] for item in stages
                   if item["attempt"] == attempt["number"])
            or result["complete"] or result["completed_cases"]):
        raise RuntimeError("cancelled preparation must stop before formal measurement")
    attempt["cancelled_measurement_rows"] = 0
    if result["result_csv"]:
        source = Path(result["result_csv"]).resolve()
        if (not result["belongs_to_attempt"]
                or not source.is_relative_to(Path(attempt["result_root"]))):
            raise RuntimeError("cancelled result does not belong to its attempt")
        _, rows = read_result_layers(source.parent)
        attempt["cancelled_measurement_rows"] = len(rows)
        if rows:
            raise RuntimeError("cancelled preparation has persisted measurement rows")


def check_result_owner(selected_csv: str, attempt: dict) -> None:
    result = attempt["run_result"]
    if (not selected_csv or selected_csv != result["result_csv"]
            or not result["belongs_to_attempt"] or not result["complete"]
            or not Path(selected_csv).resolve().is_relative_to(Path(attempt["result_root"]))):
        raise RuntimeError("selected result does not belong to the successful attempt")


class ObservingApp(AcprofTui):
    """Observe the *production* lifecycle; never alter execution callbacks."""

    def __init__(self, config: RunConfig, *, settings_path: Path) -> None:
        self.events: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
        self.download_events: list[dict] = []
        self.stage_events: list[dict] = []
        self.attempts: list[dict] = []
        self._attempt_tokens: list[object] = []
        self._attempt_processes: list[subprocess.Popen[str]] = []
        super().__init__(config, settings_path=settings_path)
        actual = self.preflight.download_progress

        def observe(progress) -> None:
            actual(progress)
            # Append an in-memory event only; do not issue I/O in measurement.
            self.download_events.append(
                {
                    "at": _utc(),
                    "execution_attempt": len(self.attempts),
                    **asdict(progress),
                    "widget_visible": bool(self.query_one("#download-progress").display),
                }
            )
            if progress.state in {"downloading", "verifying", "complete"}:
                self.events.put_nowait(("download", progress))

        self.preflight.download_progress = observe  # type: ignore[method-assign]

    def push_screen(self, screen, *args, **kwargs):
        result = super().push_screen(screen, *args, **kwargs)
        if isinstance(screen, ModalScreen):
            self.events.put_nowait(("modal", screen))
        return result

    def process_started(self, pid, kind):
        if kind == "run":
            session = self.execution.session
            process = self.execution.lifecycle.process
            if (process is None or process.pid != pid
                    or session.token is None or session.config is None):
                raise RuntimeError("started process has no matching execution ownership")
            self.attempts.append({
                "number": len(self.attempts) + 1,
                "started_at": _utc(),
                "pid": pid,
                "command": list(session.command),
                "result_root": str(session.config.result_dir(self.project_dir).resolve()),
                "fresh_token": all(session.token is not old for old in self._attempt_tokens),
            })
            self._attempt_tokens.append(session.token)
            self._attempt_processes.append(process)
        super().process_started(pid, kind)

    def consume_process_line(self, line, snapshot, state_changed):
        super().consume_process_line(line, snapshot, state_changed)
        if snapshot is not None and state_changed:
            # No screenshots, filesystem I/O or metric reads during measurement.
            self.stage_events.append(
                {
                    "at": _utc(),
                    "attempt": len(self.attempts),
                    "stage": snapshot.stage.value,
                    "measurement_active": snapshot.measurement_active,
                    "interface": snapshot.interface_status,
                    "runtime": snapshot.runtime_status,
                    "measurement": snapshot.measurement_status,
                }
            )

    def process_finished(self, kind, returncode, snapshot, launch_error):
        stopped = self.execution.session.stop_requested
        super().process_finished(kind, returncode, snapshot, launch_error)
        completion = {"kind": kind, "returncode": returncode, "launch_error": launch_error}
        if kind == "run" and self.attempts:
            session = self.execution.session
            self.attempts[-1].update({
                **completion,
                "finished_at": _utc(),
                "stop_requested": stopped,
                "child_returncode": self._attempt_processes[-1].poll(),
                "lifecycle_released": self.execution.lifecycle.process is None,
                "coordinator_idle": not session.kind and session.token is None,
                "snapshot": {f.name: getattr(session.snapshot, f.name) for f in fields(session.snapshot)},
                "run_result": asdict(session.result),
            })
        self.events.put_nowait(("finished", completion))


async def execute(root: Path, *, store: Path | None = None, require_download: bool = True,
                  scenario: str = "complete", rebuild_image: bool = False) -> dict:
    if scenario not in {"complete", "cancel-retry"}:
        raise ValueError(f"unknown acceptance scenario: {scenario}")
    if root.exists():
        raise FileExistsError(f"refusing to overwrite previous evidence: {root}")
    free = shutil.disk_usage(root.parent).free
    if free < 8 * 1024**3:
        raise RuntimeError(f"disk space too low for bounded real runtime validation: {free} bytes")
    if subprocess.check_output(["docker", "ps", "-q"], text=True).strip():
        raise RuntimeError("Docker has running containers; avoid concurrent hardware measurement")
    if store is not None and not store.is_dir():
        raise FileNotFoundError(f"provided Model Store does not exist: {store}")
    root.mkdir(parents=True)
    for directory in ("reports", "screenshots", "logs", "artifacts"):
        (root / directory).mkdir()
    config = replace(
        RunConfig.smoke(MODEL),
        revision=REVISION,
        runtime="docker",
        mems="1",
        input_scales="1",
        model_spec=str((Path.cwd() / "examples/onnxruntime/iris.model.json").resolve()),
        workload_spec=str((Path.cwd() / "examples/onnxruntime/iris.json").resolve()),
        max_download="5MB",
        model_store=str(store or root / "artifacts/model-store"),
        model_store_max="128MB",
        download_mode="auto",
        rebuild_image=rebuild_image,
        output_dir=str(root / "artifacts/result"),
    )
    config.validate(project_dir=Path.cwd())
    app = ObservingApp(config, settings_path=root / "artifacts/settings.json")
    receipt: dict = {
        "started_at": _utc(),
        "status": "started",
        "scenario": scenario,
        "single_tui_session": True,
        "measurement_resume_covered": False,
        "cancel_requested": False,
        "cancel_retry_verified": False,
        "fixture": False,
        "real_backend": True,
        "download_required": require_download,
        "existing_store": store is not None,
        "model": MODEL,
        "revision": REVISION,
        "scope": "Headless Textual Pilot 120x36, real Docker CPU basic inference; model transfer only when cache misses; not actual VS Code terminal",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "free_bytes_before": free,
        "settings": asdict(config),
        "decisions": [],
        "captures": [],
        "download_events": [],
        "stage_events": [],
    }
    path = root / "reports/receipt.json"

    def save() -> None:
        receipt["download_events"] = app.download_events[:]
        receipt["stage_events"] = app.stage_events[:]
        receipt["attempts"] = app.attempts[:]
        _write(path, receipt)

    def shot(label: str) -> None:
        if app.execution.session.snapshot.measurement_active:
            raise RuntimeError("refusing screenshots during formal measurement")
        prefix = f"attempt-{len(app.attempts)}-" if scenario == "cancel-retry" else ""
        file = f"{prefix}{label}.svg"
        app.save_screenshot(file, path=str(root / "screenshots"))
        receipt["captures"].append(file)

    save()
    try:
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            actual_config = app.configuration.collect_config()
            assert actual_config.model == MODEL
            assert actual_config.revision == REVISION
            assert actual_config.max_download == "5MB"
            assert actual_config.model_store == config.model_store
            assert actual_config.profiling_mode == "basic"
            assert not app.query_one("#start-run", Button).disabled
            receipt["command"] = build_run_command(
                actual_config, project_dir=app.project_dir, python_executable=app.python_executable
            )
            receipt["configuration_verified"] = True
            shot("01-configuration")
            save()
            assert await pilot.click("#start-run")
            start_time = time.monotonic()
            download_shot = False
            while True:
                kind, payload = await asyncio.wait_for(app.events.get(), timeout=220)
                if kind == "download":
                    # Progress callback is the original real parser and control event;
                    # the screenshot is only taken if preparation remains visible.
                    if not download_shot and not app.execution.session.snapshot.measurement_active:
                        await pilot.pause()
                        if app.query_one("#download-progress").display:
                            shot("04-download-progress")
                            download_shot = True
                    continue
                if kind == "modal":
                    screen = payload
                    await pilot.pause()
                    if app.screen is not screen:
                        continue  # Replaced by a newer screen, not a fresh decision.
                    assert not app.execution.session.snapshot.measurement_active
                    request = getattr(screen, "event", {}).get("request", {})
                    detail = {
                        "at": _utc(),
                        "type": type(screen).__name__,
                        "attempt": len(app.attempts),
                        "request": request,
                        "stage": getattr(screen, "event", {}).get("stage"),
                        "buttons": [
                            {"id": b.id, "disabled": b.disabled} for b in screen.query(Button)
                        ],
                    }
                    receipt["decisions"].append(detail)
                    shot(f"02-modal-{len(receipt['decisions'])}")
                    if (isinstance(screen, ConfirmActionScreen)
                            and receipt["cancel_requested"]
                            and not receipt["cancel_retry_verified"]
                            and len(app.attempts) == 1):
                        if screen.dialog_title != "终止当前任务？":
                            raise RuntimeError("unexpected confirmation during controlled cancellation")
                        detail["action"] = "confirm-stop"
                        save()
                        assert await pilot.click("#confirm-yes")
                        continue
                    if (rebuild_image and isinstance(screen, PreparationScreen)
                            and request.get("kind") == "error"
                            and request.get("failed_stage") == "image"):
                        attempt = app.attempts[-1]
                        rebuild = screen.query("#preparation-rebuild")
                        if attempt.get("image_rebuild_requested") or not rebuild or rebuild.first().disabled:
                            raise RuntimeError("explicit image rebuild did not recover preparation")
                        attempt["image_rebuild_requested"] = True
                        detail["action"] = "rebuild-image"
                        detail["requested_by"] = "--rebuild-image"
                        save()
                        assert await pilot.click("#preparation-rebuild")
                        continue
                    if isinstance(screen, RecoveryScreen) and receipt["cancel_retry_verified"]:
                        detail["recovery_phase"] = screen.review.phase
                        detail["recovery_directory"] = str(screen.review.directory)
                        if screen.review.phase != "preparation" or screen.review.resume is None:
                            raise RuntimeError("retry cannot resume the cancelled preparation")
                        detail["action"] = "resume-preparation"
                        save()
                        assert await pilot.click("#recovery-resume")
                        continue
                    if not isinstance(screen, PreparationScreen) or request.get("kind") != "review":
                        # Cancel only the child owned by this TUI, rather than
                        # abandoning a real preparation process awaiting input.
                        cancel = screen.query("#preparation-cancel")
                        if cancel and not cancel.first().disabled:
                            await pilot.click("#preparation-cancel")
                        raise RuntimeError(
                            f"unhandled real preparation screen: {type(screen).__name__}"
                        )
                    if scenario == "cancel-retry" and not receipt["cancel_requested"]:
                        if len(app.attempts) != 1:
                            raise RuntimeError("controlled cancellation must target the first attempt")
                        detail["action"] = "cancel"
                        receipt["cancel_requested"] = True
                        save()
                        assert await pilot.click("#preparation-cancel")
                        continue
                    report = request.get("download_report")
                    if isinstance(report, dict):
                        expected = report.get("expected_download_bytes")
                        # The download budget is enforced by the production CLI too.
                        if not isinstance(expected, int) or not 0 <= expected <= MAX_DOWNLOAD_BYTES:
                            raise RuntimeError(
                                f"download report exceeds bounded budget: {expected}"
                            )
                        detail["download_confirmation"] = True
                    else:
                        detail["download_confirmation"] = False
                    button = screen.query("#preparation-continue")
                    if not button or button.first().disabled:
                        raise RuntimeError(
                            "unexpected preparation review requires additional user input"
                        )
                    detail["action"] = "continue"
                    save()
                    assert await pilot.click("#preparation-continue")
                    continue
                if kind == "finished":
                    receipt["completion"] = payload
                    await app.workers.wait_for_complete()
                    await pilot.pause()
                    if not app.attempts or payload["kind"] != "run":
                        raise RuntimeError("missing real run process evidence")
                    attempt = app.attempts[-1]
                    check_closed_attempt(attempt)
                    app.logs.flush()
                    (root / f"logs/tui-attempt-{attempt['number']}.log").write_text(
                        app.query_one("#run-log", SelectableLog).raw_text, encoding="utf-8"
                    )
                    if scenario == "cancel-retry" and len(app.attempts) == 1:
                        if not receipt["cancel_requested"]:
                            raise RuntimeError("first attempt finished before controlled cancellation")
                        check_cancelled_attempt(attempt, app.stage_events)
                        containers = subprocess.check_output(["docker", "ps", "-q"], text=True).split()
                        attempt["running_containers_after_cancel"] = containers
                        if containers:
                            raise RuntimeError("containers remain after preparation cancellation")
                        assert not app.is_busy()
                        if app.query_one("#main-tabs", TabbedContent).active != "run-tab":
                            assert await pilot.click("#--content-tab-run-tab")
                            await pilot.pause()
                        assert not app.query_one("#start-run", Button).disabled
                        receipt["cancel_retry_verified"] = True
                        shot("03-preparation-cancelled")
                        save()
                        assert await pilot.click("#start-run")
                        download_shot = False
                        continue
                    receipt["snapshot"] = {
                        f.name: getattr(app.execution.session.snapshot, f.name)
                        for f in fields(app.execution.session.snapshot)
                    }
                    receipt["run_result"] = {
                        f.name: getattr(app.execution.session.result, f.name)
                        for f in fields(app.execution.session.result)
                    }
                    receipt["duration_s"] = round(time.monotonic() - start_time, 3)
                    app.logs.flush()
                    (root / "logs/tui-run.log").write_text(
                        app.query_one("#run-log", SelectableLog).raw_text, encoding="utf-8"
                    )
                    shot("05-monitor-complete")
                    receipt["backend_success"] = (
                        payload["returncode"] == 0
                        and app.execution.session.snapshot.stage is Stage.COMPLETED
                        and app.execution.session.result.belongs_to_attempt
                        and app.execution.session.result.complete
                    )
                    save()
                    break
            if receipt["backend_success"]:
                assert app.query_one("#main-tabs", TabbedContent).active == "monitor-tab"
                assert await pilot.click("#--content-tab-results-tab")
                await pilot.pause()
                assert app.query_one("#main-tabs", TabbedContent).active == "results-tab"
                receipt["selected_csv"] = app.query_one("#result-csv").value
                check_result_owner(receipt["selected_csv"], app.attempts[-1])
                receipt["selected_attempt"] = app.attempts[-1]["number"]
                if scenario == "cancel-retry":
                    assert receipt["cancel_retry_verified"]
                    assert len(app.attempts) == 2
                assert await pilot.click("#summarize-results")
                await app.workers.wait_for_complete()
                await pilot.pause()
                receipt["summary"] = str(app.query_one("#result-summary", Static).content)
                shot("06-results-overview")
                assert "成功 1" in receipt["summary"] or "OK 1" in receipt["summary"]
                assert await pilot.click("#report-calculate")
                await app.workers.wait_for_complete()
                await pilot.pause()
                receipt["statistics_rows"] = app.query_one("#report-table").row_count
                receipt["statistics_status"] = str(app.query_one("#report-status", Static).content)
                shot("07-results-statistics")
                receipt["status"] = "backend-and-ui-passed"
            else:
                receipt["status"] = "backend-failed"
            save()
    except BaseException:
        receipt["status"] = "driver-or-workflow-failed"
        receipt["exception"] = traceback.format_exc()
        save()
        raise

    if receipt["status"] != "backend-and-ui-passed":
        return receipt
    result_dir = str(Path(receipt["selected_csv"]).parent)
    checks = [
        ("result-verify", [sys.executable, "-m", "acprof", "results", "verify", result_dir]),
        (
            "result-audit",
            [
                sys.executable,
                "-m",
                "acprof",
                "audit",
                result_dir,
                "--require-complete",
                "--require-ok",
                "--json",
            ],
        ),
    ]
    for label, command in checks:
        result = subprocess.run(command, text=True, capture_output=True, timeout=45, check=False)
        (root / f"reports/{label}.stdout.txt").write_text(result.stdout, encoding="utf-8")
        (root / f"reports/{label}.stderr.txt").write_text(result.stderr, encoding="utf-8")
        receipt[label] = {"returncode": result.returncode, "command": command}
        save()
    receipt["real_download_observed"] = any(
        item["execution_attempt"] == receipt["selected_attempt"]
        and item["state"] in {"downloading", "verifying", "complete"}
        and isinstance(item["transferred_bytes"], int)
        and item["transferred_bytes"] > 0
        for item in receipt["download_events"]
    )
    receipt["download_confirmed"] = any(
        x.get("download_confirmation") and x.get("action") == "continue"
        for x in receipt["decisions"]
    )
    receipt["all_passed"] = (
        not require_download
        or (receipt["real_download_observed"] and receipt["download_confirmed"])
    ) and all(receipt[name]["returncode"] == 0 for name, _ in checks)
    receipt["status"] = "passed" if receipt["all_passed"] else "partial"
    receipt["finished_at"] = _utc()
    save()
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--scenario", choices=("complete", "cancel-retry"), default="complete",
        help="complete once, or cancel preparation and retry within the same TUI session",
    )
    parser.add_argument(
        "--rebuild-image", action="store_true",
        help="explicitly prepare a model image for this Store, allowing Docker build cache",
    )
    parser.add_argument(
        "--model-store",
        type=Path,
        help="reuse already verified Model Store instead of an empty one",
    )
    parser.add_argument(
        "--allow-cached",
        action="store_true",
        help="accept the already-verified cached download path",
    )
    args = parser.parse_args()
    record = asyncio.run(
        execute(
            args.output.resolve(),
            store=args.model_store.resolve() if args.model_store else None,
            require_download=not args.allow_cached,
            scenario=args.scenario,
            rebuild_image=args.rebuild_image,
        )
    )
    print(
        json.dumps(
            {
                "status": record.get("status"),
                "scenario": record.get("scenario"),
                "attempts": len(record.get("attempts", [])),
                "cancel_retry_verified": record.get("cancel_retry_verified"),
                "completion": record.get("completion"),
                "download_events": len(record.get("download_events", [])),
                "download_confirmed": record.get("download_confirmed"),
                "real_download_observed": record.get("real_download_observed"),
                "audit": record.get("result-audit"),
                "output": str(args.output),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if record.get("status") != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
