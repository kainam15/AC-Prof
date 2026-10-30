"""Single ownership of the TUI's active command and its subprocess.

The coordinator holds no Textual widgets or workers. The App may render its
immutable completion result, but only this object changes the active command
identity. A child that hasn't exited remains owned by ProcessLifecycle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from acprof.experiments.configuration import RunConfig
from acprof.tui.commands import PendingLaunch
from acprof.tui.run.process import ProcessLifecycle, StopResult
from acprof.tui.run.progress import ProgressSnapshot
from acprof.tui.run.results import (
    ProcessCompletion,
    RunResult,
    evaluate_process_completion,
)
from acprof.tui.run.stages import Stage


@dataclass
class ExecutionSession:
    """Per-attempt state; token prevents late output from a prior attempt."""

    kind: str = ""
    config: RunConfig | None = None
    command: tuple[str, ...] = ()
    started_monotonic: float = 0.0
    stop_requested: bool = False
    snapshot: ProgressSnapshot = field(default_factory=ProgressSnapshot)
    token: object | None = None
    result: RunResult = field(default_factory=RunResult)


class ExecutionCoordinator:
    """Own active command identity, completion and subprocess lifetime.

    Textual still schedules workers and renders results. The coordinator does
    not own a second scheduler or change the CLI measurement protocol.
    """

    def __init__(self) -> None:
        self.lifecycle = ProcessLifecycle()
        self.session = ExecutionSession()
        self._closing = False

    def begin(self, pending: PendingLaunch) -> object:
        if self._closing:
            raise RuntimeError("execution lifecycle is closing")
        if self.session.kind or self.lifecycle.process is not None:
            raise RuntimeError("an execution is already owned")
        if not pending.kind:
            raise ValueError("execution kind is required")
        token = object()
        self.session = ExecutionSession(
            kind=pending.kind,
            config=pending.config if pending.kind == "run" else None,
            command=pending.command,
            started_monotonic=time.monotonic(),
            snapshot=ProgressSnapshot(
                stage=Stage.STARTING, detail="正在创建子进程"
            ),
            token=token,
        )
        return token

    def accepts(self, token: object | None) -> bool:
        return token is not None and token is self.session.token and bool(self.session.kind)

    def accept_result(self, token: object, result: RunResult) -> bool:
        if not self.accepts(token):
            return False
        self.session.result = result
        return True

    def request_stop(self, token: object | None) -> bool:
        if not self.accepts(token):
            return False
        self.session.stop_requested = True
        return True

    def update_snapshot(self, snapshot: ProgressSnapshot) -> bool:
        """Return whether a measurement window boundary was crossed."""
        was_measuring = self.session.snapshot.measurement_active
        self.session.snapshot = snapshot
        return was_measuring != snapshot.measurement_active

    def finish(
        self,
        kind: str,
        returncode: int,
        snapshot: ProgressSnapshot | None,
        launch_error: str,
        *,
        analysis_ready: bool = False,
    ) -> ProcessCompletion:
        if not self.session.kind or kind != self.session.kind:
            raise RuntimeError("out-of-order execution completion")
        if self.lifecycle.process is not None:
            raise RuntimeError("cannot finish while a child is still owned")
        previous = self.session
        completion = evaluate_process_completion(
            kind,
            returncode,
            snapshot,
            previous.snapshot,
            launch_error=launch_error,
            stop_requested=previous.stop_requested,
            run_result=previous.result,
            analysis_ready=analysis_ready,
        )
        # Retain config and elapsed timestamp so the UI can show the final
        # attempt and perform post-run source changes. Invalidate old callbacks.
        self.session.kind = ""
        self.session.command = ()
        self.session.stop_requested = False
        self.session.token = None
        self.session.snapshot = completion.final_snapshot
        return completion

    def stop(self, *, expected_process=None) -> StopResult:
        """Stop only the expected process; delayed requests cannot target a replacement."""
        return self.lifecycle.stop(expected_process=expected_process)

    def close(self) -> StopResult:
        """Close the lifecycle even when the screen has already unmounted."""
        self._closing = True
        return self.lifecycle.stop(closing=True)
