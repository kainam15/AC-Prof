"""Best-effort run webhook events isolated from runtime and measurement logic."""
from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass, replace

from acprof.host.orchestrator import MatrixProgress
from acprof.host.profiler_progress import ProfilerProgress
from acprof.notifications import (
    NotificationConfigError,
    NotificationError,
    NotificationEvent,
    WeComWebhookNotifier,
)


@dataclass
class _RunNotificationContext:
    """Mutable lifecycle state; the webhook itself is never logged or persisted."""

    notifier: WeComWebhookNotifier
    model_id: str
    output_dir: str
    started_at: float
    run_command: str = ""
    total_cases: int | None = None
    event: NotificationEvent | None = None


_ACTIVE_RUN_NOTIFICATION: _RunNotificationContext | None = None


def active() -> bool:
    """Whether a current run has an enabled notification sink."""
    return _ACTIVE_RUN_NOTIFICATION is not None


def reset() -> None:
    """Clear the previous run's sensitive notification context."""
    global _ACTIVE_RUN_NOTIFICATION
    _ACTIVE_RUN_NOTIFICATION = None


def activate(
    *,
    provider: str,
    model_id: str,
    output_dir: str,
    started_at: float,
    run_command: str,
) -> None:
    """Configure a notifier without making any network request."""
    global _ACTIVE_RUN_NOTIFICATION

    _ACTIVE_RUN_NOTIFICATION = None
    if provider == "none":
        return
    if provider not in {"auto", "wecom"}:
        print(f"[notify][WARN] Unsupported notification provider: {provider}")
        return

    try:
        notifier = WeComWebhookNotifier.from_env()
    except NotificationConfigError as exc:
        if provider == "wecom":
            print(f"[notify][WARN] 企业微信通知未启用：{exc}", file=sys.stderr)
        return

    _ACTIVE_RUN_NOTIFICATION = _RunNotificationContext(
        notifier=notifier,
        model_id=model_id,
        output_dir=output_dir,
        started_at=started_at,
        run_command=run_command,
    )
    print(
        "[notify] 企业微信通知已启用；实验开始、每个 profiler 阶段和 case "
        "结束后、实验结束时发送通知"
    )


def notify_started() -> None:
    """Report the command before preflight/build work and any measurements."""
    context = _ACTIVE_RUN_NOTIFICATION
    if context is None:
        return

    event = NotificationEvent(
        status="started",
        model_id=context.model_id,
        output_dir=context.output_dir,
        elapsed_seconds=time.perf_counter() - context.started_at,
        run_command=context.run_command,
        detail="命令已启动，正在执行环境预检",
    )
    _send_notification_event(
        context,
        event,
        success_message="[notify] 企业微信实验开始通知已发送",
    )


def update_plan(
    *,
    model_id: str,
    output_dir: str,
    total_cases: int,
) -> None:
    context = _ACTIVE_RUN_NOTIFICATION
    if context is None:
        return
    context.model_id = model_id
    context.output_dir = output_dir
    context.total_cases = total_cases


def _result_status_counts(result_csv: str) -> tuple[int, int]:
    result_rows = 0
    error_rows = 0
    with open(result_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            result_rows += 1
            if str(row.get("status") or "").strip().lower() == "error":
                error_rows += 1
    return result_rows, error_rows


def record_completion(
    *,
    final_csv: str | None,
    completed_cases: int,
) -> None:
    """Create a completion event after all measurements and result writes."""
    context = _ACTIVE_RUN_NOTIFICATION
    if context is None:
        return

    result_rows: int | None = None
    error_rows: int | None = None
    detail: str | None = None
    if final_csv:
        try:
            result_rows, error_rows = _result_status_counts(final_csv)
        except (OSError, csv.Error, UnicodeError) as exc:
            detail = f"结果已生成，但通知摘要读取失败（{type(exc).__name__}）"

    if not final_csv or result_rows == 0:
        status = "no_results"
        if detail is None:
            detail = "本次运行未产生结果数据行"
    elif (
        (error_rows or 0) > 0
        or (
            context.total_cases is not None
            and completed_cases < context.total_cases
        )
    ):
        status = "partial"
    else:
        status = "success"

    context.event = NotificationEvent(
        status=status,
        model_id=context.model_id,
        output_dir=context.output_dir,
        elapsed_seconds=time.perf_counter() - context.started_at,
        total_cases=context.total_cases,
        completed_cases=completed_cases,
        result_rows=result_rows,
        error_rows=error_rows,
        final_csv=final_csv,
        detail=detail,
    )


def record_termination(status: str, detail: str) -> None:
    context = _ACTIVE_RUN_NOTIFICATION
    if context is None:
        return
    context.event = NotificationEvent(
        status=status,
        model_id=context.model_id,
        output_dir=context.output_dir,
        elapsed_seconds=time.perf_counter() - context.started_at,
        total_cases=context.total_cases,
        detail=detail,
    )


def _send_notification_event(
    context: _RunNotificationContext,
    event: NotificationEvent,
    *,
    success_message: str,
) -> None:
    """Deliver one event without allowing notification errors to escape."""
    try:
        context.notifier.send(event)
    except NotificationError as exc:
        print(f"[notify][WARN] 企业微信通知发送失败：{exc}", file=sys.stderr)
    except Exception as exc:
        # Unknown provider errors may embed request details.  Print only the
        # exception type so credentials can never be copied into terminal logs.
        print(
            "[notify][WARN] 企业微信通知发送失败："
            f"{type(exc).__name__}",
            file=sys.stderr,
        )
    else:
        print(success_message)


def notify_profiler(progress: ProfilerProgress) -> None:
    """Report one finished profiler stage before any next probe or CSV sweep."""
    context = _ACTIVE_RUN_NOTIFICATION
    if context is None:
        return

    event = NotificationEvent(
        status=f"profiler_{progress.status}",
        model_id=context.model_id,
        output_dir=context.output_dir,
        elapsed_seconds=time.perf_counter() - context.started_at,
        profiler=progress.profiler,
        profile_elapsed_seconds=progress.elapsed_seconds,
        profile_samples=progress.total_samples,
        profile_error_samples=progress.error_samples,
        detail=progress.detail or None,
    )
    _send_notification_event(
        context,
        event,
        success_message=(
            f"[notify] 企业微信 Profiler 通知已发送：{progress.profiler} "
            f"({progress.status})"
        ),
    )


def notify_case(progress: MatrixProgress) -> None:
    """Send progress only after run_single_case has stopped its resources."""
    context = _ACTIVE_RUN_NOTIFICATION
    if context is None:
        return

    result_rows: int | None = None
    error_rows: int | None = None
    summary_note = ""
    if progress.result_csv:
        try:
            result_rows, error_rows = _result_status_counts(progress.result_csv)
        except (OSError, csv.Error, UnicodeError) as exc:
            summary_note = f"；结果摘要读取失败（{type(exc).__name__}）"

    detail = (
        f"刚完成：CPU={progress.cpu}, MEM={progress.mem}GB, GPU={progress.gpu}"
        f"{summary_note}"
    )
    event = NotificationEvent(
        status="progress",
        model_id=context.model_id,
        output_dir=context.output_dir,
        elapsed_seconds=time.perf_counter() - context.started_at,
        total_cases=progress.total_cases,
        completed_cases=progress.completed_cases,
        result_rows=result_rows,
        error_rows=error_rows,
        detail=detail,
    )
    _send_notification_event(
        context,
        event,
        success_message=(
            "[notify] 企业微信进度通知已发送："
            f"case {progress.completed_cases}/{progress.total_cases}"
        ),
    )


def deliver(terminal_log: str | None = None) -> None:
    """Send the recorded event best-effort without changing command outcome."""
    context = _ACTIVE_RUN_NOTIFICATION
    if context is None or context.event is None:
        return

    event = context.event
    if terminal_log:
        event = replace(event, terminal_log=terminal_log)
    _send_notification_event(
        context,
        event,
        success_message="[notify] 企业微信最终通知已发送",
    )
