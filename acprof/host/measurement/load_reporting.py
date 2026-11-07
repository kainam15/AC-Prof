"""Post-load record completion, streaming finalization and report construction."""
from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from acprof.analysis.streaming import distribution, streaming_summary
from acprof.host.measurement.streaming import (
    OpenAIStreamingAdapter,
    StreamProtocolError,
    StreamTokenCountError,
)
from acprof.system.cleanup import cleanup_on_exit

if TYPE_CHECKING:
    from acprof.experiments.load import LoadConfig


def complete_records(
    records: list[dict[str, Any] | None], token: str, schedule: list[float] | None,
) -> list[dict[str, Any]]:
    """Retain offered request order, including requests never submitted."""
    for index, record in enumerate(records):
        if record is None:
            records[index] = {
                "request_id": f"{token}:{index}", "planned_s": schedule[index] if schedule else None,
                "sent_s": None, "completed_s": None, "status": "cancelled",
                "error": "not submitted", "connection": None,
            }
    return [record for record in records if record is not None]


def _finalize_streams(records: list[dict[str, Any]], adapter: OpenAIStreamingAdapter) -> None:
    """Derive token timings only after the load duration has been frozen."""
    for record in records:
        if record["status"] != "ok":
            adapter.invalidate(record, record.get("error_code", record["status"]))
            continue
        try:
            adapter.finalize(record)
        except StreamTokenCountError as error:
            cause = error.original
            record.update(output_tokens=None, output_token_count_source="unavailable",
                          token_count_scope="unavailable", tpot_s=None, itl_s=None,
                          tpot_precision="unavailable", itl_precision="unavailable",
                          token_count_error=f"{type(cause).__name__}: {cause}")
            record["missing_reasons"].update(tpot_s="tokenizer_error", itl_s="tokenizer_error")
        except StreamProtocolError as error:
            record.update(status="error", error=f"{type(error).__name__}: {error}",
                          error_code=error.code, stream_error_code=error.code)
            adapter.invalidate(record, error.code)


def finalize_streams(
    records: list[dict[str, Any]], adapter: OpenAIStreamingAdapter,
    primary_error: BaseException | None,
) -> BaseException | None:
    """Preserve cancellation and both failures when post-load derivation stops."""
    try:
        with cleanup_on_exit(
            lambda: _finalize_streams(records, adapter), description="streaming result finalization",
        ):
            if primary_error is not None:
                raise primary_error
    except BaseException as error:
        return error
    return None


def build_load_report(
    records: list[dict[str, Any]], config: LoadConfig, protocol: dict[str, Any],
    protocol_sha256: str, duration: float, *, streaming: bool,
    primary_error: BaseException | None, include_capacity: bool,
    slo_thresholds: dict[str, float | None] | None,
) -> dict[str, Any]:
    counts = Counter(record["status"] for record in records)
    succeeded = [record for record in records if record["status"] == "ok"]
    sent = sum(record["sent_s"] is not None for record in records)
    report = {
        "schema_version": 2 if streaming else 1,
        "kind": "stream_load_result" if streaming else "nonstream_load_result",
        "protocol": protocol, "protocol_sha256": protocol_sha256,
        "successful": primary_error is None and counts["ok"] == config.requests,
        "duration_s": duration, "target_requests_per_s": config.rate,
        "sent_requests_per_s": sent / duration if duration > 0 else None,
        "completed_requests_per_s": sum(record["http_status"] is not None for record in records
            if "http_status" in record) / duration if duration > 0 else None,
        "successful_requests_per_s": counts["ok"] / duration if duration > 0 else None,
        "success_rate": counts["ok"] / config.requests,
        "counts": {"offered": config.requests, "sent": sent, "succeeded": counts["ok"],
                   "failed": counts["error"], "dropped": counts["dropped"], "cancelled": counts["cancelled"]},
        "latency_s": distribution([record["completed_s"] - record["sent_s"] for record in succeeded]),
        "scheduled_latency_s": distribution([record["completed_s"] - record["planned_s"] for record in succeeded]),
        "queue_delay_s": distribution([record["sent_s"] - record["planned_s"] for record in records
                                       if record["sent_s"] is not None]),
        "connection_count": len({tuple(record["connection"]) for record in records if record["connection"]}),
        "requests": records,
    }
    if streaming:
        report["streaming"] = streaming_summary(records)
        report["completed_requests_per_s"] = sum(
            row.get("stream_completed_s") is not None for row in records
        ) / duration if duration > 0 else None
        if include_capacity:
            from acprof.analysis.streaming_capacity import build_capacity
            report["capacity"] = build_capacity(records, duration, config.concurrency, slo_thresholds)
    return report
