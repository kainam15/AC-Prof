"""Shared HTTP load scheduling, separate from formal energy measurement windows."""
from __future__ import annotations

import hashlib
import json
import threading
import time
import weakref
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from queue import Empty, SimpleQueue
from typing import Any
from urllib.parse import urlsplit

from acprof.experiments.load import MAX_SWEEP_POINTS, LoadConfig
from acprof.experiments.streaming import request_digest, validate_request_credentials
from acprof.host.measurement.load_reporting import (
    build_load_report,
    complete_records,
    finalize_streams,
)
from acprof.host.measurement.load_transport import LoadTransport
from acprof.host.measurement.streaming import OpenAIStreamingAdapter
from acprof.system.cleanup import cleanup_on_exit

__all__ = ["MAX_SWEEP_POINTS", "LoadConfig", "run_load"]


def _request_headers(headers: dict[str, str] | None) -> dict[str, str]:
    extra_headers = dict(headers or {})
    reserved = {"content-type", "connection", "x-req-id", "host", "content-length", "transfer-encoding"}
    if any(not isinstance(key, str) or key.lower() in reserved
           or not isinstance(value, str) or "\r" in key + value or "\n" in key + value
           for key, value in extra_headers.items()):
        raise ValueError("invalid or reserved load request header")
    return extra_headers


def _release_completed_requests(
    pending: set[Future[None]], completed: SimpleQueue[Future[None]],
) -> None:
    while True:
        try:
            future = completed.get_nowait()
        except Empty:
            return
        future.result()
        pending.remove(future)


def _submit_requests(
    transport: LoadTransport, pool: ThreadPoolExecutor, schedule: list[float] | None,
    pending: set[Future[None]],
) -> None:
    config = transport.config
    completed: SimpleQueue[Future[None]] = SimpleQueue()
    completed_ref = weakref.ref(completed)

    def record_completion(finished_future: Future[None]) -> None:
        # Futures retain callbacks; a weak reference avoids a queue/Future cycle.
        queue = completed_ref()
        if queue is not None:
            queue.put(finished_future)

    for index in range(config.requests):
        if schedule is not None:
            planned = schedule[index]
            remaining = transport.epoch + planned - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            _release_completed_requests(pending, completed)
            if len(pending) >= config.max_pending:
                transport.records[index] = {
                    "request_id": f"{transport.token}:{index}", "planned_s": planned,
                    "sent_s": None, "completed_s": None, "status": "dropped",
                    "error": "max_pending exceeded", "connection": None,
                }
                continue
        else:
            if len(pending) >= config.concurrency:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                pending.difference_update(finished)
                for future in finished:
                    future.result()
            planned = time.perf_counter() - transport.epoch
        future = pool.submit(transport.execute, index, planned)
        pending.add(future)
        if schedule is not None:
            future.add_done_callback(record_completion)
    for future in pending:
        future.result()


def _execute_load(
    transport: LoadTransport, schedule: list[float] | None,
) -> tuple[float, BaseException | None]:
    config = transport.config
    workers = min(config.concurrency, config.requests)
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="acprof-load")
    pending: set[Future[None]] = set()
    primary_error: BaseException | None = None
    transport.epoch = time.perf_counter()
    barrier = threading.Barrier(workers + 1)
    try:
        # Prestart only workers this finite request budget can use, before timing.
        ready = [pool.submit(barrier.wait) for _ in range(workers)]
        barrier.wait()
        for ready_future in ready:
            ready_future.result()
        transport.epoch = time.perf_counter()
        _submit_requests(transport, pool, schedule, pending)
    except BaseException as error:
        primary_error = error
        barrier.abort()
        transport.cancelled.set()
        for future in pending:
            future.cancel()
        transport.interrupt()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        duration = time.perf_counter() - transport.epoch
        # Workers have joined; preserve their evidence and failure while closing sockets.
        try:
            with cleanup_on_exit(transport.close, description="HTTP load connections"):
                if primary_error is not None:
                    raise primary_error
        except BaseException as error:
            primary_error = error
    return duration, primary_error


def _request_bodies(payload, adapter: OpenAIStreamingAdapter | None, count: int) -> tuple[bytes | tuple[bytes, ...], str | None]:
    if not isinstance(payload, tuple):
        prepared = adapter.prepare_payload(payload) if adapter is not None else payload
        return json.dumps(prepared, separators=(",", ":"), allow_nan=False).encode(), None
    if adapter is None:
        raise ValueError("request datasets require an OpenAI streaming adapter")
    if len(payload) < count:
        raise ValueError("request dataset has fewer entries than the request budget")
    selected = []
    model: str | None = None
    for index, value in enumerate(payload):
        row = adapter.prepare_payload(value)
        if model is not None and row["model"] != model:
            raise ValueError("request dataset must use one served model")
        model = row["model"]
        validate_request_credentials(row)
        # Validate unused rows too, retaining only this phase's normalized prefix.
        if index < count:
            selected.append(row)
    encoded = tuple(json.dumps(row, separators=(",", ":"), allow_nan=False).encode() for row in selected)
    return encoded, request_digest(selected)


def run_load(
    url, payload, config: LoadConfig, *, token,
    adapter: OpenAIStreamingAdapter | None = None,
    headers: dict[str, str] | None = None,
    slo_thresholds: dict[str, float | None] | None = None,
    include_capacity: bool = True,
) -> dict[str, Any]:
    """Schedule requests, close workers, then derive and report completed evidence.

    Timestamps are client call boundaries, not NIC timestamps. The default
    /predict JSON protocol and SSE adapter share the scheduler and transport;
    failures never contribute to successful latency distributions.
    """
    config.validate()
    if adapter is None and slo_thresholds is not None:
        raise ValueError("SLO thresholds require streaming load")
    if not include_capacity and slo_thresholds is not None:
        raise ValueError("capacity-disabled load cannot evaluate SLO thresholds")
    target = urlsplit(url)
    if target.scheme not in {"http", "https"} or not target.hostname or target.username or target.query or target.fragment:
        raise ValueError("load target must be an HTTP(S) base URL without credentials, query or fragment")
    extra_headers = _request_headers(headers)
    body, inputs_digest = _request_bodies(payload, adapter, config.requests)
    protocol = config.protocol()
    if adapter is not None:
        protocol = {**protocol, "kind": "openai_stream_http_load", "adapter": adapter.protocol(),
                    "tokenizer_count_phase": "after_load_duration"}
    if inputs_digest is not None:
        protocol.update(request_selection="listed_prefix", request_inputs_sha256=inputs_digest)
    protocol_sha256 = hashlib.sha256(json.dumps(protocol, sort_keys=True, allow_nan=False).encode()).hexdigest()
    schedule = config.schedule()
    transport = LoadTransport(target, body, config, token=token, adapter=adapter, headers=extra_headers)
    duration, primary_error = _execute_load(transport, schedule)
    records = complete_records(transport.records, token, schedule)
    if inputs_digest is not None:
        for index, record in enumerate(records):
            record["input_index"] = index
    if adapter is not None:
        primary_error = finalize_streams(records, adapter, primary_error)
    report = build_load_report(
        records, config, protocol, protocol_sha256, duration, streaming=adapter is not None,
        primary_error=primary_error, include_capacity=include_capacity, slo_thresholds=slo_thresholds,
    )
    if inputs_digest is not None:
        report["schema_version"] = 3
    if primary_error is not None:
        setattr(primary_error, "load_report", report)
        raise primary_error
    return report
