"""Shared HTTP load scheduling, separate from formal energy measurement windows."""
from __future__ import annotations

import hashlib
import json
import math
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlsplit

from acprof.host.measurement.load_reporting import (
    build_load_report,
    complete_records,
    finalize_streams,
)
from acprof.host.measurement.load_transport import LoadTransport
from acprof.host.measurement.streaming import OpenAIStreamingAdapter
from acprof.system.cleanup import cleanup_on_exit

MAX_SWEEP_POINTS = 256


@dataclass(frozen=True)
class LoadConfig:
    scenario: str = "serial"
    connections: str = "close"
    concurrency: int = 1
    requests: int = 100
    rate: float | None = None
    arrival: str = "fixed"
    seed: int = 0
    max_pending: int = 1024
    timeout_seconds: float = 300

    def validate(self):
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if self.scenario not in {"serial", "concurrent", "arrival-rate"} or self.connections not in {"close", "reuse"}:
            raise ValueError("unsupported load scenario or connection mode")
        for name in ("concurrency", "requests", "max_pending"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if self.scenario == "serial" and self.concurrency != 1:
            raise ValueError("serial requires concurrency=1")
        if self.connections == "reuse" and self.requests <= self.concurrency:
            raise ValueError("reuse requires more requests than workers to observe persistence")
        if self.scenario == "arrival-rate":
            if self.rate is None or not math.isfinite(self.rate) or self.rate <= 0:
                raise ValueError("arrival-rate requires a finite positive target rate")
        elif self.rate is not None or self.arrival != "fixed":
            raise ValueError("rate and arrival distribution apply only to arrival-rate")
        if self.arrival not in {"fixed", "poisson"} or self.max_pending < self.concurrency:
            raise ValueError("invalid arrival distribution or max_pending below concurrency")
        return self

    def protocol(self):
        self.validate()
        return {"schema_version": 1, "kind": "nonstream_http_load", **asdict(self),
                "timeout_semantics": "connect_or_read_inactivity", "retry": "none",
                "overflow": "drop_new_arrival", "clock": "perf_counter_relative_seconds"}

    def identity(self):
        return hashlib.sha256(json.dumps(self.protocol(), sort_keys=True, allow_nan=False).encode()).hexdigest()

    def schedule(self):
        self.validate()
        if self.scenario != "arrival-rate":
            return None
        if self.rate is None:
            raise ValueError("arrival-rate requires a finite positive target rate")
        rng = random.Random(self.seed) if self.arrival == "poisson" else None
        offsets = [0.0]
        for index in range(1, self.requests):
            planned = index / self.rate if rng is None else offsets[-1] + rng.expovariate(self.rate)
            if not math.isfinite(planned):
                raise ValueError("arrival-rate schedule must contain only finite offsets; increase rate or reduce requests")
            offsets.append(planned)
        return offsets


def _request_headers(headers: dict[str, str] | None) -> dict[str, str]:
    extra_headers = dict(headers or {})
    reserved = {"content-type", "connection", "x-req-id", "host", "content-length", "transfer-encoding"}
    if any(not isinstance(key, str) or key.lower() in reserved
           or not isinstance(value, str) or "\r" in key + value or "\n" in key + value
           for key, value in extra_headers.items()):
        raise ValueError("invalid or reserved load request header")
    return extra_headers


def _submit_requests(
    transport: LoadTransport, pool: ThreadPoolExecutor, schedule: list[float] | None,
    pending: set[Future[None]],
) -> None:
    config = transport.config
    for index in range(config.requests):
        if schedule is not None:
            planned = schedule[index]
            remaining = transport.epoch + planned - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            finished = {future for future in pending if future.done()}
            for future in finished:
                future.result()
            pending.difference_update(finished)
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
        pending.add(pool.submit(transport.execute, index, planned))
    for future in pending:
        future.result()


def _execute_load(
    transport: LoadTransport, schedule: list[float] | None,
) -> tuple[float, BaseException | None]:
    config = transport.config
    pool = ThreadPoolExecutor(max_workers=config.concurrency, thread_name_prefix="acprof-load")
    pending: set[Future[None]] = set()
    primary_error: BaseException | None = None
    transport.epoch = time.perf_counter()
    barrier = threading.Barrier(config.concurrency + 1)
    try:
        # Start worker threads before the timed schedule.
        ready = [pool.submit(barrier.wait) for _ in range(config.concurrency)]
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
    prepared = adapter.prepare_payload(payload) if adapter is not None else payload
    body = json.dumps(prepared, separators=(",", ":"), allow_nan=False).encode()
    protocol = config.protocol()
    if adapter is not None:
        protocol = {**protocol, "kind": "openai_stream_http_load", "adapter": adapter.protocol(),
                    "tokenizer_count_phase": "after_load_duration"}
    protocol_sha256 = hashlib.sha256(json.dumps(protocol, sort_keys=True, allow_nan=False).encode()).hexdigest()
    schedule = config.schedule()
    transport = LoadTransport(target, body, config, token=token, adapter=adapter, headers=extra_headers)
    duration, primary_error = _execute_load(transport, schedule)
    records = complete_records(transport.records, token, schedule)
    if adapter is not None:
        primary_error = finalize_streams(records, adapter, primary_error)
    report = build_load_report(
        records, config, protocol, protocol_sha256, duration, streaming=adapter is not None,
        primary_error=primary_error, include_capacity=include_capacity, slo_thresholds=slo_thresholds,
    )
    if primary_error is not None:
        setattr(primary_error, "load_report", report)
        raise primary_error
    return report
