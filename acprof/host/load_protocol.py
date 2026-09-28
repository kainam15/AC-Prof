"""Versioned non-streaming HTTP load protocol, separate from formal energy windows."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict, dataclass
import hashlib
import http.client
import json
import math
import random
import socket
import statistics
import threading
import time
from urllib.parse import urlsplit
from typing import Any


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
        if self.arrival == "fixed":
            return [index / self.rate for index in range(self.requests)]
        assert self.rate is not None
        rng, offsets = random.Random(self.seed), [0.0]
        for _ in range(self.requests - 1):
            offsets.append(offsets[-1] + rng.expovariate(self.rate))
        return offsets


def _distribution(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)

    def percentile(p):
        if not ordered:
            return None
        position = (len(ordered) - 1) * p
        low = math.floor(position)
        return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (position - low)

    return {"mean": statistics.fmean(ordered) if ordered else None,
            "p50": percentile(0.5), "p95": percentile(0.95), "p99": percentile(0.99)}


def run_load(url, payload, config: LoadConfig, *, token) -> dict[str, Any]:
    """Closed-loop workers or an open-loop scheduler with bounded pending work.

    Timestamps are client call boundaries, not NIC timestamps. Only a fully read,
    successful JSON response contributes to successful latency distributions.
    """
    config.validate()
    target = urlsplit(url)
    if target.scheme not in {"http", "https"} or not target.hostname or target.username or target.query or target.fragment:
        raise ValueError("load target must be an HTTP(S) base URL without credentials, query or fragment")
    body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    schedule = config.schedule()
    records: list[dict[str, Any] | None] = [None] * config.requests
    local, lock = threading.local(), threading.Lock()
    connections = []
    cancelled = threading.Event()
    factory = http.client.HTTPSConnection if target.scheme == "https" else http.client.HTTPConnection

    hostname = str(target.hostname)

    def connection() -> http.client.HTTPConnection:
        client: http.client.HTTPConnection | None = getattr(local, "client", None)
        if client is None:
            client = factory(hostname, target.port, timeout=config.timeout_seconds)
            with lock:
                connections.append(client)
            local.client = client
        assert client is not None
        return client

    def execute(index, planned):
        record: dict[str, Any] = {"request_id": f"{token}:{index}", "planned_s": planned, "sent_s": None,
                  "completed_s": None, "status": "cancelled", "error": "", "http_status": None,
                  "connection": None, "workload_contract": None}
        records[index] = record
        if cancelled.is_set():
            return
        client = connection()
        try:
            record["sent_s"] = time.perf_counter() - epoch
            client.request("POST", target.path.rstrip("/") + "/predict", body=body, headers={
                "Content-Type": "application/json", "Connection": "keep-alive" if config.connections == "reuse" else "close",
                "X-Req-Id": record["request_id"]})
            record["connection"] = list(client.sock.getsockname()) if client.sock is not None else None
            response = client.getresponse()
            raw = response.read()
            record["completed_s"] = time.perf_counter() - epoch
            record["http_status"] = response.status
            result = json.loads(raw)
            if response.status != 200 or not isinstance(result, dict) or result.get("error"):
                raise RuntimeError("/predict did not return a completed successful JSON response")
            if config.connections == "reuse" and response.will_close:
                raise RuntimeError("connection reuse requested but server closed the connection")
            record.update(status="ok", workload_contract=result.get("workload_contract"))
        except Exception as error:
            record.update(status="error", error=f"{type(error).__name__}: {error}")
            client.close()
        finally:
            if record["completed_s"] is None:
                record["completed_s"] = time.perf_counter() - epoch
            if config.connections == "close":
                client.close()

    pool = ThreadPoolExecutor(max_workers=config.concurrency, thread_name_prefix="acprof-load")
    pending, primary_error = set(), None
    epoch = time.perf_counter()
    barrier = threading.Barrier(config.concurrency + 1)
    try:
        # Start worker threads before the timed schedule.
        ready = [pool.submit(barrier.wait) for _ in range(config.concurrency)]
        barrier.wait()
        for future in ready:
            future.result()
        epoch = time.perf_counter()
        for index in range(config.requests):
            if schedule is not None:
                planned = schedule[index]
                remaining = epoch + planned - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                finished = {future for future in pending if future.done()}
                for future in finished:
                    future.result()
                pending.difference_update(finished)
                if len(pending) >= config.max_pending:
                    records[index] = {"request_id": f"{token}:{index}", "planned_s": planned,
                                      "sent_s": None, "completed_s": None, "status": "dropped",
                                      "error": "max_pending exceeded", "connection": None}
                    continue
            else:
                if len(pending) >= config.concurrency:
                    finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        future.result()
                planned = time.perf_counter() - epoch
            pending.add(pool.submit(execute, index, planned))
        for future in pending:
            future.result()
    except BaseException as error:
        primary_error = error
        barrier.abort()
        cancelled.set()
        for future in pending:
            future.cancel()
        with lock:
            for client in connections:
                try:
                    if client.sock is not None:
                        client.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        duration = time.perf_counter() - epoch
        for client in connections:
            client.close()
    for index, record in enumerate(records):
        if record is None:
            records[index] = {"request_id": f"{token}:{index}", "planned_s": schedule[index] if schedule else None,
                              "sent_s": None, "completed_s": None, "status": "cancelled",
                              "error": "not submitted", "connection": None}
    records = [record for record in records if record is not None]
    counts = Counter(record["status"] for record in records)
    succeeded = [record for record in records if record["status"] == "ok"]
    sent = sum(record["sent_s"] is not None for record in records)
    report = {"schema_version": 1, "kind": "nonstream_load_result", "protocol": config.protocol(),
              "protocol_sha256": config.identity(), "successful": primary_error is None and counts["ok"] == config.requests,
              "duration_s": duration, "target_requests_per_s": config.rate,
              "sent_requests_per_s": sent / duration if duration > 0 else None,
              "completed_requests_per_s": sum(record["http_status"] is not None for record in records
                  if "http_status" in record) / duration if duration > 0 else None,
              "successful_requests_per_s": counts["ok"] / duration if duration > 0 else None,
              "success_rate": counts["ok"] / config.requests,
              "counts": {"offered": config.requests, "sent": sent, "succeeded": counts["ok"],
                         "failed": counts["error"], "dropped": counts["dropped"], "cancelled": counts["cancelled"]},
              "latency_s": _distribution([record["completed_s"] - record["sent_s"] for record in succeeded]),
              "scheduled_latency_s": _distribution([record["completed_s"] - record["planned_s"] for record in succeeded]),
              "queue_delay_s": _distribution([record["sent_s"] - record["planned_s"] for record in records if record["sent_s"] is not None]),
              "connection_count": len({tuple(record["connection"]) for record in records if record["connection"]}),
              "requests": records}
    if primary_error is not None:
        setattr(primary_error, "load_report", report)
        raise primary_error
    return report
