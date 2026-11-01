"""Shared HTTP load scheduling, separate from formal energy measurement windows."""
from __future__ import annotations

import hashlib
import http.client
import json
import math
import random
import socket
import statistics
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from acprof.artifacts import loads_finite_json

if TYPE_CHECKING:
    from acprof.host.measurement.streaming import OpenAIStreamingAdapter


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
        if self.arrival == "fixed":
            return [index / self.rate for index in range(self.requests)]
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
            "p50": percentile(0.5), "p90": percentile(0.9),
            "p95": percentile(0.95), "p99": percentile(0.99), "sample_count": len(ordered)}


def _streaming_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Request metrics are equally weighted; interval metrics pool valid intervals."""
    succeeded = [row for row in records if row["status"] == "ok"]
    summary: dict[str, Any] = {"time_unit": "seconds", "percentile_method": "linear",
                              "excluded_requests": len(records) - len(succeeded),
                              "failed_requests": sum(row["status"] == "error" for row in records),
                              "cancelled_requests": sum(row["status"] == "cancelled" for row in records),
                              "dropped_requests": sum(row["status"] == "dropped" for row in records)}
    for name in ("ttft_s", "tpot_s", "output_e2e_s", "e2e_s"):
        values = [row[name] for row in succeeded if row.get(name) is not None]
        summary[name] = {**_distribution(values), "sample_unit": "request",
                         "unit": "seconds_per_token" if name == "tpot_s" else "seconds",
                         "missing_requests": len(succeeded) - len(values)}
    for field, name, unit in (("itl_s", "itl_s", "token_interval"),
                              ("output_event_intervals_s", "output_event_interval_s", "output_event_interval")):
        values = [value for row in succeeded for value in (row.get(field) or [])]
        summary[name] = {**_distribution(values), "sample_unit": unit, "unit": "seconds",
                         "requests_with_samples": sum(bool(row.get(field)) for row in succeeded),
                         "missing_requests": sum(not row.get(field) for row in succeeded)}
    summary["output_token_count_sources"] = dict(Counter(
        row.get("output_token_count_source", "unavailable") for row in succeeded))
    summary["token_count_errors"] = sum(bool(row.get("token_count_error")) for row in succeeded)
    summary["errors"] = dict(Counter(row.get("error_code", row["status"])
                                    for row in records if row["status"] != "ok"))
    return summary


def run_load(
    url, payload, config: LoadConfig, *, token,
    adapter: OpenAIStreamingAdapter | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Closed-loop workers or an open-loop scheduler with bounded pending work.

    Timestamps are client call boundaries, not NIC timestamps. The default
    /predict JSON protocol is preserved. An optional streaming adapter consumes
    SSE on the same worker connections and scheduler; failures never contribute
    to successful latency distributions.
    """
    config.validate()
    target = urlsplit(url)
    if target.scheme not in {"http", "https"} or not target.hostname or target.username or target.query or target.fragment:
        raise ValueError("load target must be an HTTP(S) base URL without credentials, query or fragment")
    extra_headers = dict(headers or {})
    reserved = {"content-type", "connection", "x-req-id", "host", "content-length", "transfer-encoding"}
    if any(not isinstance(key, str) or key.lower() in reserved
           or not isinstance(value, str) or "\r" in key + value or "\n" in key + value
           for key, value in extra_headers.items()):
        raise ValueError("invalid or reserved load request header")
    prepared = adapter.prepare_payload(payload) if adapter is not None else payload
    body = json.dumps(prepared, separators=(",", ":"), allow_nan=False).encode()
    endpoint = adapter.endpoint if adapter is not None else "/predict"
    protocol = config.protocol()
    if adapter is not None:
        protocol = {**protocol, "kind": "openai_stream_http_load", "adapter": adapter.protocol(),
                    "tokenizer_count_phase": "after_load_duration"}
    protocol_sha256 = hashlib.sha256(json.dumps(protocol, sort_keys=True, allow_nan=False).encode()).hexdigest()
    schedule = config.schedule()
    records: list[dict[str, Any] | None] = [None] * config.requests
    local, lock = threading.local(), threading.Lock()
    connections = []
    active_sockets: dict[int, socket.socket] = {}
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
        return client

    def execute(index, planned):
        record: dict[str, Any] = {"request_id": f"{token}:{index}", "planned_s": planned, "sent_s": None,
                  "completed_s": None, "status": "cancelled", "error": "", "http_status": None,
                  "connection": None, "workload_contract": None}
        records[index] = record
        if cancelled.is_set():
            return
        client = connection()
        response: http.client.HTTPResponse | None = None
        try:
            record["sent_s"] = time.perf_counter() - epoch
            client.request("POST", target.path.rstrip("/") + endpoint, body=body, headers={
                **extra_headers, "Content-Type": "application/json",
                "Connection": "keep-alive" if config.connections == "reuse" else "close",
                "X-Req-Id": record["request_id"]})
            record["connection"] = list(client.sock.getsockname()) if client.sock is not None else None
            with lock:
                if client.sock is not None:
                    active_sockets[index] = client.sock
            if cancelled.is_set():
                raise RuntimeError("load cancelled before response headers")
            response = client.getresponse()
            if adapter is None:
                raw = response.read()
                record["completed_s"] = time.perf_counter() - epoch
                record["http_status"] = response.status
                result = loads_finite_json(raw.decode("utf-8"))
                if response.status != 200 or not isinstance(result, dict) or result.get("error"):
                    raise RuntimeError("/predict did not return a completed successful JSON response")
                record["workload_contract"] = result.get("workload_contract")
            else:
                record["http_status"] = response.status
                adapter.consume(response, record, clock=lambda: time.perf_counter() - epoch,
                                cancelled=cancelled, defer_tokenization=True)
            if config.connections == "reuse" and response.will_close:
                raise RuntimeError("connection reuse requested but server closed the connection")
            record["status"] = "ok"
        except Exception as error:
            error_code = ("cancelled" if cancelled.is_set() else
                          "timeout" if isinstance(error, TimeoutError) else
                          getattr(error, "code", "transport_error"))
            record.update(status="cancelled" if error_code == "cancelled" else "error",
                          error=f"{type(error).__name__}: {error}", error_code=error_code)
            if adapter is not None:
                adapter.invalidate(record, error_code)
            client.close()
        finally:
            if record["completed_s"] is None:
                record["completed_s"] = time.perf_counter() - epoch
            with lock:
                active_sockets.pop(index, None)
            if response is not None:
                response.close()
            if config.connections == "close":
                client.close()

    pool = ThreadPoolExecutor(max_workers=config.concurrency, thread_name_prefix="acprof-load")
    pending: set[Any] = set()
    primary_error: BaseException | None = None
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
            # HTTPConnection may detach its socket after Connection: close
            # headers. Retain the active socket until its body is consumed.
            sockets = set(active_sockets.values()) | {
                client.sock for client in connections if client.sock is not None}
            for active_socket in sockets:
                try:
                    active_socket.shutdown(socket.SHUT_RDWR)
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
    completed_records = [record for record in records if record is not None]
    if adapter is not None:
        for record in completed_records:
            if record["status"] != "ok":
                adapter.invalidate(record, record.get("error_code", record["status"]))
                continue
            try:
                adapter.finalize(record)
            except Exception as error:
                # Optional local counting cannot turn a completed HTTP request
                # into a server failure or manufacture a token denominator.
                record.update(output_tokens=None, output_token_count_source="unavailable",
                              token_count_scope="unavailable", tpot_s=None, itl_s=None,
                              tpot_precision="unavailable", itl_precision="unavailable",
                              token_count_error=f"{type(error).__name__}: {error}")
                record["missing_reasons"].update(tpot_s="tokenizer_error", itl_s="tokenizer_error")
            except BaseException as error:
                primary_error = primary_error or error
                break
    counts = Counter(record["status"] for record in completed_records)
    succeeded = [record for record in completed_records if record["status"] == "ok"]
    sent = sum(record["sent_s"] is not None for record in completed_records)
    report = {"schema_version": 1, "kind": "stream_load_result" if adapter is not None else "nonstream_load_result",
              "protocol": protocol, "protocol_sha256": protocol_sha256,
              "successful": primary_error is None and counts["ok"] == config.requests,
              "duration_s": duration, "target_requests_per_s": config.rate,
              "sent_requests_per_s": sent / duration if duration > 0 else None,
              "completed_requests_per_s": sum(record["http_status"] is not None for record in completed_records
                  if "http_status" in record) / duration if duration > 0 else None,
              "successful_requests_per_s": counts["ok"] / duration if duration > 0 else None,
              "success_rate": counts["ok"] / config.requests,
              "counts": {"offered": config.requests, "sent": sent, "succeeded": counts["ok"],
                         "failed": counts["error"], "dropped": counts["dropped"], "cancelled": counts["cancelled"]},
              "latency_s": _distribution([record["completed_s"] - record["sent_s"] for record in succeeded]),
              "scheduled_latency_s": _distribution([record["completed_s"] - record["planned_s"] for record in succeeded]),
              "queue_delay_s": _distribution([record["sent_s"] - record["planned_s"] for record in completed_records if record["sent_s"] is not None]),
              "connection_count": len({tuple(record["connection"]) for record in completed_records if record["connection"]}),
              "requests": completed_records}
    if adapter is not None:
        report["streaming"] = _streaming_summary(completed_records)
        report["completed_requests_per_s"] = sum(
            row.get("stream_completed_s") is not None for row in completed_records
        ) / duration if duration > 0 else None
    if primary_error is not None:
        setattr(primary_error, "load_report", report)
        raise primary_error
    return report
