"""Per-load HTTP connections, request execution and interruptible socket ownership."""
from __future__ import annotations

import http.client
import socket
import threading
import time
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any
from urllib.parse import SplitResult

from acprof.artifacts import loads_finite_json
from acprof.host.measurement.streaming import OpenAIStreamingAdapter
from acprof.system.cleanup import cleanup_on_exit

if TYPE_CHECKING:
    from acprof.experiments.load import LoadConfig


class LoadTransport:
    """Own worker-local connections and retain active sockets through body consumption."""

    def __init__(
        self, target: SplitResult, body: bytes | tuple[bytes, ...], config: LoadConfig, *, token: str,
        adapter: OpenAIStreamingAdapter | None, headers: dict[str, str],
    ) -> None:
        self.target = target
        self.body = body
        self.config = config
        self.token = token
        self.adapter = adapter
        self.headers = headers
        self.endpoint = adapter.endpoint if adapter is not None else "/predict"
        self.records: list[dict[str, Any] | None] = [None] * config.requests
        self.cancelled = threading.Event()
        self.epoch = 0.0
        self._local = threading.local()
        self._lock = threading.Lock()
        self._connections: list[http.client.HTTPConnection] = []
        self._active_sockets: dict[int, socket.socket] = {}
        self._run_budget = adapter.new_run_budget() if adapter is not None else None
        self._factory = http.client.HTTPSConnection if target.scheme == "https" else http.client.HTTPConnection

    def now(self) -> float:
        return time.perf_counter() - self.epoch

    def _connection(self) -> http.client.HTTPConnection:
        client: http.client.HTTPConnection | None = getattr(self._local, "client", None)
        if client is None:
            client = self._factory(str(self.target.hostname), self.target.port, timeout=self.config.timeout_seconds)
            with self._lock:
                self._connections.append(client)
            self._local.client = client
        return client

    def _send(self, client: http.client.HTTPConnection, index: int, record: dict[str, Any]) -> None:
        if self._run_budget is not None:
            self._run_budget.check()
        body = self.body[index] if isinstance(self.body, tuple) else self.body
        record["sent_s"] = self.now()
        client.request("POST", self.target.path.rstrip("/") + self.endpoint, body=body, headers={
            **self.headers, "Content-Type": "application/json",
            "Connection": "keep-alive" if self.config.connections == "reuse" else "close",
            "X-Req-Id": record["request_id"]})
        record["connection"] = list(client.sock.getsockname()) if client.sock is not None else None
        with self._lock:
            if client.sock is not None:
                self._active_sockets[index] = client.sock
        if self.cancelled.is_set():
            raise RuntimeError("load cancelled before response headers")

    def _consume(self, response: http.client.HTTPResponse, record: dict[str, Any]) -> None:
        if self.adapter is None:
            raw = response.read()
            record["completed_s"] = self.now()
            record["http_status"] = response.status
            result = loads_finite_json(raw.decode("utf-8"))
            if response.status != 200 or not isinstance(result, dict) or result.get("error"):
                raise RuntimeError("/predict did not return a completed successful JSON response")
            record["workload_contract"] = result.get("workload_contract")
        else:
            record["http_status"] = response.status
            self.adapter.consume(
                response, record, clock=self.now, cancelled=self.cancelled,
                defer_tokenization=True, run_budget=self._run_budget,
            )

    def _record_error(self, record: dict[str, Any], error: Exception) -> None:
        error_code = ("cancelled" if self.cancelled.is_set() else
                      "timeout" if isinstance(error, TimeoutError) else
                      getattr(error, "code", "transport_error"))
        record.update(status="cancelled" if error_code == "cancelled" else "error",
                      error=f"{type(error).__name__}: {error}", error_code=error_code)
        if self.adapter is not None:
            self.adapter.invalidate(record, error_code)

    def execute(self, index: int, planned: float) -> None:
        record: dict[str, Any] = {
            "request_id": f"{self.token}:{index}", "planned_s": planned, "sent_s": None,
            "completed_s": None, "status": "cancelled", "error": "", "http_status": None,
            "connection": None, "workload_contract": None,
        }
        self.records[index] = record
        if self.cancelled.is_set():
            return
        client = self._connection()
        response: http.client.HTTPResponse | None = None
        try:
            self._send(client, index, record)
            response = client.getresponse()
            self._consume(response, record)
            if self.config.connections == "reuse" and response.will_close:
                raise RuntimeError("connection reuse requested but server closed the connection")
            record["status"] = "ok"
        except Exception as error:
            self._record_error(record, error)
            client.close()
        finally:
            if record["completed_s"] is None:
                record["completed_s"] = self.now()
            with self._lock:
                self._active_sockets.pop(index, None)
            if response is not None:
                response.close()
            if self.config.connections == "close":
                client.close()

    def interrupt(self) -> None:
        """Wake active reads before the scheduler joins workers."""
        with self._lock:
            # HTTPConnection may detach its socket after Connection: close headers.
            sockets = set(self._active_sockets.values()) | {
                client.sock for client in self._connections if client.sock is not None}
            for active_socket in sockets:
                try:
                    active_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def close(self) -> None:
        """Attempt every close after workers return, retaining all failure diagnostics."""
        with ExitStack() as stack:
            # Keep creation order while ExitStack unwinds in reverse registration order.
            for client in reversed(self._connections):
                stack.enter_context(cleanup_on_exit(client.close, description="HTTP load connection"))
