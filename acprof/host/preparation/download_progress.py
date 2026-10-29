"""Adapt Hub's public tqdm_class hook; never replace or patch the downloader.

    Hub >= 1.28 supplies named reconstruction and transfer bars and fixes retry
    double counting. Transfer increments describe SDK network payload, not wire
    bytes (HTTP/TLS overhead is excluded). Reconstruction includes resume data.
"""
from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from threading import RLock

from acprof.download_events import DownloadProgress, encode_event


class DownloadReporter:
    INTERVAL = 0.25

    def __init__(self, model: str, source: str, total_bytes: int, cached_bytes: int,
                 files_total: int, *, clock: Callable[[], float] = time.monotonic,
                 emit: Callable[[DownloadProgress], None] | None = None) -> None:
        self.model, self.source = model, source
        self.total_bytes, self.cached_bytes, self.files_total = total_bytes, cached_bytes, files_total
        self.clock = clock
        self.emit = emit or self._emit
        self.enabled = emit is not None or os.environ.get("ACPROF_TUI") == "1"
        self.lock = RLock()
        self.attempt = 0
        self.reconstructed: int | None = None
        self.transferred: int | None = None
        self.files_completed = 0
        self.rate: float | None = None
        self.last_emit = float("-inf")
        self.rate_time = self.clock()
        self.rate_bytes = 0
        self.state = "downloading"
        self.detail = ""

    @staticmethod
    def _emit(progress: DownloadProgress) -> None:
        # One write prevents another print from splitting this message's prefix and JSON.
        sys.stdout.write(encode_event(progress) + "\n")
        sys.stdout.flush()

    def start_attempt(self) -> None:
        with self.lock:
            self.attempt += 1
            self.reconstructed = None
            self.files_completed = 0
            self.rate = None
            self.rate_time, self.rate_bytes = self.clock(), self.transferred or 0
            self.publish(force=True)

    def publish(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = self.clock()
        if not force and now - self.last_emit < self.INTERVAL:
            return
        elapsed = now - self.rate_time
        if self.transferred is not None and elapsed >= self.INTERVAL:
            self.rate = (self.transferred - self.rate_bytes) / elapsed
            self.rate_time, self.rate_bytes = now, self.transferred
        completed = None if self.reconstructed is None else self.cached_bytes + self.reconstructed
        detail = self.detail
        if self.attempt > 1:
            # Earlier attempts may have completed files that the SDK now skips.
            # Its aggregate callback cannot identify that changed cache baseline.
            completed = None
            detail = detail or "retry_cache_baseline_unknown"
        if completed is not None and completed > self.total_bytes:
            # Expose incompatible SDK accounting rather than clamping a false percentage.
            completed, detail = None, "sdk_progress_exceeds_selected_files"
        if self.state in {"verifying", "complete"}:
            completed = self.total_bytes
            self.files_completed = self.files_total
            if detail == "retry_cache_baseline_unknown":
                detail = ""
        progress = DownloadProgress(self.model, self.source, self.state, self.total_bytes,
                                    self.cached_bytes, completed, self.transferred,
                                    self.rate if self.state == "downloading" else None,
                                    self.files_completed, self.files_total, max(1, self.attempt), detail)
        self.emit(progress)
        self.last_emit = now

    def finish(self, state: str) -> None:
        with self.lock:
            self.state = state
            self.publish(force=True)

    def tqdm_class(self):
        from huggingface_hub.utils import tqdm

        owner = self

        class ProgressBar(tqdm):
            """A silent tqdm-compatible observer; no renderer, threads or global patch."""

            monitor_interval = 0  # tqdm.__new__ must not start its periodic monitor.

            def __init__(self, *args, **kwargs):
                self.total = kwargs.get("total")
                self.n = kwargs.get("initial", 0)
                self.desc = kwargs.get("desc", "")
                self.disable = False
                name = kwargs.get("name")
                self.kind = ("transfer" if name == "huggingface_hub.snapshot_download.transfer" else
                             "data" if name == "huggingface_hub.snapshot_download" else
                             "files" if kwargs.get("unit") != "B" else "unknown")
                with owner.lock:
                    if self.kind == "data":
                        owner.reconstructed = int(self.n)
                    elif self.kind == "transfer" and owner.transferred is None:
                        owner.transferred = 0
                    elif self.kind == "unknown":
                        owner.detail = "unsupported_sdk_progress_channel"

            def update(self, n=1):
                with owner.lock:
                    self.n += n or 0
                    if self.kind == "transfer":
                        # A Range restart rolls useful progress back, not bytes already received.
                        owner.transferred = (owner.transferred or 0) + max(0, int(n or 0))
                    elif self.kind == "data":
                        owner.reconstructed = int(self.n)
                    elif self.kind == "files":
                        owner.files_completed = int(self.n)
                    owner.publish()

            @property
            def format_dict(self):
                return {"rate": None, "n": self.n, "total": self.total}

            def refresh(self, *args, **kwargs):
                with owner.lock:
                    owner.publish()

            def set_description(self, desc=None, refresh=True):
                self.desc = desc or ""

            def set_postfix_str(self, s="", refresh=True):
                pass  # SDK decoration is not a source of byte or speed evidence.

            def close(self):
                with owner.lock:
                    owner.publish(force=True)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

            def __del__(self):
                pass  # No destructor emission after verification or failure.

        return ProgressBar
