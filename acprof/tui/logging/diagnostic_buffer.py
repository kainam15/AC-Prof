"""Bounded diagnostic retention without rendering or disk I/O."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import chain


@dataclass(frozen=True, slots=True)
class DiagnosticBatch:
    entries: tuple[str, ...]
    retained_bytes: int
    discarded_entries: int
    oversized_entries: int


def _positive_bound(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class BoundedLogBuffer:
    """Keep early context and recent diagnostics within UTF-8 payload limits.

    These limits bound retained text, not the Python process RSS. Oversized
    records are omitted entirely so partial tracebacks cannot appear complete.
    """

    def __init__(self, *, max_entries: int = 512, max_bytes: int = 1024 * 1024,
                 max_entry_bytes: int = 64 * 1024) -> None:
        self.max_entries = _positive_bound(max_entries, "max_entries")
        self.max_bytes = _positive_bound(max_bytes, "max_bytes")
        self.max_entry_bytes = min(_positive_bound(max_entry_bytes, "max_entry_bytes"), self.max_bytes)
        self._prefix_entries = min(32, self.max_entries // 4)
        self._prefix_bytes = self.max_bytes // 4
        self._head: list[tuple[str, int]] = []
        self._tail: deque[tuple[str, int]] = deque()
        self._head_bytes = 0
        self._byte_size = 0
        self._tail_started = False
        self.discarded_entries = 0
        self.oversized_entries = 0

    def __len__(self) -> int:
        return len(self._head) + len(self._tail)

    def __bool__(self) -> bool:
        return bool(len(self) or self.discarded_entries)

    @property
    def byte_size(self) -> int:
        return self._byte_size

    def append(self, text: str) -> None:
        # Reject huge strings before allocating an equally large encoded copy.
        if len(text) > self.max_entry_bytes:
            self.discarded_entries += 1
            self.oversized_entries += 1
            return
        size = len(text.encode("utf-8"))
        if size > self.max_entry_bytes:
            self.discarded_entries += 1
            self.oversized_entries += 1
            return
        if not self._tail_started:
            if len(self._head) < self._prefix_entries and self._head_bytes + size <= self._prefix_bytes:
                self._head.append((text, size))
                self._head_bytes += size
                self._byte_size += size
                return
            self._tail_started = True
        if self._head_bytes + size > self.max_bytes:
            self.discarded_entries += 1
            return
        while self._tail and (len(self) >= self.max_entries or self._byte_size + size > self.max_bytes):
            _, removed_size = self._tail.popleft()
            self._byte_size -= removed_size
            self.discarded_entries += 1
        self._tail.append((text, size))
        self._byte_size += size

    def clear(self) -> None:
        self._head.clear()
        self._tail.clear()
        self._head_bytes = self._byte_size = 0
        self._tail_started = False
        self.discarded_entries = self.oversized_entries = 0

    def snapshot(self) -> DiagnosticBatch:
        """Copy outside the window; retain ownership until delivery succeeds."""
        return DiagnosticBatch(tuple(text for text, _ in chain(self._head, self._tail)),
                               self._byte_size, self.discarded_entries, self.oversized_entries)

    def drain(self) -> DiagnosticBatch:
        """Return retained text and omission counters outside the window."""
        batch = self.snapshot()
        self.clear()
        return batch
