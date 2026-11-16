"""Measurement-excluding lease shared by overlapping offline TUI readers."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from threading import Lock


class OverlappingOfflineReads:
    """Share one measurement exclusion lease across concurrent read-only jobs.

    Slow, cancelled readers must not block newer selections. Holding the lease
    until the last reader exits still excludes the collector, including when
    the latest reader finishes before an obsolete one.
    """

    def __init__(self) -> None:
        self._guard = Lock()
        self._readers = 0
        self._lease: AbstractContextManager[object] | None = None

    @contextmanager
    def read(self) -> Iterator[None]:
        from acprof.host.run_state import offline_analysis_lock

        with self._guard:
            if self._readers == 0:
                lease = offline_analysis_lock()
                lease.__enter__()
                self._lease = lease
            self._readers += 1
        try:
            yield
        finally:
            with self._guard:
                self._readers -= 1
                if self._readers == 0:
                    held_lease = self._lease
                    self._lease = None
                    assert held_lease is not None
                    held_lease.__exit__(None, None, None)
