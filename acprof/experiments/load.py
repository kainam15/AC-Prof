"""Pure HTTP load configuration and deterministic arrival plans."""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass

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
