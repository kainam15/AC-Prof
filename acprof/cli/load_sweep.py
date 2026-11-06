"""Validated load point plans and sequential endpoint sweep publication."""
from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Any
from uuid import uuid4

from acprof.artifacts import atomic_write_json
from acprof.cli.load_evidence import publish_load_report
from acprof.experiments.load import MAX_SWEEP_POINTS, LoadConfig
from acprof.system.cleanup import format_exception_notes


def _axis_values(value: str, *, integer: bool, flag: str) -> list[Any]:
    parts = value.split(",")
    if any(not part.strip() for part in parts):
        raise ValueError(f"{flag} requires a comma-separated list of positive values")
    try:
        values = [int(part) if integer else float(part) for part in parts]
    except ValueError as error:
        kind = "integer" if integer else "numeric"
        raise ValueError(f"{flag} requires {kind} values") from error
    if any(value <= 0 or (not integer and not math.isfinite(value)) for value in values):
        raise ValueError(f"{flag} values must be finite and positive")
    if len(set(values)) != len(values):
        raise ValueError(f"{flag} contains duplicate values")
    return values


def load_configurations(args: argparse.Namespace) -> list[LoadConfig]:
    """Validate every point before any endpoint preparation or output creation."""
    base = LoadConfig(
        scenario=args.scenario, connections=args.connections, concurrency=args.concurrency,
        requests=args.requests, rate=args.rate, arrival=args.arrival, seed=args.seed,
        max_pending=args.max_pending,
        timeout_seconds=args.request_timeout if args.request_timeout is not None else 300,
    )
    concurrency = [base.concurrency]
    rates = [base.rate]
    if args.sweep_concurrency is not None:
        if base.scenario == "serial":
            raise ValueError("--sweep-concurrency requires --scenario concurrent or arrival-rate")
        concurrency = _axis_values(args.sweep_concurrency, integer=True, flag="--sweep-concurrency")
    if args.sweep_rate is not None:
        if base.scenario != "arrival-rate":
            raise ValueError("--sweep-rate requires --scenario arrival-rate")
        rates = _axis_values(args.sweep_rate, integer=False, flag="--sweep-rate")
    if len(concurrency) * len(rates) > MAX_SWEEP_POINTS:
        raise ValueError(f"load sweep must contain at most {MAX_SWEEP_POINTS} points")
    configs = [replace(base, concurrency=workers, rate=rate).validate() for workers, rate in product(concurrency, rates)]
    for config in configs:
        config.schedule()  # Reject even a later point's overflowing arrival plan before sending any requests.
    return configs


def run_load_sweep(
    output: Path, configs: list[LoadConfig], execute: Callable[[LoadConfig, Path], dict[str, Any]], *,
    measurement_scope: Callable[[], AbstractContextManager[Any]],
) -> int:
    """Run validated points under one lock, retaining completed and unrun states."""
    points: list[dict[str, Any]] = [{
        "point_id": f"{index:04d}", "report_path": f"points/{index:04d}/load.json",
        "load_config": asdict(config), "status": "not_run", "successful": None, "run_id": None,
    } for index, config in enumerate(configs, start=1)]
    report: dict[str, Any] = {
        "schema_version": 1, "kind": "stream_load_sweep", "run_id": uuid4().hex,
        "status": "running", "successful": False, "points": points,
        "execution_order": "listed_concurrency_then_rate",
        "scope": "sequential endpoint load points; no service reset, independent-run statistics or hardware attribution",
    }
    path = output / "sweep.json"
    with publish_load_report(path, report):
        with measurement_scope():
            for point, config in zip(points, configs, strict=True):
                point["status"] = "running"
                try:
                    atomic_write_json(path, report)
                    point_output = output / "points" / str(point["point_id"])
                    point_output.mkdir(parents=True, exist_ok=False)
                    measured = execute(config, point_output)
                except BaseException as error:
                    point.update(
                        status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed", successful=False,
                        error=f"{type(error).__name__}: {format_exception_notes(error)}",
                    )
                    raise
                point.update(status="complete", successful=measured["successful"], run_id=measured["run_id"])
                atomic_write_json(path, report)
                print(f"Load point {point['point_id']}: {point_output / 'load.json'}; "
                      f"success rate {measured['result']['success_rate']:.1%}")
        report.update(status="complete", successful=all(point["successful"] for point in points))
    print(f"Load sweep: {path}; {len(points)} points completed")
    return 0 if report["successful"] else 1
