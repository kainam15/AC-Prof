"""Small, internally consistent sweep artifacts for offline report tests."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from acprof.analysis.streaming import streaming_summary
from acprof.analysis.streaming_capacity import build_capacity
from acprof.host.measurement.load import LoadConfig
from tests.analysis.reporting.test_load_report import experiment, request


def refresh_point(payload: dict[str, Any]) -> None:
    identity, result = payload["identity"], payload["result"]
    identity["request_body_sha256"] = hashlib.sha256(
        json.dumps(identity["request_body"], sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    payload["identity_sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    capacity = build_capacity(result["requests"], result["duration_s"], identity["load_config"]["concurrency"],
                              identity["slo_thresholds_s"])
    result.update(capacity=capacity, counts=capacity["counts"], streaming=streaming_summary(result["requests"]),
                  successful_requests_per_s=capacity["successful_requests_per_s"])
    result["successful"] = capacity["counts"]["succeeded"] == capacity["counts"]["offered"]
    result["success_rate"] = capacity["counts"]["succeeded"] / capacity["counts"]["offered"]
    payload["successful"] = result["successful"] and payload["status"] == "complete"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def make_sweep(root: Path, configs: list[LoadConfig] | None = None) -> Path:
    configs = configs or [LoadConfig(scenario="concurrent", concurrency=count, requests=4) for count in (1, 2, 4)]
    root.mkdir()
    points = []
    for index, config in enumerate(configs, start=1):
        point_id = f"{index:04d}"
        output = root / "points" / point_id
        output.mkdir(parents=True)
        payload: dict[str, Any] = experiment()
        payload.update(run_id=f"run-{point_id}", host={"hostname": "fixture-host"})
        payload["identity"].update(
            target_origin="http://fixture.invalid", endpoint="/v1/chat/completions",
            load_config=asdict(config), warmup_requests=1,
            request_body={"model": "fixture", "messages": [{"role": "user", "content": "same input"}]},
            adapter={"kind": "openai_chat_sse"}, authentication="none",
        )
        rows = []
        for _ in range(config.requests):
            row = request()
            row.update(ttft_s=0.1 * index, tpot_s=0.02 * index)
            rows.append(row)
        payload["result"].update(requests=rows, duration_s=4 / index, protocol=config.protocol())
        refresh_point(payload)
        write_json(output / "load.json", payload)
        points.append({
            "point_id": point_id, "report_path": f"points/{point_id}/load.json",
            "load_config": asdict(config), "status": "complete", "successful": True,
            "run_id": payload["run_id"],
        })
    manifest = {
        "schema_version": 1, "kind": "stream_load_sweep", "run_id": "sweep-fixture",
        "status": "complete", "successful": True, "execution_order": "listed_concurrency_then_rate",
        "scope": "sequential endpoint load points", "points": points,
    }
    write_json(root / "sweep.json", manifest)
    return root


def change_point(root: Path, index: int, change) -> None:
    path = root / f"points/{index:04d}/load.json"
    payload = json.loads(path.read_text())
    change(payload)
    refresh_point(payload)
    write_json(path, payload)


def change_manifest(root: Path, change) -> None:
    path = root / "sweep.json"
    payload = json.loads(path.read_text())
    change(payload)
    write_json(path, payload)
