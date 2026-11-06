"""Compact, source-checked offline analysis of sequential streaming load sweeps."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from acprof.analysis.load_report import analyze_load_report
from acprof.artifacts import file_sha256, read_json_object_snapshot, require_schema_version
from acprof.experiments.load import MAX_SWEEP_POINTS, LoadConfig

_STATES = {"running", "complete", "failed", "cancelled"}
_CONFIG_FIELDS = {field.name for field in fields(LoadConfig)}


@dataclass
class LoadSweepAnalysis:
    payload: dict[str, Any]
    sources: dict[Path, str | None]
    root: Path


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _validate_config(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != _CONFIG_FIELDS:
        raise ValueError("sweep point requires a full load configuration")
    try:
        if isinstance(value["rate"], bool) or isinstance(value["timeout_seconds"], bool):
            raise ValueError("numeric load settings cannot be boolean")
        LoadConfig(**value).validate()
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid sweep point configuration: {error}") from error


def _validate_point(point: Any, index: int) -> None:
    if not isinstance(point, dict) or point.get("point_id") != f"{index:04d}":
        raise ValueError("sweep point IDs must follow the listed order")
    if point.get("report_path") != f"points/{index:04d}/load.json":
        raise ValueError("sweep point path must be points/<point_id>/load.json")
    state, success, run_id = point.get("status"), point.get("successful"), point.get("run_id")
    if not isinstance(state, str) or state not in _STATES | {"not_run"}:
        raise ValueError("unknown sweep point status")
    if state in {"not_run", "running"}:
        if success is not None or run_id is not None:
            raise ValueError("pending sweep point cannot declare success or a run ID")
    elif type(success) is not bool or (state != "complete" and success):
        raise ValueError("invalid sweep point success declaration")
    if state == "complete" and (not isinstance(run_id, str) or not run_id):
        raise ValueError("completed sweep point requires a run ID")
    if run_id is not None and (not isinstance(run_id, str) or not run_id):
        raise ValueError("invalid sweep point run ID")
    _validate_config(point.get("load_config"))


def _validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    require_schema_version(manifest, 1, "stream load sweep")
    if manifest.get("kind") != "stream_load_sweep":
        raise ValueError("input must be a stream_load_sweep manifest")
    state, success = manifest.get("status"), manifest.get("successful")
    if not isinstance(state, str) or state not in _STATES or type(success) is not bool:
        raise ValueError("invalid sweep state or success declaration")
    if not isinstance(manifest.get("run_id"), str) or not manifest["run_id"]:
        raise ValueError("sweep requires a run ID")
    points = manifest.get("points")
    if not isinstance(points, list) or not 1 <= len(points) <= MAX_SWEEP_POINTS:
        raise ValueError(f"sweep must contain 1 to {MAX_SWEEP_POINTS} points")
    configs, run_ids = set(), set()
    for index, point in enumerate(points, start=1):
        _validate_point(point, index)
        config = _digest(point["load_config"])
        if config in configs or (point["run_id"] is not None and point["run_id"] in run_ids):
            raise ValueError("sweep contains duplicate point configurations or run IDs")
        configs.add(config)
        run_ids.add(point["run_id"])
    if state == "complete":
        if any(point["status"] != "complete" for point in points) or success != all(p["successful"] for p in points):
            raise ValueError("completed sweep state contradicts its points")
    elif success:
        raise ValueError("incomplete sweep cannot declare success")
    return points


def _check_child_path(path: Path, root: Path) -> None:
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"sweep point path contains a symlink: {current.relative_to(root)}")
    if path.exists() and not path.is_file():
        raise ValueError("sweep point report must be a regular file")


def _validate_child(point: dict[str, Any], report: dict[str, Any]) -> None:
    experiment, capacity = report["experiment"], report["capacity"]
    if report["version"] not in {2, 3}:
        raise ValueError("sweep requires schema v2/v3 point reports")
    run_id = experiment.get("run_id")
    if not isinstance(run_id, str) or not run_id or (point["run_id"] is not None and point["run_id"] != run_id):
        raise ValueError("sweep point run ID differs from its report")
    config, identity = point["load_config"], experiment["identity"]
    if _digest(identity.get("load_config")) != _digest(config):
        raise ValueError("sweep point declared configuration differs from its report")
    if report["version"] == 2 and (
        not isinstance(identity.get("request_body"), dict)
        or identity.get("request_body_sha256") != _digest(identity["request_body"])
    ):
        raise ValueError("sweep point request body digest differs")
    counts = capacity["counts"]
    if counts["offered"] != config["requests"]:
        raise ValueError("sweep point request counts differ from its configured request budget")
    if point["status"] == experiment["status"] == "complete" and point["successful"] != experiment["successful"]:
        raise ValueError("sweep point success differs from its manifest")


def _condition(report: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    experiment, capacity = report["experiment"], report["capacity"]
    identity = dict(experiment["identity"])
    identity["load_config"] = {k: v for k, v in identity["load_config"].items() if k not in {"concurrency", "rate"}}
    identity.pop("request_body", None)  # Validated digests retain all input/generation differences.
    if "request_dataset" in identity:
        dataset = identity["request_dataset"]
        identity["request_dataset"] = {"selection": dataset["selection"], "sha256": dataset["sha256"],
                                       "entries": len(dataset["bodies"])}
    service = experiment["service_identity"]
    probe = service.get("probe", {})
    service_evidence = {"observed": service.get("observed", {}), "probe": probe,
                        "matches": service.get("matches", {}), "mismatches": service.get("mismatches", [])}
    token_basis = {key: {name: sorted(counts) for name, counts in capacity["token_throughput"][key].items()}
                   for key in ("source_request_counts", "scope_request_counts")}
    condition = {"identity": identity, "host": experiment.get("host"),
                 "service_evidence": service_evidence, "token_count_basis": token_basis}
    condition_id = _digest(condition)
    condition["condition_id"] = condition_id
    return condition_id, condition


def _point_metrics(report: dict[str, Any]) -> dict[str, Any]:
    capacity, streaming = report["capacity"], report["streaming"]
    tokens = capacity["token_throughput"]["output"]
    dataset = report["experiment"]["identity"].get("request_dataset")
    return {
        **({"request_dataset": {key: value for key, value in dataset.items() if key != "bodies"}} if dataset else {}),
        **({"server_metrics": report["server_metrics"]} if report.get("server_metrics") is not None else {}),
        "metrics": {
            "successful_requests_per_s": capacity["successful_requests_per_s"],
            "goodput_requests_per_s": capacity["goodput_requests_per_s"],
            "output_tokens_per_s": tokens["tokens_per_s"],
            **{f"{name}_p99_s": streaming[f"{name}_s"]["p99"] for name in ("ttft", "tpot", "e2e")},
        },
        "duration_s": capacity["duration_s"], "counts": capacity["counts"],
        "success_rate": capacity["counts"]["succeeded"] / capacity["counts"]["offered"],
        "slo": capacity["slo"], "output_token_coverage": tokens["coverage_ratio"],
        "latency_samples": {f"{name}_s": streaming[f"{name}_s"]["sample_count"] for name in ("ttft", "tpot", "e2e")},
    }


def _read_point(
    point: dict[str, Any], root: Path, sources: dict[Path, str | None], conditions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    row = {key: point[key] for key in ("point_id", "report_path", "load_config", "status", "successful", "run_id")}
    row.update(metrics={}, diagnostics=[], evidence_status="not_run", curve_eligible=False, condition_id=None)
    if point.get("error"):
        row["diagnostics"].append(str(point["error"]))
    if point["status"] == "not_run":
        return row
    path = root / point["report_path"]
    try:
        _check_child_path(path, root)
        try:
            snapshot = read_json_object_snapshot(path, label="sweep point", bounded=False)
        except FileNotFoundError:
            sources[path] = None
            raise
        except ValueError:
            sources[path] = file_sha256(path)
            raise
        sources[path] = hashlib.sha256(snapshot.content).hexdigest()
        report = analyze_load_report(snapshot.payload)
        _validate_child(point, report)
        condition_id, condition = _condition(report)
        state = report["experiment"]["status"]
        complete = point["status"] == state == "complete"
        row.update(_point_metrics(report))
        conditions.setdefault(condition_id, condition)
        row.update(condition_id=condition_id, report_status=state,
                   report_run_id=report["experiment"]["run_id"], curve_eligible=complete,
                   evidence_status="valid" if complete else "partial")
        if point["status"] != state:
            row["diagnostics"].append(f"manifest status {point['status']} differs from report status {state}")
        if report["experiment"].get("error"):
            row["diagnostics"].append(str(report["experiment"]["error"]))
    except (OSError, ValueError, TypeError, KeyError) as error:
        row.update(evidence_status="missing" if isinstance(error, FileNotFoundError) else "invalid")
        row["diagnostics"].append(f"{type(error).__name__}: {error}")
    return row


def read_load_sweep(source: str | Path) -> LoadSweepAnalysis:
    path = Path(source)
    path = path / "sweep.json" if path.is_dir() else path
    if path.name != "sweep.json":
        raise ValueError("sweep report input must be sweep.json or its containing directory")
    root = path.parent.resolve()
    path = root / path.name
    _check_child_path(path, root)
    snapshot = read_json_object_snapshot(path, label="stream load sweep")
    manifest = snapshot.payload
    points = _validate_manifest(manifest)
    sources: dict[Path, str | None] = {path: hashlib.sha256(snapshot.content).hexdigest()}
    conditions: dict[str, dict[str, Any]] = {}
    rows = [_read_point(point, root, sources, conditions) for point in points]
    payload = {
        "schema_version": 1, "kind": "stream_load_sweep_analysis", "run_id": manifest["run_id"],
        "status": manifest["status"], "successful": manifest["successful"],
        "diagnostics": [str(manifest["error"])] if manifest.get("error") else [],
        "points": rows, "conditions": list(conditions.values()), "point_count": len(rows),
        "valid_points": sum(row["curve_eligible"] for row in rows),
        "partial_points": sum(row["evidence_status"] == "partial" for row in rows),
        "unavailable_points": sum(row["evidence_status"] in {"invalid", "missing", "not_run"} for row in rows),
    }
    return LoadSweepAnalysis(payload, sources, root)


def verify_load_sweep_sources(analysis: LoadSweepAnalysis) -> None:
    """Reject a mixed snapshot if retained inputs changed during rendering."""
    for path, expected in analysis.sources.items():
        _check_child_path(path, analysis.root)
        try:
            observed = file_sha256(path)
        except FileNotFoundError:
            observed = None
        if observed != expected:
            raise ValueError(f"sweep source changed during report generation: {path}")
