"""只读结果审计；保持历史字段和不可用原因，不回填测量值。"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from acprof.artifact_layout import ArtifactLayout
from acprof.artifacts import read_json_object
from acprof.capabilities import collection_outcomes
from acprof.failures import Failure
from acprof.metric_registry import METRICS, NUMERIC_FIELDS
from acprof.platform import native_only_metric, recorded_identity
from acprof.quality import read_quality, summarize_quality
from acprof.result_csv import (
    ResultCsvSnapshot,
    expected_measurements,
    measurement_key,
    read_result_csv_snapshot,
    result_csv_snapshot_unchanged,
    validate_result_rows,
)

MISSING = {"", "nan", "none", "null", "n/a"}


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


def missing_reason(field: str, row: dict, metadata: dict, *, environment_class: str) -> str:
    """Use the audit's verified recorded identity; keep row-specific evidence separate."""
    if field not in row:
        return "not_recorded"
    metric = METRICS[field]
    if row.get("runtime_kind", metadata.get("runtime_kind")) == "process":
        if field in {"cpu_cores", "mem_cap_gb"}:
            return "inherited_limit_not_controlled"
        if field.startswith("container_") or field.startswith("cold_start_"):
            return "unsupported_by_process_runtime"
        if field in {"process_cpu_time_s", "process_cpu_util_avg_pct", "process_rss_avg_bytes", "process_rss_peak_bytes"}:
            return str(row.get("process_resource_status", "unavailable"))
        if field.startswith("gpu_util_"):
            return str(row.get("gpu_utilization_status", "unavailable"))
        if field in {"gpu_mem_used_avg_bytes", "gpu_mem_used_peak_bytes"}:
            return str(row.get("gpu_memory_status", "unavailable"))
        if field in {"gpu_mem_util_avg_pct", "gpu_mem_util_peak_pct"}:
            return "not_requested"
    if environment_class == "wsl2" and native_only_metric(metric):
        return "unavailable_on_wsl2"
    if metric.applicability in {"gpu", "gpu_idle_debug"} and row.get("gpu_mode") == "off":
        return "not_applicable"
    if metric.applicability == "cpu" and row.get("gpu_mode") == "on":
        return "not_applicable"
    if str(row.get("status", "")).strip().lower() == "error":
        return "measurement_failed"
    if metric.tool:
        suffix = "torch_profiler_eager" if metric.tool == "torch" else metric.tool
        error = str(row.get(f"compute_profile_error_{suffix}", "")).strip()
        if error.lower() not in MISSING:
            return "profiler_reported_error"
        key = "compute_profile_tools" if metric.tool in {"torch", "ncu"} else "execution_profile_tools"
        enabled = metadata.get(key)
        if isinstance(enabled, str):
            try:
                enabled = json.loads(enabled)
            except ValueError:
                enabled = enabled.split(",")
        if isinstance(enabled, list) and not any(metric.tool in str(tool) for tool in enabled):
            return "tool_not_enabled"
    if row.get("runtime_kind") == "process":
        return "not_requested" if metric.row_group in {"cpu_energy", "gpu_energy", "mips", "gpu_runtime_state"} else "unavailable"
    return "unavailable_unspecified"


def _formula_checks(row: dict) -> dict[str, float]:
    """仅检查可由同一行确定的关系，不重建 RAPL 分段归因或补猜缺失来源。"""
    values = {name: number(row[name]) for name in (
        "vcpu_energy_eff_j", "gpu_energy_eff_j", "packet_request_wire_bytes_per_request",
        "packet_response_wire_bytes_per_request", "packet_total_wire_bytes_per_request",
        "packet_tcp_payload_bytes_per_request",
    ) if name in row}
    expected = {}
    cpu = values.get("vcpu_energy_eff_j")
    gpu = values.get("gpu_energy_eff_j") if row.get("gpu_mode") == "on" else 0.0
    if cpu is not None and cpu >= 0 and gpu is not None and gpu >= 0:
        expected["container_attributed_energy_eff_j"] = cpu + gpu
    request = values.get("packet_request_wire_bytes_per_request")
    response = values.get("packet_response_wire_bytes_per_request")
    if request is not None and response is not None:
        expected["packet_total_wire_bytes_per_request"] = request + response
    total = values.get("packet_total_wire_bytes_per_request")
    payload = values.get("packet_tcp_payload_bytes_per_request")
    if total is not None and payload is not None:
        expected["packet_protocol_overhead_bytes_per_request"] = total - payload
        if total > 0:
            expected["packet_protocol_overhead_ratio"] = (total - payload) / total
    return expected


def _check_workload_counts(row: dict) -> None:
    raw = row.get("workload_contract", "")
    if str(raw).strip().lower() in MISSING:
        return
    summary = json.loads(raw)
    if not isinstance(summary, dict) or summary.get("schema_version") != 1:
        raise ValueError("不支持的 workload_contract 摘要格式")
    total = summary.get("request_count")
    variants = summary.get("variants")
    if type(total) is not int or total < 0 or not isinstance(variants, list):
        raise ValueError("workload_contract 缺少合法 request_count/variants")
    counts = [item.get("count") if isinstance(item, dict) else None for item in variants]
    valid_counts = [count for count in counts if type(count) is int and count > 0]
    if len(valid_counts) != len(counts) or sum(valid_counts) != total:
        raise ValueError("workload_contract 的 variants 计数与 request_count 不一致")
    completed = number(row.get("repeat_in_window"))
    if completed is not None and completed != total:
        raise ValueError("workload_contract.request_count 与 repeat_in_window 不一致")


def audit_result(source: str | Path, *, result_snapshot: ResultCsvSnapshot | None = None,
                 verify_result_snapshot: bool = True) -> dict:
    path = Path(source)
    if path.is_dir():
        path /= "result_layers.json"
    report = {**summarize_quality(None), "schema_version": 1, "result_csv": str(path.resolve()),
              "valid": True, "issues": [], "missing_metrics": {},
              "counts": {"rows": 0, "formal_ok": 0, "warmup": 0, "warn": 0, "error": 0},
              "completion": "unknown", "measurement_status": "unknown", "run_status": "unknown", "coverage": None}

    def issue(code, message, *, severity="error", **detail):
        report["issues"].append({"code": code, "message": str(message), "severity": severity, **detail})
        if severity == "error":
            report["valid"] = False

    def read_json(name):
        try:
            artifact = layout.path(name)
            if not artifact.exists():
                return {}
            return read_json_object(artifact, label=name)
        except (ValueError, OSError) as error:
            issue("invalid_metadata", f"{name}: {error}")
            return {}

    def recorded_failure(name, location, value):
        try:
            if not isinstance(value, dict):
                raise ValueError("typed failure must be an object")
            failure = Failure(**value)
            string_fields = (failure.stage, failure.detail, failure.device, failure.runtime_profile,
                             failure.exception_type)
            if (not failure.stage or not failure.detail or any(not isinstance(item, str) for item in string_fields)
                    or not isinstance(failure.evidence, dict)):
                raise ValueError("typed failure fields have invalid types")
            return failure.to_dict()
        except (TypeError, ValueError) as error:
            issue("invalid_metadata", f"{name} {location}: {error}")
            return None

    try:
        layout = ArtifactLayout.from_csv(path)
    except (ValueError, OSError) as error:
        issue("invalid_layout", error)
        return report
    report["layout_version"] = layout.layout_version
    metadata = read_json("static_meta.json")
    validation = read_json("runtime_validation.json")
    resolution = read_json("model_resolution.json")
    report["failures"] = []
    devices = validation.get("devices", {})
    if not isinstance(devices, dict):
        issue("invalid_metadata", "runtime_validation.json devices: expected an object")
    else:
        for device, item in devices.items():
            if not isinstance(item, dict):
                issue("invalid_metadata", f"runtime_validation.json devices.{device}: expected an object")
            elif item.get("failure") is not None:
                failure = recorded_failure("runtime_validation.json", f"devices.{device}.failure", item["failure"])
                if failure is not None:
                    report["failures"].append(failure)
    recorded = read_json("runtime_failures.json").get("failures", [])
    if not isinstance(recorded, list):
        issue("invalid_metadata", "runtime_failures.json failures: expected a list")
    else:
        for index, value in enumerate(recorded):
            failure = recorded_failure("runtime_failures.json", f"failures[{index}]", value)
            if failure is not None:
                report["failures"].append(failure)
    if resolution.get("failure") is not None:
        failure = recorded_failure("model_resolution.json", "failure", resolution["failure"])
        if failure is not None:
            report["failures"].append(failure)
    report.update(read_quality(path))
    for failure in report["failures"]:
        issue(failure["reason_code"], failure["detail"], failure=failure)
    for check in report["quality_checks"]:
        issue(check["code"], check["detail"], severity=check["severity"], quality_check=check)
    report.update(recorded_identity(metadata))
    state = read_json("run_state.json")
    if state:
        report["run_status"] = state.get("status")
        report["run_id"] = state.get("run_id")
        report["completion"] = "complete" if state.get("status") == "complete" else "incomplete"
    try:
        snapshot = result_snapshot or read_result_csv_snapshot(path)
        if snapshot.path != path.resolve():
            raise ValueError("result CSV snapshot belongs to a different path")
        fields, rows = snapshot.fields, snapshot.rows
        if result_snapshot is not None:
            rows = validate_result_rows(fields, rows, snapshot.path,
                                        allow_empty=path.name == "result_layers.json")
        report["result_sha256"] = snapshot.sha256
        if verify_result_snapshot and not result_csv_snapshot_unchanged(snapshot):
            issue("changing_snapshot", "读取期间 CSV 发生变化，请采集结束后重新审计")
    except (ValueError, OSError, csv.Error) as error:
        issue("invalid_csv", error)
        return report
    report["counts"]["rows"] = len(rows)
    runtime = metadata.get("runtime_kind", "docker")
    report["runtime_kind"] = runtime
    report["resource_scope"] = "observed_process_tree" if runtime == "process" else "container_cgroup"
    hardware = {}
    if runtime == "process":
        from acprof.hardware_conditions import conditions_path
        hardware = read_json(str(conditions_path(layout).relative_to(layout.root))).get("cases", {})
        if not isinstance(hardware, dict):
            issue("invalid_metadata", "hardware conditions cases must be an object")
            hardware = {}
    if "environment_class" not in fields:
        issue("missing_environment_class", "结果缺少 environment_class 环境身份列")
    environments = {row.get("environment_class", "unknown") for row in rows}
    report["row_environment_classes"] = sorted(environments)
    if len(environments) > 1 or any(value not in {"unknown", report["environment_class"]}
                                  for value in environments):
        issue("environment_mismatch", "CSV 与元数据环境身份不一致，或包含混合环境结果")
    if report["environment_class"] == "unknown":
        issue("unknown_environment", "环境身份 unknown；不能作为 Native Linux baseline", severity="warning")
    report["execution"] = collection_outcomes(rows)
    report["unknown_columns"] = [name for name in fields if name not in METRICS]
    report["missing_columns"] = [name for name in METRICS if name not in fields]
    expected_hash = metadata.get("input_scale_plan_sha256")
    if expected_hash and str(expected_hash).lower() not in MISSING:
        try:
            plan = layout.path("input_scale_plan.json")
            if not plan.is_file() or hashlib.sha256(plan.read_bytes()).hexdigest() != expected_hash:
                issue("input_plan_hash", "输入计划与 static_meta.json 中的 SHA256 不一致")
        except (OSError, ValueError) as error:
            issue("input_plan_hash", error)
    if state.get("runtime"):
        try:
            options = state["options"]
            expected = expected_measurements(
                [None] if options["cpus"] == "auto" else [int(value) for value in options["cpus"].split(",")],
                [None] if options["mems"] == "auto" else [int(value) for value in options["mems"].split(",")], options["gpus"].split(","),
                state["runtime"]["planned"]["scales"], options["warmup"], options["repeat"],
            )
            actual = {measurement_key(row) for row in rows}
            report["coverage"] = {"expected": len(expected), "actual": len(actual),
                                  "missing": len(expected - actual), "unexpected": len(actual - expected)}
            if expected != actual:
                issue("plan_coverage", "结果行与原实验计划不完全一致")
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            issue("invalid_run_state", error)
    missing: dict[str, Counter[str]] = defaultdict(Counter)
    for index, row in enumerate(rows, 2):
        status = row.get("status", "").strip().lower()
        warmup = number(row["warmup"]) == 1
        report["counts"]["warmup"] += warmup
        if status not in {"ok", "warn", "error"}:
            issue("invalid_status", "status 应为 ok、warn 或 error", row=index)
        if status in {"warn", "error"}:
            report["counts"][status] += 1
        formal = status == "ok" and not warmup
        report["counts"]["formal_ok"] += formal
        if runtime == "process" or row.get("runtime_kind") == "process":
            from acprof.artifact_layout import resource_case_id
            if (runtime != "process" or row.get("runtime_kind") != "process"
                    or not metadata.get("runtime_id") or row.get("runtime_id") != metadata["runtime_id"]):
                issue("runtime_identity", "Process Runtime 身份与元数据不一致", row=index)
            if row.get("resource_scope") != "observed_process_tree":
                issue("resource_scope", "Process 指标必须标记 observed_process_tree", row=index)
            if status in {"ok", "warn"}:
                case_id = resource_case_id(row["cpu_cores"], row["mem_cap_gb"], row["gpu_mode"])
                record = hardware.get(case_id, {})
                plan = record.get("resource_plan", {}) if isinstance(record, dict) else {}
                if not isinstance(plan, dict) or plan.get("verified") is not True:
                    issue("resource_enforcement", "缺少已验证的 Process 资源限制证据", row=index)
                else:
                    requested = plan.get("requested", {})
                    for field, name in (("cpu_cores", "cpu_quota"), ("mem_cap_gb", "memory_gib")):
                        value = number(row.get(field))
                        if (value != requested.get(name)
                                or value is not None and plan.get("method") != "cgroup_v2"):
                            issue("resource_enforcement", f"{field} 与实际实施的限制不符", row=index, field=field)
                    if row.get("resource_limit_source") != plan.get("method"):
                        issue("resource_enforcement", "资源限制来源与执行证据不符", row=index)
        try:
            _check_workload_counts(row)
        except (ValueError, TypeError) as error:
            issue("workload_contract", error, row=index)
        for field in NUMERIC_FIELDS:
            if (runtime == "process" and (field.startswith("container_") or field.startswith("cold_start_"))
                    and number(row.get(field)) is not None):
                issue("unsupported_process_metric", f"Process 不能记录 Docker 专属指标 {field}", row=index, field=field)
            if (report["environment_class"] == "wsl2" and native_only_metric(METRICS[field])
                    and number(row.get(field)) is not None):
                issue("unsupported_wsl_metric", f"WSL2 不允许记录 Native 专属指标 {field}", row=index, field=field)
            raw = str(row.get(field, "nan")).strip().lower()
            if raw not in MISSING and number(raw) is None:
                issue("invalid_number", f"{field} 不是有限数值或 nan", row=index, field=field)
            elif formal and raw in MISSING:
                reason = missing_reason(field, row, metadata, environment_class=report["environment_class"])
                missing[field][reason] += 1
        if formal:
            for field, expected_value in _formula_checks(row).items():
                actual_value = number(row.get(field))
                # CSV 保存到 6 位小数，容忍加减的累计舍入，不把缺失分母解释为错误。
                if actual_value is not None and not math.isclose(actual_value, expected_value, rel_tol=2e-5, abs_tol=2e-6):
                    issue("formula_mismatch", f"{field}: 实际 {actual_value:g}，应为 {expected_value:g}", row=index, field=field)
    report["missing_metrics"] = {name: dict(reasons) for name, reasons in missing.items()}
    coverage = report["coverage"]
    if coverage is not None:
        formal = report["counts"]["rows"] - report["counts"]["warmup"]
        report["measurement_status"] = (
            "complete" if report["completion"] == "complete" and not coverage["missing"]
            and not coverage["unexpected"] and formal > 0 and formal == report["counts"]["formal_ok"]
            else "incomplete")
    return report
