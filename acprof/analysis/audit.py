"""只读结果审计；保持历史字段和不可用原因，不回填测量值。"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable
from decimal import InvalidOperation
from pathlib import Path

from acprof.analysis.workload import parse_workload_summary
from acprof.cancellation import cancellable, check_cancelled
from acprof.capabilities import collection_outcomes
from acprof.failures import Failure
from acprof.platform import native_only_metric, recorded_identity
from acprof.quality import read_quality, summarize_quality
from acprof.results import is_inferred_result
from acprof.results.csv import (
    MISSING_CSV_TEXT,
    ResultCsvSnapshot,
    ResultRunIdentityError,
    csv_diagnostic_text,
    measurement_key,
    read_result_csv_snapshot,
    recorded_run_id,
    require_matching_run_id,
    result_csv_snapshot_unchanged,
    run_id_column,
    validate_result_rows,
)
from acprof.results.measurement_plan import (
    RecordedMeasurementOptions,
    decimal_coordinate,
    recorded_scales,
)
from acprof.results.metadata import ResultMetadataSnapshot
from acprof.results.metrics import METRICS, NUMERIC_FIELDS
from acprof.results.profile import metric_observation

type AuditedRows = ResultCsvSnapshot | OSError | ValueError | csv.Error


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


class MissingReasons:
    """Prepare settings for one read-only source, with separate evidence per row."""

    def __init__(self, metadata: dict, *, environment_class: str):
        self._metadata = metadata
        self._wsl = environment_class == "wsl2"
        self._settings: dict[str, object] = {}
        self._enabled: dict[str, bool] = {}

    def _tool_enabled(self, tool: str) -> bool:
        if tool not in self._enabled:
            key = "compute_profile_tools" if tool in {"torch", "ncu"} else "execution_profile_tools"
            if key not in self._settings:
                enabled = self._metadata.get(key)
                if isinstance(enabled, str):
                    try:
                        enabled = json.loads(enabled)
                    except ValueError:
                        enabled = enabled.split(",")
                self._settings[key] = enabled
            enabled = self._settings[key]
            self._enabled[tool] = not isinstance(enabled, list) or any(tool in str(item) for item in enabled)
        return self._enabled[tool]

    @staticmethod
    def _process_reason(row: dict, field: str) -> str | None:
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
        return None

    def _tool_reason(self, tool: str, row: dict, reasons: dict[str, str | None]) -> str | None:
        if tool not in reasons:
            suffix = "torch_profiler_eager" if tool == "torch" else tool
            error = csv_diagnostic_text(row.get(f"compute_profile_error_{suffix}"))
            reasons[tool] = ("profiler_reported_error" if error else
                             "tool_not_enabled" if not self._tool_enabled(tool) else None)
        return reasons[tool]

    def for_row(self, row: dict) -> Callable[[str], str]:
        """Bind row state; the row must stay read-only while resolving its fields."""
        process = row.get("runtime_kind", self._metadata.get("runtime_kind")) == "process"
        row_process = row.get("runtime_kind") == "process"
        gpu = row.get("gpu_mode")
        failed = str(row.get("status", "")).strip().lower() == "error"
        tool_reasons: dict[str, str | None] = {}

        def reason(field: str) -> str:
            if field not in row:
                return "not_recorded"
            metric = METRICS[field]
            quality = metric_observation(row, field)['quality']
            if quality not in {'complete', 'unknown'}:
                return quality
            if process and (process_reason := self._process_reason(row, field)) is not None:
                return process_reason
            if self._wsl and native_only_metric(metric):
                return "unavailable_on_wsl2"
            if metric.applicability in {"gpu", "gpu_idle_debug"} and gpu == "off":
                return "not_applicable"
            if metric.applicability == "cpu" and gpu == "on":
                return "not_applicable"
            if failed:
                return "measurement_failed"
            if metric.tool and (tool_reason := self._tool_reason(metric.tool, row, tool_reasons)) is not None:
                return tool_reason
            if row_process:
                return "not_requested" if metric.row_group in {"cpu_energy", "gpu_energy", "mips"} else "unavailable"
            return "unavailable_unspecified"

        return reason


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
    summary = parse_workload_summary(row.get("workload_contract"))
    if summary is None:
        return
    try:
        completed = decimal_coordinate(str(row.get("repeat_in_window", "")))
    except InvalidOperation:
        return  # Numeric-column validation handles malformed cells.
    if completed.is_finite() and completed != summary["request_count"]:
        raise ValueError("workload_contract.request_count 与 repeat_in_window 不一致")


def _recorded_failure(name, location, value, issue):
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


def _collect_recorded_failures(validation, resolution, recorded, issue, cancelled):
    failures = []
    devices = validation.get("devices", {})
    if not isinstance(devices, dict):
        issue("invalid_metadata", "runtime_validation.json devices: expected an object")
    else:
        for device, item in cancellable(devices.items(), cancelled):
            if not isinstance(item, dict):
                issue("invalid_metadata", f"runtime_validation.json devices.{device}: expected an object")
            elif item.get("failure") is not None:
                failure = _recorded_failure("runtime_validation.json", f"devices.{device}.failure", item["failure"], issue)
                if failure is not None:
                    failures.append(failure)
    if not isinstance(recorded, list):
        issue("invalid_metadata", "runtime_failures.json failures: expected a list")
    else:
        for index, value in enumerate(cancellable(recorded, cancelled)):
            failure = _recorded_failure("runtime_failures.json", f"failures[{index}]", value, issue)
            if failure is not None:
                failures.append(failure)
    if resolution.get("failure") is not None:
        failure = _recorded_failure("model_resolution.json", "failure", resolution["failure"], issue)
        if failure is not None:
            failures.append(failure)
    return failures


def _read_preparation(state, read_json):
    attempts = state.get("attempts")
    if isinstance(attempts, list) and attempts and isinstance(attempts[-1], dict):
        prep_rel = attempts[-1].get("preparation_evidence")
        if isinstance(prep_rel, str) and prep_rel:
            return read_json(prep_rel)
    return {}


def _record_early_failure(report, state, preparation, interface, issue):
    if report["failures"]:
        return
    prep_status = preparation.get("status")
    if prep_status in {"failed", "cancelled"}:
        prep_exception = preparation.get("exception")
        prep_exc = prep_exception if isinstance(prep_exception, dict) else {}
        issue(
            preparation.get("reason_code") or f"preparation_{prep_status}",
            prep_exc.get("message") or f"preparation {prep_status}",
            stage=preparation.get("last_stage") or "preparation",
        )
    elif interface.get("status") in {"error", "cancelled"}:
        iface_status = interface["status"]
        issue(
            "interface_validation_failed" if iface_status == "error" else "interface_validation_cancelled",
            interface.get("error") or f"interface validation {iface_status}",
            stage=interface.get("failed_stage") or "interface",
        )
    elif state.get("status") in {"failed", "cancelled"} and not state.get("runtime"):
        issue(f"run_{state['status']}", f"run_state.json status is {state['status']}")


def _audit_metadata(report, path, metadata_reader, read_json, issue, cancelled):
    metadata = read_json("static_meta.json")
    validation = read_json("runtime_validation.json")
    resolution = read_json("model_resolution.json")
    interface = read_json("interface_validation.json")
    recorded_interface = resolution.get("interface_validation")
    if not interface and isinstance(recorded_interface, dict):
        interface = recorded_interface
    recorded = read_json("runtime_failures.json").get("failures", [])
    report["failures"] = _collect_recorded_failures(validation, resolution, recorded, issue, cancelled)
    report.update(read_quality(path, metadata_snapshot=metadata_reader, cancelled=cancelled))
    for failure in cancellable(report["failures"], cancelled):
        issue(failure["reason_code"], failure["detail"], failure=failure)
    for check in cancellable(report["quality_checks"], cancelled):
        issue(check["code"], check["detail"], severity=check["severity"], quality_check=check)
    report.update(recorded_identity(metadata))
    state = read_json("run_state.json")
    if state:
        report["run_status"] = state.get("status")
        report["run_id"] = state.get("run_id")
        report["completion"] = "complete" if state.get("status") == "complete" else "incomplete"
        if state.get("status") != "complete" and not state.get("runtime"):
            report["measurement_status"] = "incomplete"
    preparation = _read_preparation(state, read_json)
    _record_early_failure(report, state, preparation, interface, issue)
    return metadata, state


def require_consistent_run_identity(audit: dict) -> None:
    """Prevent analysis of results whose recorded identities contradict one another."""
    for issue in audit["issues"]:
        if issue["code"] == "run_id_mismatch":
            raise ResultRunIdentityError(issue["message"])


def _audit_source_unchanged(snapshot, rows, *, borrowed, cancelled):
    if not borrowed or snapshot.path.name == "result_layers.json":
        return result_csv_snapshot_unchanged(snapshot, verify_rows=borrowed, cancelled=cancelled)
    # The caller can edit a flat snapshot's rows without changing its original byte digest.
    try:
        current = read_result_csv_snapshot(snapshot.path, cancelled=cancelled)
    except (ValueError, OSError, csv.Error):
        check_cancelled(cancelled)
        return False
    return (snapshot.run_id is None and snapshot.sha256 == current.sha256 and snapshot.fields == current.fields
            and len(rows) == len(current.rows)
            and all(left == right for left, right in cancellable(zip(rows, current.rows, strict=True), cancelled)))


def _load_result_rows(path, result_snapshot, verify_result_snapshot, report, issue, cancelled) -> AuditedRows:
    try:
        snapshot = result_snapshot or read_result_csv_snapshot(path, cancelled=cancelled)
        if snapshot.path != path.resolve():
            raise ValueError("result CSV snapshot belongs to a different path")
        fields, rows = snapshot.fields, snapshot.rows
        report["result_sha256"] = snapshot.sha256
        state_id = recorded_run_id(report.get("run_id"))
        require_matching_run_id(state_id, snapshot.run_id, source=str(snapshot.path))
        identity_column = run_id_column(fields)
        if state_id is not None and identity_column is not None:
            for line, row in enumerate(cancellable(rows, cancelled), 2):
                require_matching_run_id(state_id, row.get(identity_column), source=f"{snapshot.path}:{line}")
        if result_snapshot is not None:
            rows = validate_result_rows(fields, rows, snapshot.path, cancelled=cancelled,
                                        allow_empty=path.name == "result_layers.json").rows
        if verify_result_snapshot and not _audit_source_unchanged(
                snapshot, rows, borrowed=result_snapshot is not None, cancelled=cancelled):
            issue("changing_snapshot", "读取期间 CSV 发生变化，请采集结束后重新审计")
        return ResultCsvSnapshot(snapshot.path, fields, rows, snapshot.sha256, snapshot.run_id)
    except ResultRunIdentityError as error:
        issue("run_id_mismatch", error)
        return error
    except (ValueError, OSError, csv.Error) as error:
        issue("invalid_csv", error)
        return error


def _process_hardware(layout, runtime, read_json, issue):
    if runtime != "process":
        return {}
    from acprof.hardware_conditions import conditions_path
    hardware = read_json(str(conditions_path(layout).relative_to(layout.root))).get("cases", {})
    if not isinstance(hardware, dict):
        issue("invalid_metadata", "hardware conditions cases must be an object")
        return {}
    return hardware


def _audit_environment(report, fields, rows, issue, cancelled):
    if "environment_class" not in fields:
        issue("missing_environment_class", "结果缺少 environment_class 环境身份列")
    environments = {row.get("environment_class", "unknown") for row in cancellable(rows, cancelled)}
    report["row_environment_classes"] = sorted(environments)
    if len(environments) > 1 or any(value not in {"unknown", report["environment_class"]}
                                  for value in environments):
        issue("environment_mismatch", "CSV 与元数据环境身份不一致，或包含混合环境结果")
    if report["environment_class"] == "unknown":
        issue("unknown_environment", "环境身份 unknown；不能作为 Native Linux baseline", severity="warning")
    report["execution"] = collection_outcomes(rows, cancelled=cancelled)
    report["unknown_columns"] = [name for name in fields if name not in METRICS]
    report["missing_columns"] = [name for name in METRICS if name not in fields]


def _audit_input_plan(metadata_reader, metadata, issue, cancelled):
    expected_hash = metadata.get("input_scale_plan_sha256")
    if not expected_hash or str(expected_hash).lower() in MISSING_CSV_TEXT:
        return
    try:
        plan = (metadata_reader.read("input_scale_plan.json", cancelled=cancelled)
                if metadata_reader.exists("input_scale_plan.json") else None)
        if plan is None or hashlib.sha256(plan.content).hexdigest() != expected_hash:
            issue("input_plan_hash", "输入计划与 static_meta.json 中的 SHA256 不一致")
    except (OSError, ValueError) as error:
        issue("input_plan_hash", error)


def _audit_plan_coverage(report, state, rows, issue, cancelled):
    if "runtime" not in state:
        return
    try:
        options = RecordedMeasurementOptions.parse(state.get("options"))
        scales = recorded_scales(state["runtime"]["planned"]["scales"])
    except (ValueError, KeyError, TypeError) as error:
        issue("invalid_run_state", error)
        return
    actual = {measurement_key(row) for row in cancellable(rows, cancelled)}
    coverage = options.coverage(scales, actual, cancelled=cancelled)
    report["coverage"] = coverage
    if coverage["missing"] or coverage["unexpected"]:
        issue("plan_coverage", "结果行与原实验计划不完全一致")


def _audit_process_row(report, runtime, metadata, hardware, index, row, status, issue):
    if runtime != "process" and row.get("runtime_kind") != "process":
        return
    from acprof.results.layout import resource_case_id
    if (runtime != "process" or row.get("runtime_kind") != "process"
            or not metadata.get("runtime_id") or row.get("runtime_id") != metadata["runtime_id"]):
        issue("runtime_identity", "Process Runtime 身份与元数据不一致", row=index)
    if row.get("resource_scope") != "observed_process_tree":
        issue("resource_scope", "Process 指标必须标记 observed_process_tree", row=index)
    if status not in {"ok", "warn"}:
        return
    case_id = resource_case_id(row["cpu_cores"], row["mem_cap_gb"], row["gpu_mode"])
    record = hardware.get(case_id, {})
    plan = record.get("resource_plan", {}) if isinstance(record, dict) else {}
    if not isinstance(plan, dict) or plan.get("verified") is not True:
        issue("resource_enforcement", "缺少已验证的 Process 资源限制证据", row=index)
        return
    requested = plan.get("requested", {})
    for field, name in (("cpu_cores", "cpu_quota"), ("mem_cap_gb", "memory_gib")):
        value = number(row.get(field))
        if (value != requested.get(name)
                or value is not None and plan.get("method") != "cgroup_v2"):
            issue("resource_enforcement", f"{field} 与实际实施的限制不符", row=index, field=field)
    if row.get("resource_limit_source") != plan.get("method"):
        issue("resource_enforcement", "资源限制来源与执行证据不符", row=index)
    if any(number(value) is not None for field, value in row.items() if field.startswith('service_cgroup_')):
        if plan.get('method') != 'cgroup_v2' or not plan.get('cgroup'):
            issue('service_cgroup_scope', '服务 cgroup 指标缺少独立 cgroup 归属证据', row=index)


def _audit_numeric_row(report, runtime, missing_reasons, missing, partial, index, row, formal, issue, cancelled):
    reason_for_row = None
    for field in cancellable(NUMERIC_FIELDS, cancelled):
        if (runtime == "process" and (field.startswith("container_") or field.startswith("cold_start_"))
                and number(row.get(field)) is not None):
            issue("unsupported_process_metric", f"Process 不能记录 Docker 专属指标 {field}", row=index, field=field)
        if (report["environment_class"] == "wsl2" and native_only_metric(METRICS[field])
                and number(row.get(field)) is not None):
            issue("unsupported_wsl_metric", f"WSL2 不允许记录 Native 专属指标 {field}", row=index, field=field)
        raw = str(row.get(field, "nan"))
        absent = raw in MISSING_CSV_TEXT
        if not absent:
            raw = raw.strip().lower()
            absent = raw in MISSING_CSV_TEXT
        if not absent and number(raw) is None:
            issue("invalid_number", f"{field} 不是有限数值或 nan", row=index, field=field)
        elif formal and absent:
            if reason_for_row is None:
                reason_for_row = missing_reasons.for_row(row)
            reason = reason_for_row(field)
            missing[field][reason] += 1
        elif partial is not None and not absent:
            observation = metric_observation(row, field)
            if observation["quality"] == "partial":
                partial[field][observation["reason"] or "partial"] += 1


def _audit_formula_row(index, row, formal, issue):
    if not formal:
        return
    for field, expected_value in _formula_checks(row).items():
        actual_value = number(row.get(field))
        # CSV 保存到 6 位小数，容忍加减的累计舍入，不把缺失分母解释为错误。
        if actual_value is not None and not math.isclose(actual_value, expected_value, rel_tol=2e-5, abs_tol=2e-6):
            issue("formula_mismatch", f"{field}: 实际 {actual_value:g}，应为 {expected_value:g}", row=index, field=field)


def _audit_rows(report, runtime, metadata, hardware, rows, issue, cancelled):
    missing: dict[str, Counter[str]] = defaultdict(Counter)
    partial: dict[str, Counter[str]] = defaultdict(Counter)
    missing_reasons = MissingReasons(metadata, environment_class=report["environment_class"])
    for index, row in enumerate(cancellable(rows, cancelled), 2):
        status = row.get("status", "").strip().lower()
        warmup = number(row["warmup"]) == 1
        report["counts"]["warmup"] += warmup
        if status not in {"ok", "warn", "error"}:
            issue("invalid_status", "status 应为 ok、warn 或 error", row=index)
        if status in {"warn", "error"}:
            report["counts"][status] += 1
        formal = status == "ok" and not warmup
        inferred = is_inferred_result(row)
        measured_ok = formal and not inferred
        report["counts"]["formal_ok"] += measured_ok
        _audit_process_row(report, runtime, metadata, hardware, index, row, status, issue)
        try:
            _check_workload_counts(row)
        except (ValueError, TypeError) as error:
            issue("workload_contract", error, row=index)
        observed = (not inferred and not warmup and status in {"ok", "warn"}
                    and row.get("resource_metric_quality", "") not in MISSING_CSV_TEXT)
        _audit_numeric_row(report, runtime, missing_reasons, missing, partial if observed else None,
                           index, row, measured_ok, issue, cancelled)
        _audit_formula_row(index, row, formal, issue)
    report["missing_metrics"] = {name: dict(reasons) for name, reasons in missing.items()}
    if partial:
        report["partial_metrics"] = {name: dict(reasons) for name, reasons in partial.items()}


def _finalize_measurement_status(report):
    coverage = report["coverage"]
    if coverage is None:
        return
    formal = report["counts"]["rows"] - report["counts"]["warmup"]
    report["measurement_status"] = (
        "complete" if report["completion"] == "complete" and not coverage["missing"]
        and not coverage["unexpected"] and formal > 0 and formal == report["counts"]["formal_ok"]
        else "incomplete")


def audit_result(source: str | Path, *, result_snapshot: ResultCsvSnapshot | None = None,
                 metadata_snapshot: ResultMetadataSnapshot | None = None,
                 verify_result_snapshot: bool = True, cancelled: Callable[[], bool] | None = None) -> dict:
    report, _ = audit_result_snapshot(source, result_snapshot=result_snapshot, metadata_snapshot=metadata_snapshot,
                                      verify_result_snapshot=verify_result_snapshot, cancelled=cancelled)
    return report


def audit_result_snapshot(source: str | Path, *, result_snapshot: ResultCsvSnapshot | None = None,
                          metadata_snapshot: ResultMetadataSnapshot | None = None,
                          verify_result_snapshot: bool = True,
                          cancelled: Callable[[], bool] | None = None) -> tuple[dict, AuditedRows]:
    """Return the audited rows (or their original read error) without reopening the source."""
    report, rows, _ = audit_with_metadata(source, result_snapshot=result_snapshot, metadata_snapshot=metadata_snapshot,
                                          verify_result_snapshot=verify_result_snapshot, cancelled=cancelled)
    return report, rows


def audit_with_metadata(source: str | Path, *, result_snapshot: ResultCsvSnapshot | None = None,
                       metadata_snapshot: ResultMetadataSnapshot | None = None,
                       verify_result_snapshot: bool = True, cancelled: Callable[[], bool] | None = None,
                       ) -> tuple[dict, AuditedRows, ResultMetadataSnapshot | None]:
    """Retain the metadata reader for comparison, including when only the table failed."""
    check_cancelled(cancelled)
    raw_path = Path(source).expanduser()
    if raw_path.is_dir():
        raw_path /= "result_layers.json"
    report = {**summarize_quality(None), "schema_version": 1, "result_csv": str(raw_path.absolute()),
              "valid": True, "issues": [], "missing_metrics": {},
              "counts": {"rows": 0, "formal_ok": 0, "warmup": 0, "warn": 0, "error": 0},
              "completion": "unknown", "measurement_status": "unknown", "run_status": "unknown", "coverage": None}

    def issue(code, message, *, severity="error", **detail):
        report["issues"].append({"code": code, "message": str(message), "severity": severity, **detail})
        if severity == "error":
            report["valid"] = False

    try:
        if metadata_snapshot is None:
            metadata_reader = ResultMetadataSnapshot(source, cancelled=cancelled)
            path = metadata_reader.result_path
        else:
            metadata_reader = metadata_snapshot
            path = metadata_reader.validate_source(source, cancelled=cancelled)
        report["result_csv"] = str(path.resolve())
        layout = metadata_reader.layout
    except (ValueError, OSError) as error:
        check_cancelled(cancelled)
        issue("invalid_layout", error)
        return report, error, None

    def read_json(name):
        check_cancelled(cancelled)
        try:
            if not metadata_reader.exists(name):
                return {}
            return metadata_reader.read(name, cancelled=cancelled).payload
        except (ValueError, OSError) as error:
            issue("invalid_metadata", f"{name}: {error}")
            return {}

    report["layout_version"] = layout.layout_version
    metadata, state = _audit_metadata(report, path, metadata_reader, read_json, issue, cancelled)
    loaded = _load_result_rows(path, result_snapshot, verify_result_snapshot, report, issue, cancelled)
    if not isinstance(loaded, ResultCsvSnapshot):
        check_cancelled(cancelled)
        return report, loaded, metadata_reader
    fields, rows = loaded.fields, loaded.rows
    report["counts"]["rows"] = len(rows)
    runtime = metadata.get("runtime_kind", "docker")
    report["runtime_kind"] = runtime
    report["resource_scope"] = "observed_process_tree" if runtime == "process" else "container_cgroup"
    hardware = _process_hardware(layout, runtime, read_json, issue)
    _audit_environment(report, fields, rows, issue, cancelled)
    _audit_input_plan(metadata_reader, metadata, issue, cancelled)
    _audit_plan_coverage(report, state, rows, issue, cancelled)
    _audit_rows(report, runtime, metadata, hardware, rows, issue, cancelled)
    _finalize_measurement_status(report)
    check_cancelled(cancelled)
    return report, loaded, metadata_reader
