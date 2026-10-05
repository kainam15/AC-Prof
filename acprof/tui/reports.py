"""将现有统计/对照 JSON 转成只读表格；不加载 Textual 或测量依赖。"""
from __future__ import annotations

import json
import math
from concurrent.futures import CancelledError
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from acprof.messages import join_messages, message
from acprof.tui.presentation import NOT_APPLICABLE, STATUS_LEGEND, UNKNOWN


@dataclass(frozen=True)
class ReportRow:
    cells: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class ReportView:
    source: Path
    title: str
    columns: tuple[str, ...]
    rows: tuple[ReportRow, ...]
    note: str


_METRIC_LABELS = {
    "latency_app_s": "应用延迟",
    "latency_s": "抓包延迟",
    "cpu_energy_total_j": "CPU 总能耗",
    "gpu_energy_total_j": "GPU 总能耗",
    "container_attributed_energy_eff_j": "容器归因有效能耗",
}
_REASONS = {
    "insufficient_windows": "窗口不足",
    "missing_windows_break_blocks": "缺失窗口打断连续块",
    "nonconsecutive_windows": "窗口序号不连续",
}

_MAX_REPORT_BYTES = 32 * 1024 * 1024
_REPORT_READ_CHUNK_BYTES = 256 * 1024


def _number(value, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(message("报告包含无效数值"))
    return float(value)


def _count(value) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(message("报告包含无效样本数"))
    return value


def _interval(row: dict, low_key: str, high_key: str) -> tuple[float | None, float | None]:
    low = _number(row.get(low_key), nullable=True)
    high = _number(row.get(high_key), nullable=True)
    if (low is None) != (high is None) or (low is not None and low > high):
        raise ValueError(message("报告置信区间无效"))
    return low, high


def _objects(value) -> list[dict]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(message("报告缺少有效的数据行"))
    return value


def _text(value, fallback: str = "") -> str:
    if value is None:
        return fallback
    if not isinstance(value, str):
        raise ValueError(message("报告包含无效文本"))
    return value


def render_quality_summary(payload: dict) -> str:
    """Keep state, observed-window completeness and output quality separate."""
    if "quality_reasons" not in payload:
        from acprof.quality import combine_quality, summarize_quality
        quality = summarize_quality(payload.get("quality_checks"))
        if payload.get("quality_status") == "unknown":
            quality = combine_quality([quality, summarize_quality(None)])
        payload = {**payload, **quality}
    lines = [message("运行：{0} · 测量完整性：{1} · 质量：{2}", payload.get("run_status", "unknown"),
                     payload.get("measurement_status", "unknown"), payload.get("quality_status", "unknown")),
             message("自动优选：{0} · 原因：{1}", message("可参与" if payload.get("auto_selection_eligible") is True else "暂停"),
                     ", ".join(payload.get("quality_reasons", ["quality_evidence_missing"])) or "—")]
    for check in payload.get("quality_checks", []):
        lines.append(message("质量证据：{0} · {1} · 来源：{2}", check.get("code", "unknown"),
                             check.get("detail", ""), json.dumps(check.get("evidence", {}), ensure_ascii=False)))
    return join_messages("\n", lines)


def _windows(source: Path, data: dict) -> ReportView:
    confidence = _number(data.get("confidence"))
    if not 0 < confidence < 1 or data.get("filter") != "status=ok and warmup=0":
        raise ValueError(message("报告的置信水平或窗口筛选口径不受支持"))
    rows = []
    for group in _objects(data.get("groups")):
        name = _text(group.get("metric"))
        if not name:
            raise ValueError(message("报告缺少指标名称"))
        unit = _text(group.get("unit"))
        multiplier, display_unit = (1000, "ms") if unit == "s" else (1, unit)
        count = _count(group.get("n_windows"))
        missing = _count(group.get("missing_windows"))
        mean = _number(group.get("mean"), nullable=True)
        std = _number(group.get("std"), nullable=True)
        low, high = _interval(group, "ci_low", "ci_high")
        if (count == 0) != (mean is None) or (low is not None and count < 3):
            raise ValueError(message("报告的样本数与统计值不一致"))
        mode = group.get("gpu_mode")
        if mode not in {"on", "off"}:
            raise ValueError(message("报告包含无效设备配置"))
        config = "/".join((f"{_number(group.get('cpu_cores')):g}c",
                           f"{_number(group.get('mem_cap_gb')):g}G",
                           "GPU" if mode == "on" else "CPU",
                           f"{_number(group.get('input_scale')):g}"))
        inapplicable = name == "gpu_energy_total_j" and mode == "off" and count == 0
        missing_value = NOT_APPLICABLE if inapplicable else UNKNOWN
        mean_text = missing_value if mean is None else f"{mean * multiplier:.6g} {display_unit}".strip()
        interval = missing_value if low is None else f"[{low * multiplier:.6g}, {high * multiplier:.6g}] {display_unit}".strip()
        reason = _text(group.get("reason"))
        state = ("不适用" if inapplicable else "无有效窗口" if not count else _REASONS.get(reason, reason or
                 ("已估计" if low is not None else "区间不可用")))
        rows.append(ReportRow(
            (config, message(_METRIC_LABELS.get(name, name)), mean_text, interval,
             f"{count}/{missing}", message(state)),
            message("字段：{0} · 标准差：{1} · 有效窗口：{2} · 缺失：{3}",
                    name, missing_value if std is None else f"{std * multiplier:.6g} {display_unit}", count, missing),
        ))
    origin = _text(data.get("result_csv"), str(source))
    note = join_messages("\n", (
        STATUS_LEGEND,
        message("仅统计正式成功窗口；少量窗口的区间可能不稳定。"),
        message("数据来源：{0}", origin),
        render_quality_summary(data),
    ))
    return ReportView(source, message("窗口统计 · {0:g}% 置信区间 · {1} 项", confidence * 100, len(rows)),
                      ("资源/输入", "指标", "均值", message("{0:g}% 区间", confidence * 100),
                       "有效/缺失", "说明"), tuple(rows), note)


def _comparison_row(label: str, row: dict, count: int, detail: str) -> ReportRow:
    change = _number(row.get("paired_mean_change_pct"))
    low, high = _interval(row, "ci_low_pct", "ci_high_pct")
    if count < 3 or low is None:
        raise ValueError(message("对照报告缺少足够的配对轮次或区间"))
    state = "方向不确定" if low <= 0 <= high else "延迟增加" if low > 0 else "本次延迟降低"
    return ReportRow((label, str(count), f"{change:+.6g}%", f"[{low:.6g}%, {high:.6g}%]", message(state)), detail)


def _comparisons(source: Path, data: dict) -> ReportView:
    if data.get("successful") is not True:
        raise ValueError(message("对照实验未完成或失败：{0}", _text(data.get("error")) or UNKNOWN))
    rows = []
    if data["kind"] == "monitor_overhead_diagnostic":
        mode = data.get("gpu_mode")
        if mode not in {"on", "off"}:
            raise ValueError(message("报告包含无效设备配置"))
        for row in _objects(data.get("comparisons")):
            label = _text(row.get("scenario"))
            if label.startswith("monitors-"):
                label = label.removeprefix("monitors-") + " Hz"
            rows.append(_comparison_row(label, row, _count(row.get("paired_rounds")),
                                        message("监测线程与关闭监测的基线对比；不包含抓包、perf 或 TUI。")))
        title = message("{0} 监测开销 · 95% 置信区间", "GPU" if mode == "on" else "CPU")
    else:
        pairs = _objects(data.get("pairs"))
        ui = data.get("ui")
        if ui not in {"terminal", "headless"}:
            raise ValueError(message("报告包含无效界面对照模式"))
        detail = message("包含终端绘制；具体终端环境以原实验记录为准。") if ui == "terminal" else message("仅 TUI 调度和日志路径，不包含终端绘制。")
        rows.append(_comparison_row("TUI / CLI", data, len(pairs), detail))
        title = message("CLI/TUI 对照 · 95% 置信区间")
    if not rows:
        raise ValueError(message("报告缺少有效的数据行"))
    note = message("正值表示更慢；跨零时方向不确定，负值不证明普遍加速。仅限本次对照。")
    return ReportView(source, title, ("对照", "配对轮数", "延迟变化", "95% 区间", "判断"), tuple(rows), note)


def _independent_comparison(source: Path, data: dict) -> ReportView:
    confidence = _number(data.get("confidence"))
    if (not 0 < confidence < 1 or data.get("resampling_unit") != "independent_run_mean"
            or data.get("direction") != "right_minus_left_and_right_divided_by_left"
            or data.get("status") not in {"compatible", "incompatible", "unknown"}):
        raise ValueError(message("独立实验比较的方向、可比性或统计口径无效"))
    rows = []
    for group in _objects(data.get("groups")):
        metric, unit = _text(group.get("metric")), _text(group.get("unit"))
        multiplier, display_unit = (1000, "ms") if unit == "s" else (1, unit)

        def value(number):
            parsed = _number(number, nullable=True)
            return UNKNOWN if parsed is None else f"{parsed * multiplier:.6g} {display_unit}".strip()

        def interval(key, *, ratio=False):
            bounds = group.get(key)
            if bounds is None:
                return UNKNOWN
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError(message("报告置信区间无效"))
            low, high = (_number(item) for item in bounds)
            if low > high:
                raise ValueError(message("报告置信区间无效"))
            factor = 1 if ratio else multiplier
            suffix = "" if ratio else f" {display_unit}"
            return f"[{low * factor:.6g}, {high * factor:.6g}]{suffix}".strip()

        left, right = group.get("left"), group.get("right")
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise ValueError(message("报告缺少有效的数据行"))
        ratio = _number(group.get("ratio"), nullable=True)
        reason = _text(group.get("reason")) or _text(group.get("comparability"))
        qualities = "/".join(_text(side.get("quality_status"), "unknown") for side in (left, right))
        explanation = f"{reason} · quality={qualities}"
        config = "/".join((f"{_number(group.get('cpu_cores')):g}c", f"{_number(group.get('mem_cap_gb')):g}G",
                           _text(group.get("gpu_mode")), f"{_number(group.get('input_scale')):g}"))
        coordinates = group.get("resource_coordinates", {})
        if (isinstance(coordinates, dict) and coordinates.get("left") and coordinates.get("right")
                and coordinates["left"] != coordinates["right"]):
            config = " → ".join(f"{_number(coordinates[side].get('cpu_cores')):g}c/"
                                 f"{_number(coordinates[side].get('mem_cap_gb')):g}G" for side in ("left", "right")) + "/" + "/".join((
                                     _text(group.get("gpu_mode")), f"{_number(group.get('input_scale')):g}"))
        detail = json.dumps(group, ensure_ascii=False, indent=2)
        rows.append(ReportRow((explanation, config, message(_METRIC_LABELS.get(metric, metric)),
            value(left.get("mean")), value(right.get("mean")), UNKNOWN if ratio is None else f"{(ratio - 1) * 100:+.6g}%",
            value(group.get("difference")), interval("difference_ci"), interval("ratio_ci", ratio=True),
            f"{_count(left.get('n_runs'))}/{_count(right.get('n_runs'))}"), detail))
    note = join_messages("\n", (
        message("左组为基线；变化为右组相对左组。区间重采样单位是独立实验，单次实验的窗口不能替代独立重复。"),
        message("可比性：{0} · {1:g}% 置信区间", data["status"], confidence * 100),
        json.dumps({key: data.get(key) for key in ("condition_checks", "experiments", "quality", "limitations", "source_sha256", "allowed_resource_dimensions")},
                   ensure_ascii=False, indent=2),
    ))
    return ReportView(source, message("独立实验比较 · {0} · {1} 项", data["status"], len(rows)),
        ("说明", "资源/输入", "指标", "基线均值", "对比均值", "变化", "差值", "差值区间", "比值区间", "独立实验数"),
        tuple(rows), note)


def read_report(path: str | Path, *, cancelled: Callable[[], bool] = lambda: False) -> ReportView:
    """读取一次完整 JSON；失败/未知报告不能冒充成功结果。"""
    source = Path(path).expanduser().resolve()
    content = bytearray()
    with source.open("rb") as stream:
        while len(content) <= _MAX_REPORT_BYTES:
            if cancelled():
                raise CancelledError()
            remaining = _MAX_REPORT_BYTES + 1 - len(content)
            chunk = stream.read(min(_REPORT_READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            content.extend(chunk)
    if len(content) > _MAX_REPORT_BYTES:
        raise ValueError(message("报告超过 32 MiB，请先缩小报告范围"))
    if cancelled():
        raise CancelledError()
    try:
        data = json.loads(content)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(message("JSON 报告损坏或编码无效")) from exc
    coverage_v2 = (isinstance(data, dict) and data.get("schema_version") == 2
                   and data.get("scope") == "selected_sample_only; no_formal_measurement"
                   and not data.get("kind") and not data.get("resampling_unit"))
    if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
            or data["schema_version"] != 1 and not coverage_v2):
        raise ValueError(message("不支持的报告类型或版本；请选择 stats、兼容性或开销对照报告"))
    if data.get("resampling_unit") == "csv_request_window":
        return _windows(source, data)
    if data.get("kind") in {"monitor_overhead_diagnostic", "ui_overhead"}:
        return _comparisons(source, data)
    if data.get("kind") == "independent_experiment_comparison":
        return _independent_comparison(source, data)
    if data.get("scope") in {"selected_sample_only; no_formal_measurement", "recorded_results; no_reexecution"}:
        from acprof.analysis.compatibility import result_status
        rows = []
        for row in _objects(data.get("rows")):
            failure = row.get("failure") or {}
            rows.append(ReportRow((row["model_id"], result_status(row), failure.get("reason_code", "")),
                join_messages("\n", (render_quality_summary(row),
                    json.dumps({"failure": failure, "quality_checks": row.get("quality_checks", []),
                        "cleanup_status": row.get("cleanup_status"), "cleanup_errors": row.get("cleanup_errors", []),
                        "attempt_id": row.get("attempt_id"), "attempts": data.get("attempts", [])}, ensure_ascii=False, indent=2)))))
        note = message("读取已有结果，不重新执行模型。") if data["scope"] == "recorded_results; no_reexecution" else message("仅覆盖所选样本的独立验证，不代表正式采集完成。")
        return ReportView(source, message("兼容性报告"), ("模型", "状态", "reason_code"), tuple(rows), note)
    raise ValueError(message("不支持的报告类型或版本；请选择 stats、兼容性或开销对照报告"))
