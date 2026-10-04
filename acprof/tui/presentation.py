"""界面数值与空值显示；不改变配置、进度或结果协议。"""

from collections.abc import Callable

from acprof.messages import message

NOT_APPLICABLE = "—"
CALCULATING = "…"
UNKNOWN = message("未知")
STATUS_LEGEND = message("— 不适用 · … 正在计算 · 未知 无法确定")


def format_bytes(value: int | None) -> str:
    """Use decimal units selected from raw bytes, with about three significant digits."""
    if value is None:
        return UNKNOWN
    for unit, scale in (("GB", 10**9), ("MB", 10**6), ("KB", 1000)):
        if value >= scale:
            amount = float(f"{value / scale:.3g}")
            precision = 2 if amount < 10 else 1 if amount < 100 else 0
            return f"{amount:.{precision}f} {unit}"
    return f"{value} B"


def format_byte_fields(value: object, translate: Callable[[str], str]) -> object:
    """Copy diagnostic data for display, leaving all source byte counts intact."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if item is None and key in {"capacity_bytes", "max_download_bytes"}:
                result[key] = translate(message("不限"))
            elif (key.endswith("_bytes") or key.endswith("_bytes_at_start")) and (item is None or type(item) is int):
                result[key] = translate(format_bytes(item))
            else:
                result[key] = format_byte_fields(item, translate)
        return result
    if isinstance(value, list):
        return [format_byte_fields(item, translate) for item in value]
    return value


def format_input_number(value: int | float | str) -> str:
    """去掉整数的 .0，保留非整数的完整精度与尚未校验的输入。"""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
