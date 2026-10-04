"""Download and capacity presentation from unmodified byte-valued reports."""
from __future__ import annotations

import json
from collections.abc import Callable

from acprof.messages import message
from acprof.tui.presentation import UNKNOWN, format_byte_fields, format_bytes


def download_fields(report: dict) -> dict[str, str]:
    return {
        "预计下载": format_bytes(report.get("expected_download_bytes")),
        "可用空间": format_bytes(report.get("disk", {}).get("free_bytes")),
        "下载源": report.get("model", {}).get("endpoint") or UNKNOWN,
    }


def download_summary(report: dict, translate: Callable[[str], str]) -> str:
    model, disk = report.get("model", {}), report.get("disk", {})
    docker = report.get("docker_storage", {})
    budget = (UNKNOWN if "max_download_bytes" not in report else message("不限")
              if report["max_download_bytes"] is None else format_bytes(report["max_download_bytes"]))
    docker_details = {key: value for key, value in docker.items()
                      if key not in {"docker_storage_total_bytes", "docker_storage_available_bytes_at_start"}}
    lines = [
        message("预计下载：{0}", format_bytes(report.get("expected_download_bytes"))),
        message("有效下载预算：{0}", budget),
        message("Model Store 路径：{0}", report.get("model_store_path", UNKNOWN)),
        message("DIRECT: {0} | PROXY: {1}", format_bytes(report.get("direct_download_bytes")),
                format_bytes(report.get("proxy_download_bytes"))),
        message("模型总量：{0} | 已缓存：{1}", format_bytes(model.get("total_bytes")),
                format_bytes(model.get("cached_bytes"))),
        message("下载源：{0}", model.get("endpoint", UNKNOWN)),
        message("Runtime: {0}", json.dumps(format_byte_fields(report.get("runtime", {}), translate), ensure_ascii=False)),
        message("Model Store: {0} | 可用：{1}", format_bytes(disk.get("total_bytes")),
                format_bytes(disk.get("free_bytes"))),
        message("可回收：{0} | 下载后剩余：{1}", format_bytes(disk.get("reclaimable_bytes")),
                format_bytes(disk.get("remaining_bytes"))),
        message("Docker 占用：{0}", format_bytes(docker.get("docker_storage_total_bytes"))),
        message("Docker 可用空间：{0}", format_bytes(docker.get("docker_storage_available_bytes_at_start"))),
    ]
    if docker_details:
        lines.append(message("Docker 存储详情：{0}", json.dumps(format_byte_fields(docker_details, translate), ensure_ascii=False)))
    lines.append(message("DIRECT/PROXY 是来源策略预期，尚未验证实际网络路由。"))
    return "\n".join(translate(line) for line in lines)


def download_result_summary(report: dict) -> str:
    return message("{0}: 已校验新增：{1} | 缓存节省：{2} | 传输流量：{3}", report["category"],
                   format_bytes(report["verified_new_payload_bytes"]), format_bytes(report["cache_savings_bytes"]),
                   format_bytes(report.get("wire_bytes")))
