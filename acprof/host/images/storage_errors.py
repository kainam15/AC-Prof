"""Actionable storage diagnostics without changing image deletion errors."""
from __future__ import annotations

import subprocess

from acprof.host.images.inventory import ImageManagementError


def storage_query_error(error: BaseException, *, filesystem: bool = False) -> ImageManagementError:
    """Keep external detail verbatim; translate only the stable reason at the UI."""
    if isinstance(error, ImageManagementError) and error.code:
        return error
    cause = error.__cause__ or error
    detail = error.detail if isinstance(error, ImageManagementError) else str(error)
    diagnostic = f"{error} {cause}".lower()
    if filesystem:
        if isinstance(cause, PermissionError):
            code, reason = "disk_permission", "磁盘查询权限不足；请检查 Docker 数据目录的访问权限。"
        else:
            code, reason = "disk", "磁盘查询失败；请检查 Docker 数据目录及挂载状态。"
    elif isinstance(cause, PermissionError) or "permission denied" in diagnostic or "access is denied" in diagnostic:
        code, reason = "permission", "Docker Socket 权限不足；请检查 Socket 权限和当前用户的 Docker 访问权限。"
    elif isinstance(cause, (subprocess.TimeoutExpired, TimeoutError)) or any(
        value in diagnostic for value in ("timed out", "timeout", "deadline exceeded", "超时")
    ):
        code, reason = "timeout", "存储查询超时；请检查 Docker 响应，稍后手动刷新。"
    elif isinstance(cause, FileNotFoundError):
        code, reason = "executable", "无法执行 Docker CLI；请检查 Docker 安装及 PATH。"
    elif "context" in diagnostic or any(value in diagnostic for value in ("ssh:", "no such host", "certificate")):
        code, reason = "context", "Docker Context 不可访问；请检查 docker context ls 和目标端点。"
    elif any(value in diagnostic for value in ("cannot connect", "connection refused", "daemon running")):
        code, reason = "daemon", "无法连接 Docker daemon；请检查 Docker 服务是否已启动。"
    elif isinstance(cause, (ValueError, TypeError)) or "无效" in diagnostic:
        code, reason = "invalid", "返回数据无效；请检查 docker info 和 docker system df 的输出。"
    elif "Docker 环境已改变" in str(error):
        code, reason = "identity", "Docker 环境已改变，请刷新后重新选择"
    else:
        code, reason = "query", "存储查询失败；请检查错误详情后手动刷新。"
    return ImageManagementError(reason, detail, code=code)
