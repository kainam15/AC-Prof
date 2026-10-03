"""Actionable lookup errors shared by host preparation, CLI and TUI."""
from __future__ import annotations

from acprof.messages import message

_MESSAGES = {
    "invalid_model_id": ("模型 ID 格式不正确：{0}", "请检查空格、字符和路径格式；支持 model 或 namespace/model。"),
    "repository_unavailable": ("未找到模型 {0}，或当前账号无权访问。", "请核对模型 ID；私有模型请检查 Hugging Face Token 和访问权限。"),
    "access_denied": ("模型 {0} 需要访问授权。", "请检查 Hugging Face Token，或在模型页面申请访问权限。"),
    "revision_not_found": ("模型 {0} 的 revision {1} 不存在。", "请检查 branch、tag 或 commit SHA，或清空 revision 后重试。"),
    "network_error": ("无法连接模型仓库，未能读取模型 {0}。", "请检查网络或代理后重试；当前无法确认模型是否存在。"),
    "hub_unavailable": ("模型仓库暂时无法响应，未能读取模型 {0}。", "服务可能限流或暂时不可用，请稍后重试。"),
    "offline": ("离线模式下无法读取模型 {0}。", "请检查本地缓存，或关闭 HF_HUB_OFFLINE 后重试。"),
    "task_unresolved": ("无法确定模型 {0} 的任务或接口。", "请补充模型声明，或明确指定任务和任务族。"),
    "lookup_failed": ("无法读取模型 {0} 的信息。", "请查看错误详情并检查模型配置。"),
}
_TERMINAL = frozenset({"invalid_model_id", "repository_unavailable", "access_denied", "revision_not_found"})


class ModelLookupError(ValueError):
    """Preserve the model identity, reason and diagnostics without exiting Python."""

    def __init__(self, model_id: str, reason_code: str, *, revision: str = "", detail: str = ""):
        self.model_id = model_id
        self.reason_code = reason_code if reason_code in _MESSAGES else "lookup_failed"
        self.revision = revision
        self.detail = detail
        summary, hint = _MESSAGES[self.reason_code]
        self.summary = message(summary, model_id, revision)
        self.hint = message(hint)
        super().__init__(f"{self.summary}\n{self.hint}\n{detail}".rstrip())

    @property
    def retryable(self) -> bool:
        return self.reason_code in {"network_error", "hub_unavailable"}

    @property
    def blocks_fallback(self) -> bool:
        return self.reason_code in _TERMINAL

    def to_dict(self) -> dict:
        return {"model_id": self.model_id, "reason_code": self.reason_code,
                "revision": self.revision, "detail": self.detail}


def _reason(exc: BaseException) -> str:
    import httpx
    from huggingface_hub.errors import (
        GatedRepoError,
        HFValidationError,
        OfflineModeIsEnabled,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    # GatedRepoError subclasses RepositoryNotFoundError. Status 401 alone cannot
    # distinguish a missing repository from a private, inaccessible repository.
    if isinstance(exc, HFValidationError):
        return "invalid_model_id"
    if isinstance(exc, GatedRepoError):
        return "access_denied"
    if isinstance(exc, RepositoryNotFoundError):
        return "repository_unavailable"
    if isinstance(exc, RevisionNotFoundError):
        return "revision_not_found"
    if isinstance(exc, OfflineModeIsEnabled):
        return "offline"
    if isinstance(exc, (httpx.TransportError, ConnectionError, TimeoutError)):
        return "network_error"
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403) or isinstance(exc, PermissionError):
        return "access_denied"
    if status == 429 or isinstance(status, int) and 500 <= status < 600:
        return "hub_unavailable"
    return "lookup_failed"


def model_lookup_error(model_id: str, exc: BaseException, *, revision: str | None = None,
                       diagnostics: list[str] | None = None) -> ModelLookupError:
    """Classify actual exception types and causes, never natural-language text."""
    if isinstance(exc, ModelLookupError):
        return exc
    reason, cause, seen = "lookup_failed", exc, set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        reason = _reason(cause)
        if reason != "lookup_failed":
            break
        cause = cause.__cause__
    detail = "\n".join(diagnostics) if diagnostics else f"{type(exc).__name__}: {exc}"
    return ModelLookupError(model_id, reason, revision=revision or "", detail=detail)
