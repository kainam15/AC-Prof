"""Actionable lookup errors shared by host preparation, CLI and TUI."""
from __future__ import annotations

from acprof.failures import Failure
from acprof.host.preparation.preparation_log import redact_text
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

    failure: Failure | None = None
    rate_limited = False

    def __init__(self, model_id: str, reason_code: str, *, revision: str = "", detail: str = "",
                 download_error: dict | None = None):
        self.model_id = model_id
        self.reason_code = reason_code if reason_code in _MESSAGES else "lookup_failed"
        self.revision = revision
        self.detail = redact_text(detail, limit=None)
        self.download_error = download_error
        summary, hint = _MESSAGES[self.reason_code]
        self.summary = message(summary, model_id, revision)
        self.hint = message(hint)
        from acprof.hub.repository import model_source
        if model_source() == "modelscope":
            if self.reason_code in {"repository_unavailable", "access_denied"}:
                self.hint = message("请核对 ModelScope 模型 ID、访问权限和 MODELSCOPE_API_TOKEN。")
            elif self.reason_code == "offline":
                self.hint = message("请检查该 revision 的本地缓存，或清除 ModelScope / Hugging Face 离线设置后重试。")
        super().__init__(f"{self.summary}\n{self.hint}\n{self.detail}".rstrip())

    @property
    def retryable(self) -> bool:
        return self.reason_code in {"network_error", "hub_unavailable"}

    @property
    def blocks_fallback(self) -> bool:
        return self.reason_code in _TERMINAL

    def to_dict(self) -> dict:
        return {"model_id": self.model_id, "reason_code": self.reason_code,
                "revision": self.revision, "detail": self.detail,
                **({"download_error": self.download_error} if self.download_error else {})}


def _reason(exc: BaseException) -> str:
    import httpx
    from huggingface_hub.errors import (
        GatedRepoError,
        HFValidationError,
        OfflineModeIsEnabled,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )
    from modelscope_hub.errors import (
        AuthenticationError,
        NotExistError,
        PermissionDeniedError,
        RateLimitError,
        ServerError,
    )

    from acprof.hub.repository import ModelScopeRevisionError

    # GatedRepoError subclasses RepositoryNotFoundError. Status 401 alone cannot
    # distinguish a missing repository from a private, inaccessible repository.
    if isinstance(exc, HFValidationError):
        return "invalid_model_id"
    if isinstance(exc, GatedRepoError):
        return "access_denied"
    if isinstance(exc, RepositoryNotFoundError):
        return "repository_unavailable"
    if isinstance(exc, (RevisionNotFoundError, ModelScopeRevisionError)):
        return "revision_not_found"
    if isinstance(exc, OfflineModeIsEnabled):
        return "offline"
    if isinstance(exc, NotExistError):
        return "repository_unavailable"
    # ModelScope also reports throttling as a permission error with a service-specific code.
    if isinstance(exc, RateLimitError) or (
        isinstance(exc, PermissionDeniedError) and exc.status_code == 403
        and isinstance(exc.response_body, dict) and str(exc.response_body.get("Code")) == "10013201001"
    ):
        return "modelscope_rate_limited"
    if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
        return "access_denied"
    if isinstance(exc, ServerError):
        return "hub_unavailable"
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
    from modelscope_hub.errors import APIError

    from acprof.hub.huggingface.download import HfDownloadError, exception_chain

    reason = "lookup_failed"
    for cause in exception_chain(exc):
        reason = _reason(cause)
        if reason != "lookup_failed":
            break
    detail = "\n".join(diagnostics) if diagnostics else f"{type(exc).__name__}: {exc}"
    download_error = next((cause.to_dict() for cause in exception_chain(exc) if isinstance(cause, HfDownloadError)), None)
    limited = reason == "modelscope_rate_limited"
    error = ModelLookupError(model_id, "hub_unavailable" if limited else reason, revision=revision or "",
                             detail=detail, download_error=download_error)
    if limited or (reason == "access_denied" and isinstance(cause, APIError)):
        error.rate_limited = limited
        error.failure = Failure("resolution", "metadata_unavailable" if limited else "access_denied", str(error),
                                retryability="transient" if limited else "after_configuration",
                                evidence={"lookup_reason": error.reason_code, "next_action": error.hint},
                                exception_type=type(cause).__name__)
    return error


def metadata_read_diagnostic(model_id: str, revision: str, filename: str, exc: Exception) -> dict:
    """Classify a listed file read at the provider boundary, before any JSON is parsed."""
    from huggingface_hub.errors import LocalEntryNotFoundError, RemoteEntryNotFoundError

    from acprof.hub.repository import MetadataSizeError

    if isinstance(exc, MetadataSizeError):
        diagnostic = {
            "reason_code": "metadata_unavailable", "file": filename, "revision": revision,
            "limit_bytes": exc.limit, "detail": redact_text(str(exc)), "exception_type": type(exc).__name__,
        }
        if exc.size is None:
            diagnostic.update(lookup_reason="metadata_size_unknown", retryability="after_configuration",
                              next_action=f"Retry with a repository endpoint or verified cache that reports file size (limit {exc.limit} bytes).")
        else:
            diagnostic["minimum_size_bytes" if exc.minimum else "size_bytes"] = exc.size
            diagnostic.update(lookup_reason="metadata_size_limit", retryability="not_retryable",
                              next_action=f"Select a revision whose {filename} fits the {exc.limit}-byte inspection limit.")
        return diagnostic

    lookup = model_lookup_error(model_id, exc, revision=revision)
    reason = lookup.reason_code
    code = "metadata_unavailable"
    if isinstance(exc, RemoteEntryNotFoundError):
        reason, code = "entry_not_found", "artifact_incomplete"
    elif reason == "access_denied":
        code = "access_denied"
    elif reason == "lookup_failed" and isinstance(exc, LocalEntryNotFoundError):
        # This SDK error can also wrap a transport failure, which takes precedence above.
        reason = "local_cache_miss"
    return {
        "reason_code": code, "lookup_reason": reason, "file": filename, "revision": revision,
        "detail": lookup.detail, "exception_type": type(exc).__name__,
        "retryability": "transient" if lookup.retryable else "after_configuration",
        **({"next_action": lookup.hint} if lookup.rate_limited else {}),
    }
