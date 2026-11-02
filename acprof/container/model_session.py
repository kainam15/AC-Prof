"""Shared configuration and staged assembly for isolated inference entrypoints."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from acprof.container.handlers import BaseHandler


@dataclass(frozen=True)
class ModelLoadConfig:
    model_id: str
    task_family: str
    task_type: str
    backend: str
    model_revision: str = "main"
    model_path: str = ""
    adapter: str = "family-default"
    use_gpu: bool = False
    threads: int = 0

    @classmethod
    def from_environment(
        cls, *, default_threads: int = 0,
        task_type_default: str = "text-generation", use_gpu: bool | None = None,
    ) -> ModelLoadConfig:
        from acprof.runtime.settings import runtime_threads

        return cls(
            model_id=os.getenv("MODEL_ID", ""), model_revision=os.getenv("MODEL_REVISION", "main"),
            task_family=os.getenv("TASK_FAMILY", "nlp"), task_type=os.getenv("TASK_TYPE", task_type_default),
            backend=os.getenv("RUNTIME_BACKEND", "transformers_pipeline"),
            model_path=os.getenv("MODEL_LOCAL_PATH", ""),
            adapter=os.getenv("ACPROF_MODEL_ADAPTER", "family-default"),
            use_gpu=bool(int(os.getenv("USE_GPU", "0"))) if use_gpu is None else use_gpu,
            threads=runtime_threads(default=default_threads),
        )

    def resolve_source(self) -> str:
        from acprof.container.handlers import resolve_model_source

        return resolve_model_source(self.model_id, self.model_path)


@dataclass(frozen=True)
class ModelSession:
    """Loaded dependencies only; each caller owns timing and request execution."""

    config: ModelLoadConfig
    handler: BaseHandler
    context: dict[str, Any]
    execution: Any
    device: str


def _require_adapter(config: ModelLoadConfig) -> None:
    # Handlers still read their adapter-specific declarations from the isolated
    # process environment. Reject a stale selection instead of mixing identities.
    if config.adapter != os.getenv("ACPROF_MODEL_ADAPTER", "family-default"):
        raise ValueError("model adapter changed after configuration; reload the model configuration")


def select_handler(config: ModelLoadConfig) -> BaseHandler:
    from acprof.container.handlers import HandlerRegistry

    _require_adapter(config)
    return HandlerRegistry.get(config.task_family, config.backend)


def prepare_execution(config: ModelLoadConfig):
    from acprof.container.execution import configured_execution

    _require_adapter(config)
    return configured_execution(
        config.task_family, config.backend, use_gpu=config.use_gpu,
        threads=config.threads, adapter=config.adapter,
    )


def load_model_session(
    config: ModelLoadConfig, execution, device: str, *,
    handler: BaseHandler | None = None, model_source: str | None = None,
    load_options: dict[str, Any] | None = None,
) -> ModelSession:
    """Load once without initializing devices, inferring, validating or profiling."""
    from acprof.container.handlers import load_handler

    _require_adapter(config)
    selected = handler if handler is not None else select_handler(config)
    source = config.resolve_source() if model_source is None else model_source
    options = {"load_options": load_options} if load_options is not None else {}
    context = load_handler(
        selected, source, config.task_type, config.backend, device,
        config.model_revision, **options,
    )
    return ModelSession(config, selected, context, execution, device)
