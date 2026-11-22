"""Static evidence for the native Transformers text-generation chat protocol."""
from __future__ import annotations

import re
from typing import Any

from acprof.failures import Failure, RuntimeFailure
from acprof.models.evidence import pinned_revision
from acprof.models.support import transformers_support

CHAT_INPUT_PROTOCOL = "single_user_chat_v1"


def external_default_template(files: set[str] | tuple[str, ...]) -> str | None:
    """A named external template set replaces embedded templates, even without a default."""
    for name in ("chat_templates/default.jinja", "chat_template.jinja"):
        if name in files:
            return name
    if any(name.startswith("chat_templates/") and name.endswith(".jinja") and name.count("/") == 1 for name in files):
        return ""  # The external set has no default for a tool-free request.
    return None


def _embedded_default(config: dict) -> object:
    template = config.get("chat_template")
    if isinstance(template, list):
        if not all(isinstance(entry, dict) and isinstance(entry.get("name"), str)
                   and isinstance(entry.get("template"), str) for entry in template):
            return None
        template = {entry["name"]: entry["template"] for entry in template}
    return template.get("default") if isinstance(template, dict) else template


def native_chat_template(task: Any, version: str) -> str:
    """Require a declared native causal head and a default for one tool-free user turn."""
    config = task.model_config
    model_type = config.get("model_type")
    mappings = transformers_support(version)["MODEL_FOR_CAUSAL_LM_MAPPING_NAMES"]
    classes = mappings.get(model_type) if isinstance(model_type, str) else None
    declared = config.get("architectures")
    supported = [classes] if isinstance(classes, str) else classes or []
    if (not isinstance(declared, list) or not declared or not all(name in supported for name in declared)
            or config.get("is_encoder_decoder") or config.get("auto_map") or config.get("custom_pipelines")):
        raise RuntimeFailure(Failure("preflight", "model_contract_required",
            "Native chat requires a declared causal LM architecture supported by the pinned Transformers loader.",
            retryability="after_configuration", evidence={"model_type": config.get("model_type"), "architectures": declared}))
    if not task.repository_files or not pinned_revision(task.model_revision):
        raise RuntimeFailure(Failure("preflight", "metadata_unavailable",
            "Native chat needs the pinned file inventory to determine which tokenizer chat template takes precedence.",
            retryability="after_configuration"))
    filename = external_default_template(task.repository_files)
    if filename:
        source = task.metadata_sources.get(filename, {})
        if source.get("chat_template_error"):
            diagnostic = source["chat_template_error"]
            raise RuntimeFailure(Failure("preflight", diagnostic["reason_code"], diagnostic["detail"],
                retryability=diagnostic.get("retryability", "after_configuration"), evidence=diagnostic))
        if (type(source.get("nonempty_text")) is not bool or source.get("revision") != task.model_revision
                or source.get("encoding") != "raw" or not re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256")))):
            raise RuntimeFailure(Failure("preflight", "metadata_unavailable",
                f"Cannot verify the pinned default chat template at {filename}.", retryability="after_configuration"))
        if source["nonempty_text"] is True:
            return filename
    elif filename is None:
        config = task.repository_metadata.get("tokenizer_config.json") or {}
        template = _embedded_default(config) if isinstance(config, dict) else None
        if isinstance(template, str) and template.strip():
            return "tokenizer_config.json"
    source_name = "tokenizer_config.json" if filename is None else filename or "chat_templates/"
    raise RuntimeFailure(Failure("preflight", "model_contract_required",
        "Native chat needs a nonempty default tokenizer chat template for a single user message.",
        retryability="after_configuration", evidence={"source": source_name,
            "next_action": "Select a pinned model with a default chat template, or choose text-generation for plain completion."}))
