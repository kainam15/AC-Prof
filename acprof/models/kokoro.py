"""Reviewed Kokoro snapshot contract shared by planning and offline loading."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from acprof.models.files import ModelFilesError


def snapshot_contract() -> dict[str, Any]:
    from acprof.extensions import CATALOG
    return CATALOG.get_extension("audio", "kokoro", "kokoro", "text-to-speech").handler_options["snapshot"]


def validate_snapshot(model_id: str, revision: str, config: dict, files: Iterable[str]) -> list[str]:
    contract = snapshot_contract()
    if model_id.lower() != contract["model_id"].lower() or revision != contract["revision"]:
        raise ModelFilesError(f"Kokoro requires {contract['model_id']} revision {contract['revision']}")
    if (not isinstance(config, dict) or config.get("model_type") not in (None, "")
            or config.get("architectures") or config.get("auto_map")
            or any(config.get(key) != value for key, value in contract["config_values"].items())
            or any(not isinstance(config.get(key), dict) or not config[key] for key in ("vocab", "plbert", "istftnet"))
            or config["plbert"].get("max_position_embeddings") != 512):
        raise ModelFilesError("Kokoro config.json does not match the registered KModel structure")
    names = set(files)
    required = {"config.json", contract["weights"], f"voices/{contract['default_voice']}.pt"}
    if missing := required - names:
        raise ModelFilesError(f"Kokoro snapshot is incomplete: missing {', '.join(sorted(missing))}")
    return sorted(name for name in names if name.startswith("voices/") and name.count("/") == 1
                  and name.endswith(".pt") and name[7:8] in contract["languages"])
