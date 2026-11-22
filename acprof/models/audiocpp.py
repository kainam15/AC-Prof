"""Pinned audio.cpp package declarations projected from the upstream model_specs catalog."""
from __future__ import annotations

import json
from collections.abc import Iterable
from functools import cache
from typing import Any

from acprof.failures import Failure
from acprof.installation import resource_root
from acprof.models.files import Checkpoint

REPOSITORY = "audio-cpp/audio.cpp-gguf"
SPEC_REVISION = "04a37f2e80ff2dbcdd84cc67848a970bd02a73e0"
DEFAULT_ASR = "Citrinet-ASR-GGUF/citrinet-asr-q8_0.gguf"
ASR = "automatic-speech-recognition"
_TASKS = {"asr": ASR, "tts": "text-to-speech", "classify": "audio-classification",
          "music": "text-to-audio", "sfx": "text-to-audio", "vc": "voice-conversion",
          "sep": "source-separation", "diar": "speaker-diarization"}


@cache
def _packages() -> dict:
    # path -> [native family, primary native task, package precision, standalone offline ASR]
    # Multi-file package precision describes the package, not its individual component.
    path = resource_root() / "acprof/models/audiocpp_packages.json"
    return json.loads(path.read_text(encoding="utf-8"))


def gguf_candidate(model_id: str, path: str) -> dict[str, Any]:
    record = _packages().get(path) if model_id == REPOSITORY else None
    if record:
        family, native_task, precision, selectable = record
        return {"path": path, "family": family, "task": _TASKS.get(native_task, native_task),
                "quantization": precision.upper() if precision else None, "selectable": selectable,
                "reason": None if selectable else "No standalone offline ASR interface for this audio.cpp package."}
    return {"path": path, "family": None, "task": None, "quantization": None, "selectable": False,
            "reason": "No matching package declaration in the registered audio.cpp catalog."}


def gguf_candidates(model_id: str, names: Iterable[str]) -> list[dict[str, Any]]:
    return [gguf_candidate(model_id, name) for name in sorted(names) if name.lower().endswith(".gguf")]


def select_package(task: Any, requested: str, siblings: Iterable[Any]) -> None:
    """Scope metadata to a selected file while preserving the raw Hub declarations."""
    if task.model_id != REPOSITORY or task.library_name != "audio.cpp":
        return
    files = set(task.repository_files)
    if not requested and files.intersection({"config.json", "model_index.json", "acprof_model.json", "modules.json"}):
        return
    if requested and not requested.lower().endswith(".gguf"):
        return
    choices = [item["path"] for item in gguf_candidates(task.model_id, files) if item["selectable"]]
    path = requested or (DEFAULT_ASR if DEFAULT_ASR in choices else next(iter(choices), ""))
    if not path:
        if any(name.lower().endswith(".gguf") for name in files):
            task.hub_metadata["task_hint_source"] = "audiocpp_inventory"
        return
    task.model_variant = path
    package = gguf_candidate(task.model_id, path)
    sibling = next((item for item in siblings if item.rfilename == path), None)
    lfs = getattr(sibling, "lfs", None)
    sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    # The repository aggregates unrelated models. Its root quantization tags, parameter counts
    # and config cannot establish the tensor characteristics of this particular GGUF.
    task.parameter_count = task.parameter_bytes = None
    task.parameter_bytes_reason = "The selected GGUF has no per-tensor inventory in the package declaration."
    task.parameter_dtype_counts = {}
    task.quantization_config = {}
    precision = package["quantization"]
    task.precision_dtype = precision
    task.quantized = True if precision and precision.startswith(("Q", "IQ", "TQ")) else (
        False if precision in {"F32", "F16", "BF16"} else None)
    task.quantization_method = precision if task.quantized else None
    task.model_config = {"audiocpp_family": package["family"], "audiocpp_artifact": {
        "path": path, "package_revision": SPEC_REVISION, "precision": precision,
        "sha256": sha, "size": getattr(sibling, "size", None)}}
    task.hub_metadata.update(task_hint=package["task"], task_hint_source="audiocpp_package",
                             gguf_selection="explicit" if requested else "automatic")


def artifact_failure(task: Any, *, interface: str | None, backend: str | None) -> Failure | None:
    path = getattr(task, "model_variant", "")
    empty_inventory = (getattr(task, "hub_metadata", {}) or {}).get("task_hint_source") == "audiocpp_inventory"
    if not path.lower().endswith(".gguf") and backend != "audio_cpp" and not empty_inventory:
        return None
    package = gguf_candidate(task.model_id, path)
    frozen = (getattr(task, "model_config", {}) or {}).get("audiocpp_artifact")
    if empty_inventory:
        why = "No GGUF in this revision matches a registered standalone offline ASR package."
    elif not path:
        why = "The audio_cpp backend requires a selected GGUF file from its registered repository."
    elif path not in (task.repository_files or ()):
        why = f"The selected GGUF {path!r} is absent from the pinned repository inventory."
    elif not package["selectable"] or getattr(task, "model_source", "huggingface") != "huggingface":
        why = package["reason"] or "The audio.cpp package declaration requires the original Hugging Face repository."
    elif frozen and (not isinstance(frozen, dict) or frozen.get("path") != path):
        why = "The selected GGUF differs from the frozen file metadata. Resolve the model again."
    elif backend != "audio_cpp":
        why = f"{backend or 'unknown'} cannot load this audio.cpp GGUF. Select the audio_cpp backend."
    elif interface != package["task"]:
        why = f"The selected {package['family']} package declares {package['task']}, incompatible with {interface}."
    else:
        return None
    return Failure("resolution", "artifact_layout_unsupported", why, retryability="after_configuration",
                   evidence={"artifact_format": "GGUF", "artifact_paths": [path] if path else [],
                             "backend": backend, "family": package["family"], "task": package["task"],
                             "next_action": "Select a registered standalone ASR GGUF file and the audio_cpp backend."})


def gguf_checkpoint(path: str) -> Checkpoint:
    return {"component": ".", "format": "gguf", "variant": path, "files": [path]}
