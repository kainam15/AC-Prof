"""Static artifact layout checks reuse the offline checkpoint and download planners."""
from __future__ import annotations

from typing import Any, NoReturn

from acprof.failures import Failure, RuntimeFailure
from acprof.models.diffusion_pipelines import diffusers_pipeline
from acprof.models.files import ModelFilesError, checkpoint_index, resolve_checkpoint, safe_path
from acprof.models.metadata_validation import (
    MetadataValidationError,
    validate_diffusers_declarations,
)


def _checkpoint_components(files: set[str], metadata: dict[str, Any]) -> list[tuple[str, bool]]:
    """Locate the declared native components without guessing from suffixes."""
    components: list[tuple[str, bool]] = []
    pipeline = metadata.get("model_index.json")
    modules = metadata.get("modules.json")
    if isinstance(pipeline, dict):
        validate_diffusers_declarations(pipeline)
        capability = diffusers_pipeline(pipeline.get("_class_name"))
        for component, declaration in pipeline.items():
            if (component.startswith("_") or not isinstance(declaration, list)
                    or len(declaration) != 2 or declaration[0] not in {"diffusers", "transformers", "stable_diffusion"}):
                continue
            prefix = safe_path(component) + "/"
            if not any(name.startswith(prefix) for name in files):
                if capability is not None and capability.root_layout and component == "unet":
                    prefix = ""
                else:
                    continue
            components.append((prefix, declaration[0] == "diffusers"))
    elif isinstance(modules, list):
        for module in modules:
            if isinstance(module, dict) and module.get("type") == "sentence_transformers.models.Transformer":
                path = module.get("path", "")
                components.append((safe_path(path) + "/" if path not in ("", ".") else "", False))
    else:
        components.append(("", False))
    return components


def checkpoint_metadata_indexes(files: set[str], metadata: dict[str, Any]) -> list[str]:
    """Select bounded index metadata for the declared loader's components."""
    components = []
    for manifest in ("model_index.json", "modules.json"):
        if manifest in metadata:
            components.extend(_checkpoint_components(files, {manifest: metadata[manifest]}))
    if "config.json" in files or not components:
        components.append(("", False))
    return sorted({index for prefix, diffusion in components
                   if (index := checkpoint_index(files, prefix, diffusion=diffusion)) is not None})


def _coreml_artifact(path: str) -> str | None:
    parts = path.split("/")
    for index, part in enumerate(parts):
        if part.lower().endswith((".mlpackage", ".mlmodelc")):
            return "/".join(parts[:index + 1])
    return path if parts[-1].lower().endswith(".mlmodel") else None


def has_coreml_artifacts(files: set[str]) -> bool:
    return any(_coreml_artifact(path) for path in files)


def coreml_only_failure(task_info: Any, files: set[str]) -> Failure | None:
    """Require positive artifact evidence; unfamiliar files may contain another loader's weights."""
    bundles: dict[str, str] = {}
    for path in sorted(files, key=lambda file: (file.count("/"), file)):
        bundle = _coreml_artifact(path)
        if bundle:
            bundles.setdefault(bundle, path)
            continue
        name = path.rsplit("/", 1)[-1].lower()
        if (name not in {".gitattributes", ".gitignore", "license", "config.json", "generation_config.json",
                         "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
                         "processor_config.json", "special_tokens_map.json", "added_tokens.json", "vocab.json"}
                and not name.startswith(("license-", "license."))
                and not name.endswith((".md", ".txt", ".mlcomputeplan.json"))):
            return None
    if not bundles:
        return None
    # The full repository listing can exceed the interactive preparation message limit.
    samples = sorted(bundles)[:3]
    evidence = {"artifact_format": "CoreML", "artifact_paths": samples,
                "files": sorted(files) if len(files) <= 3 else [bundles[bundle] for bundle in samples],
                "file_count": len(files),
                "backend": task_info.runtime_backend, "library_name": getattr(task_info, "library_name", "")}
    # A task inferred from an architecture or supplied by a user is not an author declaration.
    declared = (getattr(task_info, "hub_metadata", {}) or {}).get("pipeline_tag")
    if isinstance(declared, str) and declared and declared != "unknown":
        evidence["declared_task"] = declared
    return Failure("resolution", "artifact_layout_unsupported",
                   "Only Core ML model artifacts were found; the current configuration has no compatible loader.",
                   evidence=evidence)


def _foreign_format(files: set[str]) -> str | None:
    if any(name.lower().endswith(".gguf") for name in files):
        return "GGUF"
    if has_coreml_artifacts(files):
        return "CoreML"
    return None


def inspect_artifact_layout(task_info: Any, *, adapter: str = "family-default") -> dict[str, Any]:
    """Identify known loader layouts; an absent listing stays explicitly unknown."""
    files = set(getattr(task_info, "repository_files", ()) or ())
    backend = task_info.runtime_backend
    unknown = {
        "status": "unknown", "reason_code": "artifact_layout_unknown",
        "detail": "Repository file evidence is unavailable; the artifact layout has not been verified.",
        "weights": [],
    }
    if not files:
        return unknown
    metadata = getattr(task_info, "repository_metadata", {}) or {}

    def fail(code: str, detail: str) -> NoReturn:
        raise RuntimeFailure(Failure(
            "preflight", code, detail, runtime_profile=getattr(task_info, "runtime_profile_id", ""),
            evidence={"files": sorted(files), "backend": backend},
        ))

    def read_json(name: str) -> Any:
        if name not in metadata:
            raise LookupError(f"Checkpoint metadata was not read: {name}")
        return metadata[name]

    def reject_foreign_components(manifest: str) -> None:
        selected_metadata = {manifest: metadata[manifest]} if manifest in metadata else {}
        for prefix, diffusion in _checkpoint_components(files, selected_metadata):
            if resolve_checkpoint(files, read_json, prefix, diffusion=diffusion) is not None:
                continue
            foreign = _foreign_format({name for name in files if name.startswith(prefix)})
            if foreign:
                fail("artifact_layout_unsupported",
                     f"{backend} component {prefix or '.'} contains only {foreign} weights; a compatible registered artifact loader is required")

    def unrecognized(detail: str) -> dict[str, Any]:
        return {**unknown, "detail": detail}

    try:
        if backend == "kokoro":
            from acprof.models.kokoro import snapshot_contract, validate_snapshot
            validate_snapshot(task_info.model_id, task_info.model_revision,
                              read_json("config.json"), files)
            return {"status": "identified", "source": "registered_kokoro_snapshot", "weights": [
                {"component": ".", "format": "pth", "variant": None, "files": [snapshot_contract()["weights"]]}]}
        if backend in {"transformers_model", "transformers_pipeline", "sentence_transformers", "cross_encoder", "chronos"}:
            uses_modules = backend == "sentence_transformers" and "modules.json" in files
            config = metadata.get("config.json", getattr(task_info, "model_config", {})) or {}
            native_loader = adapter == "family-default" and not config.get("auto_map") and not config.get("custom_pipelines")
            if not uses_modules and "config.json" not in files:
                if native_loader and (failure := coreml_only_failure(task_info, files)):
                    raise RuntimeFailure(failure)
                fail("artifact_layout_unsupported", f"{backend} requires config.json at the selected loader root; this layout has no native checkpoint declaration")
            if "adapter_config.json" in files:
                base = (metadata.get("adapter_config.json") or {}).get("base_model_name_or_path")
                fail("artifact_incomplete", f"Adapter requires a separately declared offline base model ({base or 'unknown'})")
            if uses_modules:
                from acprof.models.download_plan import plan_download
                from acprof.models.support import transformers_support
                from acprof.runtime.profiles import (
                    locked_transformers_version,
                    select_runtime_profile,
                )

                reject_foreign_components("modules.json")
                profile = select_runtime_profile(task_info)
                version = locked_transformers_version(profile.environment)
                support = transformers_support(version) if version else {}
                native_types = {model_type for mapping in support.values() if isinstance(mapping, dict) for model_type in mapping}
                plan = plan_download(
                    model_id=task_info.model_id, revision=task_info.model_revision,
                    family=task_info.task_family, backend="sentence_transformers",
                    files={name: {} for name in files}, read_json=read_json,
                    native_model_types=native_types,
                )
                if plan["weights"]:
                    return {"status": "identified", "weights": plan["weights"], "source": "download_plan"}
                return unrecognized(f"Sentence-transformer layout requires additional verification: {plan['reason']}")
            checkpoint = resolve_checkpoint(files, read_json)
            if checkpoint:
                return {"status": "identified", "weights": [checkpoint], "source": "checkpoint_loader"}
            if not native_loader:
                return unrecognized("A custom loader declares its own artifact layout; native checkpoint support is unverified.")
            if failure := coreml_only_failure(task_info, files):
                raise RuntimeFailure(failure)
            foreign = _foreign_format(files) or "nonstandard or absent checkpoint weights"
            fail("artifact_layout_unsupported", f"{backend} cannot load this artifact layout ({foreign}); select a compatible artifact and registered loader")
        if backend == "diffusers":
            if "model_index.json" not in files:
                kind = "GGUF" if any(name.lower().endswith(".gguf") for name in files) else "single-file/components"
                fail("artifact_layout_unsupported", f"DiffusionPipeline requires model_index.json; {kind} needs a compatible artifact loader")
            from acprof.models.download_plan import plan_download

            reject_foreign_components("model_index.json")
            plan = plan_download(
                model_id=task_info.model_id, revision=task_info.model_revision,
                family=task_info.task_family, backend=backend,
                files={name: {} for name in files}, read_json=read_json,
            )
            if plan["weights"]:
                for checkpoint in plan["weights"]:
                    prefix = "" if checkpoint["component"] == "." else checkpoint["component"] + "/"
                    if prefix + "config.json" not in files:
                        fail("artifact_incomplete", f"Missing pipeline component configuration: {prefix}config.json")
                return {"status": "identified", "weights": plan["weights"], "source": "download_plan"}
            return unrecognized(f"Diffusers component layout requires additional verification: {plan['reason']}")
    except MetadataValidationError as exc:
        raise RuntimeFailure(Failure(
            "preflight", "metadata_invalid", str(exc),
            runtime_profile=getattr(task_info, "runtime_profile_id", ""),
            evidence={"diagnostics": exc.diagnostics, "files": sorted(files), "backend": backend},
        )) from exc
    except LookupError as exc:
        return unrecognized(str(exc))
    except ModelFilesError as exc:
        fail("artifact_incomplete", str(exc))
    return unrecognized(f"The {backend} loader validates its artifact through its own model contract.")
