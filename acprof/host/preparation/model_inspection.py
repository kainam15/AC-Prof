"""Preparation-only inspection and isolated probes shared by CLI and TUI."""
from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from acprof.artifacts import atomic_write_json
from acprof.models.contract import write_model_resolution

if TYPE_CHECKING:
    from acprof.host.run_state import MeasurementLock
    from acprof.host.runtime_validation import RuntimeValidationReport


class ProbePreparationError(RuntimeError):
    """Expose an environment failure separately from model execution failures."""
    stage = "environment"


def _configuration_lines(options: dict, *, explain: bool) -> list[str]:
    lines = ["Configuration options: static discovery; not frozen; runtime validation not_run"]
    for candidate in options.get("backend_candidates", []):
        compatibility = candidate.get("compatibility", {})
        lines.append(
            f"  Backend candidate {candidate.get('task', 'unknown')}/{candidate['backend']}: "
            f"{compatibility.get('status', 'unknown')}; selectable={candidate.get('selectable', False)}; "
            f"runtime={candidate.get('runtime_profile') or 'not_selected'}"
        )
        if explain and compatibility.get("detail"):
            lines.append("    " + compatibility["detail"])
    if not explain:
        return lines
    for item in options.get("task_options", []):
        lines.append(f"  Parameters for {item['task_id'] or item['task']}/{item['backend']}:")
        if not item["parameters"]:
            lines.append("    No editable parameter declarations for this route.")
        discovery = item["source_discovery"]
        for spec in item["parameters"]:
            consumers = ", ".join(f"{binding['stage']}:{binding['argument']}"
                                  for binding in spec["consumers"])
            lines.append(f"    {spec['name']}: {spec['kind']}; scope={spec['scope']}; consumer={consumers}")
            for default in discovery.get("parameters", {}).get(spec["name"], {}).get("default_candidates", []):
                source = default["source"]
                origin = source.get("file") or source["source_id"]
                value = json.dumps(default["value"], ensure_ascii=False, allow_nan=False)
                lines.append(f"      Default candidate: {value} <- {origin}:{source['json_pointer']}; "
                             f"value validation={default['validation']} (static)")
        for diagnostic in discovery.get("diagnostics", []):
            lines.append(f"    Source diagnostic: {diagnostic.get('file') or 'metadata'}:"
                         f"{diagnostic.get('json_pointer', '')}: {diagnostic['detail']}")
        lines.append(f"    Read-only model facts: {len(discovery.get('attributes', []))}; "
                     f"source diagnostics: {len(discovery.get('diagnostics', []))}")
    for diagnostic in options.get("diagnostics", []):
        lines.append(f"  Configuration diagnostic: {diagnostic['detail']}")
    return lines


def explain_resolution(task_info, *, explain: bool = False) -> str:
    resolution = task_info.model_resolution
    contract = resolution.get("contract", {})
    interface_status = contract.get("status", resolution.get("status", "unknown"))
    validation = resolution.get("runtime_validation", contract.get("runtime_validation"))
    runtime_status = validation.get("status", "not_run") if isinstance(validation, dict) else "not_run"
    runtime_status = {"error": "failed", "resource_limit": "failed", "not_run": "not_started"}.get(runtime_status, runtime_status)
    scope = "Representative " if len(resolution.get("task_resolutions", {})) > 1 else ""
    lines = [f"Model: {task_info.model_id}", f"Revision: {task_info.model_revision}",
             f"{scope}Task: {task_info.pipeline_tag}", f"{scope}Backend: {task_info.runtime_backend}",
             f"{scope}Adapter: {task_info.model_adapter}", f"{scope}Runtime profile: {task_info.runtime_profile_id or 'not_selected'}",
             f"Static Resolution: {resolution.get('status', 'unknown')}",
             f"Interface Resolution: {interface_status}", f"Runtime Validation: {runtime_status}",
             "Measurement: not_started (inspect/probe does not collect measurements)"]
    compatibility = resolution.get("compatibility")
    if compatibility:
        lines.append(f"Static compatibility: {compatibility['status']} (static evidence only)")
        if compatibility.get("detail"):
            lines.append(compatibility["detail"])
        if compatibility.get("next_action"):
            lines.append(f"Next action: {compatibility['next_action']}")
    variants = resolution.get("checkpoint_variants", [])
    if variants:
        lines.append("Checkpoint variants (--model-variant): " + ", ".join(variants))
    if variants or getattr(task_info, "model_variant", ""):
        lines.append("Selected checkpoint: " + (getattr(task_info, "model_variant", "") or "(default)"))
    layout = resolution.get("artifact_layout")
    if layout:
        lines.append(f"Artifact layout: {layout.get('status', 'unknown')}")
    failure = resolution.get("failure")
    if failure:
        lines.extend((f"Reason: {failure['reason_code']}", failure["detail"]))
        action = failure.get("evidence", {}).get("next_action")
        if action:
            lines.append(f"Next action: {action}")
    elif runtime_status == "not_started":
        lines.append("Evidence: static only; checkpoint loading and inference still require runtime validation.")
    for profile_id, item in resolution.get("task_resolutions", {}).items():
        state = item.get("compatibility", {}).get("status", "unknown")
        lines.append(f"Task profile {profile_id}: {item.get('status', 'unknown')}; "
                     f"static compatibility={state}; runtime={item.get('runtime_profile', 'not_selected')}")
        if item.get("failure"):
            lines.append(f"  Reason: {item['failure']['reason_code']}; {item['failure']['detail']}")
            action = item["failure"].get("evidence", {}).get("next_action")
            if action:
                lines.append(f"  Next action: {action}")
    if resolution.get("semantics"):
        lines.append(f"Semantics: {resolution['semantics']['status']} ({resolution.get('benchmark_kind', 'unknown')})")
    if task_info.task_discovery:
        from acprof.tasks.workloads import task_label
        state = task_info.task_discovery["status"]
        states = {"ready": "已确定一个推理路径", "selection_required": "存在多个独立推理路径，请选择采集范围",
                  "needs_configuration": "推理接口信息不足或相互矛盾，需要补充配置"}
        if task_info.selected_tasks:
            lines.append(f"推理路径：已选择 {len(task_info.selected_tasks)} 个推理路径")
        elif resolution.get("status") != "blocked":
            lines.append(f"推理路径：{states.get(state, state)}")
        lines.extend(f"  {item['task_id']}: {task_label(item)} ({item['interface']})"
                     for item in task_info.task_discovery['tasks'])
    if resolution.get("configuration_options"):
        lines.extend(_configuration_lines(resolution["configuration_options"], explain=explain))
    if explain and resolution.get("provenance"):
        provenance = resolution["provenance"]
        lines.append(f"Decision: {provenance['decision']} / {provenance['identity_sha256']}")
        for item in provenance["observations"]:
            kind = f" [{item['kind']}]" if item.get("kind") else ""
            lines.append(f"  <- {item['source_id']}:{item['field']} = {item.get('value', item.get('task'))}{kind}")
            if item.get("reason"):
                lines.append("     " + item["reason"])
    names = contract.get("fields", {}) if explain else contract.get("unresolved_fields", [])
    for name in names:
        field = contract["fields"][name]
        lines.append(f"{name} [{field['state']}]: {json.dumps(field['value'], ensure_ascii=False)}")
        if field.get("reason"):
            lines.append("  " + field["reason"])
        if explain:
            lines.extend("  <- " + source for source in field.get("sources", []))
    if not contract:
        lines.extend(resolution.get("conflicts", []) + resolution.get("missing", []))
    validation = resolution.get("runtime_validation", contract.get("runtime_validation"))
    if isinstance(validation, dict):
        for device, item in validation.get("devices", {}).items():
            if item.get("error"):
                lines.append(f"{device}: {item['error']}")
    return "\n".join(lines)


def _selected_nlp_generator(task_info, workload_spec_path: str | Path | None):
    from acprof.artifacts import read_json_object
    from acprof.host.configuration_transport import bound_inference_configuration
    from acprof.host.input_plan.materialize import get_task_generator
    from acprof.tasks.workloads import workload_manifest

    frozen = bound_inference_configuration(task_info)
    if frozen.get("workload", {}).get("specification_sha256") and workload_spec_path is None:
        raise ValueError("the frozen profile requires its original workload specification")
    supplied = (read_json_object(Path(workload_spec_path).expanduser(), label="workload specification")
                if workload_spec_path is not None else None)
    manifest = workload_manifest(task_info.selected_tasks[0], supplied=supplied, frozen=frozen)
    # The generator consumes the effective manifest synchronously, exactly as in a collection child.
    with TemporaryDirectory(prefix="acprof-validation-workload-") as directory:
        path = Path(directory) / "task_workload.json"
        atomic_write_json(path, manifest)
        return get_task_generator(task_info, 1, workload_spec_path=str(path))


def validate_model_runtime(task_info, output_dir: str | Path, *, cpus: int = 2,
                           memory_gb: int = 4, gpu: bool = False, timeout_seconds: float = 300,
                           reuse_existing: bool = True, workload_spec_path: str | Path | None = None,
                           measurement_lock: MeasurementLock | None = None) -> RuntimeValidationReport:
    from acprof.host.images.runtime import prepare_image
    from acprof.host.input_plan import PlannedInputScales, resolve_input_scales
    from acprof.host.input_plan.materialize import get_task_generator, write_scale_plan_file
    from acprof.host.preflight import require_collection_host, require_native_docker
    from acprof.host.preparation.task_selection import TaskSelectionRequired
    from acprof.host.preparation.task_support import require_task_support
    from acprof.host.run_state import ResultDirectoryLock, measurement_lock_scope
    from acprof.host.runtime_validation import validate_runtime
    from acprof.installation import resource_root

    if (type(cpus) is not int or cpus <= 0
            or type(memory_gb) is not int or memory_gb <= 0 or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("invalid runtime validation resources/timeout")
    if len(task_info.selected_tasks) > 1:
        raise TaskSelectionRequired([item["task_id"] for item in task_info.selected_tasks], single_profile=True)
    selected_nlp = bool(task_info.selected_tasks) and task_info.task_family == "nlp"
    if workload_spec_path is not None and not selected_nlp:
        raise ValueError("workload_spec_path requires a selected NLP profile")
    require_task_support(task_info, devices=("gpu" if gpu else "cpu",))
    require_collection_host()
    require_native_docker()
    root = Path(output_dir)
    with measurement_lock_scope(measurement_lock), ResultDirectoryLock(root):
        if (root / "runtime_validation.json").exists():
            raise ValueError("probe output already contains validation; choose a new output directory")
        generator = _selected_nlp_generator(task_info, workload_spec_path) if selected_nlp else None
        try:
            image = prepare_image(task_info, str(resource_root()), reuse_existing=reuse_existing)
        except (RuntimeError, ValueError, OSError) as exc:
            raise ProbePreparationError(str(exc)) from exc
        write_model_resolution(task_info, root)
        if generator is None:
            generator = get_task_generator(task_info, 1)
        scales = generator.default_input_scales()
        if not scales and not task_info.model_resolution.get("contract"):
            scales = resolve_input_scales(task_info.task_family)
        if not scales:
            raise ValueError("contract probe requires declared default workload scales")
        scale = min(scales)
        payload = generator.generate(scale)
        # Probe a minimal deterministic response, independent of measurement settings.
        if task_info.model_resolution.get("contract") and task_info.task_family == "multimodal":
            payload.setdefault("params", {}).update(max_new_tokens=1, do_sample=False)
        plan = root / "runtime_validation_input.json"
        entries = [{"input_scale": scale, "payload": payload}]
        planned = PlannedInputScales(scales=[float(scale)], source="runtime_validation", plan_file=str(plan))
        if selected_nlp:
            planned.workload = generator.plan_metadata()
            planned.plan_sha256 = write_scale_plan_file(str(plan), task_info, entries, workload=planned.workload,
                                                        scope="runtime_validation_only")
        else:
            atomic_write_json(plan, {"schema_version": 2, "scope": "runtime_validation_only", "entries": entries})
        return validate_runtime(task_info=task_info, image_info=image, planned=planned,
                                cpu_list=[cpus], mem_list=[memory_gb], gpu_list=["on" if gpu else "off"],
                                output_dir=str(root), timeout_seconds=timeout_seconds)
