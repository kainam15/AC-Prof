"""Resolve registered inference requests before confirmation or runtime work.

This host-side module reads existing declarations and locked environment
metadata only. It does not load a checkpoint or execute model repository code.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

from acprof.extensions import CATALOG, select_extension
from acprof.extensions.parameters import (
    ParameterConsumer,
    ParameterRequest,
    ParameterSpec,
    parameter_spec_to_dict,
    validate_parameter_value,
    validate_parameter_values,
)
from acprof.failures import RuntimeFailure
from acprof.models.configuration import parameter_declaration
from acprof.models.configuration_presets import additional_presets, apply_preset
from acprof.models.configuration_sources import discover_configuration_sources
from acprof.models.evidence import content_digest, pinned_revision
from acprof.models.inference_configuration import (
    CONFIGURATION_RULE_VERSION,
    TASK_CONFIGURATION_RULE_VERSION,
    configuration_device,
    configuration_identity,
)
from acprof.precision import resolve_precision
from acprof.runtime.profiles import PROFILES, locked_transformers_version, select_runtime_profile
from acprof.tasks.workloads import task_fingerprint, task_label

_REQUEST_KEYS = {"schema_version", "preset", "backend", "runtime_profile", "loading", "tasks"}


def _request(value: Any, path: str) -> dict:
    try:
        return ParameterRequest.from_dict(value).to_dict()
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def _object(value: Any, path: str) -> dict:
    if type(value) is not dict:
        raise ValueError(f"{path} must be an object")
    return value


def _devices(devices) -> list[str]:
    if not isinstance(devices, (list, tuple)) or not devices:
        raise ValueError("devices must be a nonempty list")
    normalized = [configuration_device(value) for value in devices]
    if len(normalized) != len(set(normalized)):
        raise ValueError("devices must not contain duplicates")
    return sorted(normalized)


def _validate_requests(requests: dict | None) -> dict:
    if requests is None:
        return {"schema_version": 1, "preset": "custom"}
    _object(requests, "requests")
    ParameterRequest("explicit", requests)
    if set(requests) - _REQUEST_KEYS:
        raise ValueError(f"unknown inference request fields: {sorted(set(requests) - _REQUEST_KEYS)}")
    if type(requests.get("schema_version")) is not int or requests["schema_version"] != 1:
        raise ValueError("inference requests require schema_version 1")
    if type(requests.get("preset", "custom")) is not str or requests.get("preset", "custom") not in {"custom", "auto", "author", "benchmark"}:
        raise ValueError("unknown inference configuration preset")
    for section in ("backend", "runtime_profile"):
        if section in requests:
            _request(requests[section], section)
    for device, fields in _object(requests.get("loading", {}), "loading").items():
        if device not in {"off", "on"}:
            raise ValueError(f"unknown loading device {device!r}")
        for name, value in _object(fields, f"loading.{device}").items():
            _request(value, f"loading.{device}.{name}")
    for task_id, fields in _object(requests.get("tasks", {}), "tasks").items():
        for name, value in _object(fields, f"tasks.{task_id}").items():
            _request(value, f"tasks.{task_id}.{name}")
    return copy.deepcopy(requests)


def _selected_value(request: dict, default: str, path: str) -> str:
    parsed = ParameterRequest.from_dict(request)
    # Unset may be previewed for editing, but _resolved_fields never accepts it.
    value = default if parsed.mode in {"auto", "unset"} else parsed.value
    if type(value) is not str or not value:
        raise ValueError(f"{path} must select a registered name")
    return value


def _route(task_info, backend: str, profile: str):
    from acprof.models.candidates import apply_model_candidate, discover_model_candidates
    from acprof.models.resolution import resolve_model_interface
    from acprof.tasks.groups import reidentify_inference_profile
    from acprof.tasks.interfaces import execution_path, validate_execution_identity

    failure = task_info.model_resolution.get("failure", {})
    if failure.get("reason_code") in {"metadata_invalid", "evidence_conflict", "model_contract_required"}:
        raise ValueError(f"inference configuration is blocked: {failure['detail']}")
    view = copy.deepcopy(task_info)
    if not view.selected_tasks:
        raise ValueError("select an inference task before configuring it")
    view.inference_configuration = {}
    changed_backend = backend != view.runtime_backend
    view.runtime_backend = backend
    if changed_backend:
        view.runtime_profile_id = ""
        for selected in view.selected_tasks:
            path = execution_path(view, selected["interface"], backend=backend)
            selected["execution_path"] = path
            if "execution_profile" in selected:
                selected["execution_profile"]["signature"]["backend"] = backend
                rebound = reidentify_inference_profile(selected)
                selected.clear()
                selected.update(rebound)
            else:
                selected["fingerprint"] = task_fingerprint(selected)
        decision = discover_model_candidates(view, override_tag=view.pipeline_tag, override_backend=backend)
        view.model_resolution = apply_model_candidate(view, decision)
    view.runtime_profile_id = profile or view.runtime_profile_id
    selected_backends = {selected.get("execution_profile", {}).get("signature", {}).get("backend", backend)
                         for selected in view.selected_tasks}
    if selected_backends != {backend}:
        raise ValueError("selected tasks require different backends; configure one backend per collection")
    validate_execution_identity(view)
    if view.model_resolution.get("status") == "selection_required":
        view.model_resolution["status"] = "candidate"
        if view.model_resolution.get("failure", {}).get("reason_code") == "task_selection_required":
            view.model_resolution.pop("failure")
    view.model_resolution["selected_task"] = copy.deepcopy(view.selected_tasks[0])
    if view.selected_tasks[0].get("execution_path"):
        view.model_resolution["execution"] = copy.deepcopy(view.selected_tasks[0]["execution_path"])
    view.model_resolution = resolve_model_interface(view)
    chosen_profile = select_runtime_profile(view)
    view.runtime_profile_id = chosen_profile.profile_id
    view.model_adapter = select_extension(view).adapter
    from acprof.models.backend_candidates import describe_backend_candidates

    for interface in sorted({selected["interface"] for selected in view.selected_tasks}):
        candidate = next((item for item in describe_backend_candidates(view, task=interface)
                          if item["backend"] == backend), None)
        if not candidate or not candidate["selectable"]:
            detail = candidate.get("compatibility", {}).get("detail", "No compatible registered route") if candidate else "No registered route"
            raise ValueError(f"{interface}/{backend}: {detail}")
    return view, chosen_profile


def _route_options(view, profile) -> tuple[list[str], list[str]]:
    registered = {backend for (family, backend, _adapter), declaration in CATALOG.routes.items()
                  if family == view.task_family
                  and all(task["interface"] in declaration.backend_tasks.get(backend, declaration.tasks)
                          for task in view.selected_tasks)}
    backends = []
    for backend in sorted(registered):
        try:
            _route(view, backend, profile.profile_id)
        except (ValueError, RuntimeFailure):
            continue
        backends.append(backend)
    profiles = []
    for candidate in PROFILES.values():
        if candidate.environment.platform.platform_id != profile.environment.platform.platform_id:
            continue
        probe = copy.copy(view)
        probe.runtime_profile_id = candidate.profile_id
        try:
            selected = select_runtime_profile(probe)
        except (ValueError, RuntimeFailure):
            continue
        if selected.profile_id == candidate.profile_id:
            profiles.append(candidate.profile_id)
    return backends, sorted(profiles)

def _field(path: str, spec: ParameterSpec, default: Any, source: dict, **scope) -> dict:
    return {"path": path, "spec": spec, "default": copy.deepcopy(default),
            "source": source, **scope}


def _fields(initial, view, profile, devices, workload) -> list[dict]:
    declarations = [parameter_declaration(view, task) for task in view.selected_tasks]
    loading_contracts = [[parameter_spec_to_dict(spec) for spec in declaration.parameters
                          if spec.scope == "loading"] for declaration in declarations]
    if len({content_digest(contract) for contract in loading_contracts}) != 1:
        raise ValueError("selected tasks have different loading declarations; configure separate task collections")
    backends, profiles = _route_options(view, profile)
    fields = [
        _field("backend", ParameterSpec("backend", "enum", scope="loading", label="Backend",
            choices=tuple(backends), consumers=(ParameterConsumer("load", "backend"),)),
            view.runtime_backend, {"kind": "registered_route"}),
        _field("runtime_profile", ParameterSpec("runtime_profile", "enum", scope="loading", label="Runtime Profile",
            choices=tuple(profiles), consumers=(ParameterConsumer("load", "runtime_profile"),)),
            profile.profile_id, {"kind": "registered_runtime", "environment": profile.environment.environment_key}),
    ]
    declaration = declarations[0]
    version = locked_transformers_version(profile.environment) or ""
    for device in devices:
        for spec in declaration.parameters:
            if spec.scope != "loading":
                continue
            source = {"kind": "parameter_declaration", "extension_id": declaration.extension_id}
            if spec.name == "dtype":
                choices = []
                for dtype in spec.choices:
                    try:
                        resolve_precision(profile, declaration, device=device, task=view.pipeline_tag,
                            model_type=view.model_config.get("model_type", ""), version=version, requested=dtype)
                    except RuntimeFailure:
                        continue
                    choices.append(dtype)
                spec = replace(spec, choices=tuple(choices))
                policy = resolve_precision(profile, declaration, device=device, task=view.pipeline_tag,
                    model_type=view.model_config.get("model_type", ""), version=version)
                default = policy["dtype"]
                source = {"kind": "registered_precision_policy", "policy": policy}
            else:
                default = spec.default.value if spec.default is not None else None
                discovery = discover_configuration_sources(view, (spec,), initial.selected_tasks[0])
                candidates = [item for item in discovery["parameters"].get(spec.name, {}).get("default_candidates", [])
                              if item["validation"] == "valid" and item["source"].get("file") is not None]
                if spec.name == "scheduler" and candidates:
                    default, source = candidates[0]["value"], candidates[0]["source"]
                    source = {"kind": "checkpoint_scheduler", "evidence": source}
            fields.append(_field(f"loading.{device}.{spec.name}", spec, default, source, device=device))
    for original, selected in zip(initial.selected_tasks, view.selected_tasks, strict=True):
        task_id = original["task_id"]
        declaration = parameter_declaration(view, selected)
        discovery = discover_configuration_sources(view, declaration.parameters, original)
        invalid = [item for item in discovery["diagnostics"] if item["reason_code"] == "metadata_invalid"]
        if invalid:
            raise ValueError("invalid configuration source: " + "; ".join(
                f"{item.get('file')}{item.get('json_pointer', '')}: {item['detail']}" for item in invalid))
        generation = selected["config"].get("generation", selected["config"].get("inference", {}))
        length_name = "max_length" if "max_length" in generation and "max_new_tokens" not in generation else "max_new_tokens"
        for spec in declaration.parameters:
            if spec.scope != "inference" or spec.read_only:
                continue
            if spec.name in {"max_new_tokens", "max_length"} and spec.name != length_name:
                continue
            if spec.name not in generation and spec.default is None:
                continue
            default = generation[spec.name] if spec.name in generation else spec.default.value
            source = ({"kind": "task_snapshot", "task_id": task_id, "fingerprint": original["fingerprint"],
                       "json_pointer": "/config/" + ("generation" if "generation" in selected["config"] else "inference") + "/" + spec.name,
                       "output_length_policy": original.get("execution_profile", {}).get("output_length_policy")}
                      if spec.name in generation else
                      {"kind": "parameter_declaration", "extension_id": declaration.extension_id})
            supplied_defaults = workload.get("parameter_defaults", {})
            if "generation" not in selected["config"] and spec.name in supplied_defaults:
                default = supplied_defaults[spec.name]
                source = {"kind": "workload_specification", "sha256": workload.get("specification_sha256"),
                          "parameter": spec.name}
            fields.append(_field(f"tasks.{task_id}.{spec.name}", spec, default, source,
                task_id=task_id, task_name=task_label(selected),
                candidates=discovery["parameters"].get(spec.name, {}).get("default_candidates", [])))
        overrides = selected["config"].get("user_experiment_overrides", {})
        for name, kind, label, default in (
            ("batch_size", "integer", "Batch Size", workload.get("batch_size", 1)),
            ("input_scales", "string", "Input scales", workload.get("input_scales", "")),
        ):
            spec = ParameterSpec(name, kind, scope="workload", label=label,
                minimum=1 if kind == "integer" else None,
                consumers=(ParameterConsumer("workload", name),), variation=("task",))
            fields.append(_field(f"tasks.{task_id}.{name}", spec, overrides.get(name, default),
                {"kind": "workload_configuration"}, task_id=task_id, task_name=task_label(selected)))
    return fields


def _supplied(requests: dict, field: dict):
    if field["path"] in {"backend", "runtime_profile"}:
        return requests.get(field["path"])
    section = requests.get("tasks" if "task_id" in field else "loading", {})
    return section.get(field.get("task_id", field.get("device")), {}).get(field["spec"].name)


def _put(requests: dict, field: dict, value: dict) -> None:
    if field["path"] in {"backend", "runtime_profile"}:
        requests[field["path"]] = value
    else:
        section = "tasks" if "task_id" in field else "loading"
        key = field.get("task_id", field.get("device"))
        name = field["spec"].name if "spec" in field else field["parameter_spec"]["name"]
        requests.setdefault(section, {}).setdefault(key, {})[name] = value


def _prepare(task_info, *, devices, requests=None, workload=None):
    requests = _validate_requests(requests)
    workload = copy.deepcopy(workload or {})
    devices = _devices(devices)
    backend = _selected_value(requests.get("backend", {"mode": "explicit", "value": task_info.runtime_backend}),
                              task_info.runtime_backend, "backend")
    profile_default = task_info.runtime_profile_id or select_runtime_profile(task_info).profile_id
    profile_request = requests.get("runtime_profile")
    if profile_request is None and backend != task_info.runtime_backend:
        profile = profile_default
    else:
        profile = _selected_value(profile_request or {"mode": "explicit", "value": profile_default},
                                  profile_default, "runtime_profile")
    view, profile = _route(task_info, backend, profile)
    if "on" in devices and profile.environment.platform.platform_id in {"cpu", "python-cpu"}:
        raise ValueError("the requested GPU device requires a GPU-capable runtime profile")
    fields = _fields(task_info, view, profile, devices, workload)
    apply_preset(fields, requests.get("preset", "custom"))
    allowed = {field["path"] for field in fields}
    supplied = {"loading." + device + "." + key for device, values in requests.get("loading", {}).items() for key in values}
    supplied.update("tasks." + task_id + "." + key for task_id, values in requests.get("tasks", {}).items() for key in values)
    if supplied - allowed:
        raise ValueError(f"unknown or unavailable inference parameters: {sorted(supplied - allowed)}")
    normalized = {"schema_version": 1, "preset": requests.get("preset", "custom"), "loading": {}, "tasks": {}}
    for field in fields:
        requested = _supplied(requests, field)
        if requested is None:
            requested = {"mode": "auto"} if normalized["preset"] == "auto" else {"mode": "explicit", "value": field["default"]}
        _put(normalized, field, _request(requested, field["path"]))
    for field in fields:
        field["request"] = _supplied(normalized, field)
        field["value"] = (field["request"].get("value") if field["request"]["mode"] == "explicit" else field["default"])
        if field["request"]["mode"] == "explicit":
            validate_parameter_value(field["spec"], field["value"], path=field["path"])
    for field in fields:
        group = {item["spec"].name: item["value"] for item in fields if item.get("task_id") == field.get("task_id")
                 and item.get("device") == field.get("device") and item["request"]["mode"] != "unset"}
        field["enabled"] = all(condition.matches(group) for condition in field["spec"].requires)
        if not field["enabled"] and _supplied(requests, field) is None:
            field["request"] = {"mode": "unset"}
            _put(normalized, field, field["request"])
    return view, profile, fields, normalized, devices, workload


def build_configuration_draft(task_info, *, devices: list[str], requests: dict | None = None,
                              workload: dict | None = None) -> dict:
    view, profile, fields, normalized, devices, workload = _prepare(
        task_info, devices=devices, requests=requests, workload=workload)
    questions, advanced = [], []
    for field in fields:
        spec = field["spec"]
        dependencies = {condition.parameter: next((item["path"] for item in fields
            if item["spec"].name == condition.parameter and item.get("task_id") == field.get("task_id")
            and item.get("device") == field.get("device")), "") for condition in spec.requires}
        label = spec.label or spec.name
        if field.get("task_name"):
            label = field["task_name"] + " · " + label
        if field.get("device"):
            label = ("CPU" if field["device"] == "off" else "GPU") + " · " + label
        question = {key: copy.deepcopy(value) for key, value in field.items()
                    if key not in {"spec", "default", "candidates", "value"}}
        question.update(label=label, caption=spec.label or spec.name, kind=spec.kind, scope=spec.scope,
            parameter_spec=parameter_spec_to_dict(spec), value=copy.deepcopy(field["value"]),
            resolved_value=copy.deepcopy(field["value"]), available=True,
            disabled_reason="; ".join(condition.reason or f"Requires {condition.parameter}" for condition in spec.requires)
                            if not field["enabled"] else "",
            dependency_paths=dependencies, reason=field["source"]["kind"],
            auto_reason="Use the reviewed task default or registered loading policy.",
            validation="static_only; runtime_not_run")
        if spec.choices:
            question["options"] = list(spec.choices)
        (advanced if spec.advanced else questions).append(question)
    auto = copy.deepcopy(normalized)
    auto["preset"] = "auto"
    for field in fields:
        _put(auto, field, {"mode": "auto"})
    return {"schema_version": 1, "questions": questions, "parameter_advanced": advanced,
            "requests": normalized, "fields": {"Model": view.model_id, "Revision": view.model_revision,
                "Backend": view.runtime_backend, "Runtime Profile": profile.profile_id},
            "presets": [{"id": "custom", "label": "Custom", "description": "Keep the displayed requests.",
                         "requests": normalized},
                        {"id": "auto", "label": "Auto", "description": "Authorize the registered rule for each available field.",
                         "requests": auto}, *additional_presets(fields)],
            "summary": "Review explicit values or choose Auto. Runtime validation has not run."}


def requests_from_answers(draft: dict, answers: dict) -> dict:
    fields = draft["questions"] + draft.get("parameter_advanced", [])
    if type(answers) is not dict or set(answers) != {field["path"] for field in fields}:
        raise ValueError("inference answers must cover exactly the displayed fields")
    result = copy.deepcopy(draft["requests"])
    for field in fields:
        _put(result, field, _request(answers[field["path"]], field["path"]))
    return result


def _resolved_fields(fields: list[dict], declarations: dict) -> tuple[dict, list[dict]]:
    values, decisions = {}, []
    for field in fields:
        request, spec, path = field["request"], field["spec"], field["path"]
        if not field["enabled"]:
            if request["mode"] == "explicit":
                raise ValueError(f"{path}: " + "; ".join(item.reason or f"requires {item.parameter}" for item in spec.requires))
            decisions.append({"path": path, "request": request, "status": "inactive",
                              "reason": "The declared dependency is not satisfied.", "rule_version": CONFIGURATION_RULE_VERSION})
            continue
        if request["mode"] == "unset":
            raise ValueError(f"{path} is unset; choose an explicit value or Auto")
        value = request["value"] if request["mode"] == "explicit" else field["default"]
        validate_parameter_value(spec, value, path=path)
        values[path] = copy.deepcopy(value)
        decisions.append({"path": path, "request": copy.deepcopy(request), "value": copy.deepcopy(value),
                          "source": ({"kind": "explicit_request", "path": path,
                                      "suggested_source": copy.deepcopy(field["source"])}
                                     if request["mode"] == "explicit" else copy.deepcopy(field["source"])),
                          "candidates": copy.deepcopy(field.get("candidates", [])),
                          "reason": "Explicit user selection." if request["mode"] == "explicit" else
                                    "Selected the fixed task value or registered policy; no runtime fallback is allowed.",
                          "rule_version": CONFIGURATION_RULE_VERSION, "validation": "static_validated",
                          "runtime_validation": "not_run"})
    for task_id in {field["task_id"] for field in fields if "task_id" in field}:
        selected = [field for field in fields if field.get("task_id") == task_id and field["spec"].scope == "inference"]
        validate_parameter_values(declarations[task_id],
            {field["spec"].name: values[field["path"]] for field in selected if field["path"] in values})
    return values, decisions


def resolve_inference_configuration(task_info, requests: dict, *, devices: list[str],
                                    workload: dict | None = None) -> tuple[Any, dict]:
    from acprof import __version__
    from acprof.installation import resource_root
    from acprof.runtime.dependency_locks import package_versions, read_python_lock
    from acprof.runtime.profiles import environment_id
    from acprof.source_identity import (
        measurement_sources,
        service_context_files,
        service_sources,
        source_fingerprint,
    )
    from acprof.tasks.groups import reidentify_inference_profile
    from acprof.tasks.parameters import apply_parameter_review, parameter_review

    if not pinned_revision(task_info.model_revision):
        raise ValueError("inference configuration requires a fixed model revision")
    view, profile, fields, normalized, devices, workload = _prepare(
        task_info, devices=devices, requests=requests, workload=workload)
    declarations = {original["task_id"]: parameter_declaration(view, selected).parameters
                    for original, selected in zip(task_info.selected_tasks, view.selected_tasks, strict=True)}
    values, decisions = _resolved_fields(fields, declarations)
    review = parameter_review(view, view.selected_tasks,
        batch_size=workload.get("batch_size", 1), input_scales=workload.get("input_scales", ""))
    answers = {field["path"]: field["value"] for field in review["questions"] + review["parameter_advanced"]}
    for index, original in enumerate(task_info.selected_tasks):
        for field in fields:
            if field.get("task_id") == original["task_id"] and field["path"] in values:
                answers[f"task_{index}.{field['spec'].name}"] = values[field["path"]]
    inactive = {f"task_{index}.{field['spec'].name}"
                for index, original in enumerate(task_info.selected_tasks)
                for field in fields if field.get("task_id") == original["task_id"]
                and field["spec"].scope == "inference" and not field["enabled"]}
    view.selected_tasks = apply_parameter_review(view, view.selected_tasks, answers,
        batch_size=workload.get("batch_size", 1), input_scales=workload.get("input_scales", ""),
        materialize_defaults=True, inactive_parameters=inactive)
    for task in view.selected_tasks:
        if not task["config"].get("generation", {}).get("do_sample", False):
            for name in ("temperature", "top_p"):
                task["config"].get("generation", {}).pop(name, None)
        if "execution_profile" in task:
            updated = reidentify_inference_profile(task)
            task.clear()
            task.update(updated)
        task["fingerprint"] = task_fingerprint(task)
    view.model_resolution["selected_task"] = copy.deepcopy(view.selected_tasks[0])
    tasks = {}
    for selected in view.selected_tasks:
        semantics = set(selected.get("execution_profile", {}).get("semantic_task_ids", [selected["task_id"]]))
        sources = [original["task_id"] for original in task_info.selected_tasks if semantics.intersection(
            original.get("execution_profile", {}).get("semantic_task_ids", [original["task_id"]]))]
        tasks[selected["task_id"]] = {"interface": selected["interface"], "fingerprint": selected["fingerprint"],
            "generation": copy.deepcopy(selected["config"].get("generation", {})),
            **({"inference": copy.deepcopy(selected["config"]["inference"])} if "inference" in selected["config"] else {}),
            "source_task_ids": sources, "snapshot": copy.deepcopy(selected)}
    root = resource_root()
    extension = select_extension(view)
    candidate = {
        "schema_version": 1, "state": "resolved", "rule_version": CONFIGURATION_RULE_VERSION,
        "model": {"source": view.model_source, "model_id": view.model_id, "revision": view.model_revision,
            "configuration_sha256": content_digest(view.model_config),
            "repository_files_sha256": content_digest(sorted(view.repository_files)),
            "base_model": copy.deepcopy(view.hub_metadata.get("base_model")),
            "license": view.model_license, "quantized": view.quantized,
            "quantization_method": view.quantization_method, "weight_storage_precision": view.precision_dtype},
        "backend": {"id": view.runtime_backend, "adapter": view.model_adapter, "extension_id": extension.extension_id,
            "runtime_profile": profile.profile_id, "environment_key": profile.environment.environment_key,
            "environment_identity_sha256": environment_id(profile.environment, root),
            "dependencies": package_versions(read_python_lock(root / profile.environment.requirements_lock)),
            "acprof_version": __version__,
            "source_identity": {
                "measurement": {"scope": "measurement-source-v2",
                    "sha256": source_fingerprint(root, measurement_sources(root), scope="measurement-source-v2")},
                "docker_service": {"scope": "service-source-v2",
                    "sha256": source_fingerprint(root,
                        [*service_context_files(root), root / "dockerfiles/runtime-model.Dockerfile"], scope="service-source-v2")},
                "process_service": {"scope": "process-service-v1",
                    "sha256": source_fingerprint(root, service_sources(root), scope="process-service-v1")},
            }},
        "loading": {device: {field["spec"].name: values[field["path"]] for field in fields
                            if field.get("device") == device and field["path"] in values} for device in devices},
        "tasks": tasks, "requests": normalized, "decisions": decisions, "workload": workload,
        "validation": {"static": "validated", "runtime": "not_run"},
    }
    if (any(task.get("inference") for task in tasks.values())
            or any("scheduler" in values for values in candidate["loading"].values())
            or normalized["preset"] in {"author", "benchmark"}):
        candidate["rule_version"] = TASK_CONFIGURATION_RULE_VERSION
        for decision in decisions:
            decision["rule_version"] = TASK_CONFIGURATION_RULE_VERSION
    candidate["identity_sha256"] = configuration_identity(candidate)
    return view, candidate
