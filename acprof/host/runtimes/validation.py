"""Use the existing model interface and output validators inside Process environments."""
from __future__ import annotations

import hashlib
import json
import uuid

from acprof.artifact_layout import ArtifactLayout
from acprof.artifacts import atomic_write_json, read_input_scale_plan
from acprof.host.runtimes.contracts import ProcessInfo
from acprof.host.runtimes.process import service_environment
from acprof.host.runtimes.process_control import ManagedProcess
from acprof.host.validation_logs import last_prefixed_record
from acprof.preparation_events import preparation_stage


def run_validation(task, info, module, prefix, output_dir, *, gpu="off", payload=None, timeout=300,
                   cpu=None, memory=None):
    from acprof.host.model_store import acquire_mount
    from acprof.host.runtimes.resources import ProcessLimits
    layout = ArtifactLayout.discover(output_dir)
    directory = layout.path("logs")
    directory.mkdir(parents=True, exist_ok=True)
    name = "process-validation-" + uuid.uuid4().hex
    command = [info.python, "-P", "-m", module]
    if payload is not None:
        path = directory / (name + ".input.json")
        atomic_write_json(path, payload)
        command.append(str(path))
    limits = ProcessLimits(cpu, memory)
    lease = None
    process = None
    try:
        lease = acquire_mount(info.runtime_environment)
        process = ManagedProcess(command, python=info.python, env=service_environment(task, info, gpu=gpu),
                                 directory=directory, name=name, pass_fds=lease.pass_fds, cgroup=limits.path)
        code = process.wait(timeout)
        with process.log_path.open(encoding="utf-8", errors="replace") as stream:
            record = last_prefixed_record(stream, prefix)
        observed = json.loads(record) if record is not None else {}
        if code or observed.get("status") != "ok":
            error = observed.get("error") or process.log_tail().replace("\r\n", "\n").replace("\r", "\n")[-4000:]
            raise RuntimeError(error or f"Process validation exit {code}")
        return observed
    finally:
        if process is not None:
            try:
                process.close()
            except BaseException:
                from acprof.host.model_store import retain_mount_for_cleanup_debt
                if lease is not None:
                    retain_mount_for_cleanup_debt(lease)
                raise
        if lease is not None:
            lease.close()
        limits.close()


@preparation_stage("interface")
def probe_interface(task, root, environment, output_dir, *, timeout_seconds=300):
    from acprof.container.model_probe import RESULT_PREFIX
    from acprof.host.source_bundle import source_bundle
    from acprof.models.contract import write_model_resolution
    report = {"schema_version": 1, "status": "running", "runtime_kind": "process",
              "scope": "import_and_signature_only", "inference": "not_run",
              "environment_id": environment["environment_id"]}
    try:
        with source_bundle(task) as bundle:
            info = ProcessInfo(str(root / "venv/bin/python"), str(root / "code"),
                               str(bundle.root), environment["environment_id"], environment)
            observed = run_validation(task, info, "acprof.container.model_probe", RESULT_PREFIX, output_dir,
                                      timeout=timeout_seconds)
            if {item["stage"] for item in observed.get("stages", []) if item.get("status") == "verified"} != {"import", "signature"}:
                raise RuntimeError("Process interface validation is incomplete")
            report.update(status="ok", validation=observed, source_sha256=bundle.identity)
    except BaseException as exc:
        report.update(status="error", error=str(exc))
        raise
    finally:
        atomic_write_json(ArtifactLayout.discover(output_dir).path("interface_validation.json"), report)
        task.model_resolution["interface_validation"] = report
        write_model_resolution(task, output_dir)


def validate_process(*, task_info, image_info, planned, cpu_list, mem_list, gpu_list, output_dir, timeout_seconds=300):
    from acprof.container.runtime_validate import RESULT_PREFIX
    from acprof.models.contract import record_runtime_validation, write_model_resolution
    from acprof.platform import detect_environment
    plan = read_input_scale_plan(planned.plan_file)
    entry = min(plan["entries"], key=lambda item: float(item["input_scale"]))
    report = {**detect_environment().metadata(), "schema_version": 1, "runtime_kind": "process",
        "runtime_id": image_info.environment_id, "status": "running", "devices": {}, "mode": "full",
        "input_scale": entry["input_scale"], "payload_sha256": hashlib.sha256(json.dumps(entry["payload"]).encode()).hexdigest(),
        "scope": "isolated_minimum_scale_predict_and_postprocess_before_measurement"}
    try:
        for gpu in dict.fromkeys(gpu_list):
            observed = run_validation(task_info, image_info, "acprof.container.runtime_validate", RESULT_PREFIX,
                                      output_dir, gpu=gpu, payload=entry["payload"], timeout=timeout_seconds,
                                      cpu=max(cpu_list), memory=max(mem_list))
            report["devices"][gpu] = observed
            required = {"execution", "load", "preprocess", "predict", "completion", "postprocess", "validate_output"}
            stages = {item["stage"] for item in observed.get("stages", []) if item.get("status") == "verified"}
            if not required <= stages or any(observed.get("validation", {}).get(layer, {}).get("status") != "verified"
                                            for layer in ("protocol", "task")):
                raise RuntimeError("Process runtime validation is incomplete")
        report["status"] = "ok"
    except BaseException as exc:
        report.update(status="error", error=str(exc))
        raise
    finally:
        atomic_write_json(ArtifactLayout.discover(output_dir).path("runtime_validation.json"), report)
        record_runtime_validation(task_info, report)
        write_model_resolution(task_info, output_dir)
    return report
