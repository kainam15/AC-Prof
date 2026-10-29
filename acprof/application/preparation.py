"""Prepare and restore immutable model/runtime evidence before a collection matrix."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from acprof.application.errors import CollectionFailed
from acprof.application.notifications import RunNotifications
from acprof.artifact_layout import ArtifactLayout
from acprof.artifacts import read_json_object
from acprof.capabilities import (
    Capability,
    CapabilityReport,
    apply_extension,
    apply_profiler_plan,
    apply_runtime_validation,
    measurement_report,
)
from acprof.config import SCALING_DIMENSIONS
from acprof.host.collection_history import (
    COLLECTION_HISTORY_NAME,
    empty_collection_history,
    write_collection_history_json,
)
from acprof.host.gpu_device import selected_gpu_device

if TYPE_CHECKING:
    from acprof.host.detect import TaskInfo
    from acprof.host.images.runtime import ImageInfo
    from acprof.host.input_plan import PlannedInputScales


@dataclass
class PreparedRuntime:
    task_info: TaskInfo
    image_info: ImageInfo
    planned_input_scales: PlannedInputScales
    compute_profile_plan_file: str
    execution_profile_plan_file: str
    capability_report: CapabilityReport


def apply_profiler_plan_file(
    capability_report: CapabilityReport,
    plan_path: str,
    *,
    source: str,
) -> None:
    if not plan_path or not Path(plan_path).is_file():
        return
    apply_profiler_plan(
        capability_report,
        read_json_object(plan_path, label=source.replace("_", " ")),
        source=source,
    )


def prepare_runtime(args, *, run_state, task_info, output_dir, cpu_list, mem_list,
                     gpu_list, run_command, cgroup_version, cgroup_collection_mode,
                     rapl_topology, preflight_measurements, latency_slo,
                     workflow=None, project_dir: str, notices: RunNotifications) -> PreparedRuntime:
    """Build or restore the runtime and persist evidence before the matrix."""
    from acprof.host.preparation.collection_workflow import PreparationWorkflow
    from acprof.host.runtimes.service import execution_runtime
    runtime = execution_runtime(getattr(args, "runtime", "docker"))
    from acprof.host.input_plan import input_plan_summary, plan_input_scales
    workflow = workflow or PreparationWorkflow()
    from acprof.host.metadata.static import (
        collect_static_meta,
        enrich_static_meta,
        enrich_static_meta_from_compute_plan,
        enrich_static_meta_from_execution_plan,
        enrich_static_meta_from_input_plan,
        write_static_meta_json,
    )

    compute_profile_disabled = args.compute_profile_tool == "none"
    total_cases = len(cpu_list) * len(mem_list) * len(gpu_list)
    layout = ArtifactLayout.discover(output_dir)
    static_meta_json = str(layout.path("static_meta.json"))
    collection_history_json = str(layout.path(COLLECTION_HISTORY_NAME))
    if run_state.ready:
        (task_info, image_info, planned_input_scales, compute_profile_plan_file,
         execution_profile_plan_file) = run_state.restore_runtime()
        if getattr(args, "runtime", "docker") == "process":
            image_info.runtime_environment["logs_dir"] = str(Path(output_dir).absolute() / "logs")
        workflow.emit("input", "passed", input_plan=input_plan_summary(task_info, planned_input_scales))
        notices.update_plan(model_id=task_info.model_id, output_dir=output_dir,
                                      total_cases=total_cases)
        print(f"[resume] 恢复实验 {run_state.data['run_id']}，复用原镜像和输入计划")
        saved_meta = read_json_object(static_meta_json, label="static metadata")
        capability_report = CapabilityReport.from_dict(saved_meta.get("capability_report", {
            "profiling_mode": args.profiling_mode,
        }))
        validation_status = saved_meta.get("runtime_validation", {}).get("status", "not_run")
        workflow.emit("runtime", "passed" if validation_status == "ok" else
                      "not_started" if validation_status == "not_run" else "failed",
                      detail=f"restored immutable runtime; saved validation: {validation_status}")
        if validation_status != "ok":
            raise RuntimeError("saved runtime validation is not successful: " + str(validation_status)
                               + "; start a new experiment to validate without changing frozen resume evidence")
    else:
        task_info.model_download_policy = args.model_download_policy
        from acprof.host.images.runtime import configure_runtime_profile
        workflow.run("interface", configure_runtime_profile, task_info)
        try:
            image_info = workflow.run("image", runtime.prepare,
                task_info, project_dir, output_dir, rebuild=(args.rebuild_image or workflow.rebuild_environment),
                timeout_seconds=args.request_timeout_seconds,
            )
        except (RuntimeError, OSError) as exc:
            print(f"\n[build][ERROR] {exc}", file=sys.stderr)
            raise CollectionFailed(str(exc)) from exc

        if args.runtime == "process":
            image_info.runtime_environment["gpu_device"] = getattr(args, "process_gpu", {})
        os.makedirs(output_dir, exist_ok=True)
        if task_info.model_resolution:
            from acprof.models.contract import write_model_resolution
            write_model_resolution(task_info, output_dir)

        scaling_cfg = SCALING_DIMENSIONS.get(task_info.task_family)
        input_scale_type = scaling_cfg.param_name if scaling_cfg else ""

        static_meta = collect_static_meta(
            task_info=task_info,
            image_info=image_info,
            batch_size=args.batch_size,
            input_scale_type=input_scale_type,
            run_command=run_command,
            cgroup_version=cgroup_version,
            cgroup_collection_mode=cgroup_collection_mode,
            compute_profile_enabled=not compute_profile_disabled,
            execution_profile_enabled=args.execution_profile_tool != "none",
            profiling_mode=args.profiling_mode,
            gpu_device=getattr(args, "process_gpu", {}) if args.runtime == "process" else selected_gpu_device(),
        )
        capability_report = measurement_report(
            args.profiling_mode, gpu_modes=gpu_list, compute_tool=args.compute_profile_tool,
            execution_tool=args.execution_profile_tool,
            dram_energy=args.dram_energy,
            rapl_topology=rapl_topology,
        )
        collection_plan = {}
        resource_allocation = {}
        if args.runtime == "process":
            from acprof.host.runtimes.capabilities import apply_process_plan, probe_collection_plan
            from acprof.host.runtimes.resources import inherited_resources
            collection_plan = probe_collection_plan("process", gpu=getattr(args, "process_gpu", {}),
                                                    profiling_mode=args.profiling_mode)
            resource_allocation = inherited_resources()
            apply_process_plan(capability_report, collection_plan)
        capability_report.measurement.update({name: item for name, item in preflight_measurements.items() if isinstance(item, Capability)})
        from acprof.extensions import select_extension
        apply_extension(capability_report, select_extension(task_info))
        static_meta = enrich_static_meta(static_meta, {
            "capability_report": capability_report.to_dict(), "latency_slo": latency_slo,
            "runtime_selection": getattr(args, "runtime_selection", {}),
            "resource_allocation": resource_allocation, "collection_plan": collection_plan,
        })
        write_static_meta_json(static_meta, static_meta_json)
        write_collection_history_json(
            empty_collection_history(),
            collection_history_json,
        )

        # ── Step 4: Run profiling matrix ──
        try:
            planned_input_scales = workflow.run("input", plan_input_scales,
                task_info=task_info,
                image_info=image_info,
                cpu_list=cpu_list,
                mem_list=mem_list,
                gpu_list=gpu_list,
                batch_size=args.batch_size,
                output_dir=output_dir,
                input_scales=args.input_scales,
                workload_spec_path=args.workload_spec,
                input_scale_policy=args.input_scale_policy,
            )
        except Exception as exc:
            print(f"\n[scale][ERROR] {exc}", file=sys.stderr)
            raise CollectionFailed(str(exc)) from exc

        workflow.emit("input", "passed", input_plan=input_plan_summary(task_info, planned_input_scales))

        static_meta = enrich_static_meta_from_input_plan(
            static_meta,
            planned_input_scales,
        )
        write_static_meta_json(static_meta, static_meta_json)
        compute_profile_plan_file = ""
        from acprof.host.metadata.static import enrich_static_meta
        def validate_selected_runtime():
            try:
                report = runtime.validate(
                    task_info=task_info, image_info=image_info, planned=planned_input_scales,
                    cpu_list=cpu_list, mem_list=mem_list, gpu_list=gpu_list, output_dir=output_dir,
                    timeout_seconds=args.request_timeout_seconds,
                )
                if report.get("status") != "ok":
                    raise RuntimeError("collection requires successful runtime validation on every requested device: "
                                       + str(report.get("status")))
            except Exception:
                if workflow.interactive:
                    from acprof.host.preparation.collection_workflow import (
                        retain_validation_failure,
                    )
                    try:
                        retain_validation_failure(output_dir)
                    except (OSError, ValueError) as archive_error:
                        print(f"[runtime-check][WARN] Cannot archive validation failure: {archive_error}", file=sys.stderr)
                raise
            return report

        try:
            validation = workflow.run("runtime", validate_selected_runtime)
            apply_runtime_validation(
                capability_report, validation,
                environment_id=image_info.runtime_environment.get("environment_id", ""),
            )
            static_meta = enrich_static_meta(static_meta, {"runtime_validation": validation,
                                                         "model_resolution": task_info.model_resolution})
            write_static_meta_json(static_meta, static_meta_json)
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"[runtime-check][ERROR] {exc}", file=sys.stderr)
            raise CollectionFailed(str(exc)) from exc
        total_cases = len(cpu_list) * len(mem_list) * len(gpu_list)
        notices.update_plan(
            model_id=task_info.model_id,
            output_dir=output_dir,
            total_cases=total_cases,
        )
        if compute_profile_disabled:
            reason = "--compute-profile-tool none"
            print(f"[compute] Compute profiling disabled by {reason}")
        else:
            workflow.emit("compute_profile", "running")
            try:
                from acprof.host.profiling.compute import collect_compute_profile_plan

                compute_profile_plan_file = collect_compute_profile_plan(
                    task_info=task_info,
                    image_tag=image_info.tag,
                    cpu_list=cpu_list,
                    mem_list=mem_list,
                    gpu_list=gpu_list,
                    output_dir=output_dir,
                    input_scale_plan_file=planned_input_scales.plan_file,
                    advisor_root=args.advisor_root,
                    ncu_root=args.ncu_root,
                    advisor_repeat=args.advisor_repeat,
                    torch_profiler_repeat=args.torch_profiler_repeat,
                    ncu_repeat=args.ncu_repeat,
                    keep_profiles=args.keep_compute_profiles,
                    compute_profile_cpus=args.compute_profile_cpus,
                    compute_profile_mem=args.compute_profile_mem,
                    compute_profile_tool=args.compute_profile_tool,
                    progress_callback=(
                        notices.notify_profiler
                        if notices.active()
                        else None
                    ),
                )
                static_meta = enrich_static_meta_from_compute_plan(
                    static_meta,
                    compute_profile_plan_file,
                )
                write_static_meta_json(static_meta, static_meta_json)
                workflow.emit("compute_profile", "passed")
            except Exception as exc:
                workflow.emit("compute_profile", "failed", detail=str(exc))
                print(f"[compute][WARN] Compute profiling unavailable: {exc}")

        execution_profile_plan_file = ""
        if args.execution_profile_tool == "none":
            print(
                "[execution-profile] Massif/Nsight Systems profiling disabled "
                "(enable with --execution-profile-tool)"
            )
        else:
            workflow.emit("execution_profile", "running")
            try:
                from acprof.host.profiling.execution import (
                    ExecutionProfileOptions,
                    ExecutionProfileRequest,
                    ProfilerSampling,
                    collect_execution_profile_plan,
                )

                execution_profile_plan_file = collect_execution_profile_plan(
                    ExecutionProfileRequest(
                        task_info=task_info, image_tag=image_info.tag,
                        cpu_list=cpu_list, mem_list=mem_list, gpu_list=gpu_list,
                        output_dir=output_dir,
                        input_scale_plan_file=planned_input_scales.plan_file,
                    ),
                    options=ExecutionProfileOptions(
                        tool_mode=args.execution_profile_tool,
                        massif=ProfilerSampling(
                            sampling=args.massif_sampling, repeat=args.massif_repeat,
                            reference_cpu=args.massif_reference_cpu,
                            reference_mem=args.massif_reference_mem,
                        ),
                        nsys=ProfilerSampling(
                            sampling=args.nsys_sampling, repeat=args.nsys_repeat,
                            reference_cpu=args.nsys_reference_cpu,
                            reference_mem=args.nsys_reference_mem,
                        ),
                        nsys_root=args.nsys_root, keep_profiles=args.keep_execution_profiles,
                    ),
                    progress_callback=(
                        notices.notify_profiler
                        if notices.active()
                        else None
                    ),
                )
                static_meta = enrich_static_meta_from_execution_plan(
                    static_meta,
                    execution_profile_plan_file,
                )
                write_static_meta_json(static_meta, static_meta_json)
                workflow.emit("execution_profile", "passed")
            except Exception as exc:
                workflow.emit("execution_profile", "failed", detail=str(exc))
                print(
                    "[execution-profile][WARN] Execution profiling unavailable: "
                    f"{exc}"
                )

        for plan_path, source in ((compute_profile_plan_file, "compute_profile_plan"),
                                  (execution_profile_plan_file, "execution_profile_plan")):
            apply_profiler_plan_file(capability_report, plan_path, source=source)
        static_meta = enrich_static_meta(static_meta, {"capability_report": capability_report.to_dict()})
        write_static_meta_json(static_meta, static_meta_json)
        from acprof.artifacts import atomic_write_json
        atomic_write_json(layout.path("capability_report.json"), capability_report.to_dict())
        if not capability_report.to_dict()["requested_measurements_available"]:
            print("[capability][WARN] 所请求的 profiler 尚有缺失或失败；详见 capability_report.json，不视为完整画像。")
        run_state.bind_runtime(task_info, image_info, planned_input_scales,
                               compute_profile_plan_file, execution_profile_plan_file)

    return PreparedRuntime(task_info, image_info, planned_input_scales,
                            compute_profile_plan_file, execution_profile_plan_file,
                            capability_report)
