#!/usr/bin/env python3
"""AC-Prof - One-click HuggingFace model profiling.

Usage:
    acprof run --model bert-base-uncased
    acprof run --model google/vit-base-patch16-224 --cpus 1,2 --mems 4,8 --gpus off
    acprof run --model amazon/chronos-bolt-base --task-family timeseries --backend chronos
"""
from __future__ import annotations

import math
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from acprof.host.detect import TaskInfo
    from acprof.host.input_plan import PlannedInputScales
    from acprof.host.runtime_images import ImageInfo

from acprof.artifact_layout import ArtifactLayout
from acprof.artifacts import read_json_object
from acprof.capabilities import (
    Capability,
    CapabilityReport,
    apply_collection_result,
    apply_extension,
    apply_profiler_plan,
    apply_runtime_validation,
    measurement_report,
    measurement_requested,
    missing_required_measurements,
)
from acprof.cli import run_notifications as notices
from acprof.cli.run_matrix import _format_elapsed, _resource_matrix
from acprof.cli.terminal_log import format_run_command, start_terminal_log, stop_terminal_log
from acprof.config import SCALING_DIMENSIONS
from acprof.host.collection_history import (
    COLLECTION_HISTORY_NAME,
    empty_collection_history,
    write_collection_history_json,
)
from acprof.host.env_utils import bootstrap_project_env
from acprof.host.gpu_device import gpu_device_scope, pin_gpu_device, selected_gpu_device
from acprof.host.orchestrator import (
    EnergyProfilingError,
    MIPSProfilingError,
)
from acprof.host.packet_capture import (
    PacketLatencyError,
    require_packet_latency_prerequisites,
)
from acprof.host.preflight import (
    require_cgroup_prerequisites,
    require_collection_host,
    require_cpu_energy_prerequisites,
    require_mips_prerequisites,
    require_native_docker,
    require_result_cgroup_compatibility,
    require_result_environment,
)
from acprof.host.result_cleanup import cleanup_intermediate_results
from acprof.host.run_state import RunState, RunStateError, load_run_state, run_options
from acprof.host.task_support import TaskSupportError
from acprof.installation import resource_root
from acprof.latency_slo import parse_latency_slo_rules, resolve_latency_slo
from acprof.run_args import build_parser as _build_parser

PROJECT_DIR = str(resource_root())
DEFAULT_NOTIFY_PROVIDER = "auto"
_ACTIVE_TMUX_TERMINAL_LOG: tuple[str, str, str] | None = None
_ACTIVE_RUN_STATE: RunState | None = None


@dataclass
class _PreparedRuntime:
    task_info: TaskInfo
    image_info: ImageInfo
    planned_input_scales: PlannedInputScales
    compute_profile_plan_file: str
    execution_profile_plan_file: str
    capability_report: CapabilityReport


def _apply_profiler_plan_file(
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


def _prepare_runtime(args, *, run_state, task_info, output_dir, cpu_list, mem_list,
                     gpu_list, run_command, cgroup_version, cgroup_collection_mode,
                     rapl_topology, preflight_measurements, latency_slo,
                     workflow=None) -> _PreparedRuntime:
    """Build or restore the runtime and persist evidence before the matrix."""
    from acprof.host.collection_workflow import PreparationWorkflow
    from acprof.host.input_plan import input_plan_summary, plan_input_scales
    from acprof.host.runtime_images import prepare_image
    workflow = workflow or PreparationWorkflow()
    from acprof.host.static_metadata import (
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
        from acprof.host.interface_probe import probe_interface
        from acprof.host.runtime_images import configure_runtime_profile
        workflow.run("interface", configure_runtime_profile, task_info)
        workflow.run("interface", probe_interface, task_info, output_dir,
                     timeout_seconds=args.request_timeout_seconds)
        try:
            image_info = workflow.run("image", prepare_image,
                task_info, PROJECT_DIR, reuse_existing=not (args.rebuild_image or workflow.rebuild_environment),
            )
        except (RuntimeError, OSError) as exc:
            print(f"\n[build][ERROR] {exc}", file=sys.stderr)
            sys.exit(1)

        os.makedirs(output_dir, exist_ok=True)
        if task_info.model_resolution:
            from acprof.model_contract import write_model_resolution
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
            gpu_device=selected_gpu_device(),
        )
        capability_report = measurement_report(
            args.profiling_mode, gpu_modes=gpu_list, compute_tool=args.compute_profile_tool,
            execution_tool=args.execution_profile_tool,
            dram_energy=args.dram_energy,
            rapl_topology=rapl_topology,
        )
        capability_report.measurement.update({name: item for name, item in preflight_measurements.items() if isinstance(item, Capability)})
        from acprof.extensions import select_extension
        apply_extension(capability_report, select_extension(task_info))
        static_meta = enrich_static_meta(static_meta, {
            "capability_report": capability_report.to_dict(), "latency_slo": latency_slo,
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
            sys.exit(1)

        workflow.emit("input", "passed", input_plan=input_plan_summary(task_info, planned_input_scales))

        static_meta = enrich_static_meta_from_input_plan(
            static_meta,
            planned_input_scales,
        )
        write_static_meta_json(static_meta, static_meta_json)
        compute_profile_plan_file = ""
        from acprof.host.runtime_validation import validate_runtime
        from acprof.host.static_metadata import enrich_static_meta
        def validate_selected_runtime():
            try:
                report = validate_runtime(
                    task_info=task_info, image_info=image_info, planned=planned_input_scales,
                    cpu_list=cpu_list, mem_list=mem_list, gpu_list=gpu_list, output_dir=output_dir,
                    timeout_seconds=args.request_timeout_seconds,
                )
                if report.get("status") != "ok":
                    raise RuntimeError("collection requires successful runtime validation on every requested device: "
                                       + str(report.get("status")))
            except Exception:
                if workflow.interactive:
                    from acprof.host.collection_workflow import retain_validation_failure
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
            sys.exit(1)
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
            try:
                from acprof.host.compute_profile import collect_compute_profile_plan

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
            except Exception as exc:
                print(f"[compute][WARN] Compute profiling unavailable: {exc}")

        execution_profile_plan_file = ""
        if args.execution_profile_tool == "none":
            print(
                "[execution-profile] Massif/Nsight Systems profiling disabled "
                "(enable with --execution-profile-tool)"
            )
        else:
            try:
                from acprof.host.execution_profile import (
                    collect_execution_profile_plan,
                )

                execution_profile_plan_file = collect_execution_profile_plan(
                    task_info=task_info,
                    image_tag=image_info.tag,
                    cpu_list=cpu_list,
                    mem_list=mem_list,
                    gpu_list=gpu_list,
                    output_dir=output_dir,
                    input_scale_plan_file=planned_input_scales.plan_file,
                    project_dir=PROJECT_DIR,
                    tool_mode=args.execution_profile_tool,
                    massif_sampling=args.massif_sampling,
                    massif_reference_cpu=args.massif_reference_cpu,
                    massif_reference_mem=args.massif_reference_mem,
                    massif_repeat=args.massif_repeat,
                    nsys_sampling=args.nsys_sampling,
                    nsys_reference_cpu=args.nsys_reference_cpu,
                    nsys_reference_mem=args.nsys_reference_mem,
                    nsys_repeat=args.nsys_repeat,
                    nsys_root=args.nsys_root,
                    keep_profiles=args.keep_execution_profiles,
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
            except Exception as exc:
                print(
                    "[execution-profile][WARN] Execution profiling unavailable: "
                    f"{exc}"
                )

        for plan_path, source in ((compute_profile_plan_file, "compute_profile_plan"),
                                  (execution_profile_plan_file, "execution_profile_plan")):
            _apply_profiler_plan_file(capability_report, plan_path, source=source)
        static_meta = enrich_static_meta(static_meta, {"capability_report": capability_report.to_dict()})
        write_static_meta_json(static_meta, static_meta_json)
        from acprof.artifacts import atomic_write_json
        atomic_write_json(layout.path("capability_report.json"), capability_report.to_dict())
        if not capability_report.to_dict()["requested_measurements_available"]:
            print("[capability][WARN] 所请求的 profiler 尚有缺失或失败；详见 capability_report.json，不视为完整画像。")
        run_state.bind_runtime(task_info, image_info, planned_input_scales,
                               compute_profile_plan_file, execution_profile_plan_file)

    return _PreparedRuntime(task_info, image_info, planned_input_scales,
                            compute_profile_plan_file, execution_profile_plan_file,
                            capability_report)


@gpu_device_scope()
def _run_main(*, args=None, prepared_task=None, preparation_artifacts=None):
    global _ACTIVE_TMUX_TERMINAL_LOG, _ACTIVE_RUN_STATE

    start_time = time.perf_counter()
    parser = _build_parser(default_notify_provider=DEFAULT_NOTIFY_PROVIDER)

    args = parser.parse_args() if args is None else args
    from acprof.cli.download_args import apply_download_arguments
    apply_download_arguments(args)
    bootstrap_project_env(Path.cwd())
    try:
        latency_slo_rules = parse_latency_slo_rules(
            args.latency_slo, environment_threshold=os.environ.get("SLOW_LATENCY_THRESHOLD_S"),
        )
    except ValueError as exc:
        parser.error(str(exc))
    from acprof.monitors.rapl_topology import discover_rapl_topology, dram_policy
    try:
        dram_enabled = dram_policy(args.profiling_mode, args.dram_energy)
    except ValueError as exc:
        parser.error(str(exc))
    from acprof.cpu_affinity import normalize_cpu_set, parse_cpu_set
    try:
        args.cpuset_cpus = normalize_cpu_set(args.cpuset_cpus)
        if args.cpuset_cpus and not parse_cpu_set(args.cpuset_cpus) <= set(os.sched_getaffinity(0)):
            raise ValueError("--cpuset-cpus includes CPUs unavailable to this process")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    run_command = format_run_command(sys.argv)
    if args.repeat_in_window < 0:
        parser.error("--repeat-in-window must be >= 0")
    if args.repeat_window_seconds <= 0.0:
        parser.error("--repeat-window-seconds must be > 0")
    if (
        args.request_timeout_seconds <= 0.0
        or not math.isfinite(args.request_timeout_seconds)
    ):
        parser.error("--request-timeout-seconds must be a finite value > 0")
    if args.torch_profiler_repeat <= 0:
        parser.error("--torch-profiler-repeat must be > 0")
    if args.ncu_repeat <= 0:
        parser.error("--ncu-repeat must be > 0")
    if args.massif_repeat <= 0:
        parser.error("--massif-repeat must be > 0")
    if args.nsys_repeat <= 0:
        parser.error("--nsys-repeat must be > 0")
    for option, value in (
        ("--massif-reference-cpu", args.massif_reference_cpu),
        ("--massif-reference-mem", args.massif_reference_mem),
        ("--nsys-reference-cpu", args.nsys_reference_cpu),
        ("--nsys-reference-mem", args.nsys_reference_mem),
    ):
        if value is not None and value <= 0:
            parser.error(f"{option} must be > 0")

    from dataclasses import asdict

    from acprof.experiment import RunConfig
    try:
        common = RunConfig.from_namespace(args).validate(project_dir=Path.cwd())
    except ValueError as exc:
        parser.error(str(exc))
    for name, value in asdict(common).items():
        setattr(args, name, value)

    require_collection_host(profiling_mode=args.profiling_mode,
                            compute_tool=args.compute_profile_tool,
                            execution_tool=args.execution_profile_tool, dram_energy=args.dram_energy)

    terminal_output_dir = os.path.join(
        os.getcwd(),
        args.output_dir,
        args.model.replace("/", "--"),
    )
    notices.activate(
        provider=args.notify,
        model_id=args.model,
        output_dir=terminal_output_dir,
        started_at=start_time,
        run_command=run_command,
    )
    notices.notify_started()

    # ── Step 1: Detect task ──
    print("=" * 60)
    print("AC-Prof")
    print("=" * 60)

    from acprof.host.collection_workflow import PreparationWorkflow, RebuildEnvironment
    workflow = PreparationWorkflow.from_environment(args.output_dir)

    saved_run = load_run_state(terminal_output_dir) if args.resume else {}
    if saved_run.get("runtime"):
        from acprof.host.detect import TaskInfo
        initial_task = TaskInfo(**saved_run["runtime"]["task"])
    else:
        initial_task = prepared_task
    task_info = workflow.resolve(args, initial=initial_task)

    def host_preflight():
        require_native_docker()
        version = require_cgroup_prerequisites()
        from acprof.platform import detect_environment
        topology = discover_rapl_topology() if detect_environment().native else {}
        measurements = {}
        if dram_enabled and args.dram_energy == "required" and topology["dram_status"] != "available":
            raise RuntimeError(f"required DRAM RAPL unavailable: {topology['dram_status']}; "
                               f"missing packages={topology['dram_missing_packages']}")
        if measurement_requested(args.profiling_mode, "packet_latency"):
            require_packet_latency_prerequisites(sniff_iface=args.sniff_iface)
        if measurement_requested(args.profiling_mode, "cpu_energy"):
            measurements["cpu_energy"] = require_cpu_energy_prerequisites()
        if measurement_requested(args.profiling_mode, "cpu_instructions"):
            measurements["cpu_instructions"] = require_mips_prerequisites()
        if "on" in [part.strip().lower() for part in args.gpus.split(",")]:
            pin_gpu_device(args.gpu_device)
        return version, topology, measurements

    cgroup_version, rapl_topology, preflight_measurements = workflow.run("preflight", host_preflight)
    from acprof.platform import detect_environment
    collection_environment = detect_environment()
    cgroup_collection_mode = "strict_v2" if collection_environment.native else "wsl2_v2"
    print(f"  Environment: {collection_environment.label}; Native benchmark: {collection_environment.native}")
    from acprof.runtime_profiles import select_runtime_profile
    latency_slo = resolve_latency_slo(
        latency_slo_rules, pipeline_tag=task_info.pipeline_tag,
        runtime_profile_id=select_runtime_profile(task_info).profile_id,
    )

    print(f"\n  Model:    {task_info.model_id}")
    print(f"  Task:     {task_info.pipeline_tag} (family={task_info.task_family})")
    print(f"  Backend:  {task_info.runtime_backend}")
    print(f"  Library:  {task_info.library_name}")
    print(f"  Revision: {task_info.model_revision}")
    print(f"  Detected: {task_info.detection_method}")
    if task_info.model_resolution:
        resolution = task_info.model_resolution
        print(f"  Interface: {resolution['loader']} / {resolution['artifact_format']} (candidate)")
        print(f"  Runtime: {task_info.runtime_profile_id}")
    print(f"  Cgroup:   {cgroup_version} (mode={cgroup_collection_mode})")
    print(f"  Profiling mode: {args.profiling_mode}")

    output_dir = os.path.join(
        os.getcwd(),
        args.output_dir,
        task_info.model_id.replace("/", "--"),
    )
    try:
        require_result_environment(output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    require_result_cgroup_compatibility(
        output_dir,
        cgroup_version=cgroup_version,
    )
    run_state = RunState(output_dir, run_options(args), resume=args.resume, project_dir=PROJECT_DIR,
                         **({"preparation_artifacts": preparation_artifacts} if preparation_artifacts else {}))
    _ACTIVE_RUN_STATE = run_state
    if run_state.complete:
        print(f"[resume] 实验已经完成：{run_state.layout.result_csv}")
        return
    _ACTIVE_TMUX_TERMINAL_LOG = start_terminal_log(output_dir, sys.argv)

    from acprof.host.input_plan import serialize_input_scales
    from acprof.host.orchestrator import merge_all_csvs, run_matrix

    cpu_list, mem_list, gpu_list = _resource_matrix(args, parser)
    prepared = None
    while prepared is None:
        try:
            prepared = _prepare_runtime(
                args, run_state=run_state, task_info=task_info, output_dir=output_dir,
                cpu_list=cpu_list, mem_list=mem_list, gpu_list=gpu_list,
                run_command=run_command, cgroup_version=cgroup_version,
                cgroup_collection_mode=cgroup_collection_mode, rapl_topology=rapl_topology,
                preflight_measurements=preflight_measurements, latency_slo=latency_slo,
                workflow=workflow,
            )
        except RebuildEnvironment:
            # No measurement has run. Preserve model/host decisions, invalidate
            # all downstream evidence affected by a rebuilt immutable image.
            continue
    task_info, image_info = prepared.task_info, prepared.image_info
    planned_input_scales = prepared.planned_input_scales
    compute_profile_plan_file = prepared.compute_profile_plan_file
    execution_profile_plan_file = prepared.execution_profile_plan_file
    capability_report = prepared.capability_report
    input_scales_arg = serialize_input_scales(planned_input_scales.scales)
    layout = run_state.layout
    static_meta_json = str(layout.path("static_meta.json"))
    collection_history_json = str(layout.path(COLLECTION_HISTORY_NAME))
    total_cases = len(cpu_list) * len(mem_list) * len(gpu_list)

    n_scales = len(planned_input_scales.scales)
    total_iters = total_cases * n_scales * (args.warmup + args.repeat)

    print(f"\n  Resource matrix: {len(cpu_list)} CPUs x {len(mem_list)} MEMs x {len(gpu_list)} GPUs = {total_cases} cases")
    print(f"  Input scales: {n_scales} levels")
    print(f"  Scale source: {planned_input_scales.source}")
    print(f"  Validated scales: {input_scales_arg}")
    print(f"  Iterations per case: {args.warmup} warmup + {args.repeat} repeat")
    print(
        "  Startup OOM pruning: "
        + (
            f"enabled (reference CPU={min(cpu_list)}, startup-only)"
            if args.prune_startup_oom
            else "disabled"
        )
    )
    if args.repeat_in_window > 0:
        repeat_desc = str(args.repeat_in_window)
    else:
        repeat_desc = f"auto target {args.repeat_window_seconds:.1f}s"
    print(f"  Requests per iteration: {repeat_desc}")
    print(f"  Request timeout: {args.request_timeout_seconds:g}s connect/read inactivity per /predict (not total deadline)")
    print(f"  Total iterations: {total_iters}")
    print(f"  Output: {output_dir}")
    print()

    try:
        csv_paths = run_matrix(
            task_info=task_info,
            image_info=image_info,
            cpu_list=cpu_list,
            mem_list=mem_list,
            gpu_list=gpu_list,
            output_dir=output_dir,
            project_dir=PROJECT_DIR,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repeat=args.repeat,
            repeat_in_window=args.repeat_in_window,
            repeat_window_seconds=args.repeat_window_seconds,
            request_timeout_seconds=args.request_timeout_seconds,
            sample_hz=args.sample_hz,
            idle_seconds=args.idle_seconds,
            idle_cooldown_seconds=args.idle_cooldown_seconds,
            idle_debug=args.idle_debug,
            sniff_iface=args.sniff_iface,
            input_scales=input_scales_arg,
            input_scale_plan_file=planned_input_scales.plan_file,
            compute_profile_plan_file=compute_profile_plan_file,
            execution_profile_plan_file=execution_profile_plan_file,
            progress_callback=(
                notices.notify_case
                if notices.active()
                else None
            ),
            prune_startup_oom=args.prune_startup_oom,
            matrix_order=args.matrix_order,
            matrix_seed=args.matrix_seed,
            dram_energy=args.dram_energy,
            cpuset_cpus=args.cpuset_cpus,
            run_state=run_state,
            profiling_mode=args.profiling_mode,
        )
    except PacketLatencyError as exc:
        print(f"\n[sniff][ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except EnergyProfilingError as exc:
        print(f"\n[energy][ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except MIPSProfilingError as exc:
        print(f"\n[mips][ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        # run_matrix has unwound all measurement/container scopes. Publish
        # evidence for completed and interrupted cases before any CSV cleanup.
        from acprof.failures import collect_failures
        from acprof.quality import collect_quality
        recorded_cases = [run_state.artifact_path(name) for name in run_state.data["cases"]]
        collect_quality(Path(output_dir), recorded_cases)
        collect_failures(Path(output_dir), recorded_cases)

    # ── Step 5: Merge all CSVs ──
    if csv_paths:
        final_csv = str(layout.result_csv)
        merge_all_csvs(csv_paths, final_csv, expected=run_state.expected())
        from acprof.result_csv import read_result_csv
        _, collected_rows = read_result_csv(final_csv)
        apply_collection_result(capability_report, collected_rows)
        from acprof.artifacts import atomic_write_json
        # static_meta is the immutable pre-matrix snapshot used by resume.
        # Publish final measurement evidence in its dedicated sidecar.
        atomic_write_json(layout.path("capability_report.json"), capability_report.to_dict())
        missing = missing_required_measurements(capability_report, collected_rows)
        if missing:
            print(f"[capability][ERROR] {args.profiling_mode} 必需指标缺少有效测量：{', '.join(missing)}。"
                  "能力报告和 CSV 已保留；本次画像失败。", file=sys.stderr)
            sys.exit(1)
        run_state.finish(final_csv)
        cleanup_intermediate_results(csv_paths, output_dir, final_csv)
        elapsed = _format_elapsed(time.perf_counter() - start_time)
        print(f"\n{'='*60}")
        print("Profiling complete!")
        print(f"  Profiling mode:   {args.profiling_mode}")
        if not capability_report.to_dict()["requested_measurements_complete"]:
            print("  [WARN] 请求指标存在缺失；能力报告保留具体状态，结果不标记为完整画像。")
        print(f"  Static meta:      {static_meta_json}")
        print(f"  Collection log:   {collection_history_json}")
        print(f"  Frozen matrix:    {layout.path('matrix_plan.json')}")
        if args.prune_startup_oom:
            print(
                "  Startup evidence: "
                f"{layout.path('startup_oom_pruning.json')}"
            )
        print(f"  Merged results:   {final_csv}")
        print(f"  Total elapsed:    {elapsed}")
        print("  Intermediate files from this run were cleaned up.")
        print(f"{'='*60}")
        notices.record_completion(
            final_csv=final_csv,
            completed_cases=len(csv_paths),
        )
    else:
        elapsed = _format_elapsed(time.perf_counter() - start_time)
        print(f"\n[WARN] No results produced after {elapsed}. Static meta is still available: {static_meta_json}")
        notices.record_completion(final_csv=None, completed_cases=0)


def _run_with_cleanup(*, args=None, prepared_task=None, preparation_artifacts=None):
    """Run profiling, finalize terminal logging, then notify best-effort."""
    global _ACTIVE_TMUX_TERMINAL_LOG, _ACTIVE_RUN_STATE

    _ACTIVE_TMUX_TERMINAL_LOG = None
    notices.reset()
    _ACTIVE_RUN_STATE = None
    try:
        return _run_main(args=args, prepared_task=prepared_task, preparation_artifacts=preparation_artifacts) if args is not None else _run_main()
    except (TaskSupportError, RunStateError) as exc:
        print(str(exc), file=sys.stderr)
        notices.record_termination("failed", str(exc))
        raise SystemExit(2) from None
    except KeyboardInterrupt:
        notices.record_termination("cancelled", "用户中断了采集")
        raise
    except SystemExit as exc:
        if exc.code not in (None, 0):
            notices.record_termination(
                "failed",
                f"采集命令以退出码 {exc.code!r} 结束",
            )
        raise
    except Exception as exc:
        detail = str(exc).strip()
        if detail:
            detail = f"{type(exc).__name__}: {detail}"
        else:
            detail = type(exc).__name__
        notices.record_termination("failed", detail)
        raise
    finally:
        terminal_log = _ACTIVE_TMUX_TERMINAL_LOG
        _ACTIVE_TMUX_TERMINAL_LOG = None
        finalized_log_path = None
        if terminal_log is not None:
            if stop_terminal_log(terminal_log):
                finalized_log_path = terminal_log[2]
        state = _ACTIVE_RUN_STATE
        _ACTIVE_RUN_STATE = None
        if state is not None:
            try:
                state.close("failed" if sys.exc_info()[0] else "interrupted")
            except OSError as exc:
                print(f"[resume][ERROR] 无法保存退出状态：{exc}", file=sys.stderr)
        notices.deliver(finalized_log_path)
        notices.reset()


def main(*, args=None, prepared_task=None, preparation_artifacts=None):
    """Translate SIGTERM to cancellation so container and result cleanup can unwind."""
    options = dict(args=args, prepared_task=prepared_task, preparation_artifacts=preparation_artifacts)
    if threading.current_thread() is not threading.main_thread():
        return _run_with_cleanup(**options)

    def interrupt(_signum, _frame):
        # A second TERM must not interrupt the cleanup that the first one started.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        return _run_with_cleanup(**options)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
