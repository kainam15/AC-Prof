"""Collect one experiment using instance-owned lifecycle and injected boundaries."""
from __future__ import annotations

import os
import sys
import time
from dataclasses import replace
from pathlib import Path

from acprof.application.errors import CollectionFailed
from acprof.application.notifications import RunNotifications
from acprof.application.outcome import RunOutcome
from acprof.application.preparation import prepare_runtime
from acprof.application.publication import publish_matrix_results, publish_run_reports
from acprof.application.resources import resolve_runtime_options, resource_matrix
from acprof.application.terminal_log import (
    format_run_command,
    start_terminal_log,
    stop_terminal_log,
)
from acprof.capabilities import measurement_requested
from acprof.events.progress import emit_event
from acprof.experiments.configuration import RunConfig
from acprof.gpu_mode import normalize_gpu_mode
from acprof.host.cases import CaseMeasurement, CasePlans, CaseWorkload, MatrixRequest
from acprof.host.env_utils import bootstrap_project_env
from acprof.host.gpu_device import (
    DockerGpuError,
    gpu_device_scope,
    pin_gpu_device,
    require_docker_gpu,
)
from acprof.host.measurement.capture import (
    PacketLatencyError,
    require_packet_latency_prerequisites,
)
from acprof.host.orchestrator import (
    EnergyProfilingError,
    MIPSProfilingError,
)
from acprof.host.preflight import (
    require_available_cpu_set,
    require_cgroup_prerequisites,
    require_collection_host,
    require_cpu_energy_prerequisites,
    require_mips_prerequisites,
    require_result_cgroup_compatibility,
    require_result_environment,
)
from acprof.host.run_state import (
    MeasurementLock,
    RunState,
    load_run_state,
    measurement_lock_scope,
    run_options,
    state_factory_measurement_scope,
)
from acprof.installation import resource_root
from acprof.latency_slo import parse_latency_slo_rules, resolve_latency_slo
from acprof.system.cleanup import cleanup_on_exit, format_exception_notes

PROJECT_DIR = str(resource_root())


def configure_environment(options) -> None:
    """Apply explicit transport values before loading the configured environment."""
    from acprof.application.download import apply_download_arguments
    apply_download_arguments(options)
    bootstrap_project_env(Path.cwd())



def inference_workload(args, cpu_list, mem_list, gpu_list) -> dict:
    """Describe requested workload and measurement axes before collection."""
    from acprof.artifacts import read_json_object
    from acprof.tasks.workloads import inference_workload_inputs

    specification = getattr(args, "workload_spec", "")
    document = read_json_object(Path(specification).expanduser(), label="workload specification") if specification else None
    return {
        **inference_workload_inputs(args.batch_size, args.input_scales, document),
        "input_scale_policy": args.input_scale_policy,
        "resources": {
            "cpus": list(cpu_list), "mems": list(mem_list), "gpus": list(gpu_list),
            "runtime": args.runtime, "cpuset_cpus": args.cpuset_cpus, "gpu_device": args.gpu_device,
        },
        "measurement": {name: getattr(args, name, None) for name in (
            "profiling_mode", "warmup", "repeat", "repeat_in_window", "repeat_window_seconds",
            "request_timeout_seconds", "sample_hz", "sampling_policy", "raw_sample_limit",
            "idle_seconds", "idle_cooldown_seconds", "compute_profile_tool", "execution_profile_tool",
            "matrix_order", "matrix_seed", "sniff_iface", "dram_energy",
        )},
    }


class RunApplication:
    """One owner for preparation, collection, evidence and finalization.

    CLI owns argv, signals and exit codes. TUI continues to run that CLI in its
    isolated child; API callers can inject expensive boundaries without argv or
    process-global mutable run state. Concurrent measurements remain prohibited
    by the existing host locks.
    """

    def __init__(self, *, notifications=None, workflow_factory=None,
                 state_factory=None, runtime_preparer=None, matrix_runner=None,
                 terminal_start=None, terminal_stop=None, environment_setup=None, restore_idle_duration=False) -> None:
        self.notifications = notifications if notifications is not None else RunNotifications()
        self.workflow_factory = workflow_factory
        self.state_factory = state_factory or RunState
        self.runtime_preparer = runtime_preparer or prepare_runtime
        self.matrix_runner = matrix_runner
        self.terminal_start = terminal_start or start_terminal_log
        self.terminal_stop = terminal_stop or stop_terminal_log
        self.environment_setup = environment_setup or configure_environment
        self.restore_idle_duration = restore_idle_duration
        self.run_state: RunState | None = None
        self.terminal_log: tuple[str, str, str] | None = None
        self._running = False

    def run(self, config: RunConfig, *, prepared_task=None, preparation_artifacts=None,
            command: str = "") -> RunOutcome:
        if self._running:
            raise RuntimeError("run application is already occupied")
        self._running = True
        terminal_status = "failed"

        def finalize() -> None:
            nonlocal terminal_status
            try:
                self._finalize(terminal_status)
            except BaseException:
                terminal_status = "failed"
                raise

        def finish() -> None:
            self._running = False
            emit_event("run_finished", status=terminal_status)

        with cleanup_on_exit(finish, description="run completion event"):
            self.notifications.reset()
            with cleanup_on_exit(finalize, description="run finalization"):
                try:
                    if not command:
                        from acprof.experiments.arguments import arguments_from_options
                        command = format_run_command(["acprof run", *arguments_from_options(vars(config.to_namespace()))])
                    outcome = self._execute(config, prepared_task=prepared_task,
                                            preparation_artifacts=preparation_artifacts, command=command)
                    terminal_status = outcome.status
                    return outcome
                except BaseException as exc:
                    terminal_status = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
                    detail = ("用户中断了采集" if terminal_status == "cancelled"
                              else f"{type(exc).__name__}: {format_exception_notes(exc)}")
                    with cleanup_on_exit(
                        lambda: self.notifications.record_termination(terminal_status, detail),
                        description="run failure notification",
                    ):
                        raise

    def _finalize(self, status: str) -> None:
        terminal_log, self.terminal_log = self.terminal_log, None
        state, self.run_state = self.run_state, None
        finalized_log_path = None
        if status == "review_required" and state is None and terminal_log is None:
            self.notifications.reset()
            return

        def close_state() -> None:
            if state is not None:
                state.close("interrupted" if status in {"cancelled", "no_results"} else "failed")

        with cleanup_on_exit(self.notifications.reset, description="notification context"), cleanup_on_exit(
            lambda: self.notifications.deliver(finalized_log_path), description="terminal notification",
        ):
            try:
                with cleanup_on_exit(close_state, description="run state"):
                    if terminal_log is not None and self.terminal_stop(terminal_log):
                        finalized_log_path = terminal_log[2]
            except BaseException as exc:
                detail = f"{type(exc).__name__}: {format_exception_notes(exc)}"
                with cleanup_on_exit(
                    lambda: self.notifications.record_termination("failed", detail),
                    description="finalization failure notification",
                ):
                    raise

    def _execute_task_child(self, child_config, child_task, *, command: str,
                            measurement_lock: MeasurementLock | None = None) -> RunOutcome:
        status = "failed"
        with cleanup_on_exit(lambda: self._finalize(status), description="task finalization"):
            try:
                result = self._execute(child_config, prepared_task=child_task, command=command, task_child=True,
                                       measurement_lock=measurement_lock)
                status = result.status
                return result
            except KeyboardInterrupt:
                status = "cancelled"
                raise

    def _run_host_preflight(self, args, *, dram_enabled: bool, workflow, measurement_lock: MeasurementLock | None = None):
        def host_preflight():
            gpu = {}
            if args.runtime == "process":
                from acprof.host.runtimes.gpu import process_gpu_device
                from acprof.host.runtimes.process_control import require_process_supervision
                from acprof.host.runtimes.resources import preflight_resources
                require_process_supervision()
                cpus, mems, gpus = resource_matrix(args)
                preflight_resources(cpus, mems, args.cpuset_cpus)
                args.process_gpu = process_gpu_device(args.gpu_device) if "on" in gpus else {}
                version, gpu = "not_used", args.process_gpu
                if "on" in gpus and args.profiling_mode == "best_effort" and not gpu.get("uuid"):
                    raise RuntimeError(f"Requested CUDA device unavailable: {gpu.get('error')}")
            else:
                version = require_cgroup_prerequisites()
                if "on" in [normalize_gpu_mode(part) for part in args.gpus.split(",")]:
                    gpu = pin_gpu_device(args.gpu_device)
                    try:
                        gpu_proof = require_docker_gpu(gpu)
                    except DockerGpuError as exc:
                        if exc.reason_code != "probe_image_missing":
                            raise
                        args.runtime_selection["docker_gpu"] = {"status": "pending", "reason": exc.reason_code}
                        print(f"[preflight] {exc.detail}. GPU check will run with the prepared image.")
                    else:
                        args.runtime_selection["docker_gpu"] = {"status": "verified", "detail": gpu_proof}
            if args.profiling_mode == "best_effort" or args.runtime == "process":
                from acprof.host.runtimes.capabilities import probe_collection_plan
                plan = probe_collection_plan(args.runtime, gpu=gpu, profiling_mode=args.profiling_mode,
                                             sniff_iface=args.sniff_iface, dram_energy=args.dram_energy,
                                             service_cgroup=args.cpus != "auto" or args.mems != "auto")
                plan["independent_tools"] = {
                    "torch": args.compute_profile_tool in {"torch", "both"},
                    "ncu": args.compute_profile_tool in {"ncu", "both", "vendor"},
                    "massif": args.execution_profile_tool in {"massif", "both"},
                    "nsys": args.execution_profile_tool in {"nsys", "both"},
                }
                args.collection_plan = plan
                if args.dram_energy == "required" and plan["probes"]["dram_energy"]["status"] != "available":
                    raise RuntimeError(f"Required DRAM RAPL unavailable: {plan['probes']['dram_energy']['detail']}")
                return version, plan["rapl_topology"], {}
            from acprof.monitors.rapl_topology import discover_rapl_topology
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
            return version, topology, measurements

        with measurement_lock_scope(measurement_lock):
            return workflow.run("preflight", host_preflight)

    def _collect_matrix(self, args, *, task_info, prepared, cpu_list, mem_list, gpu_list,
                        output_dir: str, run_state: RunState) -> list[str]:
        from acprof.host.input_plan import serialize_input_scales
        from acprof.host.orchestrator import run_matrix

        def collect_evidence() -> None:
            from acprof.failures import collect_failures
            from acprof.quality import collect_quality
            recorded_cases = list(run_state.case_paths())
            with cleanup_on_exit(lambda: collect_failures(Path(output_dir), recorded_cases),
                                 description="runtime failure evidence"):
                collect_quality(Path(output_dir), recorded_cases)

        try:
            with cleanup_on_exit(collect_evidence, description="collection evidence"):
                return (self.matrix_runner or run_matrix)(MatrixRequest(
                    task_info=task_info,
                    image_info=prepared.image_info,
                    cpu_list=cpu_list,
                    mem_list=mem_list,
                    gpu_list=gpu_list,
                    output_dir=output_dir,
                    project_dir=PROJECT_DIR,
                    workload=CaseWorkload(
                        batch_size=args.batch_size,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        repeat_in_window=args.repeat_in_window,
                        repeat_window_seconds=args.repeat_window_seconds,
                        request_timeout_seconds=args.request_timeout_seconds,
                    ),
                    measurement=CaseMeasurement(
                        sample_hz=args.sample_hz,
                        sampling_policy=args.sampling_policy,
                        raw_sample_limit=args.raw_sample_limit,
                        idle_seconds=args.idle_seconds,
                        idle_cooldown_seconds=args.idle_cooldown_seconds,
                        idle_debug=args.idle_debug,
                        sniff_iface=args.sniff_iface,
                        profiling_mode=args.profiling_mode,
                        collection_plan=getattr(args, "collection_plan", {}),
                        require_packet_latency=args.profiling_mode == "full",
                        dram_energy=args.dram_energy,
                    ),
                    plans=CasePlans(
                        input_scales=serialize_input_scales(prepared.planned_input_scales.scales),
                        input_scale_plan_file=prepared.planned_input_scales.plan_file,
                        input_scale_plan_sha256=prepared.planned_input_scales.plan_sha256,
                        compute_profile_plan_file=prepared.compute_profile_plan_file,
                        execution_profile_plan_file=prepared.execution_profile_plan_file,
                    ),
                    progress_callback=(
                        self.notifications.notify_case
                        if self.notifications.active()
                        else None
                    ),
                    prune_startup_oom=args.prune_startup_oom,
                    run_state=run_state,
                    matrix_order=args.matrix_order,
                    matrix_seed=args.matrix_seed,
                    cpuset_cpus=args.cpuset_cpus,
                ))
        except PacketLatencyError as exc:
            detail = format_exception_notes(exc)
            print(f"\n[sniff][ERROR] {detail}", file=sys.stderr)
            raise CollectionFailed(detail) from exc
        except EnergyProfilingError as exc:
            detail = format_exception_notes(exc)
            print(f"\n[energy][ERROR] {detail}", file=sys.stderr)
            raise CollectionFailed(detail) from exc
        except MIPSProfilingError as exc:
            detail = format_exception_notes(exc)
            print(f"\n[mips][ERROR] {detail}", file=sys.stderr)
            raise CollectionFailed(detail) from exc

    def _validate_execution_options(self, config: RunConfig):
        args = config.validate(project_dir=Path.cwd()).to_namespace()
        self.environment_setup(args)
        latency_slo_rules = parse_latency_slo_rules(
            args.latency_slo, environment_threshold=os.environ.get("SLOW_LATENCY_THRESHOLD_S"),
        )
        from acprof.monitors.rapl_topology import dram_policy

        dram_enabled = dram_policy(args.profiling_mode, args.dram_energy)
        require_available_cpu_set(args.cpuset_cpus)
        require_collection_host(profiling_mode=args.profiling_mode,
                                compute_tool=args.compute_profile_tool,
                                execution_tool=args.execution_profile_tool, dram_energy=args.dram_energy)
        return args, latency_slo_rules, dram_enabled

    def _prepare_inference_configuration(self, task, args, workflow, *, axes, task_child=False,
                                         historical_resume=False):
        from acprof.host.preparation.inference_configuration import prepare_execution_configuration

        return prepare_execution_configuration(
            task, args, workflow, workload=inference_workload(args, *axes),
            task_child=task_child, historical_resume=historical_resume,
        )

    @gpu_device_scope()
    def _execute(self, config: RunConfig, *, prepared_task=None,
                 preparation_artifacts=None, command: str, task_child: bool = False,
                 measurement_lock: MeasurementLock | None = None) -> RunOutcome:
        start_time = time.perf_counter()
        from acprof.results.layout import ArtifactLayout
        terminal_output_dir = str(config.result_dir(Path.cwd()))
        has_run_state = config.resume and ArtifactLayout.discover(terminal_output_dir).path("run_state.json").exists()
        saved_run = load_run_state(terminal_output_dir) if has_run_state else None
        if config.resume:
            if saved_run:
                saved_options = saved_run["options"]
                if not isinstance(saved_options, dict):
                    raise ValueError("Invalid frozen run options")
            else:
                from acprof.application.task_runs import TaskCollection
                _, manifest = TaskCollection.read_manifest(config)
                saved_options = manifest["configuration"]
            frozen_mode = saved_options.get("profiling_mode", "full")
            if self.restore_idle_duration:
                from acprof.config import DEFAULT_IDLE_SECONDS
                config = replace(config, idle_seconds=saved_options.get("idle_seconds",
                                 DEFAULT_IDLE_SECONDS if frozen_mode != "best_effort" else 0.0))
            if config.profiling_mode not in {"best_effort", "auto", frozen_mode}:
                raise ValueError("Resume requires the frozen collection policy")
            config = replace(config, profiling_mode=frozen_mode)
        elif not task_child and config.profiling_mode in {"full", "basic", "auto"}:
            config = replace(config, profiling_mode="best_effort")
        requested_args, latency_slo_rules, dram_enabled = self._validate_execution_options(config)
        run_command = command

        # ── Step 1: Detect task ──
        print("=" * 60)
        print("AC-Prof")
        print("=" * 60)

        from acprof.host.preparation.collection_workflow import (
            PreparationWorkflow,
            RebuildEnvironment,
        )
        workflow = self.workflow_factory(requested_args.output_dir) if self.workflow_factory else PreparationWorkflow.from_environment(requested_args.output_dir)
        workflow.continue_on_task_failure = task_child

        frozen_runtime = saved_run.get("runtime") if saved_run is not None else None
        collection = None
        if requested_args.resume and not saved_run and not task_child:
            from acprof.application.task_runs import TaskCollection
            from acprof.host.detect import TaskInfo
            from acprof.host.preparation.inference_configuration import (
                validate_resume_inference_requests,
            )
            from acprof.models.inference_configuration import validate_configuration_binding

            _path, manifest = TaskCollection.read_manifest(config)
            restored_task = TaskInfo(**manifest["task_snapshot"])
            if restored_task.inference_configuration:
                validate_resume_inference_requests(restored_task.inference_configuration, requested_args)
                validate_configuration_binding(restored_task.inference_configuration, restored_task)
            collection = TaskCollection.restore(config)
            prepared_task = collection.task
        args = resolve_runtime_options(
            config, requested_args,
            recorded_runtime=frozen_runtime.get("kind", "docker") if frozen_runtime is not None else "docker",
        )
        if requested_args.prune_startup_oom and not args.prune_startup_oom:
            print("[runtime] Startup OOM pruning is Docker-only; Process cases run independently")
        print(f"[runtime] {args.runtime}: {args.runtime_selection['reason']}")
        if frozen_runtime is not None:
            from acprof.host.detect import TaskInfo
            task_snapshot = frozen_runtime["task"]
            if not isinstance(task_snapshot, dict):
                raise ValueError("frozen task snapshot must be an object")
            initial_task = TaskInfo(**task_snapshot)
            if initial_task.inference_configuration:
                from acprof.host.preparation.inference_configuration import (
                    validate_resume_inference_requests,
                )
                from acprof.models.inference_configuration import validate_configuration_binding

                validate_resume_inference_requests(initial_task.inference_configuration, requested_args)
                validate_configuration_binding(initial_task.inference_configuration, initial_task, allow_task_subset=True)
        else:
            initial_task = prepared_task
        task_info = workflow.resolve(args, initial=initial_task)

        cpu_list, mem_list, gpu_list = resource_matrix(args)
        from acprof.host.preparation.inference_configuration import (
            InferenceConfigurationReviewRequired,
        )

        try:
            task_info = self._prepare_inference_configuration(
                task_info, args, workflow, axes=(cpu_list, mem_list, gpu_list),
                task_child=task_child or bool(saved_run and args.task_run_id),
                historical_resume=bool(saved_run or collection),
            )
        except InferenceConfigurationReviewRequired as review:
            return RunOutcome("review_required", inference_configuration_review={
                "identity_sha256": review.identity_sha256,
                "candidate_path": str(review.candidate_path), "requests_path": str(review.requests_path),
            })

        if task_info.selected_tasks and not task_child and not (saved_run and args.task_run_id):
            from acprof.application.task_runs import TaskCollection
            collection = collection or TaskCollection(config, task_info)
            with measurement_lock_scope() as collection_lock:
                result = collection.run(
                    lambda child_config, child_task: self._execute_task_child(
                        child_config, child_task, command=command, measurement_lock=collection_lock,
                    )
                )
            if result.status != "ok":
                raise CollectionFailed(f"Some tasks failed; see {collection.path}",
                                       task_collection_path=str(collection.path))
            return result

        if not args.task_run_id:
            terminal_output_dir = str(replace(config, model=task_info.model_id).result_dir(Path.cwd()))
        self.notifications.activate(provider=args.notify, model_id=task_info.model_id,
                                    output_dir=terminal_output_dir, started_at=start_time,
                                    run_command=run_command)
        self.notifications.notify_started()

        cgroup_version, rapl_topology, preflight_measurements = self._run_host_preflight(
            args, dram_enabled=dram_enabled, workflow=workflow, measurement_lock=measurement_lock,
        )
        from acprof.platform import detect_environment
        collection_environment = detect_environment()
        cgroup_collection_mode = ("process_tree" if args.runtime == "process" else
                                  "strict_v2" if collection_environment.native else
                                  "wsl2_v2" if collection_environment.environment == "wsl2" else "guest_v2")
        native_benchmark = collection_environment.native and args.runtime == "docker"
        print(f"  Environment: {collection_environment.label}; Native benchmark: {native_benchmark}")
        from acprof.runtime.profiles import select_runtime_profile
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

        output_dir = terminal_output_dir
        require_result_environment(output_dir)
        require_result_cgroup_compatibility(
            output_dir,
            cgroup_version=cgroup_version,
        )
        with state_factory_measurement_scope(output_dir, measurement_lock):
            run_state = self.state_factory(output_dir, run_options(args), resume=args.resume, project_dir=PROJECT_DIR,
                                           **({"preparation_artifacts": preparation_artifacts} if preparation_artifacts else {}))
        self.run_state = run_state
        if run_state.complete:
            print(f"[resume] 实验已经完成：{run_state.layout.result_csv}")
            publish_run_reports(run_state.layout)
            emit_event("result_published", final_csv=str(run_state.layout.result_csv))
            return RunOutcome("ok", str(run_state.layout.result_csv))
        self.terminal_log = self.terminal_start(output_dir, command)

        from acprof.host.input_plan import serialize_input_scales

        prepared = None
        with run_state.preparation_evidence().capture(workflow):
            while prepared is None:
                try:
                    prepared = self.runtime_preparer(
                        args, run_state=run_state, task_info=task_info, output_dir=output_dir,
                        cpu_list=cpu_list, mem_list=mem_list, gpu_list=gpu_list,
                        run_command=run_command, cgroup_version=cgroup_version,
                        cgroup_collection_mode=cgroup_collection_mode, rapl_topology=rapl_topology,
                        preflight_measurements=preflight_measurements, latency_slo=latency_slo,
                        workflow=workflow, project_dir=PROJECT_DIR, notices=self.notifications,
                    )
                except RebuildEnvironment:
                    # No measurement has run. Preserve model/host decisions, invalidate
                    # all downstream evidence affected by a rebuilt immutable image.
                    continue
        task_info = prepared.task_info
        planned_input_scales = prepared.planned_input_scales
        capability_report = prepared.capability_report
        input_scales_arg = serialize_input_scales(planned_input_scales.scales)
        layout = run_state.layout
        total_cases = len(cpu_list) * len(mem_list) * len(gpu_list)

        n_scales = len(planned_input_scales.scales)
        total_iters = total_cases * n_scales * (args.warmup + args.repeat)

        emit_event("matrix_planned", total_cases=total_cases)
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

        csv_paths = self._collect_matrix(
            args,
            task_info=task_info,
            prepared=prepared,
            cpu_list=cpu_list,
            mem_list=mem_list,
            gpu_list=gpu_list,
            output_dir=output_dir,
            run_state=run_state,
        )

        return publish_matrix_results(
            csv_paths,
            args=args,
            layout=layout,
            run_state=run_state,
            capability_report=capability_report,
            start_time=start_time,
            notifications=self.notifications,
        )
