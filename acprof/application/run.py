"""Collect one experiment using instance-owned lifecycle and injected boundaries."""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

from acprof.application.errors import CollectionFailed
from acprof.application.notifications import RunNotifications
from acprof.application.preparation import prepare_runtime
from acprof.application.resources import format_elapsed, resource_matrix
from acprof.application.terminal_log import (
    format_run_command,
    start_terminal_log,
    stop_terminal_log,
)
from acprof.capabilities import (
    apply_collection_result,
    measurement_requested,
    missing_required_measurements,
)
from acprof.experiment import RunConfig
from acprof.gpu_mode import normalize_gpu_mode
from acprof.host.cases import CaseMeasurement, CasePlans, CaseWorkload, MatrixRequest
from acprof.host.collection_history import (
    COLLECTION_HISTORY_NAME,
)
from acprof.host.env_utils import bootstrap_project_env
from acprof.host.gpu_device import gpu_device_scope, pin_gpu_device
from acprof.host.measurement.capture import (
    PacketLatencyError,
    require_packet_latency_prerequisites,
)
from acprof.host.orchestrator import (
    EnergyProfilingError,
    MIPSProfilingError,
)
from acprof.host.preflight import (
    require_cgroup_prerequisites,
    require_collection_host,
    require_cpu_energy_prerequisites,
    require_mips_prerequisites,
    require_result_cgroup_compatibility,
    require_result_environment,
)
from acprof.host.result_cleanup import cleanup_intermediate_results
from acprof.host.run_state import RunState, load_run_state, run_options
from acprof.installation import resource_root
from acprof.latency_slo import parse_latency_slo_rules, resolve_latency_slo
from acprof.progress_events import emit_event

PROJECT_DIR = str(resource_root())


@dataclass(frozen=True)
class RunOutcome:
    status: str
    final_csv: str | None = None
    completed_cases: int = 0
    task_collection_path: str | None = None


def configure_environment(options) -> None:
    """Apply explicit transport values before loading the configured environment."""
    from acprof.application.download import apply_download_arguments
    apply_download_arguments(options)
    bootstrap_project_env(Path.cwd())


class RunApplication:
    """One owner for preparation, collection, evidence and finalization.

    CLI owns argv, signals and exit codes. TUI continues to run that CLI in its
    isolated child; API callers can inject expensive boundaries without argv or
    process-global mutable run state. Concurrent measurements remain prohibited
    by the existing host locks.
    """

    def __init__(self, *, notifications=None, workflow_factory=None,
                 state_factory=None, runtime_preparer=None, matrix_runner=None,
                 terminal_start=None, terminal_stop=None, environment_setup=None) -> None:
        self.notifications = notifications if notifications is not None else RunNotifications()
        self.workflow_factory = workflow_factory
        self.state_factory = state_factory or RunState
        self.runtime_preparer = runtime_preparer or prepare_runtime
        self.matrix_runner = matrix_runner
        self.terminal_start = terminal_start or start_terminal_log
        self.terminal_stop = terminal_stop or stop_terminal_log
        self.environment_setup = environment_setup or configure_environment
        self.run_state: RunState | None = None
        self.terminal_log: tuple[str, str, str] | None = None
        self._running = False

    def run(self, config: RunConfig, *, prepared_task=None, preparation_artifacts=None,
            command: str = "") -> RunOutcome:
        if self._running:
            raise RuntimeError("run application is already occupied")
        self._running = True
        self.notifications.reset()
        outcome = None
        primary_error: BaseException | None = None
        terminal_status = "failed"
        try:
            if not command:
                from acprof.run_args import arguments_from_options
                command = format_run_command(["acprof run", *arguments_from_options(vars(config.to_namespace()))])
            outcome = self._execute(config, prepared_task=prepared_task,
                                    preparation_artifacts=preparation_artifacts, command=command)
            terminal_status = outcome.status
            return outcome
        except KeyboardInterrupt as exc:
            primary_error = exc
            terminal_status = "cancelled"
            self.notifications.record_termination("cancelled", "用户中断了采集")
            raise
        except BaseException as exc:
            primary_error = exc
            self.notifications.record_termination("failed", f"{type(exc).__name__}: {exc}")
            raise
        finally:
            try:
                self._finalize(terminal_status)
            except BaseException as cleanup_error:
                terminal_status = "failed"
                if primary_error is None:
                    raise
                primary_error.add_note(f"Run finalization also failed: {type(cleanup_error).__name__}")
            finally:
                self._running = False
                emit_event("run_finished", status=terminal_status)

    def _finalize(self, status: str) -> None:
        terminal_log, self.terminal_log = self.terminal_log, None
        state, self.run_state = self.run_state, None
        finalized_log_path = None
        try:
            if terminal_log is not None and self.terminal_stop(terminal_log):
                finalized_log_path = terminal_log[2]
        except BaseException as exc:
            self.notifications.record_termination("failed", f"{type(exc).__name__}: {exc}")
            raise
        finally:
            try:
                if state is not None:
                    state.close("interrupted" if status in {"cancelled", "no_results"} else "failed")
            except BaseException as exc:
                self.notifications.record_termination("failed", f"{type(exc).__name__}: {exc}")
                raise
            finally:
                try:
                    self.notifications.deliver(finalized_log_path)
                finally:
                    self.notifications.reset()

    def _execute_task_child(self, child_config, child_task, *, command: str) -> RunOutcome:
        status = "failed"
        primary_error: BaseException | None = None
        try:
            result = self._execute(child_config, prepared_task=child_task, command=command, task_child=True)
            status = result.status
            return result
        except BaseException as exc:
            primary_error = exc
            if isinstance(exc, KeyboardInterrupt):
                status = "cancelled"
            raise
        finally:
            try:
                self._finalize(status)
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(f"Task finalization also failed: {type(cleanup_error).__name__}")

    def _run_host_preflight(self, args, *, dram_enabled: bool, workflow):
        def host_preflight():
            if args.runtime == "process":
                from acprof.host.run_state import MeasurementLock
                from acprof.host.runtimes.gpu import process_gpu_device
                from acprof.host.runtimes.process_control import require_process_supervision
                from acprof.host.runtimes.resources import preflight_resources
                with MeasurementLock():
                    require_process_supervision()
                    cpus, mems, gpus = resource_matrix(args)
                    preflight_resources(cpus, mems, args.cpuset_cpus)
                    args.process_gpu = process_gpu_device(args.gpu_device) if "on" in gpus else {}
                return "not_used", {}, {}
            version = require_cgroup_prerequisites()
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
            if "on" in [normalize_gpu_mode(part) for part in args.gpus.split(",")]:
                pin_gpu_device(args.gpu_device)
            return version, topology, measurements

        return workflow.run("preflight", host_preflight)

    def _publish_matrix_results(self, csv_paths: list[str], *, args, output_dir: str, layout,
                                run_state: RunState, capability_report, start_time: float,
                                static_meta_json: str, collection_history_json: str) -> RunOutcome:
        from acprof.host.orchestrator import merge_all_csvs

        if csv_paths:
            final_csv = str(layout.result_csv)
            emit_event("results_merging")
            merge_all_csvs(csv_paths, final_csv, expected=run_state.expected())
            from acprof.result_csv import read_result_csv
            _, collected_rows = read_result_csv(final_csv)
            apply_collection_result(capability_report, collected_rows)
            from acprof.artifacts import atomic_write_json
            atomic_write_json(layout.path("capability_report.json"), capability_report.to_dict())
            missing = missing_required_measurements(capability_report, collected_rows)
            if missing:
                print(f"[capability][ERROR] {args.profiling_mode} 必需指标缺少有效测量：{', '.join(missing)}。"
                      "能力报告和 CSV 已保留；本次画像失败。", file=sys.stderr)
                raise CollectionFailed("required measurements missing: " + ", ".join(missing))
            run_state.finish(final_csv)
            emit_event("result_published", final_csv=final_csv)
            cleanup_intermediate_results(csv_paths, output_dir, final_csv)
            elapsed = format_elapsed(time.perf_counter() - start_time)
            print(f"\n{'='*60}")
            print("Profiling complete!")
            print(f"  Profiling mode:   {args.profiling_mode}")
            if not capability_report.to_dict()["requested_measurements_complete"]:
                print("  [WARN] 请求指标存在缺失；能力报告保留具体状态，结果不标记为完整画像。")
            print(f"  Static meta:      {static_meta_json}")
            print(f"  Collection log:   {collection_history_json}")
            print(f"  Frozen matrix:    {layout.path('matrix_plan.json')}")
            if args.prune_startup_oom:
                print(f"  Startup evidence: {layout.path('startup_oom_pruning.json')}")
            print(f"  Merged results:   {final_csv}")
            print(f"  Total elapsed:    {elapsed}")
            print("  Intermediate files from this run were cleaned up.")
            print(f"{'='*60}")
            self.notifications.record_completion(
                final_csv=final_csv,
                completed_cases=len(csv_paths),
            )
            return RunOutcome("ok", final_csv, len(csv_paths))

        elapsed = format_elapsed(time.perf_counter() - start_time)
        print(f"\n[WARN] No results produced after {elapsed}. Static meta is still available: {static_meta_json}")
        self.notifications.record_completion(final_csv=None, completed_cases=0)
        return RunOutcome("no_results")

    def _collect_matrix(self, args, *, task_info, prepared, cpu_list, mem_list, gpu_list,
                        output_dir: str, run_state: RunState) -> list[str]:
        from acprof.host.input_plan import serialize_input_scales
        from acprof.host.orchestrator import run_matrix

        try:
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
                    dram_energy=args.dram_energy,
                ),
                plans=CasePlans(
                    input_scales=serialize_input_scales(prepared.planned_input_scales.scales),
                    input_scale_plan_file=prepared.planned_input_scales.plan_file,
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
            print(f"\n[sniff][ERROR] {exc}", file=sys.stderr)
            raise CollectionFailed(str(exc)) from exc
        except EnergyProfilingError as exc:
            print(f"\n[energy][ERROR] {exc}", file=sys.stderr)
            raise CollectionFailed(str(exc)) from exc
        except MIPSProfilingError as exc:
            print(f"\n[mips][ERROR] {exc}", file=sys.stderr)
            raise CollectionFailed(str(exc)) from exc
        finally:
            from acprof.failures import collect_failures
            from acprof.quality import collect_quality
            recorded_cases = [run_state.artifact_path(name) for name in run_state.data["cases"]]
            collect_quality(Path(output_dir), recorded_cases)
            collect_failures(Path(output_dir), recorded_cases)

    def _validate_execution_options(self, config: RunConfig):
        args = config.validate(project_dir=Path.cwd()).to_namespace()
        self.environment_setup(args)
        latency_slo_rules = parse_latency_slo_rules(
            args.latency_slo, environment_threshold=os.environ.get("SLOW_LATENCY_THRESHOLD_S"),
        )
        from acprof.cpu_affinity import parse_cpu_set
        from acprof.monitors.rapl_topology import dram_policy

        dram_enabled = dram_policy(args.profiling_mode, args.dram_energy)
        if args.cpuset_cpus and not parse_cpu_set(args.cpuset_cpus) <= set(os.sched_getaffinity(0)):
            raise ValueError("--cpuset-cpus includes CPUs unavailable to this process")
        require_collection_host(profiling_mode=args.profiling_mode,
                                compute_tool=args.compute_profile_tool,
                                execution_tool=args.execution_profile_tool, dram_energy=args.dram_energy)
        return args, latency_slo_rules, dram_enabled

    @gpu_device_scope()
    def _execute(self, config: RunConfig, *, prepared_task=None,
                 preparation_artifacts=None, command: str, task_child: bool = False) -> RunOutcome:
        start_time = time.perf_counter()
        args, latency_slo_rules, dram_enabled = self._validate_execution_options(config)
        run_command = command
        terminal_output_dir = str(config.result_dir(Path.cwd()))

        # ── Step 1: Detect task ──
        print("=" * 60)
        print("AC-Prof")
        print("=" * 60)

        from acprof.host.preparation.collection_workflow import (
            PreparationWorkflow,
            RebuildEnvironment,
        )
        workflow = self.workflow_factory(args.output_dir) if self.workflow_factory else PreparationWorkflow.from_environment(args.output_dir)
        workflow.continue_on_task_failure = task_child

        from acprof.artifact_layout import ArtifactLayout
        has_run_state = ArtifactLayout.discover(terminal_output_dir).path("run_state.json").exists()
        saved_run = load_run_state(terminal_output_dir) if args.resume and has_run_state else {}
        collection = None
        if args.resume and not saved_run and not task_child:
            from acprof.application.task_runs import TaskCollection
            collection = TaskCollection.restore(config)
            prepared_task = collection.task
        from acprof.host.runtimes.selection import require_runtime_requests, select_runtime
        requested_runtime = args.runtime
        if args.resume and requested_runtime == "auto":
            requested_runtime = saved_run.get("runtime", {}).get("kind", "docker")
        selection = select_runtime(requested_runtime, profiling_mode=args.profiling_mode)
        args.runtime = selection.selected
        args.runtime_selection = selection.to_dict()
        require_runtime_requests(args)
        if args.runtime == "docker":
            args.cpus = "1,2,4,8" if args.cpus == "auto" else args.cpus
            args.mems = "2,4,8,16" if args.mems == "auto" else args.mems
        elif args.prune_startup_oom:
            print("[runtime] Startup OOM pruning is Docker-only; Process cases run independently")
            args.prune_startup_oom = False
        print(f"[runtime] {selection.selected}: {selection.reason}")
        if saved_run.get("runtime"):
            from acprof.host.detect import TaskInfo
            initial_task = TaskInfo(**saved_run["runtime"]["task"])
        else:
            initial_task = prepared_task
        task_info = workflow.resolve(args, initial=initial_task)

        if task_info.selected_tasks and not task_child and not (saved_run and args.task_run_id):
            from acprof.application.task_runs import TaskCollection
            collection = collection or TaskCollection(config, task_info)
            result = collection.run(
                lambda child_config, child_task: self._execute_task_child(
                    child_config, child_task, command=command,
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
            args, dram_enabled=dram_enabled, workflow=workflow,
        )
        from acprof.platform import detect_environment
        collection_environment = detect_environment()
        cgroup_collection_mode = ("process_tree" if args.runtime == "process" else
                                  "strict_v2" if collection_environment.native else
                                  "wsl2_v2" if collection_environment.environment == "wsl2" else "guest_v2")
        native_benchmark = collection_environment.native and args.runtime == "docker"
        print(f"  Environment: {collection_environment.label}; Native benchmark: {native_benchmark}")
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

        output_dir = terminal_output_dir
        require_result_environment(output_dir)
        require_result_cgroup_compatibility(
            output_dir,
            cgroup_version=cgroup_version,
        )
        run_state = self.state_factory(output_dir, run_options(args), resume=args.resume, project_dir=PROJECT_DIR,
                             **({"preparation_artifacts": preparation_artifacts} if preparation_artifacts else {}))
        self.run_state = run_state
        if run_state.complete:
            print(f"[resume] 实验已经完成：{run_state.layout.result_csv}")
            emit_event("result_published", final_csv=str(run_state.layout.result_csv))
            return RunOutcome("ok", str(run_state.layout.result_csv))
        self.terminal_log = self.terminal_start(output_dir, command)

        from acprof.host.input_plan import serialize_input_scales

        cpu_list, mem_list, gpu_list = resource_matrix(args)
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
        static_meta_json = str(layout.path("static_meta.json"))
        collection_history_json = str(layout.path(COLLECTION_HISTORY_NAME))
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

        return self._publish_matrix_results(
            csv_paths,
            args=args,
            output_dir=output_dir,
            layout=layout,
            run_state=run_state,
            capability_report=capability_report,
            start_time=start_time,
            static_meta_json=static_meta_json,
            collection_history_json=collection_history_json,
        )
