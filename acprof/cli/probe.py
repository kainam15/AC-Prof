#!/usr/bin/env python3
"""CLI for a one-request largest-input-scale timing probe."""
from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from acprof.artifacts import atomic_write_json, read_json_object_snapshot
from acprof.config import docker_resource_limits
from acprof.cpu_affinity import normalize_cpu_set
from acprof.events.preparation import emit_progress
from acprof.experiments.arguments import add_gpu_device_argument
from acprof.experiments.options import add_option_arguments
from acprof.host.env_utils import bootstrap_project_env
from acprof.host.gpu_device import gpu_device_scope, pin_gpu_device
from acprof.host.images.runtime import prepare_image
from acprof.host.input_plan import plan_input_scales
from acprof.host.largest_scale_probe import (
    create_probe_output_dir,
    run_largest_scale_probe,
    select_minimum_resources,
    write_probe_summary,
)
from acprof.host.preflight import (
    require_available_cpu_set,
    require_cgroup_prerequisites,
    require_collection_host,
    require_native_docker,
)
from acprof.host.preparation.collection_workflow import PreparationInterrupted, PreparationWorkflow
from acprof.host.preparation.inference_configuration import (
    InferenceConfigurationReviewRequired,
    prepare_execution_configuration,
)
from acprof.host.preparation.task_selection import select_tasks
from acprof.host.preparation.task_support import require_task_support
from acprof.installation import resource_root
from acprof.tasks.workloads import (
    inference_workload_inputs,
    task_experiment_overrides,
    workload_manifest,
)

if TYPE_CHECKING:
    from acprof.artifacts import JsonObjectSnapshot
    from acprof.host.detect import TaskInfo

PROJECT_DIR = resource_root()


def _int_list(value: str, label: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{label}必须是逗号分隔的整数"
        ) from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(f"{label}必须全部大于 0")
    return values


def _gpu_list(value: str) -> list[str]:
    values = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not values or any(item not in {"off", "on"} for item in values):
        raise argparse.ArgumentTypeError("GPU 模式只能包含 off 或 on")
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan requested memory caps from low to high, then measure the "
            "first cap that completes one largest-scale cold request."
        )
    )
    add_option_arguments(parser, group=None, purpose="probe")
    add_gpu_device_argument(parser)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help=(
            "Optional timeout for the one /predict request; omitted means "
            "wait indefinitely"
        ),
    )
    parser.add_argument(
        "--model-download-policy", choices=("auto", "full"), default="auto",
        help="Model files: auto selects verified loader formats; full keeps the complete repository",
    )
    return parser


@gpu_device_scope()
def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    from acprof.application.download import apply_download_arguments
    apply_download_arguments(args)
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.timeout_seconds is not None and (
        args.timeout_seconds <= 0.0 or not math.isfinite(args.timeout_seconds)
    ):
        parser.error("--timeout-seconds must be a finite value > 0")
    if args.runtime == "process":
        parser.error("the minimum-memory probe requires Docker; Process input scans use acprof run --runtime process --profiling-mode basic")
    try:
        cpus, mems = docker_resource_limits(args.cpus, args.mems)
        cpu_list = _int_list(cpus, "CPU 列表")
        mem_list = _int_list(mems, "内存列表")
        _, _, gpu_mode = select_minimum_resources(cpu_list, mem_list, _gpu_list(args.gpus))
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    command_started = time.perf_counter()
    emit_progress("preflight")
    require_collection_host()
    try:
        args.cpuset_cpus = normalize_cpu_set(args.cpuset_cpus)
        require_available_cpu_set(args.cpuset_cpus)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    from acprof.host.run_state import MeasurementLock, RunStateError

    try:
        with MeasurementLock():
            return _run_probe(args, cpu_list, mem_list, gpu_mode, command_started=command_started)
    except RunStateError as exc:
        print(f"[largest-probe][ERROR] {exc}", file=sys.stderr)
        return 2


def _review_probe_task(task_info: TaskInfo, args: argparse.Namespace, cpu_list: list[int], mem_list: list[int],
                       devices: list[str], gpu_device: dict, source: JsonObjectSnapshot | None) -> TaskInfo:
    """Bind confirmation to the single-device diagnostic and its input source."""
    workload = {
        **inference_workload_inputs(args.batch_size, args.input_scales or "", source.payload if source else None),
        "resources": {"runtime": "docker", "cpus": sorted({min(cpu_list), max(cpu_list)}),
                      "mems": sorted(set(mem_list)), "gpus": devices},
        "probe": {"timeout_seconds": args.timeout_seconds},
    }
    if gpu_device:
        workload["resources"]["gpu_device_uuid"] = gpu_device["uuid"]
    if args.cpuset_cpus:
        workload["resources"]["cpuset_cpus"] = args.cpuset_cpus
    if source is not None and task_info.task_family != "nlp":
        workload["specification_bytes_sha256"] = hashlib.sha256(source.content).hexdigest()
    return prepare_execution_configuration(
        task_info, args, PreparationWorkflow.from_environment(args.output_dir), workload=workload,
    )


def _run_probe(
    args: argparse.Namespace, cpu_list: list[int], mem_list: list[int], gpu_mode: str,
    *, command_started: float,
) -> int:
    bootstrap_project_env(Path.cwd())
    require_native_docker()
    cgroup_version = require_cgroup_prerequisites()

    from acprof.host.detect import detect_task

    probe_devices = [gpu_mode]
    emit_progress("resolution")
    task_info = detect_task(
        model_id=args.model, revision=args.revision,
        override_tag=args.task,
        override_family=args.task_family,
        override_backend=args.backend,
        model_spec_path=args.model_spec,
    )
    review = bool(args.inference_config or args.confirm_inference)
    batch_size, input_scales = args.batch_size, args.input_scales
    try:
        source = None
        supplied = None
        resolution = task_info.model_resolution
        failure = resolution.get("failure", {}).get("reason_code")
        selectable = failure in {None, "task_selection_required"} and resolution.get("status") not in {"blocked", "ambiguous"}
        if selectable:
            if args.workload_spec and (review or task_info.task_discovery or task_info.selected_tasks):
                source = read_json_object_snapshot(Path(args.workload_spec).expanduser(), label="workload specification")
                supplied = source.payload
            task_info = select_tasks(task_info, args, single_profile=True, supplied_workload=supplied)
        if not review or not selectable:
            require_task_support(task_info, batch_size=batch_size, devices=probe_devices)
        gpu_device = pin_gpu_device(args.gpu_device) if gpu_mode == "on" else {}
        if review:
            task_info = _review_probe_task(task_info, args, cpu_list, mem_list, probe_devices, gpu_device, source)
            overrides = task_experiment_overrides(task_info.selected_tasks[0])
            batch_size = overrides.get("batch_size", batch_size)
            input_scales = overrides.get("input_scales", input_scales)
            require_task_support(task_info, batch_size=batch_size, devices=probe_devices)
        manifest = (workload_manifest(task_info.selected_tasks[0], supplied=supplied, frozen=task_info.inference_configuration)
                    if task_info.selected_tasks and task_info.task_family == "nlp" else None)
    except InferenceConfigurationReviewRequired:
        return 0
    except PreparationInterrupted as exc:
        print(str(exc), file=sys.stderr)
        return 130
    except (RuntimeError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    output_root = Path(args.output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    output_dir = create_probe_output_dir(output_root, task_info.model_id)

    print("=" * 60)
    print("AC-Prof Minimum Viable Memory + Largest Input Scale Probe")
    print("=" * 60)
    print(f"  Model:    {task_info.model_id}")
    print(
        f"  Task:     {task_info.pipeline_tag} "
        f"(family={task_info.task_family})"
    )
    print(f"  Backend:  {task_info.runtime_backend}")
    if gpu_device:
        print(f"  GPU:      {gpu_device['name']} (index {gpu_device['index']}, {gpu_device['uuid']})")
    if task_info.selected_tasks:
        selected = task_info.selected_tasks[0]
        exemplar = selected.get("execution_profile", {}).get("representative_task_id", selected["task_id"])
        print(f"  Workload: {exemplar}")
    print(f"  Cgroup:   {cgroup_version}")
    print(f"  Output:   {output_dir}")

    task_info.model_download_policy = args.model_download_policy
    try:
        workload_path = args.workload_spec
        if manifest is not None:
            workload_file = output_dir / "task_workload.json"
            atomic_write_json(workload_file, manifest)
            workload_path = str(workload_file)
        emit_progress("image")
        image_info = prepare_image(
            task_info, str(PROJECT_DIR), reuse_existing=not args.rebuild_image,
        )
        emit_progress("input")
        planned = plan_input_scales(
            task_info=task_info, image_info=image_info, cpu_list=cpu_list, mem_list=mem_list,
            gpu_list=probe_devices, batch_size=batch_size, output_dir=str(output_dir),
            input_scales=input_scales, workload_spec_path=workload_path,
            cpuset_cpus=args.cpuset_cpus,
        )
        if task_info.inference_configuration or getattr(image_info, "runtime_environment", {}):
            from acprof.host.runtime_validation import validate_runtime

            emit_progress("runtime")
            validate_runtime(
                task_info=task_info, image_info=image_info, planned=planned,
                cpu_list=cpu_list, mem_list=mem_list, gpu_list=probe_devices, output_dir=str(output_dir),
                cpuset_cpus=args.cpuset_cpus,
            )
        summary = run_largest_scale_probe(
            task_info=task_info,
            image_info=image_info,
            planned_input_scales=planned,
            cpu_list=cpu_list,
            mem_list=mem_list,
            gpu_list=probe_devices,
            batch_size=batch_size,
            output_dir=output_dir,
            timeout_seconds=args.timeout_seconds,
            cpuset_cpus=args.cpuset_cpus,
        )
    except Exception as exc:
        print(f"[largest-probe][ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    summary["timing"]["command_s"] = time.perf_counter() - command_started
    from acprof.platform import detect_environment
    summary.update(detect_environment().metadata())
    write_probe_summary(summary["artifacts"]["summary"], summary)
    print(
        "[largest-probe] Total command elapsed: "
        f"{summary['timing']['command_s']:.3f}s"
    )
    return 0 if summary["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
