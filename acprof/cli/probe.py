#!/usr/bin/env python3
"""CLI for a one-request largest-input-scale timing probe."""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from acprof.host.env_utils import bootstrap_project_env
from acprof.host.images.runtime import prepare_image
from acprof.host.input_plan import plan_input_scales
from acprof.host.largest_scale_probe import (
    create_probe_output_dir,
    run_largest_scale_probe,
    write_probe_summary,
)
from acprof.host.preflight import (
    require_cgroup_prerequisites,
    require_collection_host,
    require_native_docker,
)
from acprof.host.task_support import TaskSupportError, require_task_support
from acprof.installation import resource_root
from acprof.run_options import add_option_arguments

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
        cpu_list = _int_list("1,2,4,8" if args.cpus == "auto" else args.cpus, "CPU 列表")
        mem_list = _int_list("2,4,8,16" if args.mems == "auto" else args.mems, "内存列表")
        gpu_list = _gpu_list(args.gpus)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    command_started = time.perf_counter()
    from acprof.preparation_events import emit_progress
    emit_progress("preflight")
    bootstrap_project_env(Path.cwd())
    require_collection_host()
    require_native_docker()
    cgroup_version = require_cgroup_prerequisites()

    from acprof.host.detect import detect_task

    emit_progress("resolution")
    task_info = detect_task(
        model_id=args.model, revision=args.revision,
        override_tag=args.task,
        override_family=args.task_family,
        override_backend=args.backend,
        model_spec_path=args.model_spec,
    )
    try:
        require_task_support(task_info, batch_size=args.batch_size, devices=gpu_list)
    except TaskSupportError as exc:
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
    print(f"  Cgroup:   {cgroup_version}")
    print(f"  Output:   {output_dir}")

    task_info.model_download_policy = args.model_download_policy
    try:
        emit_progress("image")
        image_info = prepare_image(
            task_info, str(PROJECT_DIR), reuse_existing=not args.rebuild_image,
        )
        emit_progress("input")
        planned = plan_input_scales(
            task_info=task_info,
            image_info=image_info,
            cpu_list=cpu_list,
            mem_list=mem_list,
            gpu_list=gpu_list,
            batch_size=args.batch_size,
            output_dir=str(output_dir),
            input_scales=args.input_scales,
            workload_spec_path=args.workload_spec,
        )
        if getattr(image_info, "runtime_environment", {}):
            from acprof.host.runtime_validation import validate_runtime

            emit_progress("runtime")
            validate_runtime(
                task_info=task_info, image_info=image_info, planned=planned,
                cpu_list=cpu_list, mem_list=mem_list, gpu_list=gpu_list, output_dir=str(output_dir),
            )
        summary = run_largest_scale_probe(
            task_info=task_info,
            image_info=image_info,
            planned_input_scales=planned,
            cpu_list=cpu_list,
            mem_list=mem_list,
            gpu_list=gpu_list,
            batch_size=args.batch_size,
            output_dir=output_dir,
            timeout_seconds=args.timeout_seconds,
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
