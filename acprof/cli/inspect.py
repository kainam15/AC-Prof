"""Explain model resolution and optionally validate it before any measurement."""
from __future__ import annotations

import argparse
import subprocess
import sys
import uuid
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("model", help="Model ID in the selected source")
    parser.add_argument("--model-source", choices=("huggingface", "modelscope"), default=None)
    parser.add_argument("--model-spec", help="Explicit local model declaration")
    parser.add_argument("--revision", help="Model branch, tag or full commit SHA")
    parser.add_argument("--expected-revision", help="Refuse a model commit that changed since review")
    parser.add_argument("--task", help="Explicit task selection")
    parser.add_argument("--backend", help="Explicit backend selection")
    parser.add_argument("--explain", action="store_true", help="Show field values and evidence sources")
    parser.add_argument("--output-dir", type=Path, help="Export model_resolution.json and probe evidence")
    parser.add_argument("--probe-interface", action="store_true", help="Check imports/signatures without model weights or inference")
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument("--mems", type=int, default=4, help="Probe memory limit in GiB")
    parser.add_argument("--timeout-seconds", type=float, default=300)
    args = parser.parse_args(argv)
    from acprof.cli.download_args import apply_download_arguments
    from acprof.host.detect import detect_task
    from acprof.host.env_utils import bootstrap_project_env
    from acprof.host.interface_probe import probe_interface
    from acprof.host.model_inspection import explain_resolution
    from acprof.model_contract import write_model_resolution
    apply_download_arguments(args)
    bootstrap_project_env(Path.cwd())
    task = detect_task(args.model, model_spec_path=args.model_spec, override_tag=args.task, override_backend=args.backend,
                       **({"revision": args.revision} if args.revision else {}))
    if args.expected_revision and task.model_revision != args.expected_revision:
        print("[interface][ERROR] Model revision changed; resolve and review it again", file=sys.stderr)
        return 2
    output = args.output_dir
    if args.probe_interface and output is None:
        output = Path("results/inspection") / task.model_id.replace("/", "--") / uuid.uuid4().hex[:12]
    if args.probe_interface and (output / "interface_validation.json").exists():
        print("[interface][ERROR] Choose a new output directory for this probe", file=sys.stderr)
        return 2
    if args.expected_revision and args.model_spec and output is not None:
        # An exported declaration can carry its separately saved provenance.
        # Preserve that explanation only for the same pinned model and exact spec;
        # a prior runtime observation never becomes evidence for this new probe.
        from acprof.artifacts import read_json_object
        from acprof.model_evidence import RESOLVER_VERSION
        from acprof.model_spec import task_model_spec
        previous = output / "model_resolution.json"
        if previous.is_file():
            try:
                contract = read_json_object(previous, label="model resolution").get("contract", {})
                if (contract.get("model_id") == task.model_id and contract.get("revision") == task.model_revision
                        and contract.get("resolver_version") == RESOLVER_VERSION and contract.get("status") == "resolved"
                        and contract.get("draft_spec") == task_model_spec(task)):
                    contract["runtime_validation"] = "not_run"
                    contract["fields"] = {key: value for key, value in contract["fields"].items() if not key.startswith("runtime.")}
                    task.model_resolution["contract"] = contract
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                pass
    preflight_error = None
    if task.model_resolution.get("status") not in {"ambiguous", "needs_configuration"}:
        from acprof.host.task_support import TaskSupportError, require_task_support
        try:
            require_task_support(task, devices=("cpu",))
        except TaskSupportError as exc:
            preflight_error = exc
    if output is not None:
        write_model_resolution(task, output)
    print(explain_resolution(task, explain=args.explain), flush=True)
    if preflight_error is not None:
        print(str(preflight_error), file=sys.stderr)
        return 2
    if task.model_resolution.get("status") in {"ambiguous", "needs_configuration"}:
        return 2
    if args.probe_interface:
        try:
            from acprof.host.preflight import require_collection_host, require_native_docker
            from acprof.host.run_state import MeasurementLock, ResultDirectoryLock
            from acprof.host.runtime_images import configure_runtime_profile
            require_collection_host()
            require_native_docker()
            with MeasurementLock(), ResultDirectoryLock(output):
                configure_runtime_profile(task)
                report = probe_interface(task, output, cpus=args.cpus, memory_gb=args.mems,
                                         timeout_seconds=args.timeout_seconds)
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
            print(f"[interface][ERROR] {exc}", file=sys.stderr)
            return 1
        print(explain_resolution(task), flush=True)
        print(f"Evidence: {output}", flush=True)
        return 0 if report["status"] == "ok" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
