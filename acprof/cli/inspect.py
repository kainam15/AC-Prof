"""Explain model resolution and optionally validate it before any measurement."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from contextlib import ExitStack
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("model", help="Model ID in the selected source")
    from acprof.experiments.arguments import add_download_arguments
    add_download_arguments(parser)
    parser.add_argument("--model-spec", help="Explicit local model declaration")
    parser.add_argument("--model-variant", default="", help="Named native Transformers checkpoint variant")
    parser.add_argument("--revision", help="Model branch, tag or full commit SHA")
    parser.add_argument("--expected-revision", help="Refuse a model commit that changed since review")
    parser.add_argument("--task", help="Explicit model execution interface")
    parser.add_argument("--task-family", help="Explicit task family, as in collection")
    parser.add_argument("--gpus", help="GPU modes to check against the adapter (comma-separated off/on)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size to check against the adapter")
    parser.add_argument("--tasks", default="", help="Inference profile or semantic task IDs, comma-separated, or all")
    parser.add_argument("--backend", help="Explicit backend selection")
    parser.add_argument("--summary-json", action="store_true", help="Print a compact static model summary as JSON")
    parser.add_argument("--explain", action="store_true", help="Show field values and evidence sources")
    parser.add_argument("--output-dir", type=Path, help="Export model_resolution.json and probe evidence")
    parser.add_argument("--probe-interface", action="store_true", help="Check imports/signatures without model weights or inference")
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument("--mems", type=int, default=4, help="Probe memory limit in GiB")
    parser.add_argument("--timeout-seconds", type=float, default=300)
    args = parser.parse_args(argv)
    if args.summary_json and (args.probe_interface or args.output_dir or args.explain):
        parser.error("--summary-json is a static summary and cannot be combined with --probe-interface, --output-dir or --explain")
    from acprof.application.download import apply_download_arguments
    from acprof.host.detect import detect_task
    from acprof.host.env_utils import bootstrap_project_env
    from acprof.host.preparation.model_errors import ModelLookupError
    apply_download_arguments(args)
    bootstrap_project_env(Path.cwd())
    try:
        task = detect_task(args.model, model_spec_path=args.model_spec, override_tag=args.task, override_backend=args.backend,
                           **({"override_family": args.task_family} if args.task_family else {}),
                           **({"model_variant": args.model_variant} if args.model_variant else {}),
                           **({"revision": args.revision} if args.revision else {}))
    except ModelLookupError as exc:
        from acprof.hub.repository import model_source
        if args.summary_json:
            print(json.dumps({"lookup_error": {"model_id": exc.model_id, "revision": exc.revision,
                                               "reason_code": exc.reason_code},
                              "source": model_source()}, ensure_ascii=False))
        print(f"[model-lookup][ERROR] {exc}", file=sys.stderr)
        return 1
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"[interface][ERROR] {exc}", file=sys.stderr)
        return 2
    if args.expected_revision and task.model_revision != args.expected_revision:
        print("[interface][ERROR] Model revision changed; resolve and review it again", file=sys.stderr)
        return 2
    output = args.output_dir
    if args.probe_interface and output is None:
        output = Path("results/inspection") / task.model_id.replace("/", "--") / uuid.uuid4().hex[:12]
    try:
        with ExitStack() as locks:
            if args.probe_interface:
                from acprof.host.run_state import offline_analysis_lock
                locks.enter_context(offline_analysis_lock())
            if output is not None:
                from acprof.host.run_state import ResultDirectoryLock
                from acprof.results.layout import ArtifactLayout
                locks.enter_context(ResultDirectoryLock(output))
                layout = ArtifactLayout.discover(output)
                for name in ("interface_validation.json", "runtime_validation.json", "run_state.json",
                             "static_meta.json", "result_layers.json", "result_all.csv"):
                    if layout.path(name).exists():
                        print(f"[interface][ERROR] {name} already exists; choose a new inspection directory",
                              file=sys.stderr)
                        return 2
                previous = layout.path("model_resolution.json")
                if previous.is_file():
                    try:
                        _check_previous_resolution(task, previous, review=bool(args.expected_revision and args.model_spec))
                    except ValueError as exc:
                        print(f"[interface][ERROR] {exc}; choose a new inspection directory", file=sys.stderr)
                        return 2
            return _inspect(task, args, output)
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"[interface][ERROR] {exc}", file=sys.stderr)
        return 1


def _previous_resolution(previous: Path, *, review: bool) -> dict:
    from acprof.artifacts import MAX_JSON_ARTIFACT_BYTES, read_json_object

    try:
        return read_json_object(previous, label="model resolution")
    except ValueError as exc:
        if not review:
            raise
        # Older reports may contain NaN. The identity alone can authorize a
        # fresh pinned-spec review, but none of those contract fields can carry over.
        with previous.open("rb") as stream:
            raw = stream.read(MAX_JSON_ARTIFACT_BYTES + 1)
        if len(raw) > MAX_JSON_ARTIFACT_BYTES:
            raise
        try:
            old = json.loads(raw)
        except (ValueError, RecursionError) as parse_error:
            raise exc from parse_error
        if not isinstance(old, dict):
            raise
        return {key: old[key] for key in ("model_id", "model_source", "model_revision", "model_variant", "provenance") if key in old}


def _check_previous_resolution(task, previous: Path, *, review: bool) -> None:
    """Only carry a reviewed contract forward within its original model snapshot."""
    from acprof.models.evidence import RESOLVER_VERSION
    from acprof.models.spec import task_model_spec

    saved = _previous_resolution(previous, review=review)
    try:
        snapshot = saved.get("provenance", {}).get("sources", {}).get("repository_snapshot", {})
        identity = (snapshot.get("model_id", saved.get("model_id")), snapshot.get("source", saved.get("model_source")),
                    snapshot.get("revision", saved.get("model_revision")),
                    snapshot.get("model_variant", saved.get("model_variant", "")))
    except AttributeError as exc:
        raise ValueError("Invalid repository identity in the previous model resolution") from exc
    if identity != (task.model_id, task.model_source, task.model_revision, getattr(task, "model_variant", "")):
        raise ValueError("Previous model resolution belongs to a different or unidentified repository snapshot")
    if review:
        contract = saved.get("contract")
        if (isinstance(contract, dict) and contract.get("model_id") == task.model_id
                and contract.get("revision") == task.model_revision and contract.get("resolver_version") == RESOLVER_VERSION
                and contract.get("status") == "resolved" and contract.get("draft_spec") == task_model_spec(task)
                and isinstance(contract.get("fields"), dict)):
            contract["runtime_validation"] = "not_run"
            contract["fields"] = {key: value for key, value in contract["fields"].items() if not key.startswith("runtime.")}
            task.model_resolution["contract"] = contract


def _model_summary(task) -> dict:
    """Project the selected static resolution without exploring alternate configurations."""
    resolution = task.model_resolution
    failure = resolution.get("failure", {})
    evidence = failure.get("evidence", {})
    compatibility = resolution.get("compatibility", {})
    architectures = task.model_config.get("architectures")
    formats = [item["format"] for item in resolution.get("artifact_layout", {}).get("weights", []) if item.get("format")]
    if evidence.get("artifact_format"):
        formats.append(evidence["artifact_format"])
    from acprof.models.inference_configuration import configuration_device
    devices = {"off": "CPU", "on": "CUDA"}
    platforms = [devices[configuration_device(name)] for name in resolution.get("precision", {})]
    reason = failure.get("detail")
    if not reason and compatibility.get("status") != "supported":
        reason = compatibility.get("detail")
    return {
        "source": task.model_source, "revision": task.model_revision, "backend": task.runtime_backend,
        "architecture": ", ".join(name for name in architectures if isinstance(name, str))
                        if isinstance(architectures, list) and architectures else task.model_config.get("model_type"),
        "formats": list(dict.fromkeys(formats)), "platform": evidence.get("artifact_platform") or ", ".join(platforms) or None,
        "status": resolution.get("status"), "compatibility": compatibility.get("status"),
        "reason": reason,
        "next_action": evidence.get("next_action") or (compatibility.get("next_action") if reason else None),
        "variants": resolution.get("checkpoint_variants") or evidence.get("variant_directories", []),
    }


def _inspect(task, args, output) -> int:
    from acprof.host.preparation.model_inspection import explain_resolution
    from acprof.host.preparation.task_inspection import inspect_selected_tasks
    from acprof.models.configuration import describe_configuration_options
    from acprof.models.contract import write_model_resolution

    task, preflight_error = inspect_selected_tasks(task, args)
    if getattr(args, "summary_json", False):
        print(json.dumps(_model_summary(task), ensure_ascii=False), flush=True)
        blocked = task.model_resolution.get("status") in {"ambiguous", "needs_configuration", "selection_required", "blocked"}
        return 2 if preflight_error or blocked else 0
    task.model_resolution["configuration_options"] = describe_configuration_options(task)
    if output is not None:
        if not task.model_resolution.get("provenance"):
            # Early metadata failures precede creation of the repository snapshot.
            task.model_resolution["model_source"] = task.model_source
        write_model_resolution(task, output)
    print(explain_resolution(task, explain=args.explain), flush=True)
    if preflight_error is not None:
        print(str(preflight_error), file=sys.stderr)
        return 2
    if task.model_resolution.get("status") in {"ambiguous", "needs_configuration", "selection_required", "blocked"}:
        return 2
    if args.probe_interface:
        from acprof.host.images.runtime import configure_runtime_profile
        from acprof.host.preflight import require_collection_host, require_native_docker
        from acprof.host.preparation.interface_probe import probe_interface
        require_collection_host()
        require_native_docker()
        configure_runtime_profile(task)
        report = probe_interface(task, output, cpus=args.cpus, memory_gb=args.mems,
                                 timeout_seconds=args.timeout_seconds)
        print(explain_resolution(task), flush=True)
        print(f"Evidence: {output}", flush=True)
        return 0 if report["status"] == "ok" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
