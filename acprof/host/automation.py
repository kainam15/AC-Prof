"""Preparation and final reports for the conservative automatic collection path."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from acprof.artifacts import atomic_write_json, read_json_object
from acprof.model_evidence import pinned_revision


def check_repository_access(repo_id: str, *, source: str = "huggingface", revision: str | None = None) -> dict:
    """Check access without accepting terms or exposing credentials in reports."""
    from huggingface_hub import HfApi

    from acprof.hf_download import HfDownloadError, exception_chain, try_hf_endpoints
    from acprof.hf_transport import configure_hf_transport
    from acprof.host.model_errors import model_lookup_error
    from acprof.host.model_store import cached_model_info
    from acprof.model_repository import model_source, modelscope_info
    source = model_source(source)
    if cached_model_info(repo_id, revision, source=source) is not None:
        return {"repo_id": repo_id, "source": source, "status": "cached", "scope": "local_model_store_only"}
    try:
        if source == "modelscope":
            modelscope_info(repo_id, revision)
        else:
            configure_hf_transport()
            try_hf_endpoints(repo_id, lambda endpoint: HfApi(endpoint=endpoint).auth_check(repo_id))
    except Exception as exc:
        # Provider exceptions can include URLs/headers. Persist a typed reason only.
        from acprof.failures import Failure, RuntimeFailure
        lookup = model_lookup_error(repo_id, exc, revision=revision)
        code = "access_denied" if lookup.reason_code in {"access_denied", "repository_unavailable"} else "runtime_initialization_failed"
        exhausted = next((cause for cause in exception_chain(exc) if isinstance(cause, HfDownloadError)), None)
        detail = (str(exhausted) if exhausted else
                  f"{repo_id} ({source}): {type(exc).__name__}; check repository ID, network and source credentials")
        raise RuntimeFailure(Failure("access", code, detail,
            retryability="after_configuration",
            exception_type=exhausted.attempts[-1]["exception_type"] if exhausted else type(exc).__name__)) from None
    return {"repo_id": repo_id, "source": source, "status": "accessible", "scope": "repository_read_access_only"}


def select_profiling_mode(requested: str, checks: list) -> tuple[str, list[dict]]:
    """Only an explicit auto policy may omit unavailable full-only measurements."""
    optional = {"rapl", "perf", "packet"}
    unavailable = [asdict(check) for check in checks if check.status == "unavailable"]
    blockers = [check for check in unavailable if requested != "auto" or check["name"] not in optional]
    if blockers:
        raise RuntimeError("host preflight failed: " + "; ".join(f"{item['name']}: {item['detail']}" for item in blockers))
    return ("basic" if unavailable else "full", unavailable) if requested == "auto" else (requested, [])


class AutomaticRun:
    """Write outside measurement windows; use the existing run engine for collection."""

    def __init__(self, args):
        from huggingface_hub.utils import validate_repo_id
        validate_repo_id(args.model)  # Exact ID or Hub alias only; never search by popularity.
        self.args = args
        args.output_dir = str(Path(args.output_dir).expanduser().resolve())
        self.root = Path(args.output_dir).expanduser().resolve() / args.model.replace("/", "--")
        from acprof.artifact_layout import ArtifactLayout
        self.layout = ArtifactLayout.discover(self.root) if args.resume else ArtifactLayout.for_new_run(self.root)
        self.path = self.layout.path("auto_report.json")
        self.data = {"schema_version": 1, "model_id": args.model,
                     "requested_profiling_mode": args.profiling_mode,
                     "status": "preparing", "stage": "preflight", "attempts": [],
                     "scope": "automatic_task_pipeline; no_semantic_or_plan_repairs"}
        self.started = False
        self.preparation_artifacts = {}

    def save(self):
        atomic_write_json(self.path, self.data)

    def prepare(self):
        from acprof.host.detect import TaskInfo, detect_task
        from acprof.host.doctor import collect_checks
        from acprof.host.run_state import file_sha256, load_run_state
        from acprof.host.task_support import require_task_support
        from acprof.model_contract import write_model_resolution
        from acprof.model_spec import task_model_spec

        args = self.args
        if not args.resume:
            self.layout.initialize()
        saved = load_run_state(self.root) if args.resume else {}
        if args.resume:
            previous = read_json_object(self.path, label="auto_report.json")
            if previous.get("schema_version") != 1:
                raise ValueError("unsupported auto_report schema")
            if previous.get("requested_profiling_mode") != args.profiling_mode:
                raise ValueError("resume must keep the requested profiling mode")
            self.data["attempts"] = previous.get("attempts", []) + [{
                "status": previous.get("status"), "stage": previous.get("stage"), "error": previous.get("error")}]
        self.started = True
        self.save()
        self.data["stage"] = "resolution"
        if saved.get("runtime"):
            task = TaskInfo(**saved["runtime"]["task"])
            if args.revision and args.revision != task.model_revision:
                raise ValueError("resume revision differs from the frozen snapshot")
            self.data["access"] = {"status": "not_requested", "reason": "resume reuses immutable local image"}
        else:
            self.data["stage"] = "access"
            from acprof.model_repository import model_source
            self.data["access"] = {"repositories": [check_repository_access(args.model,
                source=model_source(), revision=args.revision or None)]}
            self.data["stage"] = "resolution"
            task = detect_task(args.model, override_tag=args.task, override_family=args.task_family,
                               override_backend=args.backend, model_spec_path=args.model_spec,
                               **({"revision": args.revision} if args.revision else {}))
        write_model_resolution(task, self.root)
        require_task_support(task, batch_size=args.batch_size)
        if not pinned_revision(task.model_revision):
            raise ValueError("automatic collection requires a full model commit SHA")
        if not saved.get("runtime"):
            for dependency in task_model_spec(task).get("dependencies", []):
                self.data["access"]["repositories"].append(check_repository_access(dependency["repo_id"],
                    revision=dependency["revision"]))
        args.revision = task.model_revision
        args.resolution_identity = task.model_resolution.get("provenance", {}).get("identity_sha256")
        args.requested_profiling_mode = self.data["requested_profiling_mode"]
        self.data.update(source=task.model_source, revision=task.model_revision,
                         resolution_identity=args.resolution_identity, stage="host")
        checks = collect_checks(profiling_mode="full" if args.profiling_mode == "auto" else args.profiling_mode,
                                gpus="on" if "on" in {part.strip().lower() for part in args.gpus.split(",")} else "off",
                                sniff_iface=args.sniff_iface, output_dir=self.root)
        self.data["host_checks"] = [asdict(check) for check in checks]
        actual, reasons = select_profiling_mode(args.profiling_mode, checks)
        if saved and saved.get("options", {}).get("profiling_mode") != actual:
            raise ValueError("host capability change would alter the frozen profiling mode; use a new output directory")
        args.profiling_mode = actual
        self.data.update(profiling_mode=actual, mode_selection_reasons=reasons, stage="collection", status="running")
        write_model_resolution(task, self.root)
        self.save()
        self.preparation_artifacts = {str(self.layout.path(name).relative_to(self.root)): file_sha256(self.layout.path(name))
                                      for name in ("auto_report.json", "model_resolution.json")}
        return task

    def finish(self, error: str | None = None) -> int:
        if not self.started:
            return 2
        for name in ("runtime_validation", "capability_report", "run_state"):
            path = self.layout.path(f"{name}.json")
            if path.is_file():
                try:
                    value = read_json_object(path, label=f"{name}.json")
                    self.data[name] = {key: value.get(key) for key in (
                        "status", "outcome", "requested_measurements_complete", "collection_succeeded") if key in value}
                except (OSError, ValueError):
                    error = error or f"unreadable {name}.json"
        state = self.data.get("run_state", {})
        capabilities = self.data.get("capability_report", {})
        success = (not error and state.get("status") == "complete" and state.get("outcome") == "ok"
                   and capabilities.get("requested_measurements_complete") is True
                   and capabilities.get("collection_succeeded") is True)
        self.data["status"] = "succeeded" if success else "failed"
        if error:
            self.data["error"] = error
        self.save()
        return 0 if success else 2
