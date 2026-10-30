"""Resolve, preflight and collect one exact model using the existing run engine."""
from __future__ import annotations

import sys
from pathlib import Path

from acprof.system.cleanup import cleanup_on_exit, format_exception_notes


def main(argv=None) -> int:
    from acprof.experiments.arguments import build_parser
    from acprof.host.automation import AutomaticRun
    from acprof.host.env_utils import bootstrap_project_env
    from acprof.host.run_state import MeasurementLock, ResultDirectoryLock
    parser = build_parser(automatic=True)
    args = parser.parse_args(argv)
    from acprof.application.download import apply_download_arguments
    apply_download_arguments(args)
    bootstrap_project_env(Path.cwd())
    automatic: AutomaticRun | None = None
    failure: BaseException | None = None
    exit_code: int | None = None

    def finish():
        nonlocal exit_code
        if automatic is None:
            return
        collection_path = getattr(failure, "task_collection_path", None) or getattr(
            failure.__context__ if failure else None, "task_collection_path", None)
        if collection_path:
            automatic.collection_path = collection_path
        if failure is None or isinstance(failure, SystemExit) and failure.code in (None, 0):
            error = None
        elif isinstance(failure, (Exception, SystemExit)):
            error = f"{type(failure).__name__}: {format_exception_notes(failure)}"
        else:
            error = "interrupted"
        exit_code = automatic.finish(error)

    try:
        with cleanup_on_exit(finish, description="automatic report"):
            try:
                automatic = AutomaticRun(args)
                with MeasurementLock(), ResultDirectoryLock(automatic.root, new=not args.resume):
                    task = automatic.prepare()
                print(f"[auto] {task.model_id}@{task.model_revision}; profiling_mode={args.profiling_mode}", flush=True)
                from acprof.cli import run
                outcome = run.main(args=args, prepared_task=task, preparation_artifacts=automatic.preparation_artifacts)
                automatic.collection_path = getattr(outcome, "task_collection_path", None)
            except BaseException as exc:
                failure = exc
                raise
    except (Exception, SystemExit) as exc:
        # SystemExit raised by report finalization is cancellation, not a run result.
        if isinstance(exc, SystemExit) and exc is not failure:
            raise
        if isinstance(exc, SystemExit) and exc.code in (None, 0):
            if exit_code is not None:
                return exit_code
            if automatic is None:
                return 2
        print(f"[auto][ERROR] {type(exc).__name__}: {format_exception_notes(exc)}", file=sys.stderr)
        return 2
    return exit_code if exit_code is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
