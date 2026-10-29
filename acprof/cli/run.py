"""CLI adaptation for the shared experiment collection application."""
from __future__ import annotations

import signal
import sys
import threading

from acprof.application.errors import CollectionFailed
from acprof.application.run import RunApplication, configure_environment
from acprof.application.terminal_log import format_run_command
from acprof.experiment import RunConfig
from acprof.host.preparation.task_support import TaskSupportError
from acprof.host.run_state import RunStateError
from acprof.run_args import build_parser


def main(*, args=None, prepared_task=None, preparation_artifacts=None):
    """Parse arguments, own process signals, and translate application failures."""
    parser = build_parser()
    args = parser.parse_args() if args is None else args
    config = RunConfig.from_namespace(args)
    # Preserve omitted CLI flags (None) at the environment boundary: shared UI
    # defaults must not overwrite a configured mirror, budget or model store.
    application = RunApplication(environment_setup=lambda _options: configure_environment(args))

    def execute():
        try:
            return application.run(
                config, prepared_task=prepared_task, preparation_artifacts=preparation_artifacts,
                command=format_run_command(sys.argv),
            )
        except (ValueError, TaskSupportError, RunStateError) as exc:
            parser.error(str(exc))
        except CollectionFailed as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from None

    if threading.current_thread() is not threading.main_thread():
        return execute()

    def interrupt(_signum, _frame):
        # A second TERM must not interrupt cleanup started by the first one.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        return execute()
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
