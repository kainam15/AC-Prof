"""Offline CLI child for TUI interruption tests; never starts a model or Docker."""
import os
import signal
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

from acprof.events.progress import emit_event, measurement_boundary
from acprof.experiments.arguments import build_parser
from acprof.host import run_state
from acprof.host.images.identity import model_token
from acprof.results.layout import resource_case_id
from tests.support.run_recovery_fixtures import RunRecoveryFixture


def main():
    run_state.MEASUREMENT_LOCK_ROOT = Path(sys.argv[1])
    run_state.MEASUREMENT_LOCK_NAMESPACE = bytes.fromhex(sys.argv[2])
    command = sys.argv[3:]
    options = build_parser().parse_args(command)
    root = Path(options.output_dir)
    # Resolution and hardware boundaries are supplied by the existing fixture.
    os.environ.pop("ACPROF_INTERACTIVE_PREPARATION", None)
    signal.signal(signal.SIGINT, signal.default_int_handler)

    def timeout(_signum, _frame):
        raise TimeoutError("TUI fixture did not receive its stop signal")

    signal.signal(signal.SIGALRM, timeout)
    with ExitStack() as resources:
        fixture = RunRecoveryFixture()
        fixture.build(SimpleNamespace(addfinalizer=resources.callback), root)

        def collect(request, **kwargs):
            with (root / "case_calls.txt").open("a") as stream:
                stream.write(f"{request.cpu}\n")
            path = fixture.write_case(request, **kwargs)
            case_id = f"case_{model_token(request.task_info.model_id)}_{resource_case_id(request.cpu, request.mem, request.gpu)}"
            emit_event("case_started", case_id)
            with measurement_boundary(case_id):
                if request.cpu == 2 and not options.resume:
                    print("[WARN] fixture diagnostic retained across cancellation", flush=True)
                    signal.alarm(20)
                    try:
                        signal.pause()
                    finally:
                        signal.alarm(0)
            emit_event("case_finished", case_id, status="ok")
            return path

        try:
            fixture.invoke(*command, case=collect, output=sys.stdout)
        except KeyboardInterrupt:
            return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
