"""SIGTERM must unwind the same host cleanup path as Ctrl+C."""
import subprocess
import sys
import textwrap


def test_sigterm_unwinds_case_and_run_cleanup_and_restores_handler():
    script = textwrap.dedent('''
            import os
            import signal
            from unittest.mock import patch
            from acprof.cli import run
            from acprof.application.run import RunApplication
            import sys

            previous = signal.getsignal(signal.SIGTERM)
            class State:
                def close(self, status):
                    print('state closed', flush=True)

            application = RunApplication()
            def measure(*args, **kwargs):
                application.run_state = State()
                try:
                    os.kill(os.getpid(), signal.SIGTERM)
                finally:
                    print('case cleaned', flush=True)

            with patch.object(application, '_execute', side_effect=measure), patch.object(
                application.notifications, 'record_termination', side_effect=lambda *args: print(args[0], flush=True)
            ), patch.object(run, 'RunApplication', return_value=application), patch.object(
                sys, 'argv', ['acprof run', '--model', 'fixture/model']
            ):
                try:
                    run.main()
                except KeyboardInterrupt:
                    print('interrupted', flush=True)
            assert signal.getsignal(signal.SIGTERM) == previous
        ''')
    result = subprocess.run([sys.executable, '-c', script], capture_output=True,
                            text=True, timeout=30)
    assert (result.returncode) == (0), result.stderr
    assert (result.stdout.splitlines()) == (['case cleaned', 'cancelled', 'state closed', 'interrupted'])
