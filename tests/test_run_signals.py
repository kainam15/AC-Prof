"""SIGTERM must unwind the same host cleanup path as Ctrl+C."""
import subprocess
import sys
import textwrap
import unittest


class RunSignalTests(unittest.TestCase):
    def test_sigterm_unwinds_case_and_run_cleanup_and_restores_handler(self):
        script = textwrap.dedent('''
            import os
            import signal
            from unittest.mock import patch
            from acprof.cli import run

            previous = signal.getsignal(signal.SIGTERM)
            class State:
                def close(self, status):
                    print('state closed', flush=True)

            def measure():
                run._ACTIVE_RUN_STATE = State()
                try:
                    os.kill(os.getpid(), signal.SIGTERM)
                finally:
                    print('case cleaned', flush=True)

            with patch.object(run, '_run_main', side_effect=measure), patch.object(
                run, '_record_run_termination', side_effect=lambda *args: print(args[0], flush=True)
            ):
                try:
                    run.main()
                except KeyboardInterrupt:
                    print('interrupted', flush=True)
            assert signal.getsignal(signal.SIGTERM) == previous
        ''')
        result = subprocess.run([sys.executable, '-c', script], capture_output=True,
                                text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(),
                         ['case cleaned', 'cancelled', 'state closed', 'interrupted'])


if __name__ == '__main__':
    unittest.main()
