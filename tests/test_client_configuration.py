import os
import subprocess
import sys
from pathlib import Path


def test_import_does_not_read_runtime_environment_or_initialize_workload():
    script = '''
import os
from unittest.mock import patch
before = dict(os.environ)
with patch("acprof.workloads.get_generator", side_effect=AssertionError("workload during import")):
    import acprof.host.client
assert dict(os.environ) == before
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "WARMUP": "invalid-until-entrypoint", "NO_PROXY": "", "no_proxy": ""},
        capture_output=True, text=True, timeout=20)
    assert (result.returncode) == (0), result.stderr
    assert (result.stdout) == ("")
