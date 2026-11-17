import os
import subprocess
import sys


def test_generated_coverage_is_independent_of_python_hash_seed():
    outputs = [subprocess.check_output([sys.executable, '-c',
        'from scripts.render_runtime_coverage import render; print(render(), end="")'],
        env={**os.environ, 'PYTHONHASHSEED': str(seed)}, text=True) for seed in range(1, 7)]
    assert len(set(outputs)) == 1
