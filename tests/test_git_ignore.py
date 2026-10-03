"""Generated CSV stays ignored while reviewable fixtures can be committed."""
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.mark.parametrize('path_case', range(8))
def test_csv_locations(path_case):
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as directory:
        subprocess.run(["git", "init", "--quiet", directory], check=True)
        (Path(directory) / ".gitignore").write_bytes((root / ".gitignore").read_bytes())
        (path, ignored) = tuple({'results/foo.csv': True, 'internal-testing/foo.csv': True, 'result-past/foo.csv': True, 'ad-hoc.csv': True, 'tests/fixtures/foo.csv': False, 'tests/fixtures/nested/foo.csv': False, 'examples/foo.csv': False, 'examples/nested/foo.csv': False}.items())[path_case]
        result = subprocess.run(["git", "-c", "core.excludesFile=/dev/null", "check-ignore", "--no-index", "-q", path], cwd=directory)
        assert (result.returncode) == (0 if ignored else 1)
