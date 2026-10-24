"""Every Docker platform and environment wheel lock is covered by audits."""
from pathlib import Path

import pytest

from acprof.runtime_profiles import ENVIRONMENTS, PLATFORMS
from scripts.export_runtime_audit_requirements import export_runtime_locks

ROOT = Path(__file__).resolve().parents[1]


def test_export_matches_all_registered_target_locks_without_artifact_downloads(tmp_path):
    target = tmp_path / "requirements"
    counts = export_runtime_locks(target, root=ROOT, python_version="3.10")
    registered = {
        Path(spec.requirements_lock).name
        for spec in (*PLATFORMS.values(), *ENVIRONMENTS.values())
    }
    assert set(counts) == registered
    assert len(counts) == 31
    assert all(count > 0 for count in counts.values())
    for name, count in counts.items():
        lines = (target / name).read_text().splitlines()
        assert len(lines) == count
        assert all(" @ " not in line and " --hash=sha256:" in line for line in lines)
        assert all(line.split(" --hash=", 1)[0].count("==") == 1 for line in lines)


def test_runtime_audit_fails_for_wrong_python_baseline_and_stale_outputs(tmp_path):
    target = tmp_path / "requirements"
    with pytest.raises(ValueError, match="target Python version"):
        export_runtime_locks(target, root=ROOT, python_version="3.12")
    export_runtime_locks(target, root=ROOT, python_version="3.10")
    with pytest.raises(ValueError, match="must be empty"):
        export_runtime_locks(target, root=ROOT, python_version="3.10")
