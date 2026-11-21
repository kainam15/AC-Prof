"""Doctor checks the directory a writer can actually open, without creating it."""
import json

import pytest

from acprof.cli import doctor as cli
from acprof.host import doctor


@pytest.mark.parametrize(("relative", "usable"), [
    ("present", True),
    ("new/nested", True),
    ("mount/new/nested", True),
    ("mount/../occupied/new", True),
    ("uncreated/../present/new", True),
    ("uncreated/../occupied/new", False),
    ("detached", False),
    ("detached/new", False),
    ("cycle/new", False),
    ("occupied", False),
    ("occupied/new", False),
    ("occupied/../new", False),
])
def test_doctor_json_workspace_matches_real_directory_creation(tmp_path, monkeypatch, capsys, relative, usable):
    (tmp_path / "present" / "nested").mkdir(parents=True)
    (tmp_path / "occupied").write_text("keep")
    (tmp_path / "mount").symlink_to("present/nested", target_is_directory=True)
    (tmp_path / "detached").symlink_to("missing-mount", target_is_directory=True)
    (tmp_path / "cycle").symlink_to("cycle", target_is_directory=True)
    target = tmp_path / relative
    before = set(tmp_path.rglob("*"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ACPROF_SUDO_PASSWORD", raising=False)
    monkeypatch.setattr(doctor, "_check", lambda name, *_: doctor.DoctorCheck(name, "available", "fixture"))
    code = cli.main(["--json", "--runtime", "docker", "--profiling-mode", "basic", "--output-dir", str(target)])
    data = json.loads(capsys.readouterr().out)
    workspace, = [check for check in data["checks"] if check["name"] == "workspace"]
    assert code == (0 if usable else 1)
    assert data["ready"] is usable
    assert workspace["status"] == ("available" if usable else "unavailable")
    assert set(tmp_path.rglob("*")) == before
    if usable:
        target.mkdir(parents=True, exist_ok=True)
        assert target.is_dir()
    else:
        assert workspace["remedy"]
        with pytest.raises(OSError, match="occupied|detached|cycle"):
            target.mkdir(parents=True, exist_ok=True)
