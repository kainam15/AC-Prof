"""JSON publication preserves formatting and the existing replacement default."""

import pytest

from acprof.artifacts import atomic_write_json


@pytest.mark.parametrize("overwrite", [False, True])
def test_atomic_json_modes_preserve_utf8_format_and_complete_content(tmp_path, overwrite):
    output = tmp_path / "report.json"
    atomic_write_json(output, {"label": "中文", "missing": None}, overwrite=overwrite)
    assert output.read_bytes() == '{\n  "label": "中文",\n  "missing": null\n}\n'.encode()
    assert output.stat().st_mode & 0o777 == 0o644
    assert list(tmp_path.iterdir()) == [output]


def test_atomic_json_default_still_updates_existing_artifacts(tmp_path):
    output = tmp_path / "state.json"
    output.write_text("previous state\n", encoding="utf-8")
    output.chmod(0o600)
    atomic_write_json(output, {"status": "complete"})
    assert output.read_bytes() == b'{\n  "status": "complete"\n}\n'
    assert output.stat().st_mode & 0o777 == 0o600
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("overwrite", [False, True])
def test_atomic_json_serialization_failure_preserves_existing_files(tmp_path, overwrite):
    output = tmp_path / "state.json"
    output.write_bytes(b"previous state\n")
    before = output.stat()
    with pytest.raises(ValueError, match="Out of range float values"):
        atomic_write_json(output, {"partial": "written", "invalid": float("nan")}, overwrite=overwrite)
    assert output.read_bytes() == b"previous state\n"
    assert output.stat().st_ino == before.st_ino
    assert list(tmp_path.iterdir()) == [output]
