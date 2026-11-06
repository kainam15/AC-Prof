"""Editable configuration snapshots retain the text the user actually saw."""
import os
from pathlib import Path

import pytest

from acprof.host.env_utils import ProjectEnvSnapshot, configurable_env_values, save_project_env


def test_snapshot_values_use_captured_local_text_and_existing_precedence(tmp_path):
    (tmp_path / ".env").write_text(
        "HF_TOKEN=legacy\nACPROF_MAX_DOWNLOAD=1GB\nACPROF_MODEL_STORE=legacy-store\n",
    )
    target = tmp_path / ".env.local"
    original = "HF_TOKEN=hf_captured\nACPROF_MAX_DOWNLOAD=2GB\nUNRELATED=value\n"
    target.write_text(original)
    snapshot = ProjectEnvSnapshot.read(tmp_path)
    environment = {"ACPROF_MODEL_STORE": "process-store"}
    target.write_text("HF_TOKEN=hf_external\nACPROF_MAX_DOWNLOAD=3GB\n")

    values = snapshot.configurable_values(environ=environment)

    assert values["HF_TOKEN"] == "hf_captured"
    assert values["ACPROF_MAX_DOWNLOAD"] == "2GB"
    assert values["ACPROF_MODEL_STORE"] == "process-store"
    assert "UNRELATED" not in values
    assert snapshot.text == original
    assert environment == {"ACPROF_MODEL_STORE": "process-store"}
    assert "hf_captured" not in repr(snapshot)
    current = configurable_env_values(tmp_path, environ=environment)
    assert current["HF_TOKEN"] == "hf_external"
    assert current["ACPROF_MAX_DOWNLOAD"] == "3GB"


def test_missing_local_snapshot_retains_legacy_values_and_environment_alias(tmp_path):
    (tmp_path / ".env").write_text("ACPROF_MAX_DOWNLOAD=1GB\n")
    snapshot = ProjectEnvSnapshot.read(tmp_path)
    values = snapshot.configurable_values(environ={"HUGGING_FACE_HUB_TOKEN": "hf_process"})
    assert snapshot.text == ""
    assert values["HF_TOKEN"] == "hf_process"
    assert values["ACPROF_MAX_DOWNLOAD"] == "1GB"
    assert not snapshot.path.exists()


def test_saved_snapshot_supports_repeated_edits_and_private_backups(tmp_path):
    target = tmp_path / ".env.local"
    original = "# keep\nUNRELATED=value\nHF_TOKEN=previous\n"
    target.write_text(original)
    snapshot = ProjectEnvSnapshot.read(tmp_path)
    environment = {"HF_TOKEN": "previous"}

    first = snapshot.save({"HF_TOKEN": "hf_first"}, environ=environment)
    first_text = '# keep\nUNRELATED=value\nHF_TOKEN="hf_first"\nHUGGING_FACE_HUB_TOKEN="hf_first"\n'
    assert first.path == target
    assert first.text == first_text
    assert first.text == target.read_text()
    assert snapshot.text == original
    assert "hf_first" not in repr(first)

    second = first.save({"HF_TOKEN": "  hf_second  "}, environ=environment)
    assert second.text == '# keep\nUNRELATED=value\nHF_TOKEN="hf_second"\nHUGGING_FACE_HUB_TOKEN="hf_second"\n'
    assert target.read_text() == second.text
    assert (tmp_path / ".env.local.bak").read_text() == first_text
    assert target.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / ".env.local.bak").stat().st_mode & 0o777 == 0o600
    assert environment == {"HF_TOKEN": "hf_second", "HUGGING_FACE_HUB_TOKEN": "hf_second"}


def test_snapshot_can_retry_after_failed_publication(tmp_path, monkeypatch):
    target = tmp_path / ".env.local"
    original = "HF_TOKEN=previous\n"
    target.write_text(original)
    snapshot = ProjectEnvSnapshot.read(tmp_path)
    environment = {"HF_TOKEN": "previous"}
    original_replace = os.replace
    primary = PermissionError("publication rejected")

    def replace(source, destination, *args, **kwargs):
        if Path(destination) == target:
            raise primary
        original_replace(source, destination, *args, **kwargs)

    with monkeypatch.context() as faults:
        faults.setattr(os, "replace", replace)
        with pytest.raises(PermissionError, match="publication rejected") as caught:
            snapshot.save({"HF_TOKEN": "hf_testonly"}, environ=environment)
    assert caught.value is primary
    assert snapshot.text == original
    assert target.read_text() == original
    assert environment == {"HF_TOKEN": "previous"}
    assert not list(tmp_path.glob("*.tmp"))

    saved = snapshot.save({"HF_TOKEN": "hf_testonly"}, environ=environment)
    assert saved.text == target.read_text()
    assert environment["HF_TOKEN"] == "hf_testonly"


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_snapshot_read_rejects_non_regular_local_file(tmp_path, kind):
    target = tmp_path / ".env.local"
    if kind == "symlink":
        outside = tmp_path / "outside"
        outside.write_text("HF_TOKEN=hf_external\n")
        target.symlink_to(outside)
    else:
        target.mkdir()
    with pytest.raises(ValueError, match="regular file, not a symbolic link"):
        ProjectEnvSnapshot.read(tmp_path)


def test_legacy_save_preserves_path_return_contract(tmp_path):
    target = save_project_env(tmp_path, {"HF_TOKEN": "hf_testonly"}, environ={})
    assert isinstance(target, Path)
    assert target == tmp_path / ".env.local"
    assert 'HF_TOKEN="hf_testonly"' in target.read_text()
