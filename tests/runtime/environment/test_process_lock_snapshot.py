"""Process snapshots retain the exact dependency lock used by installation and loading."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from acprof.host.runtimes import environment
from acprof.runtime.dependency_locks import content_digest
from acprof.runtime.profiles import PROFILES, locked_transformers_version

LOCK = ("transformers @ https://example.org/transformers-4.57.6-py3-none-any.whl "
        "--hash=sha256:" + "a" * 64 + "\n")


@pytest.fixture
def process_project(tmp_path, monkeypatch):
    project = tmp_path / "project"
    profile = PROFILES["nlp-cu128"]
    lock = project / profile.environment.requirements_lock
    lock.parent.mkdir(parents=True)
    lock.write_text(LOCK)
    package = project / "acprof"
    package.mkdir()
    (package / "__init__.py").write_text("# fixture service\n")
    monkeypatch.setenv("ACPROF_PROCESS_STORE", str(tmp_path / "environments"))
    monkeypatch.setattr(environment, "select_runtime_profile", lambda task: profile)
    monkeypatch.setattr(environment.shutil, "which", lambda name: "/fixture/uv")
    monkeypatch.setattr(environment, "installed_packages", lambda python: {"transformers": "4.57.6"})
    commands = []

    def installer(command, **kwargs):
        commands.append(command)
        if command[1] == "venv":
            python = Path(command[-1]) / "bin/python"
            python.parent.mkdir(parents=True)
            python.write_bytes(b"fixture interpreter")
        elif command[1:3] == ["pip", "sync"]:
            assert Path(command[-1]).read_text() == LOCK
        else:
            raise AssertionError(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(environment, "run_command", installer)
    return project, profile.environment, commands


@pytest.fixture
def prepared_process(process_project):
    project, declaration, commands = process_project
    root, manifest = environment.prepare_environment(None, project)
    return root, manifest, declaration, commands


def test_process_loader_reads_pin_from_relocated_snapshot(prepared_process, monkeypatch):
    root, manifest, declaration, _ = prepared_process
    monkeypatch.delenv("ACPROF_DEPENDENCY_LOCK_SHA256", raising=False)
    monkeypatch.setattr("acprof.installation.resource_root", lambda: root / "code")
    locked_transformers_version.cache_clear()
    try:
        assert locked_transformers_version(declaration) == "4.57.6"
    finally:
        locked_transformers_version.cache_clear()
    environment.verify_environment(root, manifest)


def test_installer_consumes_the_staged_lock(prepared_process, process_project):
    root, _, declaration, commands = prepared_process
    project, _, _ = process_project
    install = next(command for command in commands if command[1:3] == ["pip", "sync"])
    staged = Path(install[-1])
    assert staged != project / declaration.requirements_lock
    assert staged.relative_to(staged.parents[3]).as_posix() == "code/" + declaration.requirements_lock
    assert (root / "code" / declaration.requirements_lock).read_text() == LOCK


@pytest.mark.parametrize("change", ["missing", "changed", "symlink", "escape", "legacy"])
def test_reuse_rejects_missing_or_changed_lock(prepared_process, change):
    root, manifest, declaration, _ = prepared_process
    lock = root / "code" / declaration.requirements_lock
    if change == "missing":
        lock.unlink()
    elif change == "changed":
        lock.write_text(LOCK + "# same package versions, altered lock\n")
    elif change == "symlink":
        other = lock.with_suffix(".copy")
        other.write_text(LOCK)
        lock.unlink()
        lock.symlink_to(other)
    else:
        if change == "escape":
            manifest["identity"]["requirements_lock"] = "../requirements.lock"
            (root / "requirements.lock").write_text(LOCK)
        else:
            manifest["identity"].pop("requirements_lock")
        manifest["environment_id"] = content_digest(manifest["identity"])
    with pytest.raises(ValueError, match="[Dd]ependency lock"):
        environment.verify_environment(root, manifest)


def test_lock_bytes_participate_in_environment_identity(process_project):
    project, declaration, _ = process_project
    _, before, _ = environment.environment_plan(None, project)
    lock = project / declaration.requirements_lock
    lock.write_text(LOCK + "# changed snapshot bytes\n")
    _, after, _ = environment.environment_plan(None, project)
    assert content_digest(before) != content_digest(after)


def test_source_lock_changed_after_planning_is_rejected_before_install(process_project, monkeypatch):
    project, declaration, commands = process_project
    plan = environment.environment_plan

    def changed_source(*args):
        result = plan(*args)
        (project / declaration.requirements_lock).write_text(LOCK + "# changed after planning\n")
        return result

    monkeypatch.setattr(environment, "environment_plan", changed_source)
    with pytest.raises(ValueError, match="[Dd]ependency lock"):
        environment.prepare_environment(None, project)
    assert commands == []
