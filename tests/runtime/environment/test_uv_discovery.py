import os

import pytest

from acprof.system.uv import probe_process_uv, require_process_uv
from acprof.tui.run.diagnostics import completed_command, process_uv_check


@pytest.fixture
def user_uv(tmp_path, monkeypatch):
    user_bin = tmp_path / ".local/bin"
    user_bin.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    for name in ("XDG_BIN_HOME", "XDG_DATA_HOME", "UV_INSTALL_DIR"):
        monkeypatch.delenv(name, raising=False)
    return user_bin / "uv"


def write_uv(path):
    path.write_text('#!/bin/sh\nif [ "$1" = "--version" ]; then echo "uv 0.13.5"; fi\n')
    path.chmod(0o700)


def test_process_uses_user_bin_without_changing_path(user_uv):
    before = os.environ["PATH"]
    write_uv(user_uv)
    check = process_uv_check(completed_command)
    assert check.status == "ok"
    assert check.reason_code == "cli_not_on_path"
    assert str(user_uv) in check.detail
    assert "0.13.5" in check.detail
    assert require_process_uv() == str(user_uv)
    assert os.environ["PATH"] == before


def test_uv_path_precedence_and_custom_install_directory(user_uv, monkeypatch, tmp_path):
    custom = tmp_path / "custom"
    custom.mkdir()
    monkeypatch.setenv("UV_INSTALL_DIR", str(custom))
    write_uv(custom / "uv")
    write_uv(user_uv)
    assert require_process_uv() == str(custom / "uv")
    monkeypatch.setenv("PATH", str(user_uv.parent))
    assert require_process_uv() == str(user_uv)
    assert not process_uv_check(completed_command).reason_code


@pytest.mark.parametrize("failure", ["missing", "permission", "old"])
def test_process_uv_failures_have_distinct_reasons(user_uv, failure):
    if failure != "missing":
        user_uv.write_text('#!/bin/sh\nif [ "$1" = "--version" ]; then echo "uv 0.1.0"; else echo "unexpected argument" >&2; exit 2; fi\n')
        user_uv.chmod(0o600 if failure == "permission" else 0o700)
    expected = {"missing": "not_installed", "permission": "permission_denied", "old": "version_incompatible"}
    check = process_uv_check(completed_command)
    assert check.status == "fail"
    assert check.reason_code == expected[failure]
    assert process_uv_check(completed_command, preparation=True).status == "warn"
    assert probe_process_uv().detail == check.detail
    with pytest.raises(RuntimeError) as caught:
        require_process_uv()
    assert check.detail in str(caught.value)


@pytest.mark.parametrize("runtime", ["process", "auto"])
def test_quick_preflight_uv_owns_default_probe_and_preserves_injected_runner(user_uv, monkeypatch, runtime):
    from types import SimpleNamespace

    from acprof.experiments.configuration import RunConfig
    from acprof.system import uv
    from acprof.tui.run import diagnostics

    write_uv(user_uv)
    monkeypatch.setattr(diagnostics, "inspect_docker", lambda *a: SimpleNamespace(code="cli_missing", summary="fixture"))
    monkeypatch.setattr(diagnostics, "_automatic_checks", lambda *a, **kw: [])
    monkeypatch.setattr("acprof.host.runtimes.resources.preflight_resources", lambda *a: None)
    monkeypatch.setattr("acprof.host.runtimes.process_control.require_process_supervision", lambda: None)
    config = RunConfig(model="fixture", runtime=runtime, gpus="off", profiling_mode="best_effort")
    default_calls, custom_calls = [], []

    def default_runner(command, *, timeout, kill_process_group):
        default_calls.append((command, kill_process_group))
        return SimpleNamespace(returncode=0, stdout="uv fixture", stderr="")

    def custom_runner(command, *, timeout):
        custom_calls.append(command)
        return SimpleNamespace(returncode=0, stdout="uv fixture", stderr="")

    monkeypatch.setattr(uv, "run_command", default_runner)
    for runner in (None, custom_runner):
        rows = diagnostics.quick_preflight(config, command_runner=runner)
        assert next(row for row in rows if row.check_id == "uv").status == "ok"
    assert default_calls
    assert all(owned == (os.name == "posix") for _, owned in default_calls)
    assert [command for command, _ in default_calls] == custom_calls
