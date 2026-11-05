"""A daemon can accept a queued create after its client process has exited."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.host.container_lifecycle import ContainerCleanupError, owned_container_id
from acprof.host.model_store import consumers, store
from acprof.models.evidence import content_digest
from tests.models.storage.test_model_store_docker_consumers import (
    abandon_mount,
    cached_model as cached_model,
    consumer_files,
    docker_inventory as docker_inventory,
)
from tests.support.model_store_fixtures import ModelStoreFixture

_QUEUED_LAUNCH = """
import json, os, subprocess, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
from acprof.system import command

root, manifest, runner = Path(sys.argv[1]), json.loads(sys.argv[2]), sys.argv[3]
tempfile.tempdir = str(root / 'temporary')
Path(tempfile.tempdir).mkdir()
def dispatch(argv, **kwargs):
    if argv[:2] == ['docker', 'ps']:
        return subprocess.CompletedProcess(argv, 0, '', '')
    assert argv[:2] in (['docker', 'run'], ['docker', 'create']), argv
    request = {'cidfile': argv[argv.index('--cidfile') + 1], 'argv': argv}
    (root / 'queued-create.json').write_text(json.dumps(request))
    os._exit(0)  # The queued daemon request outlives this client and its flock.
command.run_command = dispatch

from acprof.host.detect import TaskInfo
from acprof.host.images.runtime import ImageInfo
task = TaskInfo('example/test', 'fill-mask', 'nlp', 'transformers_pipeline',
                'transformers', 'a' * 40, 'fixture')
manifest['model_store']['host_path'] = str(root)
manifest['build_fingerprint'] = 'fixture'
info = ImageInfo('sha256:' + 'b' * 64, runtime_environment=manifest)
if runner == 'session':
    from acprof.host.docker_runtime import start_container_session
    start_container_session(task, 1, 1, 'off', info, 'queued', '[test]')
elif runner == 'validation':
    from acprof.host.runtime_validation import validate_runtime
    plan = root / 'input.json'
    plan.write_text(json.dumps({'schema_version': 2, 'entries': [
        {'input_scale': 1, 'payload': {'text': 'fixture'}}]}))
    validate_runtime(task_info=task, image_info=info, planned=SimpleNamespace(plan_file=str(plan)),
        cpu_list=[1], mem_list=[1], gpu_list=['off'], output_dir=str(root / 'validation'))
elif runner == 'egress':
    from acprof.host.model_store.store import acquire_mount
    from scripts.validate_measurement_egress import Experiment
    experiment = object.__new__(Experiment)
    experiment.output, experiment.identifier, experiment.image = root, 'fixture', info.tag
    experiment.cidfiles, experiment.run = [], dispatch
    mount = acquire_mount(manifest)
    experiment.model_case('queued', mount, gpu=False, controlled=False)
else:
    from acprof.host.profiling.support import profiler_container_command
    from acprof.host.profiling.commands import run_profiler_command
    task.model_store = manifest['model_store']
    with profiler_container_command(task_info=task, image_tag=info.tag, cpu=1, mem=1,
            use_gpu=False, payload_file=str(root / 'input.json'), profile_root=str(root / 'profiles'),
            tool_mount_roots=()) as argv:
        run_profiler_command([*argv, 'fixture'], check=False)
raise AssertionError('fixture must exit while creation is pending')
"""


@pytest.mark.parametrize("runner", ["session", "validation", "profiler", "egress"])
@pytest.mark.parametrize("apply", [False, True])
def test_queued_creation_keeps_models_until_creation_and_cleanup_are_confirmed(
        cached_model, docker_inventory, runner, apply):
    model = cached_model
    result = subprocess.run([sys.executable, "-c", _QUEUED_LAUNCH,
                             str(model.root), json.dumps(model.manifest), runner],
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    request = json.loads((model.root / "queued-create.json").read_text())
    docker_inventory.present = False  # Request queued; no container is registered yet.

    assert store.prune_store(root=model.root, apply=apply)["entries"] == []
    assert model.blob.read_bytes() == b"x" * 128
    assert docker_inventory.commands == []

    cidfile = Path(request["cidfile"])
    assert cidfile.parent == model.root / "entries" / model.key
    docker_inventory.present = True
    cidfile.write_text("a" * 64)  # Docker CLI acknowledges daemon registration.
    assert store.prune_store(root=model.root, apply=True)["entries"] == []
    docker_inventory.present = False
    assert store.prune_store(root=model.root, apply=True)["entries"] == [model.key]
    assert not model.blob.exists()


@pytest.mark.parametrize("phase", ["idle", "settled"])
def test_no_pending_dispatch_needs_no_daemon_after_owner_exit(cached_model, docker_inventory, phase):
    abandon_mount(cached_model, phase=phase)
    assert store.prune_store(root=cached_model.root, apply=True)["entries"] == [cached_model.key]
    docker_inventory.preflight.assert_not_called()


def test_previous_creation_receipt_cannot_release_the_next_pending_launch(cached_model, docker_inventory):
    abandon_mount(cached_model, phase="redispatched")
    docker_inventory.present = False
    assert store.prune_store(root=cached_model.root, apply=True)["entries"] == []
    assert cached_model.blob.exists()
    docker_inventory.preflight.assert_not_called()


def test_legacy_record_without_dispatch_evidence_is_preserved(cached_model, docker_inventory):
    abandon_mount(cached_model)
    path = consumer_files(cached_model)[0]
    record = json.loads(path.read_text())
    record["schema_version"] = 1
    record.pop("launch_state")
    record.pop("record_sha256")
    record["record_sha256"] = content_digest(record)
    path.write_text(json.dumps(record))
    original = path.read_bytes()
    docker_inventory.present = False
    assert store.prune_store(root=cached_model.root, apply=True)["entries"] == []
    assert path.read_bytes() == original
    docker_inventory.preflight.assert_not_called()


@pytest.mark.parametrize("damage", ["missing", "empty", "partial", "oversized", "symlink", "hardlink", "directory", "fifo"])
def test_unconfirmed_or_invalid_creation_receipt_never_releases_models(cached_model, docker_inventory, damage):
    model = cached_model
    abandon_mount(model, phase="pending")
    record = json.loads(consumer_files(model)[0].read_text())
    cidfile = model.root / "entries" / model.key / f".docker-cid-{record['consumer_id']}"
    if damage == "empty":
        cidfile.touch()
    elif damage == "partial":
        cidfile.write_text("a" * 12)
    elif damage == "oversized":
        cidfile.write_text(" " * 8192 + "a" * 64)
    elif damage == "symlink":
        cidfile.symlink_to(model.blob)
    elif damage == "hardlink":
        os.link(model.blob, cidfile)
    elif damage == "directory":
        cidfile.mkdir()
    elif damage == "fifo":
        os.mkfifo(cidfile)
    docker_inventory.present = False
    if damage in {"missing", "empty"}:
        assert store.prune_store(root=model.root, apply=True)["entries"] == []
    else:
        with pytest.raises((ValueError, OSError)):
            store.prune_store(root=model.root, apply=True)
    assert model.blob.read_bytes() == b"x" * 128
    docker_inventory.preflight.assert_not_called()


@pytest.mark.parametrize("damage", ["symlink", "hardlink", "oversized"])
def test_cleanup_cid_reader_rejects_unsafe_creation_receipts(tmp_path, damage):
    source = tmp_path / "source"
    source.write_text("a" * 64)
    cidfile = tmp_path / "container.cid"
    if damage == "symlink":
        cidfile.symlink_to(source)
    elif damage == "hardlink":
        os.link(source, cidfile)
    else:
        cidfile.write_text(" " * 8192 + "a" * 64)
    with pytest.raises(ContainerCleanupError):
        owned_container_id(cidfile)


@pytest.mark.parametrize("published", [False, True])
def test_failed_dispatch_publication_prevents_start_and_allows_pre_dispatch_close(cached_model, monkeypatch, published):
    model = cached_model
    mount = store.acquire_mount(model.manifest, model.root)
    original = consumers.atomic_write_json

    def fail(path, record, **kwargs):
        if published:
            original(path, record, **kwargs)
        raise OSError("dispatch publication failed")

    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(consumers, "atomic_write_json", fail)
            with pytest.raises(OSError, match="dispatch publication failed"):
                mount.begin_launch(model.root / "fallback.cid")
        assert mount.pass_fds
        record = json.loads(consumer_files(model)[0].read_text())
        assert record["launch_state"] == ("dispatched" if published else "idle")
    finally:
        mount.close()  # This caller has not dispatched Docker.
    assert consumer_files(model) == []
    assert mount.pass_fds == ()


def test_unsettled_launch_cannot_reuse_its_receipt(cached_model):
    model = cached_model
    with store.acquire_mount(model.manifest, model.root) as mount:
        cidfile = mount.begin_launch(model.root / "fallback.cid")
        cidfile.write_text("a" * 64)
        with pytest.raises(RuntimeError, match="unsettled launch"):
            mount.begin_launch(model.root / "fallback.cid")
        assert cidfile.read_text() == "a" * 64
        mount.finish_launch()  # Fixture confirms the old container has been removed.
        assert not cidfile.exists()


def test_cleanup_receipt_unlink_failure_can_retry_without_stale_identity(cached_model, monkeypatch):
    model = cached_model
    with store.acquire_mount(model.manifest, model.root) as mount:
        cidfile = mount.begin_launch(model.root / "fallback.cid")
        cidfile.write_text("a" * 64)
        unlink = os.unlink

        def fail(path, *args, **kwargs):
            if os.fspath(path) == cidfile.name:
                raise PermissionError("receipt unlink failed")
            return unlink(path, *args, **kwargs)

        with monkeypatch.context() as patcher:
            patcher.setattr(consumers.os, "unlink", fail)
            with pytest.raises(PermissionError, match="receipt unlink failed"):
                mount.finish_launch()
        assert cidfile.exists()
        mount.finish_launch()
        assert not cidfile.exists()
        assert mount.begin_launch(model.root / "fallback.cid") == cidfile
        assert not cidfile.exists()


@pytest.mark.parametrize("state", [None, True, [], {}, "unknown"])
def test_invalid_launch_state_cannot_be_treated_as_idle(cached_model, docker_inventory, state):
    abandon_mount(cached_model, phase="pending")
    path = consumer_files(cached_model)[0]
    record = json.loads(path.read_text())
    record["launch_state"] = state
    record.pop("record_sha256")
    record["record_sha256"] = content_digest(record)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="launch state"):
        store.prune_store(root=cached_model.root, apply=True)
    assert cached_model.blob.exists()
    docker_inventory.preflight.assert_not_called()


def test_profiler_commands_keep_the_correct_mount_across_nested_scopes(cached_model, monkeypatch):
    from acprof.host.profiling import commands

    model = cached_model
    key = ModelStoreFixture(model.root).entry(2, [model.blob])
    plan = store.read_entry(key, model.root)
    assert isinstance(plan, dict)
    other = {"model_store": {"entry_id": key, "plan_sha256": plan["plan_sha256"]}}
    cidfiles: list[Path] = []

    def run(argv, **kwargs):
        if argv[:2] == ["docker", "run"]:
            cidfile = Path(argv[argv.index("--cidfile") + 1])
            assert not cidfile.exists()
            cidfiles.append(cidfile)
            cidfile.write_text(f"{len(cidfiles):064x}")
        else:
            assert argv[:2] in (["docker", "ps"], ["docker", "rm"])
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(commands, "run_command", run)
    with store.acquire_mount(model.manifest, model.root) as first, store.acquire_mount(other, model.root) as second:
        with commands.profiler_model_mount(first):
            commands.run_profiler_command(["docker", "run", "fixture"])
            with commands.profiler_model_mount(second):
                commands.run_profiler_command(["docker", "run", "fixture"])
            commands.run_profiler_command(["docker", "run", "fixture"])
        commands.run_profiler_command(["docker", "run", "fixture"])
        assert [path.parent.name for path in cidfiles[:3]] == [model.key, key, model.key]
        assert cidfiles[3].parent.parent != model.root / "entries"
        assert all(not path.exists() for path in cidfiles)


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("failure_at", ["none", "experiment", "mount"])
def test_egress_harness_releases_mount_only_after_completed_work_and_cleanup(monkeypatch, completed, failure_at):
    from scripts import validate_measurement_egress as egress

    experiment = SimpleNamespace(close=Mock())
    mount = SimpleNamespace(close=Mock())
    retained = Mock()
    monkeypatch.setattr(egress, "retain_mount_for_cleanup_debt", retained)
    if failure_at == "experiment":
        experiment.close.side_effect = OSError("cleanup failed")
    if failure_at == "mount":
        mount.close.side_effect = OSError("cleanup failed")
    if failure_at == "experiment" or (completed and failure_at == "mount"):
        with pytest.raises(OSError, match="cleanup failed"):
            egress._close_experiment(experiment, mount, completed=completed)
    else:
        egress._close_experiment(experiment, mount, completed=completed)
    if completed and failure_at == "none":
        mount.close.assert_called_once()
        retained.assert_not_called()
    else:
        retained.assert_called_once_with(mount)
