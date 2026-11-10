"""Docker consumers outlive the client process and its advisory file lock."""
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acprof.host.model_store import consumers, store
from acprof.models.evidence import content_digest
from tests.support.model_store_fixtures import ModelStoreFixture


@dataclass(frozen=True)
class CachedModel:
    root: Path
    blob: Path
    key: str
    manifest: dict


@pytest.fixture
def cached_model(tmp_path) -> CachedModel:
    fixture = ModelStoreFixture(tmp_path)
    blob = fixture.blob("payload", 128)
    key = fixture.entry(1, [blob])
    plan = store.read_entry(key, tmp_path)
    assert isinstance(plan, dict)
    manifest = {"model_store": {"entry_id": key, "plan_sha256": plan["plan_sha256"]}}
    return CachedModel(root=tmp_path, blob=blob, key=key, manifest=manifest)


def abandon_mount(*models: CachedModel, phase="registered"):
    result = subprocess.run([
        sys.executable, "-c",
        "import json,os,sys; from pathlib import Path; "
        "from acprof.host.model_store.store import acquire_mount; "
        "leases=[acquire_mount(manifest, Path(root)) for root,manifest in json.loads(sys.argv[1])]\n"
        "if sys.argv[2] != 'idle':\n"
        " for lease in leases:\n"
        "  cidfile=lease.begin_launch(Path('unused.cid'))\n"
        "  if sys.argv[2] in {'registered','settled','redispatched'}: cidfile.write_text('a' * 64)\n"
        "  if sys.argv[2] in {'settled','redispatched'}: lease.finish_launch()\n"
        "  if sys.argv[2] == 'redispatched': lease.begin_launch(Path('unused.cid'))\n"
        "os._exit(0)",
        json.dumps([(str(model.root), model.manifest) for model in models]), phase,
    ], capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def docker_inventory(monkeypatch):
    state = SimpleNamespace(present=True, commands=[], failure=None, preflight=Mock())

    def run(argv, **kwargs):
        state.commands.append((argv, kwargs))
        assert argv[:2] == ["docker", "ps"], "GC may inspect consumers but must not stop them"
        assert 0 < kwargs["timeout"] <= 30
        if isinstance(state.failure, BaseException):
            raise state.failure
        if state.failure is True:
            return subprocess.CompletedProcess(argv, 1, "", "daemon unavailable")
        if isinstance(state.failure, str):
            return subprocess.CompletedProcess(argv, 0, state.failure, "")
        return subprocess.CompletedProcess(argv, 0, "a" * 64 if state.present else "", "")

    monkeypatch.setattr("acprof.host.preflight.require_native_docker", state.preflight)
    monkeypatch.setattr("acprof.system.command.run_command", run)
    return state


@pytest.mark.parametrize("apply", [False, True])
def test_crashed_client_cannot_make_a_live_docker_consumer_gc_eligible(cached_model, docker_inventory, apply):
    model = cached_model
    abandon_mount(model)

    protected = store.prune_store(root=model.root, apply=apply)

    assert protected["entries"] == []
    assert protected["reclaimable_bytes"] == 0
    assert model.blob.read_bytes() == b"x" * 128
    assert (model.root / "entries" / model.key).is_dir()
    owner = json.loads(consumer_files(model)[0].read_text())["owner_labels"]
    command, _kwargs = docker_inventory.commands[0]
    assert command[:4] == ["docker", "ps", "-aq", "--no-trunc"]
    assert command[4::2] == ["--filter"] * len(owner)
    assert set(command[5::2]) == {f"label={key}={value}" for key, value in owner.items()}
    docker_inventory.present = False
    cleared = store.prune_store(root=model.root, apply=True)
    assert cleared["entries"] == [model.key]
    assert not model.blob.exists()


def consumer_files(model: CachedModel) -> list[Path]:
    return sorted((model.root / "entries" / model.key).glob(".docker-consumer-*.json"))


def test_owner_inventory_is_reused_within_each_prune_but_refreshed_next_time(cached_model, docker_inventory):
    model = cached_model
    key = ModelStoreFixture(model.root).entry(2, [model.blob])
    plan = store.read_entry(key, model.root)
    assert isinstance(plan, dict)
    other = CachedModel(root=model.root, blob=model.blob, key=key, manifest={"model_store": {
        "entry_id": key, "plan_sha256": plan["plan_sha256"]}})
    abandon_mount(model, other)

    assert store.prune_store(root=model.root, apply=True)["entries"] == []
    assert len(docker_inventory.commands) == 1
    docker_inventory.preflight.assert_called_once()
    docker_inventory.present = False
    assert set(store.prune_store(root=model.root, apply=True)["entries"]) == {model.key, key}
    assert len(docker_inventory.commands) == 2
    assert not model.blob.exists()


def test_normal_close_releases_only_its_own_record(cached_model, docker_inventory):
    model = cached_model
    first = store.acquire_mount(model.manifest, model.root)
    second = store.acquire_mount(model.manifest, model.root)
    try:
        assert len(consumer_files(model)) == 2
        first.close()
        first.close()
        assert len(consumer_files(model)) == 1
        assert store.prune_store(root=model.root, apply=True)["entries"] == []
    finally:
        first.close()
        second.close()
    assert consumer_files(model) == []
    assert store.prune_store(root=model.root, apply=True)["entries"] == [model.key]
    docker_inventory.preflight.assert_not_called()
    assert docker_inventory.commands == []


def test_process_consumers_use_inherited_locks_without_docker_records(cached_model, docker_inventory):
    model = cached_model
    with store.acquire_mount(model.manifest, model.root, runtime="process") as mount:
        assert len(mount.pass_fds) == 1
        assert consumer_files(model) == []
        assert store.prune_store(root=model.root, apply=True)["entries"] == []
    assert store.prune_store(root=model.root, apply=True)["entries"] == [model.key]
    docker_inventory.preflight.assert_not_called()
    assert docker_inventory.commands == []


def test_legacy_unleased_entry_without_consumer_records_needs_no_docker(cached_model, docker_inventory):
    assert store.prune_store(root=cached_model.root, apply=True)["entries"] == [cached_model.key]
    docker_inventory.preflight.assert_not_called()


@pytest.mark.parametrize("failure", [True, PermissionError("denied"),
                                   subprocess.TimeoutExpired(["docker", "ps"], 1), "invalid-container-id"])
def test_unavailable_inventory_preserves_data_and_consumer_evidence(cached_model, docker_inventory, failure):
    model = cached_model
    abandon_mount(model)
    record = consumer_files(model)[0]
    original = record.read_bytes()
    docker_inventory.failure = failure

    with pytest.raises(RuntimeError, match="cleanup"):
        store.prune_store(root=model.root, apply=True)

    assert record.read_bytes() == original
    assert model.blob.read_bytes() == b"x" * 128


def test_unverified_daemon_blocks_gc_without_downgrading_the_failure(cached_model, docker_inventory):
    abandon_mount(cached_model)
    failure = SystemExit(2)
    docker_inventory.preflight.side_effect = failure
    with pytest.raises(RuntimeError, match="local daemon") as caught:
        store.prune_store(root=cached_model.root, apply=True)
    assert caught.value.__cause__ is failure
    assert cached_model.blob.exists()
    assert docker_inventory.commands == []


def test_cancellation_after_daemon_check_does_not_start_another_query(cached_model, docker_inventory):
    abandon_mount(cached_model)
    cancel = Event()
    docker_inventory.preflight.side_effect = cancel.set
    with pytest.raises(store.ModelStoreCancelled):
        store.prune_store(root=cached_model.root, apply=True, cancel=cancel)
    assert docker_inventory.commands == []
    assert cached_model.blob.exists()


@pytest.mark.parametrize("owner", ["live", "unreadable", "foreign_host", "foreign_user"])
def test_unconfirmed_owner_exit_cannot_release_records(cached_model, docker_inventory, monkeypatch, owner):
    from acprof.host import container_lifecycle

    model = cached_model
    abandon_mount(model)
    record = consumer_files(model)[0]
    payload = json.loads(record.read_text())
    if owner == "live":
        payload["owner_labels"] = container_lifecycle.container_owner_labels()
    elif owner == "foreign_host":
        payload["owner_labels"]["org.acprof.owner.host"] = "b" * 64
    elif owner == "foreign_user":
        payload["owner_labels"]["org.acprof.owner.uid"] = str(os.getuid() + 1)
    else:
        process_identity = container_lifecycle._process_identity

        def unreadable(pid):
            if pid != os.getpid():
                raise PermissionError("procfs denied")
            return process_identity(pid)

        monkeypatch.setattr("acprof.host.container_lifecycle._process_identity", unreadable)
    payload.pop("record_sha256")
    payload["record_sha256"] = content_digest(payload)
    record.write_text(json.dumps(payload))
    assert store.prune_store(root=model.root, apply=True)["entries"] == []
    assert record.exists()
    assert model.blob.exists()
    docker_inventory.preflight.assert_not_called()


@pytest.mark.parametrize("damage", ["json", "version", "boolean_version", "entry", "identity", "owner", "owner_identity", "size",
                                   "symlink", "hardlink", "directory"])
def test_invalid_consumer_evidence_fails_before_deleting_any_payload(cached_model, docker_inventory, damage):
    model = cached_model
    abandon_mount(model)
    record = consumer_files(model)[0]
    payload = json.loads(record.read_text())
    if damage == "json":
        record.write_text("{")
    elif damage == "size":
        record.write_text(" " * 4097)
    elif damage in {"symlink", "hardlink", "directory"}:
        record.unlink()
        if damage == "symlink":
            record.symlink_to(model.blob)
        elif damage == "hardlink":
            os.link(model.blob, record)
        else:
            record.mkdir()
    else:
        if damage in {"version", "boolean_version"}:
            payload["schema_version"] = 3 if damage == "version" else True
        elif damage == "entry":
            payload["entry_id"] = "f" * 64
        elif damage == "identity":
            payload["consumer_id"] = "f" * 32
        elif damage == "owner_identity":
            payload["owner_labels"]["org.acprof.owner.pid"] = "999999999"
            docker_inventory.present = False
        else:
            payload["owner_labels"].pop("org.acprof.owner.pid")
        record.write_text(json.dumps(payload))

    with pytest.raises((OSError, ValueError)):
        store.prune_store(root=model.root, apply=True)

    assert model.blob.read_bytes() == b"x" * 128
    assert record.exists()
    docker_inventory.preflight.assert_not_called()


def test_failed_record_release_keeps_the_lock_and_can_retry(cached_model, docker_inventory, monkeypatch):
    model = cached_model
    mount = store.acquire_mount(model.manifest, model.root)
    failure = PermissionError("cannot release consumer")
    try:
        with monkeypatch.context() as patcher:
            patcher.setattr("acprof.host.model_store.consumers.os.unlink", Mock(side_effect=failure))
            with pytest.raises(PermissionError) as caught:
                mount.close()
            assert caught.value is failure
            assert len(consumer_files(model)) == 1
            assert len(mount.pass_fds) == 1
            assert store.prune_store(root=model.root)["entries"] == []
    finally:
        mount.close()
    assert consumer_files(model) == []
    assert store.prune_store(root=model.root, apply=True)["entries"] == [model.key]


@pytest.mark.parametrize("published", [False, True])
def test_failed_record_publication_never_returns_a_mount(cached_model, monkeypatch, published):
    model = cached_model
    original_write = consumers.atomic_write_json
    failure = OSError("consumer publication failed")

    def publish(path, payload, **kwargs):
        if published:
            original_write(path, payload, **kwargs)
        raise failure

    monkeypatch.setattr("acprof.host.model_store.consumers.atomic_write_json", publish)
    with pytest.raises(OSError, match="consumer publication failed") as caught:
        store.acquire_mount(model.manifest, model.root)
    assert caught.value is failure
    assert bool(consumer_files(model)) is published
    assert store.prune_store(root=model.root)["entries"] == ([] if published else [model.key])


def test_preview_does_not_remove_an_absent_consumers_record(cached_model, docker_inventory):
    model = cached_model
    abandon_mount(model)
    record = consumer_files(model)[0]
    original = record.read_bytes()
    docker_inventory.present = False
    assert store.prune_store(root=model.root)["entries"] == [model.key]
    assert record.read_bytes() == original
    assert model.blob.exists()


def test_failed_release_sync_keeps_the_lock_until_a_durable_retry(cached_model, monkeypatch):
    model = cached_model
    mount = store.acquire_mount(model.manifest, model.root)
    sync = consumers.os.fsync
    try:
        with monkeypatch.context() as patcher:
            patcher.setattr("acprof.host.model_store.consumers.os.fsync", Mock(side_effect=OSError("sync failed")))
            with pytest.raises(OSError, match="sync failed"):
                mount.close()
        assert consumer_files(model) == []
        assert len(mount.pass_fds) == 1
        with monkeypatch.context() as patcher:
            retried = Mock(wraps=sync)
            patcher.setattr("acprof.host.model_store.consumers.os.fsync", retried)
            mount.close()
            retried.assert_called_once()
        assert mount.pass_fds == ()
    finally:
        mount.close()


def test_missing_plan_cannot_be_rebuilt_while_a_crashed_clients_container_remains(
        cached_model, docker_inventory, monkeypatch):
    from acprof.host.detect import TaskInfo
    from acprof.models.download_plan import PLAN_FILENAME

    model = cached_model
    plan = store.read_entry(model.key, model.root)
    assert isinstance(plan, dict)
    abandon_mount(model)
    (model.root / "entries" / model.key / PLAN_FILENAME).unlink()
    task = TaskInfo("example/test", "fill-mask", "nlp", "transformers_pipeline",
                    "transformers", "a" * 40, "fixture")
    task.model_download_policy = "full"
    monkeypatch.setattr(store, "entry_key", lambda _task: model.key)
    prepare = Mock(side_effect=AssertionError("active consumers must block preparation before downloads"))
    monkeypatch.setattr("acprof.host.model_store.preparation.prepare_repository", prepare)

    with pytest.raises(RuntimeError, match="Docker consumers"):
        store.prepare_model(task, plan, model.root)

    prepare.assert_not_called()
    assert model.blob.read_bytes() == b"x" * 128
    assert not (model.root / "entries" / model.key / PLAN_FILENAME).exists()
