"""A failed stop must retain its real GC lease beyond the session object's lifetime."""
import asyncio
import gc
import json
import subprocess
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_model_store_gc import TestModelStoreGc as _StoreFixture

from acprof.host import docker_runtime, model_store
from acprof.host.container_lifecycle import ContainerCleanupError


@pytest.fixture
def store(tmp_path, monkeypatch):
    retained = []
    monkeypatch.setattr(model_store, "_RETAINED_MOUNTS", retained)
    fixture = _StoreFixture()
    fixture.root = tmp_path
    blob = fixture.blob("consumer-data", 100)
    key = fixture.entry(1, [blob])
    plan = model_store.read_entry(key, tmp_path)
    manifest = {"model_store": {"entry_id": key, "plan_sha256": plan["plan_sha256"]}}
    mounts = []

    def acquire():
        mount = model_store.acquire_mount(manifest, tmp_path)
        # Test cleanup must not itself keep the lease alive for the assertions.
        mounts.append(weakref.ref(mount))
        return mount

    yield SimpleNamespace(root=tmp_path, blob=blob, key=key, acquire=acquire, retained=retained)
    for reference in mounts:
        mount = reference()
        if mount is not None:
            mount.close()


def _session(mount, identifier="a" * 64):
    return docker_runtime.RunningContainer(name="fixture", base_url="", host_port=1,
        cold_start_s=0, container_id=identifier, _model_store_mount=mount)


def _blocked(store):
    assert model_store.prune_store(root=store.root, apply=True) == {"entries": [], "reclaimable_bytes": 0}
    assert store.blob.read_bytes() == b"x" * 100


@pytest.mark.parametrize("error_type", [ContainerCleanupError, OSError, RuntimeError,
                                       KeyboardInterrupt, SystemExit, asyncio.CancelledError])
def test_failed_stop_keeps_lease_after_session_is_discarded(store, monkeypatch, error_type):
    def fail(*args, **kwargs):
        if error_type is ContainerCleanupError:
            raise ContainerCleanupError("a" * 64, [], final_state="present")
        raise error_type("cleanup interrupted")

    monkeypatch.setattr(docker_runtime, "remove_owned_container", fail)

    def discard_session():
        mount = store.acquire()
        reference = weakref.ref(mount)
        try:
            docker_runtime.stop_container_session(_session(mount))
        except error_type:
            pass
        return reference

    reference = discard_session()
    gc.collect()
    _blocked(store)
    assert reference() is not None
    assert store.retained == [reference()]


def test_retry_releases_only_its_lease_without_leaking_registry_entries(store, monkeypatch):
    first, second = store.acquire(), store.acquire()
    session = _session(first)
    failure = ContainerCleanupError(session.container_id, [], final_state="present")
    remove = Mock(side_effect=failure)
    monkeypatch.setattr(docker_runtime, "remove_owned_container", remove)
    for _ in range(3):
        with pytest.raises(ContainerCleanupError) as caught:
            docker_runtime.stop_container_session(session)
        assert caught.value is failure
    assert store.retained == [first]
    _blocked(store)

    remove.side_effect = None
    docker_runtime.stop_container_session(session)
    remove.assert_called_with(session.container_id, docker_runtime.host_command.run_command, stop=True)
    assert session._model_store_mount is None
    assert store.retained == []
    _blocked(store)  # A second consumer still owns its independent descriptor.
    second.close()
    assert model_store.prune_store(root=store.root, apply=True)["entries"] == [store.key]


def test_invalid_container_identity_never_unpins_or_removes_by_name(store, monkeypatch):
    mount = store.acquire()
    session = _session(mount, identifier="mutable-name")
    remove = Mock()
    monkeypatch.setattr(docker_runtime, "remove_owned_container", remove)
    with pytest.raises(ValueError, match="immutable ID"):
        docker_runtime.stop_container_session(session)
    remove.assert_not_called()
    assert store.retained == [mount]
    _blocked(store)


def test_repeated_retention_and_close_are_bounded_and_idempotent(store):
    mount = store.acquire()
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: model_store.retain_mount_for_cleanup_debt(mount), range(64)))
    assert store.retained == [mount]
    _blocked(store)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: mount.close(), range(16)))
    assert store.retained == []
    assert model_store.prune_store(root=store.root)["entries"] == [store.key]


def test_retained_mount_disappears_after_successful_close(store):
    mount = store.acquire()
    reference = weakref.ref(mount)
    model_store.retain_mount_for_cleanup_debt(mount)
    mount.close()
    del mount
    gc.collect()
    assert reference() is None
    assert store.retained == []


@pytest.mark.parametrize("confirmed", ["rm_success", "explicit_absence"])
def test_real_cleanup_flow_releases_only_after_positive_evidence(store, monkeypatch, confirmed):
    mount = store.acquire()
    session = _session(mount)
    commands = []
    recovered = False

    def run(command, **kwargs):
        commands.append(command)
        if recovered and confirmed == "rm_success" and command[:2] == ["docker", "rm"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:2] == ["docker", "inspect"]:
            if recovered:
                return subprocess.CompletedProcess(command, 1, "", f"No such object: {session.container_id}")
            payload = [{"Id": session.container_id, "State": {"Running": True}}]
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        return subprocess.CompletedProcess(command, 1, "", "cleanup pending")

    monkeypatch.setattr(docker_runtime.host_command, "run_command", run)
    with pytest.raises(ContainerCleanupError) as caught:
        docker_runtime.stop_container_session(session)
    assert caught.value.final_state == "present"
    assert store.retained == [mount]
    _blocked(store)
    recovered = True
    docker_runtime.stop_container_session(session)
    assert session._model_store_mount is None
    assert store.retained == []
    assert all(command[-1] == session.container_id for command in commands)
    assert model_store.prune_store(root=store.root)["entries"] == [store.key]


def test_separate_gc_process_observes_retained_lease_until_release(store, monkeypatch):
    def fail(*args, **kwargs):
        raise ContainerCleanupError("a" * 64, [], final_state="unknown")

    monkeypatch.setattr(docker_runtime, "remove_owned_container", fail)

    def abandon():
        mount = store.acquire()
        reference = weakref.ref(mount)
        try:
            docker_runtime.stop_container_session(_session(mount))
        except ContainerCleanupError:
            pass
        return reference

    reference = abandon()
    gc.collect()

    def child_gc():
        result = subprocess.run([sys.executable, "-c",
            "import json,sys; from pathlib import Path; from acprof.host.model_store import prune_store; "
            "print(json.dumps(prune_store(root=Path(sys.argv[1]), apply=True)))", str(store.root)],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(result.stdout)

    assert child_gc() == {"entries": [], "reclaimable_bytes": 0}
    assert store.blob.exists()
    # Explicit release models a caller that has positively confirmed cleanup.
    reference().close()
    assert child_gc() == {"entries": [store.key], "reclaimable_bytes": 100}
    assert not store.blob.exists()


@pytest.mark.parametrize("empty", [False, True])
def test_closed_or_empty_mount_does_not_accumulate_cleanup_debt(store, empty):
    mount = model_store.ModelStoreMount([]) if empty else store.acquire()
    mount.close()
    model_store.retain_mount_for_cleanup_debt(mount)
    assert store.retained == []


def test_concurrent_retention_cannot_resurrect_a_closed_mount(store):
    mount = store.acquire()
    with ThreadPoolExecutor(max_workers=4) as executor:
        operations = [executor.submit(mount.close if index % 2 else
            lambda: model_store.retain_mount_for_cleanup_debt(mount)) for index in range(64)]
        for operation in operations:
            operation.result(timeout=5)
    assert mount._lock is None
    assert store.retained == []


def test_close_error_preserves_retry_handle_and_registry(store, monkeypatch):
    mount = store.acquire()
    real_lock = mount._lock
    failure = OSError("descriptor close failed before release")
    wrapper = Mock()
    wrapper.close.side_effect = failure
    mount._lock = wrapper
    model_store.retain_mount_for_cleanup_debt(mount)
    session = _session(mount)
    monkeypatch.setattr(docker_runtime, "remove_owned_container", Mock())
    try:
        with pytest.raises(OSError) as caught:
            docker_runtime.stop_container_session(session)
        assert caught.value is failure
        assert session._model_store_mount is mount
        assert mount._lock is wrapper
        assert store.retained == [mount]
        _blocked(store)
        wrapper.close.side_effect = real_lock.close
        docker_runtime.stop_container_session(session)
        assert session._model_store_mount is None
        assert store.retained == []
        assert model_store.prune_store(root=store.root)["entries"] == [store.key]
    finally:
        real_lock.close()
        wrapper.close.side_effect = None
        mount.close()
