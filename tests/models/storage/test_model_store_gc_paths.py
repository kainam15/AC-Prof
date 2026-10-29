"""Cache cleanup must stay inside its configured directory, including during renames."""
import errno
import os
import shutil
from pathlib import Path
from threading import Event

import pytest

from acprof.container.model_files import PLAN_FILENAME
from acprof.host.model_store import store as model_store
from acprof.host.model_store.paths import StorePaths
from tests.support.model_store_fixtures import ModelStoreFixture


def assert_rejected(root, apply, preserved):
    error = None
    try:
        model_store.prune_store(root=root, apply=apply)
    except (ValueError, OSError) as exc:
        error = exc
    for path, contents in preserved.items():
        assert path.is_file(), f'cleanup removed a protected file: {path}'
        assert path.read_bytes() == contents
    assert error is not None, 'a symlinked cache directory must be rejected'
    assert (getattr(error, 'errno', None) in {errno.ELOOP, errno.ENOTDIR}
            or 'symlink' in str(error) or 'symbolic link' in str(error))


@pytest.mark.parametrize('apply', [False, True])
@pytest.mark.parametrize(('source', 'component'), [
    ('hf', 'hf'),
    ('hf', 'hf/models--outside'),
    ('hf', 'hf/models--outside/blobs'),
    ('modelscope', 'modelscope'),
    ('modelscope', 'modelscope/models--outside'),
    ('modelscope', 'modelscope/models--outside/snapshots'),
    ('modelscope', 'modelscope/models--outside/snapshots/revision'),
    ('modelscope', 'modelscope/models--outside/snapshots/revision/nested'),
])
def test_cache_directory_symlinks_cannot_authorize_external_file_deletion(tmp_path, source, component, apply):
    root = tmp_path / 'store'
    parent = 'hf/models--outside/blobs' if source == 'hf' else 'modelscope/models--outside/snapshots/revision/nested'
    payload = root / parent / 'payload'
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b'outside sentinel')
    alias = root / component
    outside = tmp_path / 'outside'
    alias.rename(outside)
    alias.symlink_to(outside, target_is_directory=True)
    external_file = outside / payload.relative_to(alias)

    # A valid entry must also survive when any cache deletion path is invalid.
    safe_source = 'modelscope/models--owned/snapshots/revision/payload' if source == 'hf' else 'hf/models--owned/blobs/payload'
    owned = root / safe_source
    owned.parent.mkdir(parents=True, exist_ok=True)
    owned.write_bytes(b'owned sentinel')
    key = ModelStoreFixture(root).entry(1, [owned])
    plan = root / 'entries' / key / PLAN_FILENAME
    preserved = {external_file: external_file.read_bytes(), owned: owned.read_bytes(), plan: plan.read_bytes()}
    assert_rejected(root, apply, preserved)


@pytest.mark.parametrize('apply', [False, True])
@pytest.mark.parametrize('component', ['entries', 'entry'])
def test_entry_directory_symlinks_cannot_authorize_external_tree_deletion(tmp_path, component, apply):
    root = tmp_path / 'store'
    store = ModelStoreFixture(root)
    blob = store.blob('payload', 100)
    key = store.entry(1, [blob])
    alias = root / 'entries' if component == 'entries' else root / 'entries' / key
    plan = root / 'entries' / key / PLAN_FILENAME
    outside = tmp_path / 'outside'
    alias.rename(outside)
    alias.symlink_to(outside, target_is_directory=True)
    external_plan = outside / plan.relative_to(alias)
    assert_rejected(root, apply, {external_plan: external_plan.read_bytes(), blob: blob.read_bytes()})


@pytest.mark.parametrize('component', ['entries', 'blobs'])
def test_replaced_parent_is_rejected_before_any_deletion(tmp_path, monkeypatch, component):
    root, foreign = tmp_path / 'store', tmp_path / 'foreign'
    owned, outside = ModelStoreFixture(root), ModelStoreFixture(foreign)
    blob, external_blob = owned.blob('payload', 100), outside.blob('payload', 200)
    key = owned.entry(1, [blob])
    outside.entry(1, [external_blob])
    relative = Path('entries') if component == 'entries' else Path('hf/models--example--test/blobs')
    alias, moved = root / relative, root / f'original-{component}'
    original = model_store.prune_candidates

    def replace_parent(*args, **kwargs):
        result = original(*args, **kwargs)
        alias.rename(moved)
        alias.symlink_to(foreign / relative, target_is_directory=True)
        return result

    monkeypatch.setattr(model_store, 'prune_candidates', replace_parent)
    owned_plan = (moved / key / PLAN_FILENAME if component == 'entries'
                  else root / 'entries' / key / PLAN_FILENAME)
    owned_blob = blob if component == 'entries' else moved / 'payload'
    foreign_plan = foreign / 'entries' / key / PLAN_FILENAME
    original_plan = (root / 'entries' / key / PLAN_FILENAME).read_bytes()
    assert_rejected(root, True, {owned_plan: original_plan, owned_blob: b'x' * 100,
                                foreign_plan: foreign_plan.read_bytes(), external_blob: b'x' * 200})


@pytest.mark.parametrize('apply', [False, True])
def test_configured_store_root_can_itself_be_a_symlink(tmp_path, apply):
    root = tmp_path / 'store'
    store = ModelStoreFixture(root)
    blob = store.blob('payload', 100)
    key = store.entry(1, [blob])
    plan = root / 'entries' / key / PLAN_FILENAME
    expected_bytes = 100 + len(plan.read_bytes())
    alias = tmp_path / 'store-alias'
    alias.symlink_to(root, target_is_directory=True)
    assert model_store.prune_store(root=alias, apply=apply) == {'entries': [key], 'reclaimable_bytes': expected_bytes}
    assert blob.exists() is not apply
    assert plan.exists() is not apply


def test_unlink_cannot_be_redirected_after_the_parent_is_opened(tmp_path, monkeypatch):
    store = ModelStoreFixture(tmp_path / 'store')
    blob = store.blob('payload', 100)
    outside = tmp_path / 'outside'
    outside.mkdir()
    sentinel = outside / 'payload'
    sentinel.write_bytes(b'external sentinel')
    original = os.unlink
    moved = store.root / 'original-blobs'
    swapped = False

    def replace_parent(path, *args, **kwargs):
        nonlocal swapped
        if Path(path).name == 'payload' and not swapped:
            blob.parent.rename(moved)
            blob.parent.symlink_to(outside, target_is_directory=True)
            swapped = True
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, 'unlink', replace_parent)
    result = model_store.prune_store(root=store.root, apply=True)
    assert swapped
    assert sentinel.is_file(), 'unlink followed a replaced cache directory'
    assert sentinel.read_bytes() == b'external sentinel'
    assert not (moved / 'payload').exists()
    assert result == {'entries': [], 'reclaimable_bytes': 100}


def test_rmtree_cannot_be_redirected_after_the_parent_is_opened(tmp_path, monkeypatch):
    store, outside = ModelStoreFixture(tmp_path / 'store'), ModelStoreFixture(tmp_path / 'outside')
    blob, external_blob = store.blob('payload', 100), outside.blob('payload', 200)
    key = store.entry(1, [blob])
    outside.entry(1, [external_blob])
    external_plan = outside.root / 'entries' / key / PLAN_FILENAME
    sentinel = external_plan.read_bytes()
    original = shutil.rmtree
    moved = store.root / 'original-entries'
    swapped = False

    def replace_parent(path, *args, **kwargs):
        nonlocal swapped
        if Path(path).name == key and not swapped:
            (store.root / 'entries').rename(moved)
            (store.root / 'entries').symlink_to(outside.root / 'entries', target_is_directory=True)
            swapped = True
        return original(path, *args, **kwargs)

    monkeypatch.setattr(shutil, 'rmtree', replace_parent)
    result = model_store.prune_store(root=store.root, apply=True)
    assert swapped
    assert external_plan.is_file(), 'rmtree followed a replaced entries directory'
    assert external_plan.read_bytes() == sentinel
    assert external_blob.read_bytes() == b'x' * 200
    assert not (moved / key).exists()
    assert not blob.exists()
    assert result['entries'] == [key]


def test_replaced_store_root_is_rejected_before_any_deletion(tmp_path, monkeypatch):
    store, outside = ModelStoreFixture(tmp_path / 'store'), ModelStoreFixture(tmp_path / 'outside')
    blob, external_blob = store.blob('payload', 100), outside.blob('payload', 200)
    key = store.entry(1, [blob])
    outside.entry(1, [external_blob])
    original = model_store.prune_candidates
    moved = tmp_path / 'original-store'
    plan_bytes = (store.root / 'entries' / key / PLAN_FILENAME).read_bytes()
    external_plan = outside.root / 'entries' / key / PLAN_FILENAME

    def replace_root(*args, **kwargs):
        result = original(*args, **kwargs)
        store.root.rename(moved)
        store.root.symlink_to(outside.root, target_is_directory=True)
        return result

    monkeypatch.setattr(model_store, 'prune_candidates', replace_root)
    assert_rejected(store.root, True, {
        moved / blob.relative_to(store.root): b'x' * 100,
        moved / 'entries' / key / PLAN_FILENAME: plan_bytes,
        external_blob: b'x' * 200, external_plan: external_plan.read_bytes(),
    })


@pytest.mark.parametrize('abort', ['cancel', 'scan-error'])
def test_interrupted_directory_scan_closes_descriptors_without_deleting(tmp_path, monkeypatch, abort):
    store = ModelStoreFixture(tmp_path / 'store')
    blob = store.blob('payload', 100)
    key = store.entry(1, [blob])
    entry = store.root / 'entries' / key
    plan_bytes = (entry / PLAN_FILENAME).read_bytes()
    cancel = Event()
    original = os.scandir

    def interrupt(path):
        if isinstance(path, int) and os.readlink(f'/proc/self/fd/{path}') == str(entry / 'hf'):
            if abort == 'cancel':
                cancel.set()
            else:
                raise OSError('directory scan interrupted')
        return original(path)

    monkeypatch.setattr(os, 'scandir', interrupt)
    for _ in range(10):
        cancel.clear()
        with pytest.raises(model_store.ModelStoreCancelled if abort == 'cancel' else OSError):
            model_store.prune_store(root=store.root, apply=True, cancel=cancel)
        assert blob.read_bytes() == b'x' * 100
        assert (entry / PLAN_FILENAME).read_bytes() == plan_bytes
        for descriptor in Path('/proc/self/fd').iterdir():
            try:
                target = os.readlink(descriptor)
            except FileNotFoundError:
                continue  # The descriptor used by iterdir is already closed.
            assert not target.startswith(str(store.root)), f'leaked Model Store descriptor: {target}'


def test_disappearing_entry_during_preview_is_not_treated_as_unreferenced(tmp_path, monkeypatch):
    store = ModelStoreFixture(tmp_path / 'store')
    blob = store.blob('payload', 100)
    key = store.entry(1, [blob])
    entry = store.root / 'entries' / key
    original = StorePaths.walk
    moved = store.root / 'original-entry'
    plan_bytes = (entry / PLAN_FILENAME).read_bytes()

    def move_entry(paths, relative, **kwargs):
        if relative == Path('entries') / key:
            entry.rename(moved)
        yield from original(paths, relative, **kwargs)

    monkeypatch.setattr(StorePaths, 'walk', move_entry)
    with pytest.raises(FileNotFoundError):
        model_store.prune_store(root=store.root)
    assert blob.read_bytes() == b'x' * 100
    assert (moved / PLAN_FILENAME).read_bytes() == plan_bytes
