"""Incomplete capacity scans must not authorize downloads or cache eviction."""
import errno
import os
import shutil
from collections import Counter
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from threading import Event

import pytest

from acprof.container.model_files import PLAN_FILENAME
from acprof.host.model_store import store as model_store
from tests.support.model_store_fixtures import ModelStoreFixture


def fail_scan(monkeypatch, directory, failure, *, during_iteration=False):
    original = os.scandir

    def broken_entries(entries):
        yield next(entries)
        raise failure

    @contextmanager
    def interrupted_scan(path):
        with original(path) as entries:
            yield broken_entries(entries)

    def scan(path):
        actual = Path(os.readlink(f'/proc/self/fd/{path}')) if isinstance(path, int) else Path(path)
        if actual == directory:
            if during_iteration:
                return interrupted_scan(path)
            raise failure
        return original(path)

    monkeypatch.setattr(os, 'scandir', scan)


@pytest.mark.parametrize('operation', ['report', 'space', 'preview', 'apply'])
@pytest.mark.parametrize('error_number', [errno.EACCES, errno.EIO])
@pytest.mark.parametrize('during_iteration', [False, True], ids=['open', 'iterate'])
def test_capacity_scan_failure_stops_consumers_without_removing_data(
    tmp_path, monkeypatch, operation, error_number, during_iteration,
):
    fixture = ModelStoreFixture(tmp_path)
    blob = fixture.blob('weights', 40)
    key = fixture.entry(1, [blob])
    plan = tmp_path / 'entries' / key / PLAN_FILENAME
    plan_bytes = plan.read_bytes()
    residue = tmp_path / 'interrupted-download'
    residue.mkdir()
    payload = residue / 'weights.incomplete'
    payload.write_bytes(b'x' * 1024)
    failure = OSError(error_number, 'capacity scan failed', str(residue))
    fail_scan(monkeypatch, residue, failure, during_iteration=during_iteration)
    invoke = {
        'report': partial(model_store.disk_report, tmp_path),
        'space': partial(model_store.require_space, [], tmp_path),
        'preview': partial(model_store.prune_store, root=tmp_path, target_bytes=0),
        'apply': partial(model_store.prune_store, root=tmp_path, target_bytes=0, apply=True),
    }[operation]

    with pytest.raises(OSError, match='capacity scan failed') as caught:
        invoke()

    assert caught.value is failure
    assert plan.read_bytes() == plan_bytes
    assert blob.read_bytes() == b'x' * 40
    assert payload.read_bytes() == b'x' * 1024


@pytest.mark.parametrize('target', ['root', 'file'])
@pytest.mark.parametrize('error_number', [errno.EACCES, errno.EIO])
def test_capacity_stat_failures_are_not_reported_as_absence(tmp_path, monkeypatch, target, error_number):
    payload = tmp_path / 'weights.bin'
    payload.write_bytes(b'weights')
    failed = tmp_path if target == 'root' else payload
    failure = OSError(error_number, 'capacity stat failed', str(failed))
    original = Path.stat

    def stat_path(path: Path, *args, **kwargs):
        if path == failed:
            raise failure
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'stat', stat_path)
    with pytest.raises(OSError, match='capacity stat failed') as caught:
        model_store.disk_report(tmp_path)
    assert caught.value is failure


@pytest.mark.parametrize('kind', ['file', 'directory'])
def test_capacity_scan_tolerates_concurrent_prune(tmp_path, monkeypatch, kind):
    (tmp_path / 'retained').write_bytes(b'keep')
    directory = tmp_path / 'evicted'
    directory.mkdir()
    payload = directory / 'weights.bin'
    payload.write_bytes(b'old weights')
    if kind == 'file':
        original_stat = Path.stat

        def stat_path(path: Path, *args, **kwargs):
            if path == payload:
                payload.unlink()
            return original_stat(path, *args, **kwargs)

        monkeypatch.setattr(Path, 'stat', stat_path)
    else:
        original_scan = os.scandir

        def scan(path):
            if path == directory:
                shutil.rmtree(directory)
            return original_scan(path)

        monkeypatch.setattr(os, 'scandir', scan)

    report = model_store.disk_report(tmp_path)
    assert report['total_bytes'] == 4
    assert report['models'] == []


def test_capacity_report_does_not_create_a_missing_store(tmp_path):
    root = tmp_path / 'absent' / 'store'
    report = model_store.disk_report(root)
    assert report['total_bytes'] == 0
    assert report['models'] == []
    assert report['free_bytes'] > 0
    assert not root.parent.exists()


def test_capacity_report_rejects_a_non_directory_root(tmp_path):
    root = tmp_path / 'file'
    root.write_bytes(b'not a store')
    with pytest.raises(NotADirectoryError):
        model_store.disk_report(root)
    assert root.read_bytes() == b'not a store'


def test_capacity_scan_ignores_links_and_special_files(tmp_path):
    root = tmp_path / 'store'
    root.mkdir()
    (root / 'payload').write_bytes(b'weights')
    external = tmp_path / 'outside'
    external.mkdir()
    (external / 'large').write_bytes(b'x' * 1024)
    (root / 'external').symlink_to(external, target_is_directory=True)
    (root / 'cycle').symlink_to(root, target_is_directory=True)
    (root / 'broken').symlink_to(tmp_path / 'missing')
    os.mkfifo(root / 'fifo')
    alias = tmp_path / 'configured-store'
    alias.symlink_to(root, target_is_directory=True)

    report = model_store.disk_report(alias)
    assert report['total_bytes'] == 7
    assert report['path'] == str(alias)
    assert report['models'] == []
    assert (external / 'large').read_bytes() == b'x' * 1024


@pytest.mark.parametrize('component', ['entries', 'entry', 'plan'])
def test_inventory_still_rejects_linked_published_metadata(tmp_path, component):
    root = tmp_path / 'store'
    fixture = ModelStoreFixture(root)
    key = fixture.entry(1, [fixture.blob('weights', 40)])
    original = root / 'entries'
    if component in {'entry', 'plan'}:
        original /= key
    if component == 'plan':
        original /= PLAN_FILENAME
    external = original.rename(tmp_path / 'external')
    original.symlink_to(external, target_is_directory=component != 'plan')

    with pytest.raises(ValueError, match='symlink|metadata.*path'):
        model_store.disk_report(root)
    assert original.is_symlink()
    assert external.exists()


def test_capacity_and_published_inventory_share_one_directory_scan(tmp_path, monkeypatch):
    fixture = ModelStoreFixture(tmp_path)
    keys = [fixture.entry(index, [fixture.blob(str(index), 10)]) for index in (3, 1, 2)]
    expected_bytes = 30 + sum(len((tmp_path / 'entries' / key / PLAN_FILENAME).read_bytes()) for key in keys)
    staging = tmp_path / 'entries' / '.preparing-interrupted'
    staging.mkdir()
    (staging / PLAN_FILENAME).write_bytes(b'{}')
    expected = {path: 1 for path in tmp_path.rglob('*') if path.is_dir()}
    expected[tmp_path] = 1
    observed = Counter()
    original = os.scandir

    def scan(path):
        observed[Path(path)] += 1
        return original(path)

    monkeypatch.setattr(os, 'scandir', scan)
    report = model_store.disk_report(tmp_path)
    assert report['total_bytes'] == expected_bytes + 2
    assert [model['entry_id'] for model in report['models']] == sorted(keys)
    assert observed == expected


def test_capacity_scan_cancels_before_opening_another_directory(tmp_path, monkeypatch):
    child = tmp_path / 'child'
    child.mkdir()
    (child / 'payload').write_bytes(b'weights')
    cancel = Event()
    original = os.scandir
    observed = []

    def scan(path):
        observed.append(path)
        cancel.set()
        return original(path)

    monkeypatch.setattr(os, 'scandir', scan)
    with pytest.raises(model_store.ModelStoreCancelled, match='cancelled'):
        model_store.disk_report(tmp_path, cancel=cancel)
    assert observed == [tmp_path]
