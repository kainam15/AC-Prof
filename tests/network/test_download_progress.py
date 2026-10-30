import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.container.model_files import plan_download, seal_plan
from acprof.host.model_store import store as model_store


@pytest.fixture
def download_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv('ACPROF_TUI', '1')
    task = SimpleNamespace(model_id='example/test', model_revision='a' * 40, task_family='nlp',
                           runtime_backend='transformers_pipeline', model_download_policy='full',
                           model_adapter='family-default', model_resolution={})
    plan = plan_download(model_id=task.model_id, revision=task.model_revision, family='nlp',
                         backend=task.runtime_backend, policy='full',
                         files={'config.json': {'size': 2}}, read_json=lambda _: {})
    plan['endpoint'] = 'https://huggingface.co'
    seal_plan(plan)
    target = tmp_path / 'snapshot'
    target.mkdir()
    (target / 'config.json').write_text('{}')
    return task, plan, tmp_path, target


def test_model_preparation_reports_live_sdk_bytes_then_verification(download_fixture, capsys):
    task, plan, root, target = download_fixture
    observed = []

    def snapshot(**kwargs):
        assert 'tqdm_class' in kwargs, 'preparation must attach the SDK progress adapter'
        cls = kwargs['tqdm_class']
        transfer = cls(name='huggingface_hub.snapshot_download.transfer', unit='B', total=2)
        data = cls(name='huggingface_hub.snapshot_download', unit='B', total=2)
        files = cls(total=1)
        transfer.update(2)
        data.update(2)
        files.update(1)
        files.close()
        observed.extend(capsys.readouterr().out.splitlines())
        return str(target)

    with patch('huggingface_hub.snapshot_download', side_effect=snapshot):
        model_store.prepare_model(task, plan, root)
    observed.extend(capsys.readouterr().out.splitlines())
    events = [json.loads(line.split(' ', 1)[1]) for line in observed if line.startswith('ACPROF_DOWNLOAD ')]
    assert events[0]['state'] == 'downloading'
    assert any(event['state'] == 'downloading' and event['transferred_bytes'] == 2 for event in events)
    assert [event['state'] for event in events][-2:] == ['verifying', 'complete']
    assert events[-1]['completed_bytes'] == 2
    assert events[-1]['files_completed'] == 1
    report = json.loads(next(line.split(' ', 1)[1] for line in observed if line.startswith('[network-download] ')))
    assert report['wire_bytes'] is None


def test_failed_download_is_visible_and_keeps_failure(download_fixture, capsys):
    task, plan, root, _ = download_fixture
    with patch('huggingface_hub.snapshot_download', side_effect=ValueError('invalid metadata')):
        with pytest.raises(ValueError, match='invalid metadata'):
            model_store.prepare_model(task, plan, root)
    events = [json.loads(line.split(' ', 1)[1]) for line in capsys.readouterr().out.splitlines()
              if line.startswith('ACPROF_DOWNLOAD ')]
    assert events
    assert events[-1]['state'] == 'failed'
    assert not any(event['state'] == 'complete' for event in events)


def test_sdk_snapshot_callback_contract_separates_cached_resumed_and_transferred_bytes(tmp_path):
    from huggingface_hub import snapshot_download

    from acprof.host.preparation.download_progress import DownloadReporter

    events = []
    reporter = DownloadReporter('example/test', 'huggingface', 140, 40, 2, emit=events.append)
    reporter.start_attempt()

    def download(_repo_id, *, filename, tqdm_class, **kwargs):
        if filename == 'new.bin':
            with tqdm_class(total=100, initial=20) as bar:
                bar.update(10)
                bar.update_transfer(7)
        return filename

    with patch('huggingface_hub._snapshot_download.read_tree_cache', return_value={'cached.bin': None, 'new.bin': None}), \
            patch('huggingface_hub._snapshot_download.hf_hub_download', side_effect=download):
        snapshot_download('example/test', revision='a' * 40, cache_dir=tmp_path,
                          tqdm_class=reporter.tqdm_class())
    latest = events[-1]
    assert latest.completed_bytes == 70  # 40 cached + 20 resumed + 10 newly reconstructed
    assert latest.transferred_bytes == 7
    assert latest.files_completed == 2


def test_transfer_speed_and_retry_rollback_never_include_reconstruction():
    from acprof.host.preparation.download_progress import DownloadReporter

    now = [0.0]
    events = []
    reporter = DownloadReporter('example/test', 'huggingface', 100, 20, 1,
                                clock=lambda: now[0], emit=events.append)
    reporter.start_attempt()
    cls = reporter.tqdm_class()
    transfer = cls(name='huggingface_hub.snapshot_download.transfer', unit='B', total=100)
    data = cls(name='huggingface_hub.snapshot_download', unit='B', total=100)
    data.update(50)  # SDK's resume offset is reconstruction only.
    now[0] = 0.25
    transfer.update(10)
    assert events[-1].rate_bytes_s == 40
    assert events[-1].completed_bytes == 70
    transfer.update(-10)
    now[0] = 0.5
    transfer.update(5)
    assert events[-1].transferred_bytes == 15
    assert events[-1].rate_bytes_s == 20
    assert len(events) == 3


def test_endpoint_retry_keeps_transferred_bytes_but_does_not_guess_changed_cache():
    from acprof.host.preparation.download_progress import DownloadReporter

    events = []
    reporter = DownloadReporter('example/test', 'huggingface', 100, 20, 2, emit=events.append)
    reporter.start_attempt()
    cls = reporter.tqdm_class()
    transfer = cls(name='huggingface_hub.snapshot_download.transfer', unit='B')
    data = cls(name='huggingface_hub.snapshot_download', unit='B')
    transfer.update(30)
    data.update(30)
    reporter.start_attempt()
    # The first attempt may have completed a file that the new SDK call skips
    # as cached. Aggregate callbacks alone cannot recover that new baseline.
    cls(name='huggingface_hub.snapshot_download', unit='B').update(10)
    reporter.publish(force=True)
    assert events[-1].attempt == 2
    assert events[-1].transferred_bytes == 30
    assert events[-1].cached_bytes == 20
    assert events[-1].completed_bytes is None
    assert events[-1].detail == 'retry_cache_baseline_unknown'
    reporter.finish('verifying')
    assert events[-1].completed_bytes == 100
    assert events[-1].detail == ''


@pytest.mark.parametrize('changes', [
    {'completed_bytes': 101}, {'rate_bytes_s': float('nan')}, {'cached_bytes': True},
    {'files_completed': 2}, {'attempt': 0}, {'version': True}, {'state': 'measuring'},
])
def test_download_messages_reject_invalid_counts(changes):
    from dataclasses import replace

    from acprof.events.download import DownloadProgress, encode_event

    progress = DownloadProgress('example/test', 'huggingface', 'downloading', 100, 20, 30, 10, 40.0, 0, 1)
    with pytest.raises(ValueError, match='download progress'):
        encode_event(replace(progress, **changes))


def test_plain_cli_keeps_sdk_output_and_has_no_new_progress_messages(download_fixture, monkeypatch, capsys):
    task, plan, root, target = download_fixture
    monkeypatch.delenv('ACPROF_TUI')

    def snapshot(**kwargs):
        assert 'tqdm_class' not in kwargs
        return str(target)

    with patch('huggingface_hub.snapshot_download', side_effect=snapshot):
        model_store.prepare_model(task, plan, root)
    assert 'ACPROF_DOWNLOAD ' not in capsys.readouterr().out


def test_adapter_creates_no_tqdm_monitor_and_exposes_unknown_channels():
    from acprof.host.preparation.download_progress import DownloadReporter

    events = []
    reporter = DownloadReporter('example/test', 'huggingface', 100, 0, 1, emit=events.append)
    reporter.start_attempt()
    cls = reporter.tqdm_class()
    with patch('tqdm.std.TMonitor', side_effect=AssertionError('unexpected background monitor')):
        bar = cls(name='future-sdk-channel', unit='B', total=100)
        bar.update(10)
        bar.close()
    assert events[-1].detail == 'unsupported_sdk_progress_channel'
    assert events[-1].completed_bytes is None
    assert events[-1].transferred_bytes is None
