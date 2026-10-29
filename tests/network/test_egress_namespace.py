from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import egress_namespace, validate_measurement_egress as validation


@pytest.mark.parametrize('case', ['host', 'pid-reused', 'valid'])
def test_namespace_guard_pins_identity_and_never_enters_host(monkeypatch, case):
    monkeypatch.setattr(egress_namespace, 'process_start_ticks', Mock(side_effect=[42, 43 if case == 'pid-reused' else 42]))
    monkeypatch.setattr(egress_namespace.os, 'open', Mock(return_value=77))
    monkeypatch.setattr(egress_namespace.os, 'fstat', Mock(return_value=SimpleNamespace(st_dev=1, st_ino=1 if case == 'host' else 2)))
    monkeypatch.setattr(egress_namespace.os, 'stat', Mock(return_value=SimpleNamespace(st_dev=1, st_ino=1)))
    enter = Mock()
    close = Mock()
    monkeypatch.setattr(egress_namespace.os, 'setns', enter)
    monkeypatch.setattr(egress_namespace.os, 'close', close)
    if case == 'valid':
        egress_namespace.enter_namespace(1234, 42)
        enter.assert_called_once_with(77, 0)
    else:
        with pytest.raises(RuntimeError, match='host network namespace' if case == 'host' else 'identity changed'):
            egress_namespace.enter_namespace(1234, 42)
        enter.assert_not_called()
    close.assert_called_once_with(77)


def test_pid_reuse_before_namespace_open_is_rejected(monkeypatch):
    monkeypatch.setattr(egress_namespace, 'process_start_ticks', lambda _pid: 43)
    opened = Mock()
    monkeypatch.setattr(egress_namespace.os, 'open', opened)
    with pytest.raises(RuntimeError, match='identity changed'):
        egress_namespace.enter_namespace(1234, 42)
    opened.assert_not_called()


def test_policy_failure_does_not_release_model_startup_gate(tmp_path, monkeypatch):
    experiment = validation.Experiment(tmp_path, 'unused-image', 'unused-gpu')
    monkeypatch.setattr(experiment, 'create', Mock(return_value=('container', {})))
    monkeypatch.setattr(experiment, 'policy', Mock(side_effect=RuntimeError('cannot install policy')))
    ready = Mock()
    monkeypatch.setattr(experiment, 'wait_ready', ready)
    try:
        with pytest.raises(RuntimeError, match='cannot install policy'):
            experiment.model_case('controlled', SimpleNamespace(args=[]), gpu=False, controlled=True)
        assert (tmp_path / 'controlled-gate').is_dir()
        assert not (tmp_path / 'controlled-gate' / 'release').exists()
        ready.assert_not_called()
    finally:
        experiment.client.close()


def test_cleanup_failure_still_releases_model_store_lease(tmp_path, monkeypatch):
    from contextlib import nullcontext

    experiment = Mock()
    experiment.results = {}
    experiment.helper = tmp_path / 'helper.py'
    experiment.helper.write_text('test helper')
    experiment.run.side_effect = ['image', 'head', 'docker-version', 'nft-version', '{}', 'old-parser']
    experiment.close.side_effect = RuntimeError('cleanup failed')
    mount = Mock()
    monkeypatch.setattr(validation, 'Experiment', Mock(return_value=experiment))
    monkeypatch.setattr(validation, 'MeasurementLock', nullcontext)
    monkeypatch.setattr(validation, 'acquire_mount', Mock(return_value=mount))
    monkeypatch.setattr(validation.sys, 'argv', ['validate', '--image', 'image', '--gpu-uuid', 'gpu',
                                               '--output', str(tmp_path / 'output'), '--protocols-only'])
    with pytest.raises(RuntimeError, match='cleanup failed'):
        validation.main()
    mount.close.assert_called_once_with()
