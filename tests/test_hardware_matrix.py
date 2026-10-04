"""Validate hardware plans and failure evidence without starting real workloads."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from acprof.experiment import RunConfig
from scripts import check_hardware


def matrix_file(tmp_path, cases=None):
    cases = cases if cases is not None else [
        {"name": name, "model": f"demo/{name}", "revision": str(index) * 40,
         "task": "fill-mask", "input_scales": "64"}
        for index, name in enumerate(("first", "second", "third"), 1)]
    path = tmp_path / 'matrix.json'
    path.write_text(json.dumps({'schema_version': 1, 'cases': cases}), encoding='utf-8')
    return path


def test_matrix_preserves_pinned_revisions_and_separates_output_roots(tmp_path):
    path = matrix_file(tmp_path)
    cases = check_hardware.load_hardware_matrix(path, RunConfig(gpus='off,on'), tmp_path / 'output')
    assert [config.revision for _, config in cases] == [str(i) * 40 for i in (1, 2, 3)]
    assert all(config.gpus == 'off,on' for _, config in cases)
    assert len({config.output_dir for _, config in cases}) == 3
    assert not (tmp_path / 'output').exists()


def test_checked_in_matrix_has_a_successful_serial_execution_path(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'examples/hardware-validation.json'
    output = tmp_path / 'output'
    with patch.object(check_hardware, 'run_config', return_value={'successful': True}) as run:
        assert check_hardware.main(['--matrix', str(path), '--output-dir', str(output)]) == 0
        assert run.call_count == 2
        assert [call.args[0].task for call in run.call_args_list] == ['fill-mask', 'image-classification']
        assert all(call.args[0].skip_build and call.args[0].gpus == 'off,on'
                   for call in run.call_args_list)
    report = json.loads((output / 'hardware_matrix.json').read_text())
    assert report['successful'] is True
    assert [case['status'] for case in report['cases']] == ['passed', 'passed']


@pytest.mark.parametrize('bad', ('unresolved', 'duplicate', 'escape', 'unknown', 'empty'))
def test_invalid_matrix_is_rejected_before_running_any_case(tmp_path, bad):
    path = matrix_file(tmp_path)
    payload = json.loads(path.read_text())
    if bad == 'unresolved':
        payload['cases'][0]['revision'] = 'main'
    elif bad == 'duplicate':
        payload['cases'][1]['name'] = 'first'
    elif bad == 'escape':
        payload['cases'][0]['name'] = '../outside'
    elif bad == 'unknown':
        payload['cases'][0]['input_scale'] = 'wrong field'
    else:
        payload['cases'] = []
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        check_hardware.load_hardware_matrix(path, RunConfig(), tmp_path / 'output')
    with patch.object(check_hardware, 'run_config') as run:
        with pytest.raises(SystemExit) as exc:
            check_hardware.main(['--matrix', str(path), '--output-dir', str(tmp_path / 'output')])
        assert exc.value.code == 2
        run.assert_not_called()
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('failure', ('audit', 'exception', 'cancelled'))
def test_matrix_stops_on_failure_and_records_unrun_cases(tmp_path, failure):
    path = matrix_file(tmp_path)
    output = tmp_path / 'output'
    bad = {'successful': False} if failure == 'audit' else (
        RuntimeError('cleanup incomplete') if failure == 'exception' else KeyboardInterrupt())
    with patch.object(check_hardware, 'run_config', side_effect=[{'successful': True}, bad]) as run:
        if failure == 'cancelled':
            with pytest.raises(KeyboardInterrupt):
                check_hardware.main(['--matrix', str(path), '--output-dir', str(output)])
        else:
            assert check_hardware.main(['--matrix', str(path), '--output-dir', str(output)]) == 1
        assert run.call_count == 2
    report = json.loads((output / 'hardware_matrix.json').read_text())
    assert report['successful'] is False
    assert [case['status'] for case in report['cases']] == [
        'passed', 'cancelled' if failure == 'cancelled' else 'failed', 'not_run']
    assert report['source']['source_sha256']
    assert all(Path(case['output_dir']).is_relative_to(output) for case in report['cases'])
