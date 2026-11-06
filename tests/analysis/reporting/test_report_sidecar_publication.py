"""A failed report removes only its own unpublished bundle's window inputs."""

import hashlib
import json
import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from acprof.analysis.model import load_analysis
from acprof.analysis.report_statistics import write_window_inputs
from acprof.cli.main import main
from acprof.plotting.report import write_report
from tests.analysis.reporting.test_report_publication import create_competitor
from tests.analysis.reporting.test_report_statistics import row, write_rows


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr('acprof.analysis.report_statistics.MAX_EMBEDDED_WINDOW_BYTES', 1)
    monkeypatch.setattr('plotly.offline.get_plotlyjs', lambda: '/* fixture */')
    sources = [tmp_path / f'result-{index}.csv' for index in range(2)]
    for index, source in enumerate(sources):
        write_rows(source, [row(0, index + 1)])
    output = tmp_path / 'report.html'
    sidecars = [tmp_path / f'report.windows-{index}.json' for index in (1, 2)]
    return sources, load_analysis(sources), output, sidecars


def assert_complete_bundle(output, sidecars):
    raw = output.read_text().split('<script id="report-data" type="application/json">')[1].split('</script>')[0]
    reports = json.loads(raw)['window_statistics']
    assert len(reports) == len(sidecars) == 2
    for report, sidecar in zip(reports, sidecars, strict=True):
        inputs = report['recalculation']
        assert inputs['file'] == sidecar.name
        assert inputs['sha256'] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
        assert json.loads(sidecar.read_bytes())['result_sha256'] == report['result_sha256']


@pytest.mark.parametrize('phase', ['first-input', 'second-input', 'html'])
@pytest.mark.parametrize('cancel', [False, True])
def test_post_publication_sync_failure_cleans_inputs_until_html_is_visible(bundle, monkeypatch, phase, cancel):
    sources, model, output, sidecars = bundle
    original_sources = [source.read_bytes() for source in sources]
    target = {'first-input': sidecars[0], 'second-input': sidecars[1], 'html': output}[phase]
    failure = KeyboardInterrupt('directory sync cancelled') if cancel else OSError('directory sync failed')
    original_sync = os.fsync

    def fail_after_publication(descriptor: int) -> None:
        # Observe the public destination, not a call count or private staging order.
        if stat.S_ISDIR(os.fstat(descriptor).st_mode) and target.is_file():
            raise failure
        original_sync(descriptor)

    with monkeypatch.context() as scoped:
        scoped.setattr(os, 'fsync', fail_after_publication)
        with pytest.raises(type(failure), match=str(failure)) as caught:
            write_report(model, output)
    assert caught.value is failure
    assert [source.read_bytes() for source in sources] == original_sources
    assert not list(output.parent.glob('.*.tmp'))
    if phase == 'html':
        assert_complete_bundle(output, sidecars)
        with pytest.raises(ValueError, match='new file'):
            write_report(model, output)
    else:
        assert set(output.parent.iterdir()) == set(sources)
        write_report(model, output)
        assert_complete_bundle(output, sidecars)


@pytest.mark.parametrize('kind', ['file', 'directory', 'symlink', 'dangling_symlink'])
@pytest.mark.parametrize('timing', ['before-publication', 'after-publication'])
def test_input_cleanup_preserves_competing_entries(bundle, kind, timing):
    sources, model, output, sidecars = bundle
    target = sidecars[1]
    failure = OSError('HTML rendering failed')
    competing = []

    def compete():
        if timing == 'after-publication':
            # Retain the old inode elsewhere, preventing fixture inode reuse.
            target.rename(target.with_suffix('.original'))
        create_competitor(target, kind)
        competing.append(target.lstat())

    def write_inputs(*args):
        if sidecars[0].exists() and timing == 'before-publication':
            compete()
        write_window_inputs(*args)

    from acprof.artifacts import atomic_write

    def publish(path, writer, **options):
        if path == output:
            compete()
            raise failure
        atomic_write(path, writer, **options)

    error = OSError if timing == 'after-publication' else FileExistsError
    with patch('acprof.plotting.report.write_window_inputs', write_inputs), \
            patch('acprof.plotting.report.atomic_write', publish), pytest.raises(error) as caught:
        write_report(model, output)
    if timing == 'after-publication':
        assert caught.value is failure
    assert len(competing) == 1
    assert target.lstat() == competing[0]
    assert not output.exists()
    assert not sidecars[0].exists()
    assert all(source.exists() for source in sources)
    assert not list(output.parent.glob('.*.tmp'))


@pytest.mark.parametrize('kind', ['file', 'directory', 'symlink', 'dangling_symlink'])
def test_competing_html_does_not_keep_this_attempts_orphan_inputs(bundle, monkeypatch, kind):
    _sources, model, output, sidecars = bundle
    original_link = os.link
    competing = []

    def compete_at_publication(source, destination, **options) -> None:
        if Path(destination) == output:
            create_competitor(output, kind)
            competing.append(output.lstat())
        original_link(source, destination, **options)

    monkeypatch.setattr(os, 'link', compete_at_publication)
    with pytest.raises(FileExistsError):
        write_report(model, output)
    assert output.lstat() == competing[0]
    assert all(not sidecar.exists() for sidecar in sidecars)
    assert not list(output.parent.glob('.*.tmp'))


@pytest.mark.parametrize('cancel_render', [False, True])
@pytest.mark.parametrize('cancel_cleanup', [False, True])
def test_input_cleanup_attempts_all_files_and_preserves_failures(bundle, monkeypatch, cancel_render, cancel_cleanup):
    _sources, model, output, sidecars = bundle
    primary = KeyboardInterrupt('render cancelled') if cancel_render else OSError('render failed')
    secondary = KeyboardInterrupt('cleanup cancelled') if cancel_cleanup else PermissionError('cleanup denied')
    original_unlink = os.unlink

    def fail_unlink(path: str | Path, **options) -> None:
        if Path(path) == sidecars[1]:
            raise secondary
        original_unlink(path, **options)

    from acprof.artifacts import atomic_write

    def fail_render(path, writer, **options):
        if path == output:
            raise primary
        atomic_write(path, writer, **options)

    monkeypatch.setattr(os, 'unlink', fail_unlink)
    expected = secondary if cancel_cleanup and not cancel_render else primary
    with patch('acprof.plotting.report.atomic_write', fail_render), \
            pytest.raises(type(expected), match=str(expected)) as caught:
        write_report(model, output)
    assert caught.value is expected
    if expected is secondary:
        assert expected.__cause__ is primary
    notes = '\n'.join(expected.__notes__)
    assert str(sidecars[1]) in notes
    assert str(primary if expected is secondary else secondary) in notes
    assert not sidecars[0].exists()
    assert sidecars[1].exists()
    assert not output.exists()
    assert not list(output.parent.glob('.*.tmp'))


@pytest.mark.parametrize('command', ['report', 'compare'])
def test_report_cli_shows_cleanup_diagnostics_without_success_receipt(bundle, capsys, command):
    sources, _model, output, sidecars = bundle
    failure = OSError('render failed')
    failure.add_note(f'Cleanup failed for {sidecars[1]}: PermissionError: cleanup denied')
    arguments = ([command, *(str(source) for source in sources), '--output', str(output)] if command == 'report'
                 else [command, '--left', str(sources[0]), '--right', str(sources[1]), '--html-output', str(output)])
    # Isolate CLI presentation from independent-comparison metadata validation.
    with patch('acprof.analysis.independent_comparison.compare_experiments', return_value={}), \
            patch('acprof.plotting.report.write_report', side_effect=failure), pytest.raises(SystemExit) as caught:
        main(arguments)
    assert caught.value.code == 1
    captured = capsys.readouterr()
    assert 'render failed' in captured.err
    assert 'cleanup denied' in captured.err
    assert str(sidecars[1]) in captured.err
    assert 'Traceback' not in captured.err
    assert 'ACPROF_REPORT' not in captured.out
    assert '报告已生成' not in captured.out
    assert not output.exists()
