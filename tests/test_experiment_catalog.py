"""Rebuildable local discovery and exact frozen-argument recovery."""
import json
import shutil
from pathlib import Path

import pytest
from catalog_fixtures import CatalogFixture

from acprof.run_args import build_parser


class TestExperimentCatalog(CatalogFixture):
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path):
        self.build(request, tmp_path)

    @pytest.mark.parametrize('broken', ('json', 'schema', 'symlink', 'artifact_symlink', 'missing_v2_manifest'))
    def test_invalid_layout_does_not_hide_healthy_experiments(self, broken):
        from acprof.tui.experiment_catalog import scan_experiments
        good = self.record('z-good', run_id='healthy')
        bad = self.record('a-bad', run_id='broken')
        manifest = bad / 'result_manifest.json'
        if broken == 'json':
            manifest.write_text('{broken', encoding='utf-8')
        elif broken == 'schema':
            manifest.write_text('{"schema_version":999}', encoding='utf-8')
        elif broken == 'symlink':
            manifest.symlink_to(good / 'static_meta.json')
        elif broken == 'artifact_symlink':
            (bad / 'run_state.json').unlink()
            (bad / 'run_state.json').symlink_to(good / 'run_state.json')
        else:
            (bad / '.acprof').mkdir()
            (bad / 'run_state.json').rename(bad / '.acprof/run_state.json')
        report = scan_experiments([self.root])
        assert [record.run_id for record in report.records] == ['healthy']
        assert any(str(bad) in warning and 'invalid_experiment' in warning
                   for warning in report.warnings)

    def test_search_many_records_by_model_date_device_status_and_keep_run_identity(self):
        from acprof.tui.experiment_catalog import scan_experiments
        for i in range(25):
            self.record(f'batch-{i}', run_id=f'run-{i}', status='failed' if i == 18 else 'complete', model=f'demo/model-{i}')
        report = scan_experiments([self.root])
        assert (len(report.records)) == (25)
        found = report.search('model-18 2026-10 failed cpu')
        assert (len(found)) == (1)
        assert (found[0].run_id) == ('run-18')
        renamed = self.root / 'moved' / 'renamed'
        renamed.parent.mkdir()
        found[0].directory.rename(renamed)
        selected = scan_experiments([self.root]).search('model-18')[0]
        assert (selected.run_id) == ('run-18')
        assert (selected.directory) == (renamed)

    def test_duplicate_run_collapses_aliases_and_content_conflict_is_explicit(self):
        from acprof.tui.experiment_catalog import scan_experiments
        original = self.record('a')
        copy = self.root / 'b' / original.name
        shutil.copytree(original, copy)
        report = scan_experiments([self.root])
        assert (len(report.records)) == (1)
        assert (set(report.records[0].aliases)) == ({original, copy})
        from acprof.result_layers import publish_result_rows, read_result_layers
        fields, rows = read_result_layers(copy)
        rows[0]["status"] = "error"
        publish_result_rows(fields, rows, copy)
        report = scan_experiments([self.root])
        assert ('run_id_content_conflict') in (report.records[0].issues)
        assert not (report.records[0].has_recovery_state)

    def test_scan_is_bounded_skips_symlink_and_reports_bad_json(self):
        from acprof.tui.experiment_catalog import scan_experiments
        path = self.record('known')
        outside = self.root / 'outside'
        outside.mkdir()
        (path.parent / 'escape').symlink_to(outside, target_is_directory=True)
        (outside / 'static_meta.json').write_text('{}')
        (path / 'runtime_failures.json').write_text('{broken')
        report = scan_experiments([path.parent])
        assert (len(report.records)) == (1)
        assert ('runtime_failures.json') in (' '.join(report.warnings))
        limited = scan_experiments([self.root], max_directories=1)
        assert (any('scan_limit' in warning for warning in limited.warnings))

    def test_resume_restores_nonform_options_and_reuse_gets_new_output(self):
        from acprof.tui.experiment_catalog import (
            config_from_record,
            resume_command,
            scan_experiments,
        )
        path = self.record('batch')
        record = scan_experiments([path]).records[0]
        command = resume_command(record, python_executable=Path('/fixed/python'))
        parsed = build_parser().parse_args(command[command.index('--model'):])
        assert (parsed.resume)
        assert (parsed.matrix_seed) == (71)
        assert (parsed.matrix_order) == ('declared')
        assert (parsed.latency_slo) == (['default=3'])
        assert not (parsed.prune_startup_oom)
        assert (parsed.revision) == ('a' * 40)
        assert (Path(parsed.output_dir)) == (path.parent)
        first, second = config_from_record(record, reuse=True), config_from_record(record, reuse=True)
        assert not (first.resume)
        assert (first.output_dir) != (second.output_dir)
        assert (first.output_dir) != (str(path.parent))
        assert (first.model) == ('demo/model')
        assert (first.revision) == ('a' * 40)
        from acprof.experiment import build_run_command
        reused = build_run_command(first, project_dir=Path.cwd())
        parsed_reuse = build_parser().parse_args(reused[reused.index('--model'):])
        assert (parsed_reuse.matrix_seed) == (71)
        assert (parsed_reuse.matrix_order) == ('declared')
        assert (parsed_reuse.latency_slo) == (['default=3'])
        assert not (parsed_reuse.prune_startup_oom)

    def test_legacy_missing_identity_is_unknown_and_cannot_resume(self):
        from acprof.tui.experiment_catalog import resume_command, scan_experiments
        path = self.root / 'legacy'
        path.mkdir()
        (path / 'static_meta.json').write_text(json.dumps({'model_name': 'old/model'}))
        record = scan_experiments([path]).records[0]
        assert (record.run_id) is None
        with pytest.raises(ValueError, match='run_id'):
            resume_command(record, python_executable=Path('/fixed/python'))

    def test_directory_enumeration_and_duplicate_hashing_have_budgets(self):
        from acprof.tui.experiment_catalog import scan_experiments
        original = self.record('a')
        shutil.copytree(original, self.root / 'b' / original.name)
        report = scan_experiments([self.root], max_duplicate_bytes=3)
        assert (report.records[0].issues) == (('duplicate_evidence_unverified',))
        assert not (report.records[0].has_recovery_state)
        for index in range(30):
            (self.root / f'file-{index}').touch()
        report = scan_experiments([self.root], max_entries=5)
        assert ('scan_entry_limit') in (' '.join(report.warnings))
        calls = []
        def cancelled():
            calls.append(True)
            return len(calls) > 4
        with pytest.raises(InterruptedError):
            scan_experiments([self.root], cancelled=cancelled)



def test_duplicate_run_detects_tampered_layer_even_when_manifest_unchanged(tmp_path):
    from acprof.tui.experiment_catalog import scan_experiments
    fixture = CatalogFixture()
    fixture.root = tmp_path
    original = fixture.record("original")
    clone = tmp_path / "clone" / original.name
    shutil.copytree(original, clone)
    (clone / "summary.csv").write_text("corrupted\n")
    report = scan_experiments([tmp_path])
    assert len(report.records) == 1
    assert "duplicate_evidence_unverified" in report.records[0].issues
