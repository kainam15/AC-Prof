"""Rebuildable local discovery and exact frozen-argument recovery."""
import json
from pathlib import Path

from acprof.run_args import build_parser


class CatalogFixture:
    def build(self, request, tmp_path):
        self._request = request
        self.temporary = tmp_path
        self.root = Path(str(self.temporary))

    def record(self, name, *, run_id='run-a', status='interrupted', model='demo/model', revision='a' * 40):
        path = self.root / name / model.replace('/', '--')
        path.mkdir(parents=True)
        options = vars(build_parser().parse_args(['--model', model, '--cpus', '1', '--mems', '4', '--gpus', 'off',
            '--matrix-order', 'declared', '--matrix-seed', '71', '--latency-slo', 'default=3', '--no-prune-startup-oom']))
        for key in ('resume', 'output_dir', 'skip_build', 'notify'):
            options.pop(key)
        options['revision'] = revision
        options['measurement_environment'] = {'execution_environment': 'native_linux'}
        state = {'schema_version': 1, 'run_id': run_id, 'status': status, 'created_at': '2026-10-01T12:00:00Z',
            'options': options, 'artifacts': {}, 'runtime': {'task': {'model_id': model, 'model_revision': revision,
                'runtime_profile_id': 'transformers-cpu'}, 'image': {'runtime_environment': {'environment_id': 'env-a'}}}}
        (path / 'run_state.json').write_text(json.dumps(state))
        (path / 'static_meta.json').write_text(json.dumps({'model_name': model, 'model_revision': revision, 'cpu_model': 'Fixture CPU'}))
        (path / 'result_all.csv').write_text('fixture\n')
        return path
