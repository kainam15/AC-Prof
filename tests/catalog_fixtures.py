"""Rebuildable local discovery and exact frozen-argument recovery."""
import json
from dataclasses import asdict
from pathlib import Path

from acprof.experiment import RunConfig
from acprof.host.run_state import host_identity, run_options
from acprof.run_args import build_parser


class CatalogFixture:
    def build(self, request, tmp_path):
        self._request = request
        self.temporary = tmp_path
        self.root = Path(str(self.temporary))

    def record(self, name, *, run_id='run-a', status='interrupted', model='demo/model', revision='a' * 40):
        path = self.root / name / model.replace('/', '--')
        path.mkdir(parents=True)
        args = build_parser().parse_args(['--model', model, '--cpus', '1', '--mems', '4', '--gpus', 'off',
            '--matrix-order', 'declared', '--matrix-seed', '71', '--latency-slo', 'default=3',
            '--no-prune-startup-oom', '--revision', revision])
        for key, value in asdict(RunConfig.from_namespace(args).validate(project_dir=self.root)).items():
            setattr(args, key, value)
        options = run_options(args)
        state = {'schema_version': 1, 'run_id': run_id, 'status': status, 'created_at': '2026-10-01T12:00:00Z',
            'host': host_identity(Path.cwd()),
            'options': options, 'artifacts': {}, 'runtime': {'task': {'model_id': model, 'model_revision': revision,
                'runtime_profile_id': 'transformers-cpu'}, 'image': {'runtime_environment': {'environment_id': 'env-a'}}}}
        (path / 'run_state.json').write_text(json.dumps(state))
        (path / 'static_meta.json').write_text(json.dumps({'model_name': model, 'model_revision': revision, 'cpu_model': 'Fixture CPU'}))
        # A recorded experiment publishes independently hashed result layers.
        from acprof.result_layers import publish_result_rows
        publish_result_rows(
            ["cpu_cores", "mem_cap_gb", "gpu_mode", "input_scale", "warmup",
             "repeat_idx", "status"],
            [dict(cpu_cores="1", mem_cap_gb="4", gpu_mode="off", input_scale="2",
                  warmup="0", repeat_idx="0", status="ok")],
            path,
        )
        return path
