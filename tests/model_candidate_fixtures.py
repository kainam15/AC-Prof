"""Model suggestions keep cache presence separate from conditional success."""
import json

import catalog_fixtures as fixtures


class ModelCandidateFixture:
    def build(self, request, tmp_path):
        self._request = request
        self.fixture_root = tmp_path
        self.fixture = fixtures.CatalogFixture()
        self.fixture.build(self._request, self.fixture_root)

    def recorded(self, *, code=None):
        from acprof.tui.experiment_catalog import scan_experiments
        path = self.fixture.record('history')
        state = json.loads((path / 'run_state.json').read_text())
        from acprof.host.execution_conditions import measurement_environment
        state['options']['measurement_environment'] = measurement_environment()
        state['host'] = {'machine_id_sha256': 'host-a', 'source_sha256': 'source-a'}
        (path / 'run_state.json').write_text(json.dumps(state))
        (path / 'runtime_validation.json').write_text(json.dumps({'status': 'ok' if code is None else 'error',
            'scope': 'isolated_minimum_scale_predict_and_postprocess_before_measurement'}))
        if code:
            (path / 'runtime_failures.json').write_text(json.dumps({'failures': [{
                'stage': 'runtime', 'reason_code': code, 'detail': 'original failure', 'evidence': {'memory_gb': 4}}]}))
        return scan_experiments([path]).records[0]
