import json
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from acprof.host import orchestrator
from acprof.host.cases import CaseMeasurement, CasePlans, CaseWorkload, MatrixRequest
from acprof.host.detect import TaskInfo
from acprof.host.images.runtime import ImageInfo


class TestMatrixPlan:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        self.task = TaskInfo('org/model', 'fill-mask', 'nlp', 'transformers_pipeline',
                             'transformers', 'a' * 40, 'manual')
        self.image = ImageInfo('sha256:' + 'b' * 64)

    def run_matrix(self, directory, **options):
        request = MatrixRequest(
            task_info=self.task,
            image_info=self.image,
            cpu_list=[4, 1, 2],
            mem_list=[8, 2, 4],
            gpu_list=['off'],
            output_dir=str(directory),
            project_dir=str(directory),
            workload=CaseWorkload(warmup=0, repeat=1),
            measurement=CaseMeasurement(profiling_mode='basic'),
            plans=CasePlans(input_scales='16,32,64,128'),
        )
        return orchestrator.run_matrix(replace(request, **options))

    def test_matrix_freezes_actual_case_and_scale_order(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            orchestrator, 'run_single_case', return_value=''
        ) as case:
            self.run_matrix(tmp)
            path = Path(tmp) / 'matrix_plan.json'
            assert (path.is_file()), 'formal collection needs a frozen plan'
            plan = json.loads(path.read_text())
            assert (len(plan['cases'])) == (9)
            assert (len(plan['plan_sha256'])) == (64)
            assert (plan['algorithm_version'])
            assert ([(c['cpu_cores'], c['mem_cap_gb']) for c in plan['cases']]) == ([(c.args[0].cpu, c.args[0].mem) for c in case.call_args_list])
            for item, call in zip(plan['cases'], case.call_args_list, strict=True):
                assert (item['input_scales']) == (call.kwargs['plans'].input_scale_order)

    def test_same_seed_produces_byte_identical_plan(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            orchestrator, 'run_single_case', return_value=''
        ):
            paths = [Path(tmp) / name for name in ('first', 'second')]
            for path in paths:
                self.run_matrix(path, matrix_order='seeded', matrix_seed=37)
            assert ((paths[0] / 'matrix_plan.json').read_bytes()) == ((paths[1] / 'matrix_plan.json').read_bytes())

    def test_changed_options_do_not_overwrite_a_frozen_plan(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            orchestrator, 'run_single_case', return_value=''
        ):
            self.run_matrix(tmp, matrix_seed=3)
            original = (Path(tmp) / 'matrix_plan.json').read_bytes()
            with pytest.raises(ValueError, match='matrix.*(identity|options)'):
                self.run_matrix(tmp, matrix_seed=4)
            assert ((Path(tmp) / 'matrix_plan.json').read_bytes()) == (original)

    def test_reuse_does_not_rebuild_or_repeat_probes(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            orchestrator, 'run_single_case', return_value=''
        ) as case:
            self.run_matrix(tmp, matrix_seed=37)
            original = case.call_args_list[:]
            case.reset_mock()
            with patch('acprof.host.matrix_plan.build_matrix_plan', side_effect=AssertionError('reshuffled')), \
                 patch('acprof.host.startup_probe.run_startup_probes', side_effect=AssertionError('reprobed')):
                self.run_matrix(tmp, matrix_seed=37)
            assert (case.call_args_list) == (original)

    def test_scale_seed_does_not_depend_on_number_or_order_of_cases(self):
        from acprof.host.matrix_plan import build_matrix_plan, matrix_identity
        plans = []
        for cpus in ([1, 2], [8, 2, 1]):
            identity = matrix_identity(self.task, self.image, cpus, [4], ['off'],
                                       [16., 32., 64., 128.], order='seeded', seed=37, prune=False)
            plans.append(build_matrix_plan(identity, {}))
        def common(plan):
            return {c['cpu_cores']: (c['input_scales'], c['input_scale_seed'])
                    for c in plan['cases'] if c['cpu_cores'] in (1, 2)}
        assert (common(plans[0])) == (common(plans[1]))
        assert (plans[0]['cases'][0]['input_scale_seed']) != ('37')

    def test_corrupted_plan_is_rejected_before_formal_collection(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            orchestrator, 'run_single_case', return_value=''
        ) as case:
            self.run_matrix(tmp)
            path = Path(tmp) / 'matrix_plan.json'
            plan = json.loads(path.read_text())
            plan['cases'].reverse()
            path.write_text(json.dumps(plan))
            case.reset_mock()
            with pytest.raises(ValueError, match='hash'):
                self.run_matrix(tmp)
            case.assert_not_called()
