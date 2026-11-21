import hashlib
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

    @pytest.mark.parametrize("recorded", [True, False])
    def test_each_case_receives_digest_of_validated_plan_bytes(self, tmp_path, recorded):
        path = tmp_path / "input_scale_plan.json"
        path.write_bytes(json.dumps({"schema_version": 2, "entries": [
            {"input_scale": value, "payload": {"text": "original"}} for value in (16, 32)
        ]}, indent=2).encode("utf-8-sig"))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        options = {"input_scale_plan_sha256": digest} if recorded else {}
        plans = CasePlans(input_scales="32,16", input_scale_plan_file=str(path), **options)
        with patch.object(orchestrator, "run_single_case", return_value="") as case:
            self.run_matrix(tmp_path, plans=plans)
        identity = json.loads((tmp_path / "matrix_plan.json").read_text())["identity"]
        assert identity["input_plan_sha256"] == digest
        assert identity["input_scales"] == [16, 32]
        assert case.call_count == 9
        assert all(call.kwargs["plans"].input_scale_plan_sha256 == digest for call in case.call_args_list)

    @pytest.mark.parametrize(("change", "error"), [
        ("payload", "SHA256"), ("scales", "scale.*match"), ("no_file", "plan file"), ("oversize", "read limit"),
    ])
    def test_invalid_input_plan_stops_before_probes_and_matrix_freeze(self, tmp_path, monkeypatch, change, error):
        path = tmp_path / "input_scale_plan.json"
        plan = {"schema_version": 2, "entries": [{"input_scale": 16, "payload": {"text": "original"}}]}
        path.write_text(json.dumps(plan))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        options = {} if change == "scales" else {"input_scale_plan_sha256": digest}
        if change == "payload":
            plan["entries"][0]["payload"]["text"] = "different"
            path.write_text(json.dumps(plan))
        if change == "oversize":
            monkeypatch.setattr("acprof.artifacts.MAX_JSON_ARTIFACT_BYTES", len(path.read_bytes()) - 1)
        plans = CasePlans(input_scales="32" if change == "scales" else "16",
                          input_scale_plan_file=None if change == "no_file" else str(path), **options)
        with patch("acprof.host.startup_probe.run_startup_probes") as probe, \
             patch.object(orchestrator, "run_single_case") as case:
            with pytest.raises(ValueError, match=error):
                self.run_matrix(tmp_path, plans=plans, prune_startup_oom=True)
        probe.assert_not_called()
        case.assert_not_called()
        assert not (tmp_path / "matrix_plan.json").exists()

    def test_changed_legacy_input_plan_cannot_replace_frozen_matrix(self, tmp_path):
        path = tmp_path / "input_scale_plan.json"
        plan = {"schema_version": 2, "entries": [{"input_scale": 16, "payload": {"text": "original"}}]}
        path.write_text(json.dumps(plan))
        plans = CasePlans(input_scales="16", input_scale_plan_file=str(path))
        with patch.object(orchestrator, "run_single_case", return_value="") as case:
            self.run_matrix(tmp_path, plans=plans)
            frozen = tmp_path / "matrix_plan.json"
            original = frozen.read_bytes()
            plan["entries"][0]["payload"]["text"] = "different"
            path.write_text(json.dumps(plan))
            case.reset_mock()
            with pytest.raises(ValueError, match="matrix plan identity"):
                self.run_matrix(tmp_path, plans=plans)
            case.assert_not_called()
            assert frozen.read_bytes() == original

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
