"""Model suggestions keep cache presence separate from conditional success."""
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from tests.support.model_candidate_fixtures import ModelCandidateFixture


class TestModelCandidate(ModelCandidateFixture):
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path):
        self.build(request, tmp_path)

    def test_coverage_sources_keep_legacy_rows_selectable_and_unknown_rows_viewable(self, monkeypatch):
        from acprof.tui.models.candidates import coverage_candidates, model_choice

        monkeypatch.setenv("ACPROF_MODEL_SOURCE", "modelscope")
        report = self.fixture.root / "coverage.json"
        row = {"model_id": "demo/model", "revision": "a" * 40}
        report.write_text(json.dumps({
            "schema_version": 1, "scope": "recorded_results; no_reexecution",
            "rows": [row, {**row, "source": "modelscope"}, {**row, "source": "unknown"}, {**row, "source": None}],
        }))
        assert [(item.model_source, model_choice(item).actions) for item in coverage_candidates(report)] == [
            ("huggingface", {"view", "use"}), ("modelscope", {"view", "use"}),
            ("unknown", {"view"}), ("unknown", {"view"}),
        ]

    @pytest.mark.parametrize("metadata_source", [None, "huggingface"])
    @pytest.mark.parametrize("embedded", [False, True], ids=["standalone", "embedded"])
    def test_resolution_only_history_and_conflicting_source_evidence(self, metadata_source, embedded):
        from acprof.tui.experiments.catalog import read_experiment_record
        from acprof.tui.models.candidates import candidate_from_record, model_choice

        root = self.fixture.root
        resolution = {
            "model_id": "demo/model", "model_revision": "a" * 40,
            "provenance": {"sources": {"repository_snapshot": {"source": "modelscope"}}},
        }
        metadata = {}
        if embedded:
            metadata["model_resolution"] = resolution
        else:
            (root / "model_resolution.json").write_text(json.dumps(resolution))
        if metadata_source is not None:
            metadata["model_source"] = metadata_source
        if metadata:
            (root / "static_meta.json").write_text(json.dumps(metadata))
        record = read_experiment_record(root)
        candidate = candidate_from_record(record, {"model_source": "modelscope"})
        if metadata_source is None:
            assert candidate.model_source == "modelscope"
            assert (candidate.model_id, candidate.revision) == ("demo/model", "a" * 40)
            assert "use" in model_choice(candidate).actions
        else:
            assert candidate.model_source == "unknown"
            assert model_choice(candidate).actions == {"view"}

    def test_coverage_candidate_retains_attempt_resources_and_failed_access_evidence(self):
        from acprof.tui.models.candidates import coverage_candidates
        root = self.fixture.root
        report = root / 'coverage.json'
        report.write_text(json.dumps({'schema_version': 2, 'scope': 'selected_sample_only; no_formal_measurement',
            'attempts': [{'attempt_id': 2, 'configuration': {'resources': {'cpus': 2, 'memory_gb': 8, 'gpu': False},
                'budgets': {'max_download_bytes': 5000}, 'host': {'machine_id_sha256': 'host-a'}}}],
            'rows': [{'model_id': 'demo/gated', 'revision': 'b' * 40, 'attempt_id': 2,
                'failure': {'reason_code': 'access_denied', 'detail': 'accept model terms'}}]}))
        candidates = coverage_candidates(report)
        assert (candidates[0].status) == ('revalidate')
        assert (candidates[0].history_status) == ('access_required')
        assert ('access_denied') in (candidates[0].reason)
        assert ('current conditions') in (candidates[0].reason)
        from acprof.tui.models.candidates import model_choice
        assert ('use') in (model_choice(candidates[0]).actions)
        assert (candidates[0].conditions['mems']) == ('8')
        assert (candidates[0].report) == (report)
        assert ('max_download_bytes') in (candidates[0].detail)
        assert ('accept model terms') in (candidates[0].detail)
        payload = json.loads(report.read_text())
        payload['rows'][0]['failure'] = {'reason_code': 'resource_limit', 'detail': 'original 8 GiB limit'}
        report.write_text(json.dumps(payload))
        limited = coverage_candidates(report)[0]
        assert (limited.status) == ('revalidate')
        assert (limited.history_status) == ('resource_limited')
        assert (limited.conditions['mems']) == ('8')
        assert ('use') in (model_choice(limited).actions)

    @pytest.mark.parametrize('key_case', range(7), ids=["('revision', 'b' * 40)", "('runtime_environment', 'env-b')", "('device', 'GPU-other')", "('mems', '8')", "('host_id', 'host-b')", "('execution_options', {'cpuset_cpus': '0-1'})", "('execution_options', {'input_scales': '128'})"])
    def test_success_is_bound_to_revision_runtime_device_and_resources(self, key_case):
        from acprof.tui.models.candidates import candidate_from_record, record_conditions
        record = self.recorded()
        conditions = record_conditions(record)
        candidate = candidate_from_record(record, conditions)
        assert (candidate.status) == ('verified_current')
        (key, changed) = tuple((('revision', 'b' * 40), ('runtime_environment', 'env-b'), ('device', 'GPU-other'), ('mems', '8'), ('host_id', 'host-b'), ('execution_options', {'cpuset_cpus': '0-1'}), ('execution_options', {'input_scales': '128'})))[key_case]
        current = {**conditions, key: changed}
        candidate = candidate_from_record(record, current)
        assert (candidate.status) == ('revalidate')
        assert (key) in (candidate.reason)
        candidate = candidate_from_record(record, {**conditions, 'revision': ''})
        assert (candidate.status) == ('revalidate')
        assert ('revision') in (candidate.reason)

    def test_failures_remain_selectable_with_original_resources_and_source(self):
        from acprof.tui.models.candidates import candidate_from_record, record_conditions
        record = self.recorded(code='resource_limit')
        candidate = candidate_from_record(record, record_conditions(record))
        assert (candidate.status) == ('resource_limited')
        assert (candidate.experiment) == (record.directory)
        changed = candidate_from_record(record, {**record_conditions(record), 'mems': '8'})
        assert (changed.status) == ('revalidate')
        assert (changed.history_status) == ('resource_limited')
        assert ('original failure') in (candidate.detail)
        assert ('memory_gb') in (candidate.detail)
        access = replace(record, failures=({'reason_code': 'access_denied', 'detail': 'gated'},))
        assert (candidate_from_record(access, record_conditions(access)).status) == ('access_required')

    def test_cache_is_not_inference_evidence_and_missing_blobs_are_not_cached(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from acprof.tui.models.candidates import cached_candidates
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry = root / 'entries' / ('a' * 64)
            entry.mkdir(parents=True)
            (entry / 'model_download_plan.json').write_text('{}')
            plan = {'model_id': 'demo/cached', 'model_revision': 'a' * 40}
            with patch('acprof.host.model_store.store.read_entry', return_value=plan), patch(
                'acprof.host.model_store.store.model_sources', return_value=[SimpleNamespace(cache_status='hit')]
            ):
                values, warnings = cached_candidates(root)
                assert (len(values)) == (1)
                assert (values[0].status) == ('cached_unverified')
                assert not (warnings)
            with patch('acprof.host.model_store.store.read_entry', return_value=plan), patch(
                'acprof.host.model_store.store.model_sources', return_value=[SimpleNamespace(cache_status='partial')]
            ):
                values, warnings = cached_candidates(root)
                assert not (values)

    @pytest.mark.parametrize(('gpus', 'expected'), [('off', 'nlp-cpu'), ('on', 'nlp-cu124')])
    def test_actual_runtime_routing_and_driver_query_are_shared_and_bounded(self, gpus, expected):
        from types import SimpleNamespace
        from unittest.mock import patch

        from acprof.experiments.configuration import RunConfig
        from acprof.host.images.runtime import select_nlp_torch_index_url
        from acprof.tui.models.candidates import current_conditions, current_context
        record = self.recorded()
        task = {'model_id': record.model_id, 'pipeline_tag': 'fill-mask', 'task_family': 'nlp',
                'runtime_backend': 'transformers_model', 'library_name': 'transformers',
                'model_revision': record.revision, 'detection_method': 'config_infer',
                'model_config': {'model_type': 'bert'}, 'model_resolution': {'runtime_profile': 'historical', 'requested_devices': ['cpu']}}
        record = replace(record, state={**record.state, 'runtime': {'task': task}})
        config = replace(RunConfig.smoke(record.model_id), gpus=gpus)
        with patch.dict('os.environ', {}, clear=True), patch('acprof.host.images.runtime.shutil.which', return_value='/usr/bin/nvidia-smi'), patch(
                'acprof.system.command.run_command', return_value=SimpleNamespace(returncode=0, stdout='CUDA Version: 12.4')) as command:
            assert (select_nlp_torch_index_url().endswith('/cu124'))
            assert (command.call_args.kwargs['timeout']) == (10)
            command.reset_mock()
            with patch('acprof.host.run_state.host_identity', return_value={'machine_id_sha256': 'host-a', 'source_sha256': 'source-a'}), \
                 patch('acprof.host.gpu_device.resolve_gpu_device', return_value={'uuid': 'gpu-a'}) as device, \
                 patch('acprof.host.preparation.task_support.require_task_support') as support:
                context = current_context(config, Path.cwd())
                frozen = json.dumps(record.state, sort_keys=True)
                first = current_conditions(record, config, context, Path.cwd())
                second = current_conditions(record, config, context, Path.cwd())
                assert first['runtime_profile'] == expected
                assert (first['runtime_environment']) == (second['runtime_environment'])
                assert (json.dumps(record.state, sort_keys=True)) == (frozen)
                assert support.call_args.kwargs['devices'] == [gpus]
                assert command.call_count == (1 if gpus == 'on' else 0)
                assert device.call_count == (1 if gpus == 'on' else 0)

    def test_missing_thread_evidence_and_changed_thread_or_input_options_require_revalidation(self):
        from acprof.tui.models.candidates import candidate_from_record, record_conditions
        record = self.recorded()
        conditions = record_conditions(record)
        for changes in ({'input_scales': '128'}, {'input_scale_policy': 'minimal'}, {'cpuset_cpus': '0-1'},
                        {'measurement_environment': {'OMP_NUM_THREADS': '3'}}):
            current = {**conditions, 'execution_options': {**conditions['execution_options'], **changes}}
            assert (candidate_from_record(record, current).status) == ('revalidate')
        legacy = replace(record, options={**record.options, 'measurement_environment': {}})
        assert (candidate_from_record(legacy, record_conditions(legacy)).status) == ('revalidate')

    def test_driver_timeout_stays_unknown_and_non_torch_cpu_avoids_probe(self):
        from subprocess import TimeoutExpired
        from unittest.mock import patch

        from acprof.experiments.configuration import RunConfig
        from acprof.tui.models.candidates import current_context
        config = RunConfig.smoke('demo/model')
        with patch('acprof.host.run_state.host_identity', return_value={'machine_id_sha256': 'host-a'}), \
             patch('acprof.host.images.runtime.select_nlp_torch_index_url', side_effect=TimeoutExpired('nvidia-smi', 10)) as select, \
             patch('acprof.host.gpu_device.resolve_gpu_device') as gpu:
            timed_out = current_context(config, Path.cwd())
            assert (timed_out['torch_index_url']) is None
            assert ('10') in (timed_out['warnings'][0])
            select.assert_called_once()
            select.reset_mock()
            current_context(config, Path.cwd(), needs_torch_platform=False)
            select.assert_not_called()
            gpu.assert_not_called()


def _candidate_for_workload(tmp_path, monkeypatch, recorded_selection, selected_members, requested_tasks,
                            available_profiles=None, discovery=None):
    from types import SimpleNamespace

    from acprof.experiments.configuration import RunConfig
    from acprof.tui.models.candidates import candidate_from_record, current_conditions

    fixture = ModelCandidateFixture()
    fixture.build(None, tmp_path)
    record = fixture.recorded()
    config = replace(RunConfig.from_namespace(SimpleNamespace(**record.options)), tasks=requested_tasks)
    options = dict(record.options)
    if recorded_selection:
        options["tasks"] = recorded_selection
    history = dict(record.state["runtime"]["task"])
    if selected_members:
        history["task_discovery"] = {"status": "selection_required" if available_profiles else "ready", "model_revision": record.revision, "profiles": available_profiles or [
            {"profile_id": recorded_selection, "semantic_task_ids": list(selected_members)}]}
        history["selected_tasks"] = [{"task_id": recorded_selection, "interface": history["pipeline_tag"],
                                      "execution_profile": {"semantic_task_ids": list(selected_members)}}]
    if discovery is not None:
        history["task_discovery"] = {"model_revision": record.revision, **discovery}
    record = replace(record, options=options, state={**record.state, "runtime": {"task": history}})
    context = {"host": record.state["host"], "gpu_uuid": None, "torch_index_url": "https://download.pytorch.org/whl/cpu"}
    monkeypatch.setattr("acprof.host.preparation.task_support.require_task_support", lambda *args, **kwargs: None)
    current = current_conditions(record, config, context, Path.cwd())
    assert current["runtime_profile"]
    assert current["runtime_environment"]
    record = replace(record, runtime_profile=current["runtime_profile"],
                     metadata={**record.metadata, "runtime_environment": {"environment_id": current["runtime_environment"]}})
    return candidate_from_record(record, current)


@pytest.mark.parametrize(("recorded_selection", "selected_members", "requested_tasks", "expected"), [
    ("summarization", (), "summarization", "verified_current"),
    ("summarization", (), "translation_en_to_de", "revalidate"),
    ("profile-a", ("summarization",), "summarization", "verified_current"),
    ("profile-a", ("summarization",), "profile-a", "verified_current"),
    ("profile-a", ("summarization",), "summarization,translation_en_to_de", "revalidate"),
    ("profile-a", ("summarization",), "all", "verified_current"),
    ("profile-a", ("summarization",), "", "verified_current"),
    ("profile-a", ("summarization", "translation_en_to_de"), "translation_en_to_de", "revalidate"),
    ("profile-a", ("summarization", "translation_en_to_de"), "translation_en_to_de,summarization", "verified_current"),
    ("", (), "summarization", "revalidate"),
    ("", (), "", "verified_current"),
])
def test_candidate_validation_follows_selected_workload(tmp_path, monkeypatch, recorded_selection,
                                                       selected_members, requested_tasks, expected):
    candidate = _candidate_for_workload(tmp_path, monkeypatch, recorded_selection, selected_members, requested_tasks)
    assert candidate.status == expected
    if expected == "revalidate":
        assert candidate.history_status == "previous_success"
        assert "tasks" in candidate.reason
        assert json.loads(candidate.detail)["validation"]["status"] == "ok"



@pytest.mark.parametrize(("requested_tasks", "expected"), [
    ("", "revalidate"), ("all", "revalidate"), ("profile-a", "revalidate"),
    ("translation_en_to_de", "revalidate"), ("summarization", "verified_current"),
])
def test_single_child_does_not_validate_other_exemplars_or_siblings(tmp_path, monkeypatch, requested_tasks, expected):
    profiles = [
        {"profile_id": "profile-a", "semantic_task_ids": ["summarization", "translation_en_to_de"]},
        {"profile_id": "profile-b", "semantic_task_ids": ["translation_en_to_fr"]},
    ]
    candidate = _candidate_for_workload(tmp_path, monkeypatch, "profile-a", ("summarization",), requested_tasks,
                                        available_profiles=profiles)
    assert candidate.status == expected


@pytest.mark.parametrize(("profiles", "conflicts", "expected"), [
    ([{"profile_id": "a", "semantic_task_ids": ["summarization"]}], [], "verified_current"),
    ([{"profile_id": "a", "semantic_task_ids": ["summarization"]},
      {"profile_id": "b", "semantic_task_ids": ["translation_en_to_de"]}], [], "revalidate"),
    ([{"profile_id": "a", "semantic_task_ids": ["summarization"]}], ["conflicting work"], "revalidate"),
    ([], [], "revalidate"),
])
def test_unselected_history_requires_one_unambiguous_default(tmp_path, monkeypatch, profiles, conflicts, expected):
    discovery = {"status": "ready", "profiles": profiles, "conflicts": conflicts}
    candidate = _candidate_for_workload(tmp_path, monkeypatch, "", (), "", discovery=discovery)
    assert candidate.status == expected


def test_legacy_default_requires_a_recorded_interface(tmp_path):
    from acprof.tui.models.candidates import candidate_from_record, record_conditions

    fixture = ModelCandidateFixture()
    fixture.build(None, tmp_path)
    record = fixture.recorded()
    del record.state["runtime"]["task"]["pipeline_tag"]
    assert candidate_from_record(record, record_conditions(record)).status == "revalidate"
