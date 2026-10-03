"""Dependency resolution pins role-specific files without loading repository code."""
import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import test_model_contract as fixture

from acprof.model_spec import task_model_spec


class TestModelDependency:
    def discover(self, source, files, *, sha="b" * 40):
        hub = SimpleNamespace(sha=sha, siblings=[SimpleNamespace(rfilename=name) for name in files])
        lookup = Mock(return_value=hub)
        task = fixture.TestModelContract().discover(source, dependency_lookup=lookup)
        return task, lookup

    def test_tokenizer_is_pinned_without_downloading_weights(self):
        source = fixture.SOURCE + '\nAutoTokenizer.from_pretrained("example/tokenizer")\n'
        task, lookup = self.discover(source, ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"])
        spec = task_model_spec(task)
        assert (spec), task.model_resolution
        assert (task.model_revision) == (fixture.SHA)
        dep = spec["dependencies"][0]
        assert (dep["revision"]) == ("b" * 40)
        assert (dep["allow_patterns"]) == (["config.json", "tokenizer.json", "tokenizer_config.json"])
        lookup.assert_called_once_with("example/tokenizer", revision="main", files_metadata=False)
        assert (task.model_resolution["contract"]["runtime_validation"]) == ("not_run")

    def test_roles_merge_but_metadata_never_selects_weights(self):
        source = fixture.SOURCE + '\nAutoConfig.from_pretrained("example/shared")\nAutoProcessor.from_pretrained("example/shared")\n'
        task, lookup = self.discover(source, ["config.json", "preprocessor_config.json", "model.safetensors"])
        assert (task_model_spec(task)["dependencies"]) == ([{"repo_id": "example/shared", "revision": "b" * 40,
                          "allow_patterns": ["config.json", "preprocessor_config.json"]}])
        assert (lookup.call_count) == (1)

    def test_tokenizer_prefix_does_not_include_nested_repositories_or_weights(self):
        source = fixture.SOURCE + '\nAutoTokenizer.from_pretrained("example/tokenizer")\n'
        files = ["tokenizer.json", "tokenizer.model", "chat_templates/default.jinja", "tokenizer_weights.bin",
                 "tokenizer/model.safetensors", "tokenizer/backups/tokenizer.json", "backup/spiece.model",
                 "chat_templates/backup/template.jinja"]
        task, _ = self.discover(source, files)
        assert (task_model_spec(task)["dependencies"][0]["allow_patterns"]) == (["chat_templates/default.jinja", "tokenizer.json", "tokenizer.model"])

    def test_dynamic_and_unknown_loader_never_trigger_repository_lookup(self):
        for call in ('AutoTokenizer.from_pretrained(dynamic_repo())', 'CustomLoader.from_pretrained("example/unknown")'):
            task, lookup = self.discover(fixture.SOURCE + '\n' + call, ["tokenizer.json"])
            assert not (task_model_spec(task))
            lookup.assert_not_called()

    def test_unpinned_response_and_empty_selection_remain_actionable(self):
        for sha, files in (("main", ["tokenizer.json"]), ("b" * 40, ["model.safetensors"])):
            task, _ = self.discover(fixture.SOURCE + '\nAutoTokenizer.from_pretrained("example/tokenizer")', files, sha=sha)
            assert not (task_model_spec(task))
            assert ("dependencies") in (task.model_resolution["contract"]["unresolved_fields"])

    def test_dependency_commit_changes_contract_identity(self):
        source = fixture.SOURCE + '\nAutoTokenizer.from_pretrained("example/tokenizer")'
        first, _ = self.discover(source, ["tokenizer.json"])
        second, _ = self.discover(source, ["tokenizer.json"], sha="c" * 40)
        assert (first.model_resolution["contract"]["cache_key"]) != (second.model_resolution["contract"]["cache_key"])

    def test_author_dependencies_are_not_re_resolved(self):
        declaration = copy.deepcopy(fixture.EXPECTED)
        declaration["dependencies"] = [{"repo_id": "example/tokenizer", "revision": "c" * 40}]
        lookup = Mock(side_effect=AssertionError("Author dependencies must stay pinned"))
        task = fixture.TestModelContract().discover(spec=declaration, dependency_lookup=lookup)
        lookup.assert_not_called()
        assert (task_model_spec(task)) == (declaration)

    @pytest.mark.parametrize('suffix', ('\nif dynamic_training_mode():\n    AutoModel.from_pretrained("example/base")', '\nmodels = [AutoModel.from_pretrained("example/base") for item in dynamic_items()]', '\nmatch dynamic_mode():\n    case "train":\n        AutoModel.from_pretrained("example/base")'))
    def test_conditional_weight_load_does_not_download_a_potential_base_model(self, suffix):
        task, lookup = self.discover(fixture.SOURCE + suffix, ["config.json", "model.safetensors"])
        assert not (task_model_spec(task))
        lookup.assert_not_called()

    def test_source_revision_cannot_be_discarded_or_overridden_by_dynamic_kwargs(self):
        for arguments in ('revision="release"', '**load_options'):
            task, lookup = self.discover(fixture.SOURCE + '\nAutoTokenizer.from_pretrained("example/tokenizer", ' + arguments + ')',
                                        ["tokenizer.json"])
            assert not (task_model_spec(task))
            lookup.assert_not_called()


def test_custom_index_shards_match_primary_checkpoint_selection():
    from acprof.container.model_files import plan_download
    from acprof.model_dependencies import dependency_files
    index = 'model.safetensors.index.json'
    names = ['config.json', index, 'weights/encoder.safetensors', 'weights/decoder.safetensors', 'pytorch_model.bin']
    metadata = {'config.json': {'model_type': 'bert'}, index: {
        'weight_map': {'encoder': names[2], 'decoder': names[3]}}}
    expected = sorted(names[:-1])
    assert (dependency_files(names, {'weights'}, read_json=metadata.__getitem__)) == (expected)
    plan = plan_download(model_id='fixture/model', revision='a' * 40, family='nlp',
        backend='transformers_pipeline', files={name: {'size': 1} for name in names},
        read_json=metadata.__getitem__, native_model_types={'bert'})
    assert ([item['path'] for item in plan['files']]) == (expected)

def test_missing_shard_does_not_fall_back_to_incomplete_numbered_files():
    from acprof.model_dependencies import dependency_files
    index = 'model.safetensors.index.json'
    first = 'model-00001-of-00002.safetensors'
    missing = 'model-00002-of-00002.safetensors'
    with pytest.raises(ValueError, match='missing checkpoint shards'):
        dependency_files(['config.json', index, first, 'pytorch_model.bin'], {'weights'},
                         read_json=lambda _: {'weight_map': {'a': first, 'b': missing}})

@pytest.mark.parametrize('name', ('../outside.safetensors', '/outside.safetensors', 'bad\\name.safetensors'))
def test_invalid_index_paths_and_selection_limit_still_fail(name):
    from acprof.model_dependencies import dependency_files
    index = 'model.safetensors.index.json'
    with pytest.raises(ValueError, match='invalid model file path'):
        dependency_files(['config.json', index], {'weights'}, read_json=lambda _: {'weight_map': {'a': name}})
    shards = [f'shard{i}.safetensors' for i in range(127)]
    with pytest.raises(ValueError, match='128 files'):
        dependency_files(['config.json', index, *shards], {'weights'},
                         read_json=lambda _: {'weight_map': dict(enumerate(shards))})

def test_declared_dependency_is_checked_before_weight_download():
    import json
    import tempfile
    from pathlib import Path
    from unittest.mock import patch

    from acprof.container.download_model import _prepare_repository_plan
    index = 'model.safetensors.index.json'
    names = ['config.json', index, 'first.safetensors']
    hub = SimpleNamespace(sha='a' * 40, siblings=[SimpleNamespace(rfilename=name, size=100) for name in names])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / index
        path.write_text(json.dumps({'weight_map': {'a': 'first.safetensors', 'b': 'missing.safetensors'}}))
        with patch('huggingface_hub.HfApi.model_info', return_value=hub), patch(
            'huggingface_hub.hf_hub_download', return_value=str(path),
        ) as downloaded, pytest.raises(ValueError, match='missing checkpoint shards'):
            _prepare_repository_plan('https://fixture.invalid', 'fixture/model', 'a' * 40,
                                     dependency={'allow_patterns': names}, native_types=set(), library_versions={})
        assert (all(call.args[1] == index for call in downloaded.call_args_list))


def test_metadata_byte_limit_is_checked_after_download():
    import tempfile
    from pathlib import Path
    from unittest.mock import patch

    from acprof.container.download_model import _prepare_repository_plan
    index = 'model.safetensors.index.json'
    hub = SimpleNamespace(sha='a' * 40, siblings=[SimpleNamespace(rfilename=index, size=100)])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / index
        path.write_bytes(b' ' * (4 * 1024 * 1024) + b'{"weight_map":{"a":"missing"}}')
        with patch('huggingface_hub.HfApi.model_info', return_value=hub), patch(
            'huggingface_hub.hf_hub_download', return_value=str(path),
        ), pytest.raises(ValueError, match='invalid model metadata'):
            _prepare_repository_plan('https://fixture.invalid', 'fixture/model', 'a' * 40,
                                     dependency={'allow_patterns': [index]}, native_types=set(), library_versions={})


@pytest.mark.parametrize('size_case', range(4), ids=['None', '-1', "'100'", '4 * 1024 * 1024 + 1'])
def test_unknown_or_oversized_index_is_rejected_before_download(size_case):
    from unittest.mock import patch

    from acprof.host.detect import dependency_metadata
    index = 'model.safetensors.index.json'
    hub = SimpleNamespace(sha='a' * 40, siblings=[SimpleNamespace(rfilename=index)])
    size = tuple((None, -1, '100', 4 * 1024 * 1024 + 1))[size_case]
    with patch('huggingface_hub.HfApi.model_info', return_value=hub), patch(
        'huggingface_hub.hf_hub_download',
        return_value=SimpleNamespace(file_size=size, commit_hash='a' * 40),
    ) as download:
        metadata = dependency_metadata('fixture/model', 'main')
        with pytest.raises(ValueError, match='size'):
            metadata['read_json'](index)
        assert download.call_count == 1
        assert download.call_args.kwargs['dry_run'] is True

def test_valid_index_budget_lookup_is_lazy_and_uses_the_pinned_revision():
    import tempfile
    from pathlib import Path
    from unittest.mock import patch

    from acprof.host.detect import dependency_metadata
    index = 'model.safetensors.index.json'
    hub = SimpleNamespace(sha='a' * 40, siblings=[SimpleNamespace(rfilename=index)])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / index
        path.write_text(' ' * (1024 * 1024) + '{"weight_map":{"a":"custom.safetensors"}}')
        with patch('huggingface_hub.HfApi.model_info', return_value=hub), patch(
            'huggingface_hub.hf_hub_download',
            side_effect=[SimpleNamespace(file_size=path.stat().st_size, commit_hash='a' * 40), str(path)],
        ) as download:
            metadata = dependency_metadata('fixture/model', 'main')
            download.assert_not_called()
            assert (metadata['read_json'](index)['weight_map']) == ({'a': 'custom.safetensors'})
            assert download.call_count == 2
            assert download.call_args_list[0].kwargs['dry_run'] is True
            assert all(call.kwargs['revision'] == 'a' * 40 for call in download.call_args_list)
