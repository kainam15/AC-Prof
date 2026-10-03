"""Contract synthesis must remain static, pinned, conservative and explainable."""
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.host.detect import detect_task
from acprof.host.task_support import TaskSupportError, require_task_support
from acprof.model_spec import task_model_spec, validate_model_spec

FIXTURE = Path(__file__).parent / "fixtures" / "custom_pipeline_audio_like"
SOURCE = (FIXTURE / "pipeline.py").read_text()
CONFIG = json.loads((FIXTURE / "config.json").read_text())
SHA = "a" * 40
EXPECTED: dict = {
    "schema_version": 1, "format": "transformers-pipeline", "task": "audio-text-to-text",
    "pipeline_task": "listen-and-answer",
    "multimodal": {"inputs": {"prompt": "text", "audio": "audio", "sampling_rate": "sampling_rate"},
                   "forward_kwargs": {"max_new_tokens": "$max_new_tokens", "temperature": 0}},
}
GENERIC_LOADER = {"auto_model": "AutoModel", "pipeline_tag": "feature-extraction"}


class TestModelContract:
    def discover(self, source=SOURCE, config=None, *, revision=SHA, spec=None, readme=None,
                 dependency_lookup=None, transformers_info=None, **options):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = {"config.json": json.dumps(CONFIG if config is None else config), "pipeline.py": source}
            if spec is not None:
                documents["acprof_model.json"] = json.dumps(spec)
            if readme is not None:
                documents["README.md"] = readme
            for name, text in documents.items():
                (root / name).write_text(text)
            hub = SimpleNamespace(sha=revision, pipeline_tag="audio-text-to-text", library_name="transformers",
                                  config={}, transformers_info=transformers_info,
                                  siblings=[SimpleNamespace(rfilename=name) for name in documents])
            self.downloads = []

            def download(**kwargs):
                assert (kwargs["revision"]) == (SHA if revision == SHA else revision)
                path = root / kwargs["filename"]
                if kwargs.get("dry_run"):
                    return SimpleNamespace(commit_hash=revision, file_size=path.stat().st_size)
                self.downloads.append(kwargs["filename"])
                return str(path)

            def model_info(repo_id, **kwargs):
                # Route all Hub reads through one mock so dependency responses
                # cannot be overwritten by a nested main-repository patch.
                if repo_id == "arbitrary/audio-model":
                    return hub
                if dependency_lookup is None:
                    raise AssertionError(f"Unexpected dependency lookup: {repo_id}")
                return dependency_lookup(repo_id, **kwargs)

            with patch("huggingface_hub.HfApi.model_info", side_effect=model_info), patch(
                "huggingface_hub.hf_hub_download", side_effect=download,
            ):
                task = detect_task("arbitrary/audio-model", **options)
                # Author specs bypass synthesis, but dependency preflight still
                # needs the pinned source after this temporary snapshot closes.
                task.repository_sources["pipeline.py"] = source
                return task

    def test_generates_v1_contract_without_checkpoint_specific_routing(self):
        task = self.discover()
        assert (task_model_spec(task)) == (EXPECTED)
        validate_model_spec(task_model_spec(task))
        require_task_support(task)
        assert (task.runtime_profile_id) == ("custom-multimodal-cu128")
        assert (task.model_resolution["status"]) == ("candidate")
        report = task.model_resolution["contract"]
        assert (report["status"]) == ("resolved")
        assert (report["fields"]["task"]["state"]) == ("declared")
        assert (report["fields"]["multimodal.inputs.prompt"]["state"]) == ("derived")
        assert (report["fields"]["multimodal.forward_kwargs.temperature"]["value"]) == (0)
        assert not (any(field["state"] == "verified" for field in report["fields"].values()))
        assert ("pipeline.py") in (self.downloads)
        assert ("turns") not in (task_model_spec(task)["multimodal"]["inputs"])

    @pytest.mark.parametrize('model_classes', (['AutoModel'], 'AutoModel'))
    def test_generic_loader_hint_allows_unique_pipeline_contract_synthesis(self, model_classes):
        config = copy.deepcopy(CONFIG)
        config["custom_pipelines"]["listen-and-answer"]["pt"] = model_classes
        task = self.discover(config=config, transformers_info=GENERIC_LOADER)
        require_task_support(task)
        assert (task_model_spec(task)) == (EXPECTED)
        assert (task.runtime_backend) == ("transformers_pipeline")
        assert (task.model_resolution["runtime_validation"]["status"]) == ("not_run")
        assert not (task.model_resolution["conflicts"])
        assert ({item["task"] for item in task.model_resolution["candidates"]}) == ({"audio-text-to-text"})
        observations = task.model_resolution["provenance"]["observations"]
        hints = [item for item in observations if item.get("kind") == "loader_hint"]
        assert ([(item["field"], item["value"]) for item in hints]) == ([("hub.transformers_info.pipeline_tag", "feature-extraction")])
        assert (task.hub_metadata["transformers_info"]) == (GENERIC_LOADER)

    def test_author_spec_does_not_need_task_override_for_generic_loader_hint(self):
        task = self.discover(spec=EXPECTED, transformers_info=GENERIC_LOADER)
        require_task_support(task)
        assert (task_model_spec(task)) == (EXPECTED)
        assert ("pipeline.py") in (self.downloads)

    def test_pipeline_task_conflict_is_not_dismissed_as_a_generic_loader(self):
        task = self.discover(transformers_info={"auto_model": "AutoModelForCausalLM",
                                                "pipeline_tag": "text-generation"})
        with pytest.raises(TaskSupportError, match="conflict"):
            require_task_support(task)
        assert (task.model_resolution["status"]) == ("ambiguous")
        assert ("pipeline.py") not in (self.downloads)

    @pytest.mark.parametrize('source_case', range(2))
    def test_generic_loader_hint_still_requires_complete_inputs_and_dependencies(self, source_case):
        sources = (
            SOURCE.replace('inputs.get("prompt", "Listen.")', 'inputs["messages"]'),
            SOURCE.replace('class AudioPipeline(Pipeline):', '''class AudioPipeline(Pipeline):
    def __init__(self, model):
        if model.config.use_external_processor:
            self.processor = AutoProcessor.from_pretrained("example/audio")
'''),
        )
        source, missing = tuple(zip(sources, ('multimodal.inputs', 'dependencies')))[source_case]
        task = self.discover(source, transformers_info=GENERIC_LOADER)
        assert (task.model_resolution["status"]) == ("needs_configuration")
        assert not (task.model_resolution["conflicts"])
        assert not (task_model_spec(task))
        assert ("pipeline.py") in (self.downloads)
        with pytest.raises(TaskSupportError, match=missing):
            require_task_support(task)

    def test_required_unknown_input_is_actionable_and_not_executable(self):
        task = self.discover(SOURCE.replace('turns = inputs.get("turns", [])', 'speaker = inputs["speaker"]\n        turns = []'))
        assert (task.model_resolution["status"]) == ("needs_configuration")
        assert not (task_model_spec(task))
        with pytest.raises(TaskSupportError, match="speaker"):
            require_task_support(task)

    @pytest.mark.parametrize('index_case', range(4))
    def test_dynamic_keys_and_unproven_sampling_do_not_get_guessed(self, index_case):
        (index, source) = tuple(enumerate((SOURCE.replace('inputs.get("prompt", "Listen.")', 'inputs.get(runtime_key(), "Listen.")'), SOURCE.replace('temperature = temperature or None', 'temperature = temperature + 1'), SOURCE.replace('do_sample = temperature is not None', 'do_sample = True'), SOURCE.replace('generation_keys = ["temperature", "max_new_tokens", "repetition_penalty"]', 'generation_keys = get_keys()'))))[index_case]
        task = self.discover(source)
        assert (task.model_resolution["status"]) == ("needs_configuration")
        assert not (task_model_spec(task))

    @pytest.mark.parametrize('index_case', range(5))
    def test_dynamic_mapping_and_generation_mutations_are_unresolved(self, index_case):
        changes = (
            ('generation_kwargs = {k:', 'generation_keys.append(dynamic_key())\n        generation_kwargs = {k:'),
            ('return {}, generation_kwargs, {}', 'generation_kwargs.update(dynamic_kwargs())\n        return {}, generation_kwargs, {}'),
            ('turns = inputs.get("turns", [])', 'inputs = rewrite(inputs)\n        turns = inputs.get("turns", [])'),
            ('turns = inputs.get("turns", [])', 'eval(dynamic_code())\n        turns = inputs.get("turns", [])'),
            ('max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty,',
             'max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty, **dynamic_kwargs(),'),
        )
        (index, (old, new)) = tuple(enumerate(changes))[index_case]
        task = self.discover(SOURCE.replace(old, new))
        assert (task.model_resolution["status"]) == ("needs_configuration")
        assert not (task_model_spec(task))

    def test_literal_inputs_readme_evidence_and_report_export(self):
        from acprof.model_contract import write_model_resolution
        source = SOURCE.replace('inputs.get("audio", None)', 'inputs["audio"]')
        task = self.discover(source, readme='''```python
pipe = pipeline("listen-and-answer")
pipe({"turns": turns, "audio": audio, "sampling_rate": rate}, max_new_tokens=3)
```''')
        assert (task_model_spec(task)) == (EXPECTED)
        report = task.model_resolution["contract"]
        assert (report["fields"]["pipeline.inputs.audio"]["value"]["required"])
        assert (report["fields"]["documentation.example.0"]["state"]) == ("derived")
        with tempfile.TemporaryDirectory() as directory:
            path = write_model_resolution(task, directory)
            assert (json.loads(path.read_text())) == (task.model_resolution)
            assert not (Path(directory, "acprof_model.json").exists())

    def test_config_loader_candidates_include_plain_config_and_remain_unpinned(self):
        from acprof.model_source_analysis import dependency_candidates
        source = '''AutoConfig.from_pretrained(config.audio_model_id)
AutoModel.from_pretrained(config.text_model_name_or_path)
GenerationConfig.from_pretrained("example/generation")
AutoProcessor.from_pretrained(dynamic_repo())
'''
        candidates = dependency_candidates(source, "module.py", {
            "audio_model_id": "example/audio", "text_model_name_or_path": "example/text",
        }, "example/main")
        assert ([(item["repo_id"], item["role"]) for item in candidates]) == ([
            ("example/audio", "metadata"), ("example/text", "weights"),
            ("example/generation", "generation_metadata"), (None, "processor"),
        ])
        assert (all("revision" not in item for item in candidates))

    def test_cache_identity_changes_with_revision_and_source_and_is_not_mutable(self):
        from acprof.model_source_analysis import analyze_pipeline
        first = self.discover().model_resolution["contract"]
        second = self.discover().model_resolution["contract"]
        assert (first["cache_key"]) == (second["cache_key"])
        revised = self.discover(revision="b" * 40).model_resolution["contract"]
        changed = self.discover(SOURCE + "\n# another source\n").model_resolution["contract"]
        assert (first["cache_key"]) != (revised["cache_key"])
        assert (first["cache_key"]) != (changed["cache_key"])
        result = analyze_pipeline(SOURCE, "pipeline.py", "AudioPipeline")
        result["inputs"].clear()
        assert ("audio") in (analyze_pipeline(SOURCE, "pipeline.py", "AudioPipeline")["inputs"])

    @pytest.mark.parametrize('hint_case', range(2), ids=['None', 'GENERIC_LOADER'])
    def test_local_declaration_and_hub_conflict_stays_visible_in_provenance(self, hint_case):
        spec = copy.deepcopy(EXPECTED)
        spec["task"] = "image-text-to-text"
        spec["multimodal"]["inputs"] = {"prompt": "text", "image": "image"}
        hint = tuple((None, GENERIC_LOADER))[hint_case]
        task = self.discover(spec=spec, transformers_info=hint)
        assert (task.model_resolution["status"]) == ("ambiguous")
        assert (task.model_resolution["contract"]["status"]) == ("needs_confirmation")

    def test_reanalysis_cannot_reuse_a_previous_generated_spec_as_authority(self):
        from acprof.model_contract import apply_model_contract
        task = self.discover()
        changed_source = SOURCE.replace('inputs.get("prompt", "Listen.")', 'inputs["messages"]')
        apply_model_contract(task, lambda _: changed_source)
        assert (task.model_resolution["status"]) == ("needs_configuration")
        assert not (task_model_spec(task))

    def test_structured_chat_and_oversized_source_remain_unresolved(self):
        for source in (SOURCE.replace('inputs.get("prompt", "Listen.")', 'inputs.get("prompt", [])'),
                       SOURCE + "\n#" + "x" * (256 * 1024)):
            task = self.discover(source)
            assert (task.model_resolution["status"]) == ("needs_configuration")
            assert not (task_model_spec(task))

    def test_explicit_sampling_parameter_is_forwarded_without_temperature_guess(self):
        source = SOURCE.replace('"temperature", "max_new_tokens"', '"do_sample", "max_new_tokens"')
        source = source.replace('temperature=None, max_new_tokens=None', 'do_sample=False, max_new_tokens=None')
        source = source.replace('        temperature = temperature or None\n        do_sample = temperature is not None\n', '')
        source = source.replace('do_sample=do_sample, temperature=temperature,', 'do_sample=do_sample,')
        task = self.discover(source)
        assert (task_model_spec(task)["multimodal"]["forward_kwargs"]) == ({"max_new_tokens": "$max_new_tokens", "do_sample": False})

    @pytest.mark.parametrize('hint_case', range(2), ids=['None', 'GENERIC_LOADER'])
    def test_multiple_pipelines_require_selection_without_reading_code(self, hint_case):
        config = copy.deepcopy(CONFIG)
        config["custom_pipelines"]["other-pipeline"] = {"impl": "other.OtherPipeline", "pt": ["AutoModel"]}
        hint = tuple((None, GENERIC_LOADER))[hint_case]
        task = self.discover(config=config, transformers_info=hint)
        assert (task.model_resolution["status"]) == ("ambiguous")
        assert ("multiple custom pipelines") in (str(task.model_resolution["conflicts"]))
        assert ("pipeline.py") not in (self.downloads)

    def test_author_spec_wins_and_no_remote_python_is_read(self):
        spec = copy.deepcopy(EXPECTED)
        spec["multimodal"]["inputs"] = {"text": "text", "audio": "audio", "rate": "sampling_rate"}
        task = self.discover(spec=spec)
        assert (task_model_spec(task)) == (spec)
        assert ("pipeline.py") in (self.downloads)

    def test_remote_source_is_never_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory, "executed")
            source = f'open({str(marker)!r}, "w").write("bad")\n' + SOURCE
            script = '''import json, sys
sys.path.insert(0, "tests")
from test_model_contract import TestModelContract, task_model_spec
task = TestModelContract().discover(sys.stdin.read())
assert "torch" not in sys.modules and "transformers" not in sys.modules
print(json.dumps(task_model_spec(task)))
'''
            process = subprocess.run([sys.executable, "-c", script], input=source, capture_output=True,
                                     text=True, cwd=FIXTURE.parents[2], timeout=30, check=True)
            assert not (marker.exists())
            assert (json.loads(process.stdout)) == (EXPECTED)

    @pytest.mark.parametrize('reference', ('../evil.Pipeline', 'other/repo--evil.Pipeline'))
    def test_unsafe_code_references_are_rejected_before_source_download(self, reference):
        config = copy.deepcopy(CONFIG)
        config["custom_pipelines"]["listen-and-answer"]["impl"] = reference
        task = self.discover(config=config)
        assert (task.model_resolution["status"]) == ("needs_configuration")
        assert ("pipeline.py") not in (self.downloads)

    def test_unpinned_revision_never_reads_contract_sources(self):
        task = self.discover(revision="main")
        assert (self.downloads) == ([])
        with pytest.raises(TaskSupportError, match="SHA|revision"):
            require_task_support(task)

    def test_dependency_candidates_block_execution_but_keep_generated_draft(self):
        source = SOURCE.replace("class AudioPipeline(Pipeline):", '''class AudioPipeline(Pipeline):
    def __init__(self, model, **kwargs):
        self.processor = AutoProcessor.from_pretrained(model.config.audio_model_id)
        self.tokenizer = AutoTokenizer.from_pretrained("example/tokenizer")
''')
        config = dict(CONFIG, audio_model_id="example/audio")
        with patch("acprof.host.detect.dependency_metadata", side_effect=OSError("metadata unavailable")):
            task = self.discover(source, config)
        assert (task.model_resolution["status"]) == ("needs_configuration")
        report = task.model_resolution["contract"]
        assert (report["draft_spec"]) == (EXPECTED)
        assert ({(d["repo_id"], d["role"]) for d in report["dependency_candidates"]}) == ({("example/audio", "processor"), ("example/tokenizer", "tokenizer")})
        assert not (task_model_spec(task))
