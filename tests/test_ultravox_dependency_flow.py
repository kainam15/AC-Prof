"""Pinned upstream source is text-only acceptance input, never imported."""
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from acprof.host.detect import TaskInfo
from acprof.model_contract import apply_model_contract
from acprof.model_resolution import discover_model_candidates
from acprof.model_spec import task_model_spec

FIXTURE = Path(__file__).parent / "fixtures" / "ultravox_dependency_snapshot"
MODEL = "fixie-ai/ultravox-v0_5-llama-3_2-1b"


class TestUltravoxDependencyFlow:
    def discover(self, model=MODEL, version="4.57.6"):
        repositories = json.loads((FIXTURE / "repositories.json").read_text())
        metadata = {p.name: json.loads(p.read_text()) for p in FIXTURE.glob("*.json")
                    if p.name not in {"repositories.json", "source_sha256.json"}}
        config = metadata["config.json"]
        task = TaskInfo(model, "audio-text-to-text", "multimodal", "transformers_model", "transformers",
                        repositories[MODEL]["revision"], "hub_api", model_config=config,
                        repository_files=tuple(repositories[MODEL]["files"]), repository_metadata=metadata,
                        hub_metadata={"transformers_info": {"auto_model": "AutoModel", "pipeline_tag": "feature-extraction"}})
        task.model_resolution = discover_model_candidates(task)

        def read_text(name):
            return (FIXTURE / (name + ".txt" if name.endswith(".py") else name)).read_text()

        lookup = Mock(side_effect=lambda repo, revision: copy.deepcopy(repositories[repo]))
        with patch("acprof.runtime_profiles.locked_transformers_version", return_value=version):
            apply_model_contract(task, read_text, resolve_repository=lookup)
        return task, lookup

    @pytest.mark.parametrize("name,digest", [
        pytest.param(name, digest, id=name)
        for name, digest in sorted(json.loads((FIXTURE / "source_sha256.json").read_text()).items())
    ])
    def test_snapshot_hashes_match_fixed_source_input(self, name, digest):
        path = FIXTURE / (name + ".txt" if name.endswith(".py") else name)
        data = path.read_bytes()
        # The upstream config lacks a final newline; the fixture adds
        # one for repository text hygiene without changing JSON data.
        if name == "config.json":
            data = data.removesuffix(b"\n")
        assert (hashlib.sha256(data).hexdigest()) == (digest)

    @pytest.mark.parametrize('model_case', range(2), ids=['MODEL', "'arbitrary/composite-audio'"])
    def test_zero_unresolved_dependencies_without_checkpoint_name_routing(self, model_case):
        example = json.loads((Path(__file__).parents[1] / "examples/multimodal/ultravox.model.json").read_text())
        model = tuple((MODEL, 'arbitrary/composite-audio'))[model_case]
        task, lookup = self.discover(model)
        report = task.model_resolution["contract"]
        assert (report["unresolved_fields"]) == ([])
        assert (report["status"]) == ("resolved"), report["fields"].get("dependencies")
        assert (task.model_resolution["status"]) == ("candidate")
        assert (task.runtime_backend) == ("transformers_pipeline")
        spec = task_model_spec(task)
        for field in ("task", "pipeline_task", "format", "multimodal"):
            assert (spec[field]) == (example[field])
        dependencies = {d["repo_id"]: d for d in spec["dependencies"]}
        assert ({repo: d["revision"] for repo, d in dependencies.items()}) == ({d["repo_id"]: d["revision"] for d in example["dependencies"]})
        assert (dependencies["meta-llama/Llama-3.2-1B-Instruct"]["allow_patterns"]) == (["config.json", "generation_config.json", "model.safetensors"])
        assert (dependencies["openai/whisper-large-v3-turbo"]["allow_patterns"]) == (["config.json", "preprocessor_config.json"])
        candidates = report["dependency_candidates"]
        assert not (any(c["activation"] == "unknown" for c in candidates))
        assert (any(c["dependency_kind"] == "main_model" and c["loader"] == "super()" for c in candidates))
        assert (any(c.get("alternative", {}).get("branch") == "fallback"
                            and c["activation"] == "inactive" for c in candidates))
        assert (any(c["role"] == "weights" and c["activation"] == "inactive"
                            and "audio_model_id" in c["expression"] for c in candidates))
        assert (lookup.call_count) == (2)
        assert (report["runtime_validation"]) == ("not_run")

    def test_unreviewed_transformers_version_keeps_weight_lifecycle_unknown(self):
        task, _ = self.discover(version="99.0.0")
        report = task.model_resolution["contract"]
        assert ("dependencies") in (report["unresolved_fields"])
        assert not (task_model_spec(task))
        assert (any(c["role"] == "weights" and c["activation"] == "unknown"
                            for c in report["dependency_candidates"]))
        assert not (any("model.safetensors" in d["allow_patterns"]
                             for d in report["draft_spec"].get("dependencies", [])))
