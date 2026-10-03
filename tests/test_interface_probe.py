"""Source-only inspection must never enter the model preparation path."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import test_model_contract as contracts


def bundle_module():
    assert importlib.util.find_spec("acprof.host.source_bundle"), "source-only bundle is missing"
    from acprof.host import source_bundle
    return source_bundle


def test_interface_probe_does_not_prepare_a_model(tmp_path):
    from acprof.host.interface_probe import probe_interface
    task = contracts.TestModelContract().discover()
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["docker", "run"]:
            Path(command[command.index("--cidfile") + 1]).write_text("c" * 64)
            assert "--network" in command and command[command.index("--network") + 1] == "none"
            assert "--read-only" in command
            assert command[command.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
            assert "--gpus" not in command
            mounts = [command[i + 1] for i, value in enumerate(command) if value == "-v"]
            assert all(value.endswith(":ro") for value in mounts)
            assert all("model-store" not in value for value in mounts)
            report = {"status": "ok", "stages": [{"stage": name, "status": "verified"} for name in ("import", "signature")]}
            return SimpleNamespace(returncode=0, stdout="ACPROF_INTERFACE_VALIDATION=" + json.dumps(report), stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    with patch("acprof.host.runtime_images.prepare_image", side_effect=AssertionError("weights path entered")), patch(
        "acprof.host.model_store.prepare_model", side_effect=AssertionError("model store entered")), patch(
        "acprof.host.model_store.plan_model", side_effect=AssertionError("model planning entered")), patch(
        "acprof.host.interface_probe.prepare_environment_image", return_value=SimpleNamespace(image_id="sha256:" + "b" * 64)), patch(
        "acprof.host.interface_probe.run_command", side_effect=run), patch(
        "acprof.host.interface_probe.recover_abandoned_containers"):
        assert probe_interface(task, tmp_path)["status"] == "ok"
    assert len([command for command in calls if command[:2] == ["docker", "run"]]) == 1
    assert calls[-1] == ["docker", "rm", "-f", "c" * 64]
    assert not list(tmp_path.rglob("*.csv"))


def test_dependency_image_failure_is_an_environment_failure(tmp_path):
    from acprof.host.interface_probe import InterfaceProbeError, probe_interface
    task = contracts.TestModelContract().discover()
    with patch("acprof.host.interface_probe.prepare_environment_image", side_effect=RuntimeError("registry unavailable")), patch(
        "acprof.host.interface_probe.run_command") as run, pytest.raises(InterfaceProbeError) as caught:
        probe_interface(task, tmp_path)
    assert caught.value.stage == "environment"
    assert json.loads((tmp_path / "interface_validation.json").read_text())["failed_stage"] == "environment"
    run.assert_not_called()



def test_bundle_reuses_exact_cached_sources_and_cleans_up():
    task = contracts.TestModelContract().discover()
    task.repository_sources = {"pipeline.py": contracts.SOURCE}
    with patch("acprof.host.detect.read_model_source", side_effect=AssertionError("network accessed")):
        with bundle_module().source_bundle(task) as bundle:
            root = bundle.root
            assert (root / "pipeline.py").read_text() == contracts.SOURCE
            assert json.loads((root / "config.json").read_text())["custom_pipelines"]
            assert not list(root.rglob("*.safetensors"))
    assert not root.exists()


@pytest.mark.parametrize("filename", ["model.safetensors", "model.bin", "model.pt", "model.pth", "model.ckpt", "model.gguf", "model.onnx", "../escape.py", "/escape.py", "weights.bin.py"])
def test_bundle_rejects_non_source_and_unsafe_paths_before_fetch(filename):
    task = contracts.TestModelContract().discover()
    task.model_resolution["contract"]["fields"]["custom_code"]["value"]["files"].append(filename)
    with patch("acprof.host.detect.read_model_source") as fetch, pytest.raises(ValueError):
        with bundle_module().source_bundle(task):
            pass
    fetch.assert_not_called()


def test_missing_graph_does_not_fall_back_to_repository_download():
    task = contracts.TestModelContract().discover()
    del task.model_resolution["contract"]["fields"]["custom_code"]
    with patch("huggingface_hub.snapshot_download") as snapshot, patch("acprof.host.detect.read_model_source") as fetch:
        with pytest.raises(ValueError, match="interface probe source graph incomplete"):
            with bundle_module().source_bundle(task):
                pass
    snapshot.assert_not_called()
    fetch.assert_not_called()


def test_missing_declared_source_fetches_only_that_file():
    task = contracts.TestModelContract().discover()
    task.repository_sources = {}
    with patch("acprof.host.detect.read_model_source", return_value=contracts.SOURCE) as fetch:
        with bundle_module().source_bundle(task) as bundle:
            assert (bundle.root / "pipeline.py").is_file()
    fetch.assert_called_once_with(task.model_id, "pipeline.py", task.model_revision)


def test_source_content_must_match_static_evidence():
    task = contracts.TestModelContract().discover()
    task.repository_sources = {"pipeline.py": contracts.SOURCE + "\n# changed"}
    with pytest.raises(ValueError, match="source.*changed"):
        with bundle_module().source_bundle(task):
            pass


def test_explicit_declaration_keeps_a_complete_source_graph():
    task = contracts.TestModelContract().discover()
    from acprof.model_spec import task_model_spec
    explicit = contracts.TestModelContract().discover(spec=task_model_spec(task))
    assert explicit.model_resolution["contract"]["fields"]["custom_code"]["value"]["files"] == ["pipeline.py"]


def test_bundle_cancellation_removes_temporary_directory():
    task = contracts.TestModelContract().discover()
    task.repository_sources = {"pipeline.py": contracts.SOURCE}
    with pytest.raises(KeyboardInterrupt):
        with bundle_module().source_bundle(task) as bundle:
            root = bundle.root
            raise KeyboardInterrupt()
    assert not root.exists()


def test_static_graph_includes_package_initializers_and_imported_submodules():
    from acprof.model_evidence import ModelEvidence
    from acprof.model_metadata_analysis import collect_source_evidence
    task = contracts.TestModelContract().discover()
    sources = {"pipeline.py": "from .processors import audio\n", "processors/__init__.py": "",
               "processors/audio.py": "from .helpers.codec import decode\n",
               "processors/helpers/__init__.py": "from . import constants\n",
               "processors/helpers/constants.py": "RATE = 16000\n",
               "processors/helpers/codec.py": "def decode(): pass\n"}
    task.repository_files = tuple(sources)
    evidence = ModelEvidence(task.model_id, task.model_revision)
    graph = collect_source_evidence(task, evidence, contracts.CONFIG, sources.__getitem__)
    assert set(graph) == set(sources)


def test_static_graph_accepts_name_exported_by_package_initializer():
    from acprof.model_evidence import ModelEvidence
    from acprof.model_metadata_analysis import collect_source_evidence
    task = contracts.TestModelContract().discover()
    sources = {
        "pipeline.py": "from . import RATE\n",
        "__init__.py": "from .constants import RATE\n",
        "constants.py": "RATE = 16000\n",
    }
    task.repository_files = tuple(sources)
    evidence = ModelEvidence(task.model_id, task.model_revision)
    graph = collect_source_evidence(task, evidence, contracts.CONFIG, sources.__getitem__)
    assert set(graph) == set(sources)


def test_static_graph_accepts_name_defined_by_package_initializer():
    from acprof.model_evidence import ModelEvidence
    from acprof.model_metadata_analysis import collect_source_evidence
    task = contracts.TestModelContract().discover()
    sources = {
        "pipeline.py": "from . import RATE\n",
        "__init__.py": "RATE = 16000\n",
    }
    task.repository_files = tuple(sources)
    evidence = ModelEvidence(task.model_id, task.model_revision)
    graph = collect_source_evidence(task, evidence, contracts.CONFIG, sources.__getitem__)
    assert set(graph) == set(sources)


def test_static_graph_rejects_unbound_package_name():
    from acprof.model_evidence import ModelEvidence
    from acprof.model_metadata_analysis import collect_source_evidence
    task = contracts.TestModelContract().discover()
    sources = {
        "pipeline.py": "from . import MISSING\n",
        "__init__.py": "RATE = 16000\n",
    }
    task.repository_files = tuple(sources)
    evidence = ModelEvidence(task.model_id, task.model_revision)
    with pytest.raises(ValueError, match="relative source module is missing"):
        collect_source_evidence(task, evidence, contracts.CONFIG, sources.__getitem__)


@pytest.mark.parametrize("initializer", [
    "if True:\n    RATE = 16000\n",
    "RATE = 16000\ndel RATE\n",
    "from . import RATE\n",
])
def test_static_graph_does_not_treat_unproven_initializer_bindings_as_exports(initializer):
    from acprof.model_evidence import ModelEvidence
    from acprof.model_metadata_analysis import collect_source_evidence
    task = contracts.TestModelContract().discover()
    sources = {"pipeline.py": "from . import RATE\n", "__init__.py": initializer}
    task.repository_files = tuple(sources)
    evidence = ModelEvidence(task.model_id, task.model_revision)
    with pytest.raises(ValueError, match="relative source module is missing"):
        collect_source_evidence(task, evidence, contracts.CONFIG, sources.__getitem__)


def test_bundle_rejects_graph_with_missing_declared_dependency():
    task = contracts.TestModelContract().discover()
    task.model_resolution["contract"]["sources"]["helpers.py"] = {"sha256": "a" * 64}
    with pytest.raises(ValueError, match="source graph incomplete"):
        with bundle_module().source_bundle(task):
            pass


def test_tokenizer_metadata_keeps_its_source_graph_and_small_metadata():
    from acprof.host.detect import TaskInfo
    from acprof.model_contract import apply_model_contract
    task = TaskInfo("fixture/tokenizer", "fill-mask", "nlp", "transformers_pipeline", "transformers", "a" * 40, "fixture",
                    model_config={"model_type": "bert"}, repository_files=("custom_tokenizer.py",),
                    repository_metadata={"tokenizer_config.json": {"auto_map": {"AutoTokenizer": ["custom_tokenizer.Tokenizer", None]}}})
    apply_model_contract(task, lambda name: "class Tokenizer: pass\n")
    with bundle_module().source_bundle(task) as bundle:
        assert (bundle.root / "custom_tokenizer.py").is_file()
        assert json.loads((bundle.root / "tokenizer_config.json").read_text())["auto_map"]


@pytest.mark.parametrize("interrupted", [True, False])
def test_interface_failure_cleans_container_and_both_temporary_directories(tmp_path, interrupted):
    import subprocess

    from acprof.host.interface_probe import probe_interface
    task = contracts.TestModelContract().discover()
    directories = []
    def run(command, **kwargs):
        if command[:2] == ["docker", "run"]:
            cidfile = Path(command[command.index("--cidfile") + 1])
            cidfile.write_text("c" * 64)
            directories.append(cidfile.parent)
            directories.extend(Path(command[i + 1].split(":")[0]) for i, value in enumerate(command) if value == "-v")
            if interrupted:
                raise KeyboardInterrupt()
            raise subprocess.TimeoutExpired(command, 1, output="partial probe log")
        assert command == ["docker", "rm", "-f", "c" * 64]
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    with patch("acprof.host.interface_probe.prepare_environment_image", return_value=SimpleNamespace(image_id="sha256:" + "b" * 64)), patch(
        "acprof.host.interface_probe.recover_abandoned_containers"), patch(
        "acprof.host.interface_probe.run_command", side_effect=run), pytest.raises(KeyboardInterrupt if interrupted else subprocess.TimeoutExpired):
        probe_interface(task, tmp_path)
    assert directories and all(not directory.exists() for directory in directories)
    report = json.loads((tmp_path / "interface_validation.json").read_text())
    assert report["status"] == ("cancelled" if interrupted else "error")
    if not interrupted:
        assert "partial probe log" in (tmp_path / "logs" / "interface_validation.log").read_text()
