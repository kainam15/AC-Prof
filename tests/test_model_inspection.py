"""Inspection exposes provenance without starting measurement or loading models."""
import contextlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import test_model_contract as fixture

from acprof.cli.main import main


@pytest.mark.parametrize('family,tag,expected_scale,field', (('nlp', 'fill-mask', 64, 'text'), ('cv', 'image-classification', 0.1, 'image_base64')))
def test_native_probe_uses_family_defaults_when_generator_has_no_declared_scales(family, tag, expected_scale, field):
    from types import SimpleNamespace

    from acprof.host.detect import TaskInfo
    from acprof.host.model_inspection import probe_model_contract
    task = TaskInfo("example/native", tag, family, "transformers_pipeline", "transformers", "a" * 40, "hub")
    with tempfile.TemporaryDirectory() as directory, patch(
            "acprof.host.runtime_images.prepare_image", return_value=SimpleNamespace(tag="sha256:" + "b" * 64)), patch(
            "acprof.host.preflight.require_collection_host"), patch(
            "acprof.host.preflight.require_native_docker"), patch(
            "acprof.host.run_state.MeasurementLock"), patch(
            "acprof.host.runtime_validation.validate_runtime", return_value={"status": "ok"}) as validate:
        probe_model_contract(task, directory, mode="full")
        entry = json.loads(Path(validate.call_args.kwargs["planned"].plan_file).read_text())["entries"][0]
        assert (entry["input_scale"]) == (expected_scale)
        assert (entry["payload"][field])
        if family == "nlp":
            assert ("[MASK]") in (entry["payload"][field])

def test_full_probe_accepts_native_models_and_preserves_feature_width():
    from types import SimpleNamespace

    from acprof.host.detect import TaskInfo
    from acprof.host.model_inspection import probe_model_contract
    task = TaskInfo("example/iris", "tabular-classification", "structured", "onnxruntime",
                    "onnx", "a" * 40, "manual")
    task.model_spec = {"schema_version": 1, "format": "onnxruntime", "task": "tabular-classification",
                       "model_file": "iris.onnx", "feature_dim": 4}
    with tempfile.TemporaryDirectory() as directory, patch("acprof.host.runtime_images.prepare_image",
            return_value=SimpleNamespace(tag="sha256:" + "b" * 64)), patch(
            "acprof.host.preflight.require_collection_host"), patch(
            "acprof.host.preflight.require_native_docker"), patch(
            "acprof.host.run_state.MeasurementLock"), patch(
            "acprof.host.runtime_validation.validate_runtime", return_value={"status": "ok"}) as validate:
        report = probe_model_contract(task, directory, mode="full")
        assert (report["status"]) == ("ok")
        plan = json.loads(Path(validate.call_args.kwargs["planned"].plan_file).read_text())
        assert (len(plan["entries"][0]["payload"]["features"][0])) == (4)
        assert ("params") not in (plan["entries"][0]["payload"])

def test_review_provenance_survives_export_and_subprocess_inspection():
    from acprof.model_contract import write_model_resolution
    from acprof.model_spec import task_model_spec
    task = fixture.TestModelContract().discover()
    original = task.model_resolution["contract"]
    declaration = task_model_spec(task)
    author_task = fixture.TestModelContract().discover(spec=declaration)
    with tempfile.TemporaryDirectory() as directory, patch("acprof.host.detect.detect_task", return_value=author_task), patch(
        "acprof.host.env_utils.bootstrap_project_env",
    ), contextlib.redirect_stdout(io.StringIO()):
        write_model_resolution(task, directory)
        path = Path(directory, "acprof_model.json")
        path.write_text(json.dumps(declaration))
        assert (main(["inspect", task.model_id, "--model-spec", str(path),
                               "--expected-revision", task.model_revision, "--output-dir", directory])) == (0)
        saved = json.loads(Path(directory, "model_resolution.json").read_text())["contract"]
        assert (saved["cache_key"]) == (original["cache_key"])
        assert (saved["fields"]["multimodal.inputs.prompt"]["sources"]) == (original["fields"]["multimodal.inputs.prompt"]["sources"])

def test_changed_revision_refuses_probe_and_preserves_previous_report():
    task = fixture.TestModelContract().discover()
    with tempfile.TemporaryDirectory() as directory, patch("acprof.host.detect.detect_task", return_value=task), patch(
        "acprof.host.env_utils.bootstrap_project_env",
    ), patch("acprof.host.runtime_images.prepare_image") as build:
        path = Path(directory, "model_resolution.json")
        path.write_text('{"preserve":true}')
        assert (main(["inspect", task.model_id, "--expected-revision", "b" * 40,
                               "--probe", "full", "--output-dir", directory])) == (2)
        assert (path.read_text()) == ('{"preserve":true}')
        build.assert_not_called()

def test_inspect_explains_and_exports_the_static_contract():
    task = fixture.TestModelContract().discover(transformers_info=fixture.GENERIC_LOADER)
    with tempfile.TemporaryDirectory() as directory, patch("acprof.host.detect.detect_task", return_value=task), patch(
        "acprof.host.env_utils.bootstrap_project_env",
    ), patch("acprof.host.runtime_images.prepare_image") as build, contextlib.redirect_stdout(io.StringIO()) as output:
        code = main(["inspect", task.model_id, "--explain", "--output-dir", directory])
        assert (code) == (0)
        report = json.loads(Path(directory, "model_resolution.json").read_text())
        assert (report["contract"]["runtime_validation"]) == ("not_run")
        assert ("multimodal.inputs.prompt") in (output.getvalue())
        assert ("pipeline.py") in (output.getvalue())
        assert ("feature-extraction [loader_hint]") in (output.getvalue())
        assert ("not a task declaration") in (output.getvalue())
        build.assert_not_called()

def test_unresolved_inspection_exports_draft_and_returns_two():
    task = fixture.TestModelContract().discover(fixture.SOURCE.replace('inputs.get("prompt", "Listen.")', 'inputs["turns"]'))
    with tempfile.TemporaryDirectory() as directory, patch("acprof.host.detect.detect_task", return_value=task), patch(
        "acprof.host.env_utils.bootstrap_project_env",
    ), contextlib.redirect_stdout(io.StringIO()) as output:
        assert (main(["inspect", task.model_id, "--output-dir", directory])) == (2)
        assert ("multimodal.inputs.turns") in (output.getvalue())
        assert (Path(directory, "model_resolution.json").is_file())
