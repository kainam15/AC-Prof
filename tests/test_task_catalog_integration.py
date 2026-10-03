"""Task detection, preflight and scale semantics for non-vision Hub tasks."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.host import detect, input_plan, model_schema, runtime_images
from acprof.host.task_support import TaskSupportError, require_task_support


def info(task, family="nlp", backend="transformers_pipeline"):
    return detect.TaskInfo("example/model", task, family, backend,
                           "transformers", "fixed-revision", "hub_api")


def test_chronos_default_scales_respect_loaded_model_context():
    with tempfile.TemporaryDirectory() as tmp, patch.object(input_plan, "_start_probe_session", return_value=SimpleNamespace(name="probe")), patch.object(
        input_plan, "stop_container_session"
    ), patch.object(input_plan, "_request_scale_meta", return_value={
        "input_scale_type": "context_length", "max_effective_input_scale": 512,
        "reason": "model context length",
    }) as metadata:
        planned = input_plan.plan_input_scales(
            info("time-series-forecasting", "timeseries", "chronos"), runtime_images.ImageInfo("unused"),
            [1], [2], ["off"], 1, tmp,
        )
        metadata.assert_called_once()
        assert (max(planned.scales)) == (512)
        assert (len(planned.scales)) == (6)
        assert (planned.workload["model_constraints"]["max_effective_input_scale"]) == (512)

@pytest.mark.parametrize('task,library,family,backend', (('table-question-answering', 'transformers', 'nlp', 'transformers_pipeline'), ('text-ranking', 'sentence-transformers', 'nlp', 'cross_encoder'), ('sentence-similarity', 'sentence-transformers', 'nlp', 'sentence_transformers'), ('text-to-audio', 'transformers', 'audio', 'transformers_pipeline'), ('audio-to-audio', 'transformers', 'audio', 'transformers_model'), ('voice-activity-detection', 'unknown', 'audio', 'torchscript'), ('tabular-classification', 'unknown', 'structured', 'torchscript'), ('tabular-regression', 'sklearn', 'structured', 'skops'), ('reinforcement-learning', 'unknown', 'structured', 'torchscript'), ('robotics', 'unknown', 'structured', 'torchscript'), ('graph-ml', 'unknown', 'structured', 'torchscript')))
def test_new_hub_tasks_select_the_appropriate_runtime(task, library, family, backend):
    with patch("huggingface_hub.HfApi.model_info", return_value=SimpleNamespace(
        pipeline_tag=task, library_name=library, sha="fixed-revision",
    )):
        actual = detect.detect_task("example/model")
        assert ((actual.task_family, actual.runtime_backend)) == ((family, backend))
        assert (actual.model_revision) == ("fixed-revision")
        if family == "structured" and backend == "torchscript":
            with pytest.raises(TaskSupportError) as caught:
                require_task_support(actual)
            assert (caught.value.failure.reason_code) == ("model_contract_required")
            assert (actual.model_resolution["status"]) == ("needs_configuration")
        else:
            require_task_support(actual)

def test_explicit_backend_override_is_preserved_and_incompatibility_rejected():
    with patch.object(detect, "_detect_from_hub", return_value=info("robotics", "structured")):
        actual = detect.detect_task("example/model", override_backend="diffusers")
    assert (actual.runtime_backend) == ("diffusers")
    with pytest.raises(TaskSupportError, match="后端"):
        require_task_support(actual)

def test_table_planning_uses_rows_instead_of_token_binary_search():
    generator = SimpleNamespace(
        default_input_scales=lambda: [1, 4],
        generate=lambda scale: {"table": {"name": ["a"] * int(scale)}, "query": "How many?"},
        effective_input_scale=lambda scale, payload: float(len(payload["table"]["name"])),
        scale_label=lambda scale: f"rows{int(scale)}",
        plan_metadata=lambda: {"input_scale_type": "table_rows"},
        input_metadata=lambda scale, payload: {"table_rows": len(payload["table"]["name"])},
    )
    with tempfile.TemporaryDirectory() as tmp, patch("acprof.workloads.get_generator", return_value=generator), patch.object(
        input_plan, "_start_probe_session", side_effect=AssertionError("table rows entered token planner")
    ):
        planned = input_plan.plan_input_scales(
            info("table-question-answering"), runtime_images.ImageInfo("unused"),
            [1], [2], ["off"], 1, tmp,
        )
        assert (planned.scales) == ([1.0, 4.0])
        assert (planned.workload["input_scale_type"]) == ("table_rows")
        saved = json.loads(Path(planned.plan_file).read_text())
        assert (saved["entries"][1]["input_metadata"]["table_rows"]) == (4)

@pytest.mark.parametrize('task', ('text-to-speech', 'text-to-audio'))
def test_text_audio_uses_token_planning_and_forwards_manifest(task):
    with tempfile.TemporaryDirectory() as tmp, patch.object(
        input_plan, "_plan_manual_nlp_scales", return_value="planned tokens"
    ) as plan, patch.object(input_plan, "_plan_audio_scales", side_effect=AssertionError("text treated as waveform")):
        result = input_plan.plan_input_scales(
            info(task, "audio"), runtime_images.ImageInfo("unused"),
            [1], [2], ["off"], 1, tmp, input_scales="8,16", workload_spec_path="text.json",
        )
        assert (result) == ("planned tokens")
        assert (plan.call_args.kwargs["workload_spec_path"]) == ("text.json")

@pytest.mark.parametrize('task,family,required,output_type', (('table-question-answering', 'nlp', {'table', 'query'}, 'table_answer'), ('sentence-similarity', 'nlp', {'query', 'documents'}, 'similarity'), ('text-ranking', 'nlp', {'query', 'documents'}, 'ranking'), ('text-to-speech', 'audio', {'text'}, 'audio'), ('text-to-audio', 'audio', {'text'}, 'audio')))
def test_static_input_contracts_describe_task_inputs(task, family, required, output_type):
    inputs, outputs = model_schema._model_io_formats(info(task, family))
    assert (required.issubset(inputs["json_schema"]["required"]))
    assert (output_type) in (outputs["json_schema"]["properties"]["output_type"]["enum"])
