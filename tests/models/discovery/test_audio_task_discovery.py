"""Speech checkpoints must not acquire text-only profiles from generic suffixes."""
from types import SimpleNamespace

import pytest

from acprof.host.detect import TaskInfo
from acprof.host.preparation.collection_workflow import PreparationWorkflow
from acprof.host.preparation.task_selection import select_tasks
from acprof.models.candidates import apply_model_candidate, discover_model_candidates

ASR = "automatic-speech-recognition"


@pytest.fixture
def whisper():
    return TaskInfo(
        "openai/whisper-tiny", ASR, "audio", "transformers_model", "transformers",
        "169d4a4341b33bc18d8881c4b69c2e104e1cc0af", "hub_api",
        model_config={
            "model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"],
            "is_encoder_decoder": True,
        },
        hub_metadata={
            "pipeline_tag": ASR,
            "transformers_info": {"auto_model": "AutoModelForSpeechSeq2Seq", "pipeline_tag": ASR},
        },
    )


@pytest.mark.parametrize("metadata", ["hub", "loader", "architecture"])
def test_whisper_exposes_only_its_speech_profile(whisper, metadata):
    if metadata != "hub":
        whisper.hub_metadata["pipeline_tag"] = None
    if metadata == "architecture":
        whisper.hub_metadata.pop("transformers_info")
        whisper.pipeline_tag = "unknown"
        whisper.detection_method = "config_infer"

    decision = discover_model_candidates(whisper, override_backend="transformers_model")

    assert decision["status"] == "candidate"
    assert decision["conflicts"] == decision["missing"] == []
    assert [item["interface"] for item in decision["task_discovery"]["tasks"]] == [ASR]
    assert len(decision["task_discovery"]["profiles"]) == 1


def test_run_can_select_whisper_profile_from_explicit_inspect(whisper):
    inspected = discover_model_candidates(whisper, override_tag=ASR, override_backend="transformers_model")
    profile_id = inspected["task_discovery"]["profiles"][0]["profile_id"]
    whisper.model_resolution = apply_model_candidate(
        whisper, discover_model_candidates(whisper, override_backend="transformers_model")
    )
    args = SimpleNamespace(task="", tasks=profile_id, backend="transformers_model",
                           task_family="", model_spec="", workload_spec="")

    selected = select_tasks(whisper, args, PreparationWorkflow())

    assert len(selected.selected_tasks) == 1
    assert selected.selected_tasks[0]["interface"] == ASR
    assert selected.runtime_backend == "transformers_model"


def test_whisper_still_rejects_explicit_text_only_protocol(whisper):
    decision = discover_model_candidates(whisper, override_tag="text2text-generation",
                                        override_backend="transformers_model")
    assert decision["status"] != "candidate"
    assert any("incompatible" in conflict for conflict in decision["conflicts"])
