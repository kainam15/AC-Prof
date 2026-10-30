import json
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from acprof.tasks.workloads import parse_task, workload_manifest
from acprof.workloads.nlp import NLPWorkloadGenerator


@pytest.mark.parametrize(("task_id", "prefix"), [
    ("translation_en_to_de", "translate English to German: "),
    ("summarization", "summarize: "),
    ("generative-question-answering", "question: When does the library open? context: "),
])
def test_task_templates_and_checkpoint_generation_survive_materialization(tmp_path, task_id, prefix):
    task = parse_task(task_id, {"num_beams": 4, "max_length": 200, "min_length": 30},
                      encoder_decoder=True, source="fixture")
    path = tmp_path / "workload.json"
    path.write_text(json.dumps(workload_manifest(task)))
    generator = NLPWorkloadGenerator("fixture/model", task["interface"], 2, str(path))
    payload = generator.generate_for_word_count(48)
    assert payload["text"].startswith(prefix)
    assert payload["params"] == {"num_beams": 4, "max_length": 200, "min_length": 30, "do_sample": False}
    assert "max_new_tokens" not in payload["params"]
    assert generator.plan_metadata()["task_workload"]["task_id"] == task_id
    assert generator.plan_metadata()["template_sha256"]


def test_modified_workload_fingerprint_is_rejected(tmp_path):
    task = parse_task("summarization", {}, encoder_decoder=True, source="fixture")
    task["config"]["prefix"] = "translate: "
    path = tmp_path / "workload.json"
    path.write_text(json.dumps(workload_manifest(task)))
    with pytest.raises(ValueError, match="fingerprint"):
        NLPWorkloadGenerator("fixture/model", task["interface"], 1, str(path))


def test_decoder_only_generative_question_answering_has_question_and_context(tmp_path):
    task = parse_task("generative-question-answering", {}, encoder_decoder=False, source="explicit_task")
    assert task["interface"] == "text-generation"
    path = tmp_path / "workload.json"
    path.write_text(json.dumps(workload_manifest(task)))
    payload = NLPWorkloadGenerator("fixture/model", task["interface"], 1, str(path)).generate_for_word_count(30)
    assert "question:" in payload["text"]
    assert "context:" in payload["text"]


@pytest.mark.parametrize(("version", "dtype_key"), [("4.57.6", "torch_dtype"), ("5.10.4", "dtype")])
def test_seq2seq_loader_uses_native_generate_on_both_transformers_versions(version, dtype_key):
    from acprof.container.handlers.text_generation import load_seq2seq
    transformers = MagicMock(__version__=version)
    with patch.dict("sys.modules", {"transformers": transformers}):
        load_seq2seq("fixture/model", "a" * 40, device="cpu", dtype="float32", model_kwargs={})
    assert transformers.AutoModelForSeq2SeqLM.from_pretrained.call_args.kwargs[dtype_key] == "float32"
    assert transformers.AutoTokenizer.from_pretrained.call_args.kwargs["revision"] == "a" * 40
    transformers.pipeline.assert_not_called()


def test_direct_seq2seq_generation_counts_actual_tokens_and_respects_parameters():
    from acprof.container.handlers.text_generation import generate

    class Tensor:
        def __init__(self, rows):
            self.rows = rows
            self.shape = (len(rows), len(rows[0]))

        def to(self, _device):
            return self

        def tolist(self):
            return self.rows

    config = SimpleNamespace(eos_token_id=1, decoder_start_token_id=0, is_encoder_decoder=True)
    output = Tensor([[0, 17, 19, 1], [0, 21, 1, 0]])
    model = Mock(device="cpu", config=config, generation_config=config)
    model.generate.return_value = output
    tokenizer = Mock(return_value={"input_ids": Tensor([[5, 6, 1]] * 2),
                                   "attention_mask": Tensor([[1, 1, 1]] * 2)})
    tokenizer.batch_decode.return_value = ["Hallo Welt", "Hallo"]
    ctx = {"pipeline": SimpleNamespace(model=model, tokenizer=tokenizer, generation_config=config)}
    with patch.dict("sys.modules", {"torch": MagicMock()}):
        result = generate(ctx, {"text": "translate English to German: Hello world", "batch_size": 2,
                                "params": {"max_length": 10, "num_beams": 4}})
    assert result == [{"generated_text": "Hallo Welt"}, {"generated_text": "Hallo"}]
    assert model.generate.call_args.kwargs["num_beams"] == 4
    assert "max_new_tokens" not in model.generate.call_args.kwargs
    assert ctx["generation_evidence"]["actual_input_tokens"] == 6
    assert ctx["generation_evidence"]["actual_output_tokens"] == 5
    assert ctx["generation_evidence"]["actual_output_tokens_per_sequence"] == [3, 2]


def test_decoder_only_evidence_uses_executed_generation_parameters():
    import numpy as np

    from acprof.container.handlers.nlp import NLPHandler
    pipeline = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=1),
                               model=SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=False)))
    context = {'task_type': 'text-generation', 'pipeline': pipeline, '_generation_observation': {
        'batch_size': 1, 'params': {'max_length': 5, 'eos_token_id': 9},
        'inputs': [{'input_ids': np.array([[10, 11, 12]])}],
        'outputs': [{'input_ids': np.array([[10, 11, 12]]),
                     'generated_sequence': np.array([[10, 11, 12, 4, 9]])}]}}
    response = NLPHandler().postprocess(context, [{'generated_text': 'answer'}])
    assert response['actual_input_tokens'] == 3
    assert response['actual_output_tokens'] == 2
    assert response['stop_reason'] == 'eos'
    assert response['generation_parameters'] == {'max_length': 5, 'eos_token_id': 9}


def test_task_length_parameters_override_checkpoint_defaults_in_either_notation():
    task = parse_task('summarization', {'max_length': 80, 'min_length': 20}, encoder_decoder=True,
                      source='fixture', default_generation={'max_new_tokens': 32, 'min_new_tokens': 4})
    assert task['config']['generation'] == {'max_length': 80, 'min_length': 20, 'do_sample': False}


def test_non_generation_task_does_not_silently_discard_generation_parameters():
    with pytest.raises(ValueError, match='unsupported task configuration'):
        parse_task('text-classification', {'num_beams': 4}, encoder_decoder=False, source='fixture')
