import pytest

from acprof.container.validation import OutputValidationError, validate_output
from acprof.workloads.contract import workload_contract


def test_native_chat_uses_observed_generation_evidence():
    context = {"task_type": "conversational", "native_chat": True}
    raw = {"text": "Hello", "input_scale": 9, "batch_size": 2, "params": {"max_new_tokens": 32},
           "task_workload": {"interface": "conversational", "task_id": "fixture-chat"}}
    prepared = {"_effective_input_scale": 9, "_input_token_scope": "templated_prompt_per_sample_including_special_tokens"}
    output = [[{"generated_text": "Hi"}], [{"generated_text": "Hi"}]]
    response = {"task": "conversational", "output_type": "text", "n_results": 2, "actual_input_tokens": 18,
                "actual_output_tokens": 4, "actual_output_tokens_per_sequence": [2, 2],
                "output_token_count": 2, "inference_interface": "TextGenerationPipeline", "stop_reason": "eos"}
    result = validate_output(context, raw, prepared, output, response)
    assert "observed_input_and_output_tokens" in result["task"]["checks"]
    contract = result["workload_contract"]
    assert contract["input"]["text"] == {"tokens": 18, "token_scope": "model_input_including_special_tokens"}
    assert contract["generation"] == {"parameters": {"max_new_tokens": 32}, "interface": "TextGenerationPipeline",
                                      "max_output_tokens": 32, "actual_output_tokens": 4,
                                      "actual_output_tokens_per_sequence": [2, 2], "actual_output_tokens_status": "available",
                                      "stop_reason": "eos", "decoded_text_token_count": 2}

    missing = {**response, "actual_input_tokens": None, "actual_output_tokens": None, "actual_output_tokens_per_sequence": None}
    partial = workload_contract(context, raw, prepared, missing)
    assert partial["input"]["text"] == {"tokens": 9, "token_scope": prepared["_input_token_scope"]}
    assert partial["generation"]["actual_output_tokens"] is None
    assert partial["generation"]["actual_output_tokens_status"] == "unavailable"
    with pytest.raises(OutputValidationError, match="selected task requires observed"):
        validate_output(context, raw, prepared, output, missing)


def test_native_chat_rejects_empty_reply_and_preserves_custom_output_protocol():
    raw = {"text": "Hello"}
    prepared = {"_effective_input_scale": 9}
    response = {"task": "conversational", "output_type": "text", "n_results": 1}
    with pytest.raises(OutputValidationError, match="empty text"):
        validate_output({"task_type": "conversational", "native_chat": True},
                        raw, prepared, [{"generated_text": ""}], response)
    custom = validate_output({"task_type": "conversational"}, raw, prepared, [{"custom_output": 1}], response)
    assert "prediction_records_or_tensors" in custom["task"]["checks"]
    assert "generation" not in custom["workload_contract"]
