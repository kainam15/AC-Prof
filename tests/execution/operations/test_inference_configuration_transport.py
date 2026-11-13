"""Inference requests and confirmation proofs cross the CLI boundary without loss."""
from dataclasses import replace

import pytest

from acprof.events.preparation import encode_event, encode_reply, parse_event
from acprof.experiments.arguments import build_parser
from acprof.experiments.configuration import RunConfig, RunConfigError, build_run_command


def test_inference_request_and_confirmation_round_trip_without_becoming_preset_identity(tmp_path):
    request_path = tmp_path / "inference requests.json"
    request_path.write_text('{"schema_version": 1}', encoding="utf-8")
    original = RunConfig.smoke("fixture/model")
    requested = replace(original, inference_config=str(request_path), confirm_inference="a" * 64)
    command = build_run_command(requested, project_dir=tmp_path)
    namespace = build_parser().parse_args(command[command.index("run") + 1:])
    restored = RunConfig.from_namespace(namespace).validate(project_dir=tmp_path)
    assert restored.inference_config == str(request_path)
    assert restored.confirm_inference == "a" * 64
    assert requested.experiment_parameters() == original.experiment_parameters()


@pytest.mark.parametrize("digest", ["yes", "A" * 64, "a" * 63, "a" * 65])
def test_confirmation_requires_an_exact_lowercase_sha256(digest, tmp_path):
    with pytest.raises(RunConfigError, match="confirm_inference"):
        replace(RunConfig.smoke("fixture/model"), confirm_inference=digest).validate(
            project_dir=tmp_path,
        )


def test_missing_inference_request_file_has_a_field_error(tmp_path):
    with pytest.raises(RunConfigError, match="inference_config"):
        replace(RunConfig.smoke("fixture/model"), inference_config="absent.json").validate(
            project_dir=tmp_path,
        )


def test_confirmation_protocol_preserves_the_reviewed_digest():
    digest = "b" * 64
    event = parse_event(encode_event("resolution", "waiting", request={
        "id": 3, "kind": "review", "purpose": "inference_confirmation",
        "questions": [], "resolved": True, "confirmation_digest": digest,
    }))
    assert event["request"]["confirmation_digest"] == digest
    reply = encode_reply(3, "confirm", confirmation_digest=digest)
    assert '"confirmation_digest": "' + digest + '"' in reply


@pytest.mark.parametrize("digest", ["", "b" * 63, "B" * 64])
def test_final_confirmation_event_cannot_omit_or_mangle_its_digest(digest):
    with pytest.raises(ValueError, match="confirmation"):
        encode_event("resolution", "waiting", request={
            "id": 3, "kind": "review", "purpose": "inference_confirmation",
            "questions": [], "resolved": True, "confirmation_digest": digest,
        })


def test_final_confirmation_cannot_contain_unapplied_editable_questions():
    with pytest.raises(ValueError, match="confirmation"):
        encode_event("resolution", "waiting", request={
            "id": 3, "kind": "review", "purpose": "inference_confirmation",
            "questions": [{"path": "dtype"}], "resolved": True,
            "confirmation_digest": "b" * 64,
        })


def test_confirmation_digest_is_only_transmitted_with_confirm():
    with pytest.raises(ValueError, match="confirmation"):
        encode_reply(3, "answer", confirmation_digest="b" * 64)
