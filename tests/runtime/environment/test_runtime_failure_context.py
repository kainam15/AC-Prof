"""Typed runtime failures survive SDK wrappers and reach public consumers."""
import json
import sys
from unittest.mock import Mock

import httpx
import pytest

from acprof.container import runtime_validate
from acprof.container.load_policy import load_processor
from acprof.failures import Failure, RuntimeFailure, compatibility_status, failure_from_exception
from acprof.input_budget import InputBudgetExceeded


def wrapped_error(cause: Exception, *, suppressed: bool = False) -> RuntimeError:
    try:
        raise cause
    except Exception:
        try:
            if suppressed:
                raise RuntimeError("fixture SDK wrapper") from None
            raise RuntimeError("fixture SDK wrapper")
        except RuntimeError as wrapped:
            return wrapped


def http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://fixture.invalid/model")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("opaque HTTP failure", request=request, response=response)


@pytest.mark.parametrize("suppressed", [False, True], ids=["implicit", "suppressed-display"])
@pytest.mark.parametrize(("cause", "reason", "retry", "status", "evidence"), [
    (ModuleNotFoundError("opaque dependency failure", name="fixture_backend"),
     "runtime_dependency_missing", "after_configuration", "failed", {"module": "fixture_backend"}),
    (InputBudgetExceeded("opaque construction limit"),
     "input_budget_exceeded", "after_configuration", "needs_configuration", {}),
    (TimeoutError("opaque timeout"), "request_timeout", "higher_budget", "inconclusive", {}),
    (MemoryError("opaque allocation failure"), "resource_limit", "higher_budget", "unverified", {}),
    (PermissionError("opaque permission failure"), "access_denied", "after_configuration", "access_denied", {}),
    (http_error(401), "access_denied", "after_configuration", "access_denied", {"http_status": 401}),
    (http_error(403), "access_denied", "after_configuration", "access_denied", {"http_status": 403}),
    (http_error(429), "runtime_initialization_failed", "not_retryable", "failed", {}),
], ids=["dependency", "input-budget", "timeout", "memory", "permission", "http-401", "http-403", "http-429"])
def test_context_preserves_typed_reason_and_retry_advice(cause, reason, retry, status, evidence, suppressed):
    wrapped = wrapped_error(wrapped_error(cause, suppressed=suppressed))

    failure = failure_from_exception(wrapped, stage="load", device="cpu", runtime_profile="fixture-profile",
                                     evidence={"origin": "caller"})

    assert (failure.reason_code, failure.retryability, compatibility_status(failure)) == (reason, retry, status)
    assert (failure.stage, failure.device, failure.runtime_profile) == ("load", "cpu", "fixture-profile")
    assert failure.detail == "fixture SDK wrapper"
    assert failure.evidence == {"origin": "caller", **evidence}
    expected_type = type(cause).__name__ if reason != "runtime_initialization_failed" else "RuntimeError"
    assert failure.exception_type == expected_type


@pytest.mark.parametrize(("device", "profile"), [("unknown", ""), ("cpu", "source-profile")])
def test_structured_failure_keeps_its_source_fields_and_merges_observations(device, profile):
    original = Failure("preprocess", "input_budget_exceeded", "source diagnostic", device, profile,
                       "after_configuration", {"origin": "source", "retained": True}, "InputBudgetExceeded")
    wrapped = wrapped_error(RuntimeFailure(original))

    failure = failure_from_exception(wrapped, stage="processor", device="cuda", runtime_profile="caller-profile",
                                     evidence={"origin": "caller", "observed": True})

    assert (failure.stage, failure.reason_code, failure.detail) == (
        "preprocess", "input_budget_exceeded", "source diagnostic")
    assert failure.device == ("cuda" if device == "unknown" else device)
    assert failure.runtime_profile == (profile or "caller-profile")
    assert failure.evidence == {"origin": "caller", "retained": True, "observed": True}
    assert failure.exception_type == "InputBudgetExceeded"
    assert original.evidence == {"origin": "source", "retained": True}


@pytest.mark.parametrize("falsey", [False, True])
def test_explicit_cause_wins_over_an_incidental_context(falsey):
    from acprof.hub.huggingface.download import exception_chain

    class ExplicitTimeout(TimeoutError):
        def __bool__(self):
            return not falsey

    wrapped = wrapped_error(PermissionError("incidental context"))
    wrapped.__cause__ = ExplicitTimeout("explicit timeout")

    failure = failure_from_exception(wrapped, stage="predict")

    assert failure.reason_code == "request_timeout"
    assert failure.exception_type == "ExplicitTimeout"
    assert list(exception_chain(wrapped)) == [wrapped, wrapped.__cause__]


@pytest.mark.parametrize(("stage", "reason"), [("load", "runtime_initialization_failed"),
                                             ("predict", "inference_failed")])
def test_cyclic_unknown_context_does_not_parse_message_text(stage, reason):
    failure = RuntimeError("PermissionError timeout missing dependency")
    context = ValueError("unknown context")
    failure.__context__ = context
    context.__context__ = failure

    classified = failure_from_exception(failure, stage=stage)

    assert classified.reason_code == reason
    assert classified.retryability == "not_retryable"


@pytest.mark.parametrize(("cause", "reason"), [
    (ModuleNotFoundError("opaque dependency failure", name="fixture_processor"), "runtime_dependency_missing"),
    (TimeoutError("opaque timeout"), "request_timeout"),
    (PermissionError("opaque permission failure"), "access_denied"),
])
def test_processor_loader_reports_original_failure_instead_of_incompatibility(cause, reason):
    wrapped = wrapped_error(cause)
    loader = Mock(side_effect=wrapped)

    with pytest.raises(RuntimeFailure) as caught:
        load_processor(loader, "fixture-model", local_files_only=True)

    assert caught.value.failure.reason_code == reason
    assert caught.value.failure.stage == "processor"
    assert caught.value.__cause__ is wrapped
    loader.assert_called_once_with("fixture-model", local_files_only=True)


@pytest.mark.parametrize(("cause", "stage", "reason", "retry"), [
    (ModuleNotFoundError("opaque dependency failure", name="fixture_runtime"),
     "load", "runtime_dependency_missing", "after_configuration"),
    (InputBudgetExceeded("opaque construction limit"),
     "preprocess", "input_budget_exceeded", "after_configuration"),
    (TimeoutError("opaque timeout"), "completion", "request_timeout", "higher_budget"),
])
def test_validation_cli_emits_typed_failure_for_implicit_context(tmp_path, monkeypatch, capsys,
                                                               cause, stage, reason, retry):
    payload = tmp_path / "input.json"
    payload.write_text('{"text":"fixture"}')
    wrapped = wrapped_error(cause)

    def validate(_payload, *, stages):
        if stage != "load":
            stages.append({"stage": "load", "status": "verified"})
        stages.append({"stage": stage, "status": "error"})
        raise wrapped

    monkeypatch.setattr(sys, "argv", ["runtime_validate", str(payload)])
    monkeypatch.setattr(runtime_validate, "validate", validate)
    monkeypatch.setenv("USE_GPU", "0")
    monkeypatch.setenv("ACPROF_RUNTIME_PROFILE", "fixture-profile")
    assert runtime_validate.main() == 1

    output = capsys.readouterr().out.split(runtime_validate.RESULT_PREFIX, 1)[1]
    record = json.loads(output)
    assert (record["status"], record["failed_stage"]) == ("error", stage)
    failure = record["failure"]
    assert (failure["reason_code"], failure["retryability"], failure["exception_type"]) == (
        reason, retry, type(cause).__name__)
    assert (failure["device"], failure["runtime_profile"]) == ("cpu", "fixture-profile")
    if stage == "completion":
        assert failure["evidence"]["timeout_scope"] == "completion_wait"
        assert failure["evidence"]["model_loaded"] is True
    assert list(tmp_path.glob("*.csv")) == []
