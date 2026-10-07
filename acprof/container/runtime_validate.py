"""独立容器中的完整接口验证，不产生性能测量行。"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any

from acprof.artifacts import loads_finite_json

RESULT_PREFIX = "ACPROF_RUNTIME_VALIDATION="
STAGE_PREFIX = "ACPROF_RUNTIME_STAGE="


def validate(payload: dict, *, stages: list[dict] | None = None) -> dict:
    from acprof.container.execution import complete_prediction, configured_execution
    from acprof.container.handlers import HandlerRegistry, load_handler, resolve_model_source
    from acprof.extensions import get_extension
    from acprof.runtime_settings import runtime_threads

    stages = [] if stages is None else stages

    def check(stage, operation):
        print(STAGE_PREFIX + json.dumps({"stage": stage, "status": "running"}), flush=True)
        try:
            result = operation()
        except Exception as exc:
            stages.append({"stage": stage, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            # Preserve typed errors, especially TimeoutError used by callers.
            raise
        stages.append({"stage": stage, "status": "verified"})
        print(STAGE_PREFIX + json.dumps(stages[-1]), flush=True)
        return result

    use_gpu = os.getenv("USE_GPU", "0") == "1"
    execution, device = check("execution", lambda: configured_execution(
        os.environ["TASK_FAMILY"], os.environ["RUNTIME_BACKEND"], use_gpu=use_gpu,
        threads=runtime_threads(default=1),
        adapter=os.getenv("ACPROF_MODEL_ADAPTER", "family-default"),
    ))

    def load():
        handler = HandlerRegistry.get(os.environ["TASK_FAMILY"], os.environ["RUNTIME_BACKEND"])
        context = load_handler(handler,
            resolve_model_source(os.environ["MODEL_ID"]), os.environ["TASK_TYPE"],
            os.environ["RUNTIME_BACKEND"], device, os.environ["MODEL_REVISION"],
        )
        context["_validation_entrypoint"] = get_extension(
            os.environ["TASK_FAMILY"], os.environ["RUNTIME_BACKEND"],
            adapter=os.getenv("ACPROF_MODEL_ADAPTER", "family-default"),
            task=os.environ["TASK_TYPE"],
        ).validation_entrypoint
        return handler, context

    handler, context = check("load", load)
    processed = check("preprocess", lambda: handler.preprocess(context, payload))
    with execution.inference_context():
        output = check("predict", lambda: handler.predict(context, processed))
        output = check("completion", lambda: complete_prediction(execution, context, output))
    response = check("postprocess", lambda: handler.postprocess(context, output))

    def validate_output():
        validation = handler.validate_output(context, payload, processed, output, response)
        for layer in ('protocol', 'task'):
            if not isinstance(validation, dict) or not isinstance(validation.get(layer), dict) or validation[layer].get('status') != 'verified':
                raise ValueError(f'{layer} validation must be verified before profiling: {validation!r}')
        return validation

    validation = check("validate_output", validate_output)
    runtime_metadata = check("metadata", execution.metadata)
    from acprof.container.load_policy import actual_dtype
    return {
        "status": "ok", "mode": "full", "device": device, "stages": stages,
        "dtype": actual_dtype(context),
        "quality_checks": context.get("quality_checks", []),
        "attention_implementation": context.get("attention_implementation", "model_default"),
        **runtime_metadata, "validation": validation,
        "runtime_parameters": context.get("runtime_parameters", runtime_metadata.get("runtime_parameters", {})),
        "artifact": context.get("artifact", {}),
        "model_spec": context.get("model_spec", context.get("manifest", {})),
        "workload_contract": validation["workload_contract"],
        "adapter": type(handler).__name__, "response": response,
        "effective_input_scale": processed.get("_effective_input_scale") if isinstance(processed, dict) else None,
    }


def main() -> int:
    stages = []
    invalid_input = False
    try:
        with open(sys.argv[1], encoding="utf-8") as stream:
            content = stream.read()
        try:
            payload = loads_finite_json(content)
            if not isinstance(payload, dict):
                raise ValueError("runtime validation input must be a JSON object")
        except (TypeError, ValueError, RecursionError):
            invalid_input = True
            raise
        result = validate(payload, stages=stages)
    except Exception as exc:
        traceback.print_exc()
        failed_stage = (stages[-1]["stage"] if stages and stages[-1]["status"] == "error"
                        else "input_or_execution_context")
        result: dict[str, Any] = {
            "status": "error", "error": f"{type(exc).__name__}: {exc}",
            "failed_stage": failed_stage, "stages": stages,
        }
        from acprof.failures import Failure, failure_from_exception
        if invalid_input:
            failure = Failure(
                failed_stage, "recorded_evidence_invalid", result["error"],
                device="cuda" if os.getenv("USE_GPU") == "1" else "cpu",
                runtime_profile=os.getenv("ACPROF_RUNTIME_PROFILE", ""),
                exception_type=type(exc).__name__,
            )
        else:
            failure = failure_from_exception(exc, stage=failed_stage,
                device="cuda" if os.getenv("USE_GPU") == "1" else "cpu",
                runtime_profile=os.getenv("ACPROF_RUNTIME_PROFILE", ""))
        if failure.reason_code == "request_timeout":
            from dataclasses import replace

            from acprof.runtime_settings import request_timeout_s
            completion = failed_stage == "completion"
            failure = replace(failure, evidence={**failure.evidence,
                "timeout_seconds": request_timeout_s() if completion else None,
                "timeout_scope": "completion_wait" if completion else "unknown",
                "request_phase": failed_stage,
                "model_loaded": True if any(s["stage"] == "load" and s["status"] == "verified" for s in stages) else None})
        result["failure"] = failure.to_dict()
    try:
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        traceback.print_exc()
        from acprof.failures import Failure
        failure = Failure(
            "result_serialization", "recorded_evidence_invalid", f"{type(exc).__name__}: {exc}",
            device="cuda" if os.getenv("USE_GPU") == "1" else "cpu",
            runtime_profile=os.getenv("ACPROF_RUNTIME_PROFILE", ""),
            exception_type=type(exc).__name__,
        )
        result = {
            "status": "error", "error": failure.detail, "failed_stage": "result_serialization",
            "stages": stages, "failure": failure.to_dict(),
        }
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
    print(RESULT_PREFIX + encoded, flush=True)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
