"""Import/signature probe for declared Pipelines, executed only in a container."""
from __future__ import annotations

import inspect
import os


def validate_basic(payload: dict) -> dict:
    from acprof.container.dynamic_modules import load_local_pipeline_class
    from acprof.container.handlers import resolve_model_source
    from acprof.model_spec import load_model_spec, pipeline_task

    source = resolve_model_source(os.environ["MODEL_ID"])
    task = os.environ["TASK_TYPE"]
    spec = load_model_spec(source, task, expected_format="transformers-pipeline")
    if "multimodal" not in spec:
        raise ValueError("basic contract probe requires a declared multimodal Pipeline")
    selected = pipeline_task(source, task)
    pipeline_class = load_local_pipeline_class(source, selected)
    params = {"max_new_tokens": 1, "do_sample": False, **payload.get("params", {})}
    kwargs = {key: params[value[1:]] if isinstance(value, str) and value.startswith("$") else value
              for key, value in spec["multimodal"].get("forward_kwargs", {
                  "max_new_tokens": "$max_new_tokens", "do_sample": "$do_sample"}).items()}
    signatures = {}
    for method, arguments, keywords in (("preprocess", (None, {}), {}),
                                       ("_forward", (None, {}), kwargs),
                                       ("postprocess", (None, {}), {})):
        signature = inspect.signature(getattr(pipeline_class, method))
        signature.bind(*arguments, **keywords)
        signatures[method] = str(signature)
    return {"status": "ok", "mode": "basic", "device": "cpu", "signatures": signatures,
            "stages": [{"stage": stage, "status": "verified"} for stage in ("import", "signature")],
            "inference": "not_run", "preprocess": "not_run", "output_validation": "not_run"}
