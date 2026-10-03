"""Import/signature inspection, executed offline without loading a model."""
from __future__ import annotations

import inspect
import json
import os
import traceback
from pathlib import Path

RESULT_PREFIX = "ACPROF_INTERFACE_VALIDATION="


def validate_interface(payload: dict) -> dict:
    from acprof.container.handlers import HandlerRegistry, resolve_model_source
    from acprof.container.local_pipeline import load_local_pipeline_class
    from acprof.model_source_analysis import SOURCE_METADATA_FILES
    from acprof.model_spec import load_model_spec, pipeline_task

    source = resolve_model_source(os.environ["MODEL_ID"])
    task = os.environ["TASK_TYPE"]
    config = json.loads((Path(source) / "config.json").read_text())
    auto_maps = [json.loads(path.read_text()).get("auto_map", {}) for name in SOURCE_METADATA_FILES
                if (path := Path(source) / name).is_file()]
    checks = []
    if config.get("custom_pipelines") or any(auto_maps):
        from acprof.container.load_policy import registered_policy
        from acprof.failures import Failure, RuntimeFailure
        profile, _, trust = registered_policy(task, os.environ["RUNTIME_BACKEND"], "cpu")
        if not trust:
            raise RuntimeFailure(Failure("import", "remote_code_disallowed", "Interface inspection requires a registered remote-code policy",
                                         "cpu", profile.profile_id))
    if config.get("custom_pipelines"):
        spec = load_model_spec(source, task, expected_format="transformers-pipeline")
        pipeline_class = load_local_pipeline_class(source, pipeline_task(source, task))
        params = {"max_new_tokens": 1, "do_sample": False, **payload.get("params", {})}
        kwargs = {key: params[value[1:]] if isinstance(value, str) and value.startswith("$") else value
                  for key, value in spec.get("multimodal", {}).get("forward_kwargs", {}).items()}
        checks.extend((getattr(pipeline_class, method), arguments, keywords)
                      for method, arguments, keywords in (("preprocess", (None, {}), {}),
                          ("_forward", (None, {}), kwargs), ("postprocess", (None, {}), {})))
    else:
        handler = HandlerRegistry.get(os.environ["TASK_FAMILY"], os.environ["RUNTIME_BACKEND"])
        checks.extend((getattr(handler, method), arguments, {}) for method, arguments in (
            ("load", (source, task, os.environ["RUNTIME_BACKEND"], "cpu", os.environ["MODEL_REVISION"])),
            ("preprocess", ({}, {})), ("predict", ({}, {})), ("postprocess", ({}, {})),
            ("validate_output", ({}, {}, {}, {}, {})),
        ))
        if any(auto_maps):
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            for entry in [entry for mapping in auto_maps for entry in mapping.values()]:
                for reference in entry if isinstance(entry, (list, tuple)) else [entry]:
                    if reference:
                        # The source graph rejects cross-repository references.
                        cls = get_class_from_dynamic_module(reference, source, local_files_only=True)
                        inspect.signature(cls)
    signatures = {}
    for method, arguments, keywords in checks:
        signature = inspect.signature(method)
        signature.bind(*arguments, **keywords)
        signatures[method.__name__] = str(signature)
    return {"status": "ok", "signatures": signatures,
            "stages": [{"stage": stage, "status": "verified"} for stage in ("import", "signature")],
            "inference": "not_run", "preprocess": "not_run", "output_validation": "not_run"}


def main() -> int:
    try:
        result = validate_interface({})
    except Exception as exc:
        traceback.print_exc()
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
