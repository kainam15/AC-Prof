"""Stable entry point for capability-routed local Transformers Pipeline loading."""
from __future__ import annotations

import json
from pathlib import Path

from acprof.model_resolution import transformers_capabilities
from acprof.model_spec import custom_code_files


def load_local_pipeline_class(model_source: str, pipeline_name: str) -> type:
    """Load declared local code offline, using only reviewed runtime capabilities."""
    root = Path(model_source).absolute()
    if not root.is_dir():
        raise ValueError("custom pipeline requires a baked local model snapshot")
    config = json.loads((root / "config.json").read_text())
    entry = config["custom_pipelines"][pipeline_name]
    module_file, = custom_code_files({"custom_pipelines": {pipeline_name: entry}})
    class_name = entry["impl"].rsplit(".", 1)[1]

    import transformers

    capabilities = transformers_capabilities(transformers.__version__)
    if all(capabilities.values()):
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        return get_class_from_dynamic_module(f"{module_file[:-3]}.{class_name}", str(root), local_files_only=True)

    from acprof.container.compat.transformers_dynamic import load_pipeline_class_compat

    return load_pipeline_class_compat(root, module_file, class_name)
