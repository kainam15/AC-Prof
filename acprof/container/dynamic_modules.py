"""Prepare local Pipeline dependencies before Transformers imports custom code.

Transformers 4.57.6 copies only direct relative imports for local snapshots.
Reuse its resolver and cache layout while preparing the recursive closure, as
newer upstream loaders do (Transformers dynamic_module_utils, Apache-2.0).
"""
from __future__ import annotations

import filecmp
import importlib
import json
from pathlib import Path
import shutil

from acprof.model_spec import custom_code_files


def load_local_pipeline_class(model_source: str, pipeline_name: str) -> type:
    """Import a declared local Pipeline without loading weights or changing its snapshot."""
    root = Path(model_source).absolute()
    if not root.is_dir():
        raise ValueError("custom pipeline requires a baked local model snapshot")
    config = json.loads((root / "config.json").read_text())
    entry = config["custom_pipelines"][pipeline_name]
    module_file, = custom_code_files({"custom_pipelines": {pipeline_name: entry}})
    class_name = entry["impl"].rsplit(".", 1)[1]

    from transformers.dynamic_module_utils import (
        HF_MODULES_CACHE, get_cached_module_file, get_class_in_module, get_relative_import_files,
    )

    # Keep snapshot filenames: resolving Hub symlinks would lose relative names
    # in the content-addressed blob directory. Discover missing files before import.
    dependencies = sorted({Path(name) for name in get_relative_import_files(root / module_file)})
    relative_files = [path.relative_to(root) for path in dependencies]
    if any(".." in path.parts for path in relative_files):
        raise ValueError("custom pipeline relative imports must stay within the model snapshot")
    cached_module = get_cached_module_file(str(root), module_file, local_files_only=True)
    cached_root = (Path(HF_MODULES_CACHE) / cached_module).parents[len(Path(module_file).parts) - 1]
    for source, relative in zip(dependencies, relative_files):
        target = cached_root / relative
        if not target.is_file() or not filecmp.cmp(source, target, shallow=False):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    importlib.invalidate_caches()
    return get_class_in_module(class_name, cached_module)
