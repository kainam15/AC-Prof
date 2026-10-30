"""Recover metadata accompanying a posthoc commit after process death.

This coordinates recovery, not whole-directory reader transactions.
"""
from __future__ import annotations

from acprof.artifacts import read_json_object, sync_directory
from acprof.result_store.protocol import write_bytes
from acprof.results.csv import ResultValidationError


def recover_metadata(layout, manifest):
    pending = layout.result_store_path("metadata_pending.json")
    if not pending.exists():
        return
    intent = read_json_object(pending, bounded=False)
    if intent.get("run_id") != manifest["run_id"]:
        raise ResultValidationError("metadata recovery belongs to another run")
    selected = intent["before"] if manifest["commit"] == intent["commit"] else intent["after"]
    allowed = {layout.path(name).relative_to(layout.root).as_posix()
               for name in ("static_meta.json", "collection_history.json")}
    if set(selected) != allowed:
        raise ResultValidationError("invalid posthoc metadata recovery paths")
    for name, content in selected.items():
        path = layout.contained(name)
        if content is None:
            path.unlink(missing_ok=True)
            sync_directory(path.parent)
        elif isinstance(content, str):
            write_bytes(path, content.encode("utf-8"))
        else:
            raise ResultValidationError("invalid metadata recovery content")
    pending.unlink()
    sync_directory(pending.parent)
