"""Posthoc publication against an incremental store, preserving frozen snapshots."""
from __future__ import annotations

import json
from uuid import uuid4

from acprof.artifacts import atomic_write_json, sync_directory
from acprof.host.collection_history import normalize_collection_history
from acprof.result_store.metadata import prepare_metadata_recovery, write_metadata
from acprof.result_store.protocol import reference
from acprof.result_store.writer import ResultWriter
from acprof.results.layout import ArtifactLayout


def commit_incremental(context, *, rows, static_meta, collection_history):
    layout = ArtifactLayout.discover(context.result_dir)
    normalize_collection_history(collection_history)
    pending = layout.result_store_path("metadata_pending.json")
    try:
        with ResultWriter(layout.root) as writer:
            complete = writer.manifest["state"] == "complete"
            paths = (context.static_meta_path, context.collection_history_path)
            before = {path.relative_to(layout.root).as_posix(): path.read_bytes().decode("utf-8") if path.exists() else None for path in paths}
            after = {path.relative_to(layout.root).as_posix(): json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
                     for path, payload in zip(paths, (static_meta, collection_history), strict=True)}
            intent = {"run_id": writer.manifest["run_id"], "commit": writer.manifest["commit"],
                      "before": before, "after": after}
            plan = prepare_metadata_recovery(layout, intent, run_id=writer.manifest["run_id"])
            provenance = layout.result_update_path(uuid4().hex)
            atomic_write_json(provenance, {"schema_version": 1, "origin": "posthoc",
                                           "static_meta": static_meta, "collection_history": collection_history})
            atomic_write_json(pending, intent)
            write_metadata(plan.after)
            writer.update_rows(rows, reason="posthoc profiler derivation",
                               evidence=[reference(layout, provenance)], force_commit=True)
            if complete:
                writer.freeze()
            pending.unlink()
            sync_directory(pending.parent)
    except BaseException as error:
        # HEAD replacement may have succeeded even if its final fsync failed.
        try:
            with ResultWriter(layout.root):
                pass
        except BaseException as recovery_error:
            error.add_note(f"Posthoc recovery incomplete; retained journal: {recovery_error}")
        raise
