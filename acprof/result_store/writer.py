"""Single-writer append and module replacement transactions.

Opening/recovering a writer may read previous results. append_window never reads,
serializes or hashes earlier CSV rows. Call it only after all collectors stopped.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import os
from uuid import uuid4

from acprof.artifact_layout import ArtifactLayout
from acprof.artifacts import atomic_write_json, read_json_object
from acprof.result_csv import ResultValidationError, measurement_key
from acprof.result_store.protocol import (
    csv_bytes,
    publication_lock,
    reference,
    save_commit,
    write_bytes,
)
from acprof.result_store.recovery import clear_pending, projected_rows, recover_projection
from acprof.result_store.sources import apply_source


class ResultWriter:
    def __init__(self, root):
        self.layout = ArtifactLayout.discover(root)
        self.failed = False
        self.lease = None

    def __enter__(self):
        self.lease = self.layout.result_store_path("writer.lock").open("a+b")
        try:
            fcntl.flock(self.lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with publication_lock(self.layout, exclusive=True):
                self.manifest, self.rows, self.windows, payloads = recover_projection(self.layout)
            self.hashes = {layer: hashlib.sha256(raw) for layer, raw in payloads.items()}
            return self
        except BaseException:
            self.lease.close()
            self.lease = None
            raise

    def __exit__(self, *_):
        if self.lease is not None:
            self.lease.close()
            self.lease = None

    def _ready(self):
        if self.failed or self.lease is None:
            raise ResultValidationError("result writer requires recovery before another window")

    def append_window(self, path):
        self._ready()
        source: dict = {"kind": "window", "reference": reference(self.layout, path)}
        # Validate this source once, without copying or reserializing old rows.
        new_rows: dict[tuple[str, ...], dict[str, str]] = {}
        new_windows: dict[tuple[str, ...], str] = {}
        apply_source(self.layout, self.manifest, source, new_rows, new_windows)
        identity, receipt_digest = next(iter(new_windows.items()))
        if identity in self.windows:
            if self.windows[identity] != receipt_digest:
                raise ResultValidationError("conflicting committed window identity")
            return False
        key, normalized = next(iter(new_rows.items()))
        if key in self.rows:
            raise ResultValidationError("duplicate planned measurement across attempts")
        changes, counts = {}, {}
        for layer, info in self.manifest["layers"].items():
            selected = projected_rows(self.manifest, layer, [normalized])
            if selected:
                changes[layer] = csv_bytes(info["fields"], selected)
                counts[layer] = info["row_count"] + 1
        self._commit(changes, counts, [source], replace=False,
                     row_count=len(self.rows) + 1, state="collecting")
        self.rows[key] = normalized
        self.windows.update(new_windows)
        return True

    def update_rows(self, rows, *, reason: str, evidence=(), state="processing", force_commit=False):
        """Persist only changed fields; rewrite only modules affected by the patch."""
        self._ready()
        inserts, changes = [], []
        seen = set()
        for incoming in rows:
            if set(incoming) - set(self.manifest["wide_fields"]):
                raise ResultValidationError("unknown derivation fields")
            key = measurement_key(incoming)
            if key in seen:
                raise ResultValidationError("duplicate measurement in derivation")
            seen.add(key)
            if key not in self.rows:
                inserts.append({name: str(incoming.get(name, "nan")) for name in self.manifest["wide_fields"]})
            else:
                delta = {name: str(value) for name, value in incoming.items()
                         if name in self.manifest["wide_fields"] and str(value) != self.rows[key].get(name, "nan")}
                if delta:
                    changes.append({"key": list(key), "values": delta})
        if not inserts and not changes and not force_commit:
            if self.manifest["state"] != state:
                self._commit({}, {}, [], replace=False, row_count=len(self.rows), state=state)
            return False
        update = {"schema_version": 1, "run_id": self.manifest["run_id"], "reason": reason,
                  "evidence": list(evidence), "insert": inserts, "changes": changes}
        path = self.layout.result_update_path(uuid4().hex)
        atomic_write_json(path, update)
        source = {"kind": "derivation", "reference": reference(self.layout, path)}
        new_rows = dict(self.rows)
        apply_source(self.layout, self.manifest, source, new_rows, dict(self.windows))
        payloads, counts = {}, {}
        for layer, info in self.manifest["layers"].items():
            affected = bool(inserts) or any(set(change["values"]) & set(info["fields"]) for change in changes)
            if not affected:
                continue
            selected = projected_rows(self.manifest, layer, list(new_rows.values()))
            raw = csv_bytes(info["fields"], selected, header=True)
            if hashlib.sha256(raw).hexdigest() != info["sha256"]:
                payloads[layer], counts[layer] = raw, len(selected)
        self._commit(payloads, counts, [source], replace=True, row_count=len(new_rows), state=state)
        self.rows = new_rows
        return True

    def _append_bytes(self, layer, raw):
        path = self.layout.layer_csv(layer)
        with path.open("ab") as stream:
            if stream.tell() != self.manifest["layers"][layer]["committed_bytes"]:
                raise ResultValidationError(f"unexpected {layer} tail; recovery required")
            if stream.write(raw) != len(raw):
                raise OSError("short result CSV write")
            stream.flush()
            os.fsync(stream.fileno())

    def freeze(self):
        """Create an immutable snapshot once at completion or explicit freeze."""
        self._ready()
        from acprof.result_layers import prepare_result_snapshot
        if not self.rows:
            raise ResultValidationError("cannot freeze an experiment without measurements")
        snapshot = prepare_result_snapshot(self.manifest["wide_fields"], list(self.rows.values()), self.layout.root)
        path = self.layout.result_store_path(f"snapshots/{snapshot['generation']}.json")
        if path.exists() and read_json_object(path) != snapshot:
            raise ResultValidationError("immutable snapshot manifest changed")
        if not path.exists():
            atomic_write_json(path, snapshot)
        self._commit({}, {}, [], replace=False, row_count=len(self.rows), state="complete",
                     snapshot=reference(self.layout, path))
        return snapshot

    def _commit(self, payloads, counts, sources, *, replace, row_count, state, snapshot=None):
        after = copy.deepcopy({k: v for k, v in self.manifest.items() if k != "commit"})
        after.update(sequence=after["sequence"] + 1, state=state, row_count=row_count)
        if snapshot is not None:
            after["snapshot"] = snapshot
        elif sources:
            after.pop("snapshot", None)
        new_hashes = dict(self.hashes)
        transaction = uuid4().hex
        replacements = list(payloads) if replace else []
        with publication_lock(self.layout, exclusive=True):
            if read_json_object(self.layout.result_csv) != self.manifest:
                raise ResultValidationError("result commit changed under its single writer")
            try:
                for layer, raw in payloads.items():
                    info = after["layers"][layer]
                    checksum = hashlib.sha256() if replace else self.hashes[layer].copy()
                    checksum.update(raw)
                    new_hashes[layer] = checksum
                    info.update(sha256=checksum.hexdigest(), row_count=counts[layer],
                                committed_bytes=len(raw) + (0 if replace else info["committed_bytes"]),
                                revision=info["revision"] + int(replace))
                atomic_write_json(self.layout.result_store_path("pending.json"), {
                    "before": self.manifest, "after": after, "replacements": replacements,
                })
                for layer, raw in payloads.items():
                    if replace:
                        write_bytes(self.layout.layer_csv(layer), raw)
                    else:
                        self._append_bytes(layer, raw)
                manifest = save_commit(self.layout, self.manifest, after, sources, transaction=transaction)
                atomic_write_json(self.layout.result_csv, manifest)
                clear_pending(self.layout)
                self.manifest, self.hashes = manifest, new_hashes
            except BaseException:
                self.failed = True
                raise
