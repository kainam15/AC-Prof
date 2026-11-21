"""Single-writer append and module replacement transactions.

Opening/recovering a writer may read previous results. append_window never reads,
serializes or hashes earlier CSV rows. Call it only after all collectors stopped.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import os
from collections import ChainMap
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from uuid import uuid4

from acprof.artifacts import atomic_write_json, read_json_object, sync_directory
from acprof.result_store.protocol import (
    csv_bytes,
    publication_lock,
    reference,
    save_commit,
    validate_pending_transaction,
    write_bytes,
)
from acprof.result_store.recovery import clear_pending, projected_rows, recover_projection
from acprof.result_store.sources import EXACT_KEY_ENCODING, apply_source, prepare_window
from acprof.result_store.types import ResultReference, ResultSource
from acprof.results.csv import (
    ResultValidationError,
    measurement_key,
    require_error_diagnostic,
    require_matching_run_id,
    run_id_column,
)
from acprof.results.layout import ArtifactLayout
from acprof.system.cleanup import cleanup_on_exit


class ResultWriter:
    def __init__(self, root, *, expected_run_id: str | None = None):
        self.layout = ArtifactLayout.discover(root)
        self.expected_run_id = expected_run_id
        self.failed = False
        self._resources: ExitStack | None = None

    def __enter__(self):
        if self._resources is not None:
            raise ResultValidationError("result writer is already active")
        with ExitStack() as resources:
            lease = self.layout.result_store_path("writer.lock").open("a+b")
            resources.enter_context(cleanup_on_exit(
                lease.close, description=f"result writer lease {self.layout.root}"))
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with publication_lock(self.layout, exclusive=True):
                self.manifest, self.rows, self.windows, self.hashes = recover_projection(
                    self.layout, expected_run_id=self.expected_run_id)
            self.failed = False
            self._resources = resources.pop_all()
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        resources, self._resources = self._resources, None
        if resources is not None:
            return resources.__exit__(exc_type, exc_value, traceback)
        return False

    def _ready(self):
        if self.failed or self._resources is None:
            raise ResultValidationError("result writer requires recovery before another window")

    def verified_receipts(self, layout: ArtifactLayout, manifest: dict) -> Mapping[tuple[str, ...], str]:
        """Receipt digests fully validated while owning this writer's current directory and HEAD."""
        self._ready()
        if layout != self.layout or manifest is not self.manifest:
            raise ResultValidationError("result source verification requires the active writer's current manifest")
        return self.windows

    @contextmanager
    def _publication(self):
        """Require recovery until durable publication and bookkeeping both finish."""
        self.failed = True
        yield
        self.failed = False

    def append_window(self, path):
        self._ready()
        source: ResultSource = {"kind": "window", "reference": reference(self.layout, path)}
        # Validate this source once, without copying or reserializing old rows.
        window = prepare_window(self.layout, self.manifest, source, self.rows, self.windows)
        if window is None:
            return False
        key, normalized, identity, receipt_digest = window
        changes, counts = {}, {}
        for layer, info in self.manifest["layers"].items():
            selected = projected_rows(self.manifest, layer, [normalized])
            if selected:
                changes[layer] = csv_bytes(info["fields"], selected)
                counts[layer] = info["row_count"] + 1
        with self._publication():
            self._commit(changes, counts, [source], replace=False,
                         row_count=len(self.rows) + 1, state="collecting")
            self.rows[key] = normalized
            self.windows.update({identity: receipt_digest})
        return True

    def update_rows(self, rows, *, reason: str, evidence=(), state="processing", force_commit=False):
        """Persist only changed fields; rewrite only modules affected by the patch."""
        self._ready()
        inserts, changes = [], []
        seen = set()
        fields = self.manifest["wide_fields"]
        allowed_fields = set(fields)
        identity_column = run_id_column(fields) or "run_id"
        source_path = str(self.layout.root)
        for incoming in rows:
            if not allowed_fields.issuperset(incoming):
                raise ResultValidationError("unknown derivation fields")
            require_matching_run_id(self.manifest["run_id"], incoming.get(identity_column), source=source_path)
            key = measurement_key(incoming)
            if key in seen:
                raise ResultValidationError("duplicate measurement in derivation")
            seen.add(key)
            existing = self.rows.get(key)
            require_error_diagnostic(incoming if existing is None else ChainMap(incoming, existing), source=source_path)
            if existing is None:
                inserts.append({name: str(incoming.get(name, "nan")) for name in fields})
            else:
                delta = {name: str(value) for name, value in incoming.items()
                         if str(value) != existing.get(name, "nan")}
                if delta:
                    changes.append({"key": list(key), "values": delta})
        if not inserts and not changes and not force_commit:
            if self.manifest["state"] != state:
                with self._publication():
                    self._commit({}, {}, [], replace=False, row_count=len(self.rows), state=state)
            return False
        update = {"schema_version": 1, "run_id": self.manifest["run_id"], "reason": reason,
                  "measurement_key_encoding": EXACT_KEY_ENCODING,
                  "evidence": list(evidence), "insert": inserts, "changes": changes}
        path = self.layout.result_update_path(uuid4().hex)
        atomic_write_json(path, update)
        source: ResultSource = {"kind": "derivation", "reference": reference(self.layout, path)}
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
        with self._publication():
            self._commit(payloads, counts, [source], replace=True, row_count=len(new_rows), state=state)
            self.rows = new_rows
        return True

    def _append_bytes(self, layer, raw):
        path = self.layout.layer_csv(layer)
        stream = path.open("ab")
        with cleanup_on_exit(stream.close, description=f"result CSV append stream {path}"):
            if stream.tell() != self.manifest["layers"][layer]["committed_bytes"]:
                raise ResultValidationError(f"unexpected {layer} tail; recovery required")
            if stream.write(raw) != len(raw):
                raise OSError("short result CSV write")
            stream.flush()
            os.fsync(stream.fileno())

    def freeze(self):
        """Create an immutable snapshot once at completion or explicit freeze."""
        self._ready()
        from acprof.results.layers import prepare_result_snapshot
        if not self.rows:
            raise ResultValidationError("cannot freeze an experiment without measurements")
        snapshot = prepare_result_snapshot(self.manifest["wide_fields"], list(self.rows.values()), self.layout.root)
        path = self.layout.result_store_path(f"snapshots/{snapshot['generation']}.json")
        if path.exists():
            if read_json_object(path) != snapshot:
                raise ResultValidationError("immutable snapshot manifest changed")
            sync_directory(path.parent)
        else:
            atomic_write_json(path, snapshot)
        with self._publication():
            self._commit({}, {}, [], replace=False, row_count=len(self.rows), state="complete",
                         snapshot=reference(self.layout, path))
        return snapshot

    def _commit(self, payloads: dict[str, bytes], counts: dict[str, int], sources: list[ResultSource], *,
                replace: bool, row_count: int, state: str, snapshot: ResultReference | None = None) -> None:
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
            for layer, raw in payloads.items():
                info = after["layers"][layer]
                checksum = hashlib.sha256() if replace else self.hashes[layer].copy()
                checksum.update(raw)
                new_hashes[layer] = checksum
                info.update(sha256=checksum.hexdigest(), row_count=counts[layer],
                            committed_bytes=len(raw) + (0 if replace else info["committed_bytes"]),
                            revision=info["revision"] + int(replace))
            pending = {"before": self.manifest, "after": after, "replacements": replacements}
            if not validate_pending_transaction(pending, self.layout, self.manifest):
                raise ResultValidationError("invalid pending result transaction")
            atomic_write_json(self.layout.result_store_path("pending.json"), pending)
            for layer, raw in payloads.items():
                if replace:
                    write_bytes(self.layout.layer_csv(layer), raw)
                else:
                    self._append_bytes(layer, raw)
            manifest = save_commit(self.layout, pending["before"], pending["after"], sources, transaction=transaction)
            atomic_write_json(self.layout.result_csv, manifest)
            clear_pending(self.layout)
            self.manifest, self.hashes = manifest, new_hashes
