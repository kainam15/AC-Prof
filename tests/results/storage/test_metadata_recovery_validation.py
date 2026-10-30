"""Reject corrupt recovery authority before changing either metadata document."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from acprof.host.posthoc.incremental import commit_incremental
from acprof.result_store.writer import ResultWriter
from acprof.results.csv import ResultValidationError
from tests.results.storage.test_incremental_results import initialized


def _pending(root, *, committed=False):
    layout = initialized(root)
    original = json.loads(layout.result_csv.read_bytes())
    if committed:
        with ResultWriter(root) as writer:
            writer.update_rows([], reason="metadata recovery fixture", force_commit=True)
    names = [layout.path(name).relative_to(root).as_posix()
             for name in ("static_meta.json", "collection_history.json")]
    intent = {"run_id": original["run_id"], "commit": original["commit"],
              "before": {names[0]: '\ufeff{"old":"原始元数据"}\r\n', names[1]: None},
              "after": {name: '{"new":true}\n' for name in names}}
    for name in names:
        layout.contained(name).write_bytes(b'{"live":"unchanged"}\n')
    return layout, intent, names


def _assert_rejected(layout, intent):
    pending = layout.result_store_path("metadata_pending.json")
    pending.write_text(json.dumps(intent), encoding="utf-8")
    paths = [layout.path(name) for name in ("static_meta.json", "collection_history.json")]
    paths += [layout.result_csv, pending]
    before = {path: path.read_bytes() for path in paths}
    for _attempt in range(2):
        with pytest.raises(ResultValidationError, match="metadata"):
            with ResultWriter(layout.root):
                pytest.fail("invalid recovery journal was accepted")
        assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("phase", ["before", "after"])
@pytest.mark.parametrize("bad_content", [42, "{", "[]", "null", '{"value":NaN}', '{"value":1e999}', "\ud800"])
def test_invalid_later_content_cannot_partially_restore_metadata(tmp_path, committed, phase, bad_content):
    layout, intent, names = _pending(tmp_path, committed=committed)
    intent[phase][names[1]] = bad_content
    _assert_rejected(layout, intent)


@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("damage", ["missing-phase", "list-phase", "missing-path", "extra-path", "after-null", "wrong-run"])
def test_invalid_metadata_recovery_envelope_is_retained(tmp_path, committed, damage):
    layout, intent, names = _pending(tmp_path, committed=committed)
    if damage == "missing-phase":
        del intent["before"]
    elif damage == "list-phase":
        intent["before"] = names
    elif damage == "missing-path":
        del intent["before"][names[1]]
    elif damage == "extra-path":
        intent["after"]["unrelated.json"] = "{}"
    elif damage == "after-null":
        intent["after"][names[1]] = None
    else:
        intent["run_id"] = "another-run"
    _assert_rejected(layout, intent)


@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("damage", ["missing", "null", "invalid-hash", "orphan"])
def test_metadata_recovery_requires_a_commit_on_the_current_chain(tmp_path, committed, damage):
    layout, intent, _names = _pending(tmp_path, committed=committed)
    if damage == "missing":
        del intent["commit"]
    elif damage == "null":
        intent["commit"] = None
    elif damage == "invalid-hash":
        intent["commit"]["sha256"] = "0" * 64
    else:
        raw = layout.contained(intent["commit"]["path"]).read_bytes()
        orphan = layout.result_store_path("commits/orphan.json")
        orphan.write_bytes(raw)
        intent["commit"] = {"path": orphan.relative_to(layout.root).as_posix(),
                            "sha256": hashlib.sha256(raw).hexdigest()}
    _assert_rejected(layout, intent)


@pytest.mark.parametrize("committed", [False, True])
def test_valid_recovery_preserves_original_bytes_and_missing_history(tmp_path, committed):
    layout, intent, names = _pending(tmp_path, committed=committed)
    pending = layout.result_store_path("metadata_pending.json")
    pending.write_text(json.dumps(intent), encoding="utf-8")
    head = layout.result_csv.read_bytes()
    expected = intent["after" if committed else "before"]
    with ResultWriter(tmp_path):
        pass
    for name in names:
        path = layout.contained(name)
        if expected[name] is None:
            assert not path.exists()
        else:
            assert path.read_bytes() == expected[name].encode("utf-8")
    assert not pending.exists()
    assert layout.result_csv.read_bytes() == head


@pytest.mark.parametrize("invalid_part", ["prior-metadata", "new-metadata"])
def test_posthoc_checks_recovery_content_before_publishing_it(tmp_path, invalid_part):
    layout, _intent, _names = _pending(tmp_path)
    context = SimpleNamespace(result_dir=tmp_path, static_meta_path=layout.path("static_meta.json"),
                              collection_history_path=layout.path("collection_history.json"))
    if invalid_part == "prior-metadata":
        context.static_meta_path.write_bytes(b"{")
    before = {path: path.read_bytes() for path in
              (layout.result_csv, context.static_meta_path, context.collection_history_path)}
    with pytest.raises(ResultValidationError, match="metadata"):
        commit_incremental(context, rows=[], static_meta=[] if invalid_part == "new-metadata" else {"new": True},
                           collection_history={"schema_version": 1})
    assert {path: path.read_bytes() for path in before} == before
    assert not layout.result_store_path("metadata_pending.json").exists()
    assert list(layout.path("raw/result_updates").glob("*.json")) == []


@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("failure_type", [OSError, KeyboardInterrupt])
def test_recovery_can_retry_after_a_metadata_write_fails(tmp_path, monkeypatch, committed, failure_type):
    from acprof.result_store import metadata

    layout, intent, names = _pending(tmp_path, committed=committed)
    pending = layout.result_store_path("metadata_pending.json")
    pending.write_text(json.dumps(intent), encoding="utf-8")
    journal = pending.read_bytes()
    failure = failure_type("interrupted metadata recovery")
    write = metadata.write_bytes

    def fail_after_write(path, raw):
        write(path, raw)
        raise failure

    with monkeypatch.context() as guard:
        guard.setattr(metadata, "write_bytes", fail_after_write)
        with pytest.raises(failure_type) as caught:
            with ResultWriter(tmp_path):
                pass
    assert caught.value is failure
    assert pending.read_bytes() == journal
    with ResultWriter(tmp_path):
        pass
    expected = intent["after" if committed else "before"]
    for name in names:
        path = layout.contained(name)
        assert (path.read_bytes() if path.exists() else None) == (
            expected[name].encode("utf-8") if expected[name] is not None else None)
    assert not pending.exists()


@pytest.mark.parametrize("committed", [False, True])
def test_posthoc_process_death_keeps_a_valid_recovery_journal(tmp_path, committed):
    layout = initialized(tmp_path)
    meta, history = (layout.path(name) for name in ("static_meta.json", "collection_history.json"))
    original = '\ufeff{"old":"原始元数据"}\r\n'.encode("utf-8")
    meta.write_bytes(original)
    script = '''
import os, signal, sys
from pathlib import Path
from types import SimpleNamespace
from acprof.host.posthoc.incremental import commit_incremental
from acprof.result_store.writer import ResultWriter
from acprof.results.layout import ArtifactLayout
root = Path(sys.argv[1])
layout = ArtifactLayout.discover(root)
context = SimpleNamespace(result_dir=root, static_meta_path=layout.path("static_meta.json"),
                          collection_history_path=layout.path("collection_history.json"))
original = ResultWriter.update_rows
def crash(writer, *args, **kwargs):
    if sys.argv[2] == "True":
        original(writer, *args, **kwargs)
    os.kill(os.getpid(), signal.SIGKILL)
ResultWriter.update_rows = crash
commit_incremental(context, rows=[], static_meta={"new": True}, collection_history={"schema_version": 1})
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path), str(committed)],
                            capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == -9, result.stderr
    pending = layout.result_store_path("metadata_pending.json")
    assert pending.exists()
    with ResultWriter(tmp_path):
        pass
    if committed:
        assert json.loads(meta.read_bytes()) == {"new": True}
        assert json.loads(history.read_bytes()) == {"schema_version": 1}
    else:
        assert meta.read_bytes() == original
        assert not history.exists()
    assert not pending.exists()
