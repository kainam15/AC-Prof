"""Malformed pending transactions fail consistently without changing evidence."""
import json

import pytest

from acprof.analysis.audit import audit_result
from acprof.result_store.protocol import publication_lock
from acprof.result_store.reader import committed_content_digest, read_snapshot
from acprof.result_store.recovery import recover_pending
from acprof.result_store.writer import ResultWriter
from acprof.results.csv import ResultValidationError, read_result_csv_snapshot, verify_result_source
from tests.results.storage.test_committed_digest import make_store
from tests.results.storage.test_incremental_results import receipt
from tests.results.storage.test_sequential_result_reads import artifact_hashes
from tests.support.result_store_fixtures import assert_writer_lease_available, leave_pending_append

_MISSING = object()
OPERATIONS = ("snapshot", "verify", "digest", "audit", "recovery", "writer")


@pytest.fixture
def interrupted_append(tmp_path, monkeypatch):
    layout = make_store(tmp_path)
    snapshot = read_result_csv_snapshot(layout.result_csv)
    leave_pending_append(layout, receipt(tmp_path, 1), monkeypatch)
    return layout, snapshot


def assert_pending_rejected(layout, snapshot, operation):
    before = artifact_hashes(layout)
    if operation == "audit":
        report = audit_result(layout.root)
        assert not report["valid"]
        assert report["counts"]["rows"] == 0
        assert any(issue["code"] == "invalid_csv" and "pending" in issue["message"]
                   for issue in report["issues"])
    else:
        def consume():
            if operation == "snapshot":
                read_snapshot(layout.root)
            elif operation == "verify":
                verify_result_source(layout.result_csv, snapshot.sha256)
            elif operation == "digest":
                committed_content_digest(layout.root, cancelled=lambda: False,
                                         budget=[32 * 1024 * 1024])
            elif operation == "recovery":
                with publication_lock(layout, exclusive=True):
                    recover_pending(layout)
            else:
                with ResultWriter(layout.root):
                    pytest.fail("malformed pending transaction was accepted")
        with pytest.raises(ResultValidationError, match="pending"):
            consume()
    assert artifact_hashes(layout) == before
    assert_writer_lease_available(layout)
    with publication_lock(layout, exclusive=True, blocking=False):
        pass


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(("path", "value"), [
    pytest.param((), {}, id="empty-object"),
    pytest.param(("before",), _MISSING, id="missing-before"),
    pytest.param(("before",), [], id="before-not-object"),
    pytest.param(("before", "commit"), _MISSING, id="missing-before-commit"),
    pytest.param(("before", "commit", "path"), 1, id="commit-path-type"),
    pytest.param(("before", "commit", "sha256"), "bad", id="commit-hash"),
    pytest.param(("before", "commit", "path"), "../outside.json", id="commit-escape"),
    pytest.param(("before", "commit", "path"), "raw/elsewhere.json", id="commit-outside-journal"),
    pytest.param(("before", "state"), [], id="before-state-type"),
    pytest.param(("before", "run_id"), True, id="before-run-id-type"),
    pytest.param(("before", "state"), "interrupted", id="before-current-disagrees"),
    pytest.param(("after",), _MISSING, id="missing-after"),
    pytest.param(("after",), {}, id="empty-after"),
    pytest.param(("after",), [], id="after-not-object"),
    pytest.param(("after", "run_id"), "another-run", id="after-run-id"),
    pytest.param(("after", "sequence"), 99, id="after-sequence"),
    pytest.param(("after", "schema_version"), True, id="after-schema-type"),
    pytest.param(("after", "row_count"), 2.0, id="after-count-type"),
    pytest.param(("after", "layers"), [], id="after-layers-type"),
    pytest.param(("after", "layers", "summary"), [], id="after-layer-type"),
    pytest.param(("after", "layers", "summary", "path"), "raw/wrong.csv", id="after-layer-path"),
    pytest.param(("after", "layers", "summary", "revision"), 99, id="undeclared-replacement"),
    pytest.param(("after", "commit"), {}, id="after-has-commit"),
    pytest.param(("after", "snapshot"), {"path": 1, "sha256": "a" * 64}, id="snapshot-reference"),
    pytest.param(("replacements",), _MISSING, id="missing-replacements"),
    pytest.param(("replacements",), None, id="null-replacements"),
    pytest.param(("replacements",), {}, id="object-replacements"),
    pytest.param(("replacements",), "summary", id="string-replacements"),
    pytest.param(("replacements",), [{}], id="replacement-element-type"),
    pytest.param(("replacements",), ["unknown"], id="unknown-replacement"),
    pytest.param(("replacements",), ["summary", "summary"], id="duplicate-replacement"),
    pytest.param(("replacements",), ["summary"], id="replacement-without-revision"),
])
def test_malformed_pending_is_rejected_at_every_entry(interrupted_append, operation, path, value):
    layout, snapshot = interrupted_append
    pending = layout.result_store_path("pending.json")
    transaction = json.loads(pending.read_bytes())
    if not path:
        transaction = value
    else:
        target = transaction
        for part in path[:-1]:
            target = target[part]
        if value is _MISSING:
            target.pop(path[-1])
        else:
            target[path[-1]] = value
    pending.write_text(json.dumps(transaction))
    assert_pending_rejected(layout, snapshot, operation)


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("content", [b"[]", b"null", b'{"before":', b"\xff", b'{"invalid":NaN}'])
def test_pending_json_damage_is_a_domain_error(interrupted_append, operation, content):
    layout, snapshot = interrupted_append
    layout.result_store_path("pending.json").write_bytes(content)
    assert_pending_rejected(layout, snapshot, operation)
