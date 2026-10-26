import random

import pytest

from acprof.tui.log_records import LogRecords


def test_pinned_error_excerpt_is_bounded_and_independent_of_raw_history():
    records = LogRecords(max_lines=20, max_bytes=100_000)
    for number in range(100):
        records.append(f"[ERROR] failure {number}")
    records.append("[ERROR] " + "中" * 2000)
    for number in range(30):
        records.append(f"[INFO] cleanup {number}")
    batch = records.error_context.snapshot()
    assert len(batch.entries) == 12
    assert batch.retained_bytes <= 16 * 1024
    assert batch.discarded_entries == 89
    assert batch.oversized_entries == 1
    assert batch.entries[0] == "[ERROR] failure 0"
    assert batch.entries[-1] == "[ERROR] failure 99"
    assert records.raw_text.startswith("[INFO] cleanup 10\n")


def update(records, rows, **view):
    change = records.project_update(**view)
    rows[change.start:] = change.rows
    assert rows == records.project(**view)
    return change


def test_later_build_error_restores_only_affected_scope_and_warning_counts_cross_batches():
    records = LogRecords(100, 10000)
    rows = []
    view = dict(level='ALL', details=False, query='')
    records.append('[cmd] first\n#1 0.1 old hidden detail\n#1 DONE\n[cmd] second\n#2 0.1 required diagnostic')
    update(records, rows, **view)
    assert not any('diagnostic' in row.text for row in rows)
    prefix = rows[:2]
    records.append('#2 ERROR: compiler failed')
    change = update(records, rows, **view)
    assert not change.reset
    assert change.start == 2
    assert rows[:2] == prefix
    assert any('required diagnostic' in row.text for row in rows)
    assert not any('old hidden detail' in row.text for row in rows)
    records.append('[case] next\n[WARN] retry')
    update(records, rows, **view)
    records.append('[WARN] retry\n[WARN] retry')
    change = update(records, rows, **view)
    assert change.start == len(rows) - 1
    assert rows[-1].count == 3
    assert rows[-1].text.endswith('[×3]')


@pytest.mark.parametrize('max_lines', [7, 1000])
@pytest.mark.parametrize('level', ['ALL', 'ERROR', 'WARN', 'INFO', 'DEBUG'])
def test_incremental_and_full_projections_match_across_batches_eviction_and_view_changes(max_lines, level):
    records = LogRecords(max_lines, 1600)
    rows = []
    rng = random.Random(194)
    entries = [
        '[cmd] build', '#1 0.2 detail', '#1 ERROR: failure', '[INFO] done', '[WARN] retry',
        '[WARN] retry', '[DEBUG] details', '[case] next', '[network-preflight] {"ok": true}',
        '[ERROR] failed', '  traceback continuation', '模型 error context', 'x' * 1800,
    ]
    for batch in range(80):
        records.append('\n'.join(rng.choices(entries, k=rng.randint(1, 4))))
        view = dict(level=level, details=batch % 17 == 0, query='detail' if batch % 13 == 0 else '')
        update(records, rows, **view)
    records.max_lines = 2
    records.trim()
    change = update(records, rows, **view)
    assert change.reset


def test_ordinary_append_projects_only_new_records(monkeypatch):
    # Count actual input records consumed by the projection, not wall time.
    from acprof.tui import log_records

    records = LogRecords(10000, 1000000)
    records.append('\n'.join(f'[INFO] line {i}' for i in range(2000)))
    rows = []
    view = dict(level='ALL', details=False, query='')
    update(records, rows, **view)
    original = log_records._Projection.extend
    consumed = []

    def traced(self, entries):
        entries = list(entries)
        consumed.extend(entries)
        return original(self, entries)

    monkeypatch.setattr(log_records._Projection, 'extend', traced)
    records.append('[INFO] one new line')
    change = records.project_update(**view)
    assert [record.text for record in consumed] == ['[INFO] one new line']
    assert change.start == 2000
    assert len(change.rows) == 1
