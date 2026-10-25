from textual.widgets.text_area import Selection

from acprof.tui.log import SelectableLog
from tests.test_tui_log import LogApp


async def test_level_colors_do_not_modify_text_or_copy():
    app = LogApp()
    async with app.run_test(size=(70, 12)) as pilot:
        log = app.query_one(SelectableLog)
        raw = '[hf][WARNING] retry /tmp/a.json\n[INFO] 0 errors; [red]literal[/red]\n#2 CACHED'
        log.write(raw)
        await pilot.pause()
        assert log.get_line(0).spans
        assert log.get_line(1).plain == raw.splitlines()[1]
        assert log.copy_all()
        assert app.clipboard == raw


async def test_filter_search_and_compaction_preserve_retained_originals():
    app = LogApp()
    async with app.run_test(size=(70, 12)) as pilot:
        log = app.query_one(SelectableLog)
        raw = '[INFO] start\n[WARN] retry\n[WARN] retry\n#3 0.1 downloading dependency\n#3 DONE 1.2s\n[ERROR] failed\n  File "模型.py", line 2\nValueError: bad'
        log.write(raw)
        log.set_view(details=False)
        assert '×2' in log.text
        assert '0.1 downloading dependency' not in log.text
        log.set_view(level='ERROR')
        assert 'ValueError: bad' in log.text
        assert '[INFO]' not in log.text
        log.set_view(level='ALL', query='dependency')
        assert 'downloading dependency' in log.text
        assert log.match_count == 1
        assert log.find_match()
        assert log.selected_text == 'dependency'
        assert not log.following
        assert log.copy_all()
        assert app.clipboard == raw
        log.set_view(query='', details=True)
        assert log.text == raw
        await pilot.pause()


async def test_filtered_append_keeps_selection_and_bounded_raw_history():
    app = LogApp(max_lines=5)
    async with app.run_test(size=(60, 10)) as pilot:
        log = app.query_one(SelectableLog)
        log.write('[INFO] start\n[WARN] selected\n[INFO] next')
        log.set_view(level='WARN')
        log.selection = Selection((0, 7), (0, 15))
        log.write('[INFO] more\n[INFO] last\n[INFO] trim')
        assert log.selected_text == 'selected'
        assert log.discarded_lines == 1
        log.copy_all()
        assert '[INFO] start' not in app.clipboard
        assert '[INFO] trim' in app.clipboard
        log.clear_log()
        assert not log.copy_all()
        assert log.discarded_lines == 0
        await pilot.pause()


async def test_search_is_literal_and_byte_retention_is_explicit():
    app = LogApp(max_bytes=100)
    async with app.run_test(size=(60, 10)) as pilot:
        log = app.query_one(SelectableLog)
        log.write('[INFO] [literal] 模型')
        log.set_view(query='[literal]')
        assert log.match_count == 1
        log.write('x' * 200)
        assert log.discarded_lines >= 1
        assert len(log.raw_text.encode()) <= 100
        await pilot.pause()


async def test_match_navigation_starts_at_each_end_and_wraps():
    app = LogApp()
    async with app.run_test(size=(60, 10)) as pilot:
        log = app.query_one(SelectableLog)
        log.write('[INFO] hit one\n[INFO] hit two\n[INFO] hit three')
        log.set_view(query='hit')
        assert log.find_match(previous=True)
        assert log.selection.start == (2, 7)
        assert log.find_match()
        assert log.selection.start == (0, 7)
        assert log.find_match(previous=True)
        assert log.selection.start == (2, 7)
        await pilot.pause()


async def test_build_failure_restores_diagnostics_and_case_warnings_do_not_merge():
    app = LogApp(details=False)
    async with app.run_test(size=(70, 12)) as pilot:
        log = app.query_one(SelectableLog)
        log.write('[cmd] docker build\n#1 0.1 compiler diagnostic\n#1 ERROR: failed')
        assert 'compiler diagnostic' in log.text
        log.write('[case] first\n[WARN] retry\n[case] second\n[WARN] retry')
        assert '×2' not in log.text
        log.write('[INFO] no error; literal [ERROR] in a message')
        log.set_view(level='ERROR')
        assert 'literal' not in log.text
        assert 'compiler diagnostic' in log.text
        await pilot.pause()


async def test_large_batches_remain_bounded_and_append_without_reloading_document():
    app = LogApp(max_lines=1000, max_bytes=100000)
    async with app.run_test(size=(70, 12)) as pilot:
        from unittest.mock import patch

        log = app.query_one(SelectableLog)
        with patch.object(log, 'load_text', side_effect=AssertionError('full document reload')):
            for start in range(0, 5000, 100):
                log.write('\n'.join(f'[INFO] record {number}' for number in range(start, start + 100)))
        assert len(log.lines) == 1000
        assert log.discarded_lines == 4000
        assert log.lines[0] == '[INFO] record 4000'
        assert log.lines[-1] == '[INFO] record 4999'
        await pilot.pause()
