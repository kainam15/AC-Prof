"""Storage previews remain dismissible while another process owns the cache lock."""
import asyncio
import tempfile
import threading
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.cells import cell_len
from textual.containers import VerticalScroll
from textual.widgets import Button, Input, Select, Static

from acprof.container.model_files import PLAN_FILENAME
from acprof.experiments.configuration import RunConfig
from acprof.host.model_store import store as model_store
from acprof.tui.i18n import translate
from acprof.tui.models.store import ModelStoreScreen
from acprof.tui.models.store_summary import model_store_summary
from acprof.tui.ui.dialogs import ConfirmActionScreen
from tests.support.model_store_fixtures import ModelStoreFixture, locked_store
from tests.support.tui_fixtures import AcprofTui


@pytest.mark.parametrize('language', ['zh', 'en'])
@pytest.mark.parametrize('size', [(80, 24), (120, 30), (150, 45)])
async def test_locked_preview_shows_capacity_then_closes_before_unlock(language, size, tmp_path):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        app = AcprofTui(RunConfig.smoke('example/model'), settings_path=root / 'settings.json')
        with locked_store(root) as holder:
            async with app.run_test(size=size) as pilot:
                try:
                    app.query_one('#ui-language', Select).value = language
                    await pilot.pause()
                    assert app.configuration.preferences.language == language
                    app.query_one('#model-store', Input).value = str(root)
                    app.preflight.open_model_store()
                    await pilot.pause()
                    assert isinstance(app.screen, ModelStoreScreen)
                    screen = app.screen
                    app.save_screenshot(path=str(tmp_path), filename='model-store-close.svg')
                    close = screen.query_one('#store-close', Button)
                    expected_close = '关闭弹窗' if language == 'zh' else 'Close'
                    assert str(close.label) == expected_close
                    assert cell_len(expected_close) <= close.content_region.width
                    region = close.content_region
                    lines = screen._compositor.render_strips()
                    assert expected_close in ''.join(
                        line.crop(region.x, region.right).text for line in lines[region.y:region.bottom]
                    )
                    content = str(screen.query_one('#store-report', Static).content)
                    assert ('缓存占用：' if language == 'zh' else 'Cache used:') in content
                    assert 'total_bytes' not in content
                    assert (app.tr('缓存正在由另一任务处理；可以关闭窗口取消等待。')) in (str(screen.query_one('#store-status', Static).content))
                    assert not (screen.query_one('#store-close', Button).disabled)
                    status = screen.query_one('#store-status', Static)
                    assert (status.region.y) >= (0)
                    assert (status.region.bottom) <= (screen.query_one('#store-close', Button).region.y)
                    assert (app.is_busy())
                    if language == 'zh':
                        assert (await pilot.click('#store-close'))
                    else:
                        await pilot.press('escape')
                    await app.workers.wait_for_complete()
                    await pilot.pause()
                    assert (holder.poll()) is None, 'cache owner must still be holding the lock'
                    assert not isinstance(app.screen, ModelStoreScreen)
                    assert not (app.is_busy())
                finally:
                    holder.stdin.close()
                    await app.workers.wait_for_complete()

async def test_closing_retains_busy_state_until_the_reader_releases_resources():
    started, acknowledged, release = threading.Event(), threading.Event(), threading.Event()

    def scan(root, *, cancel):
        started.set()
        if not cancel.wait(5):
            raise AssertionError('close did not signal cancellation')
        acknowledged.set()
        if not release.wait(5):
            raise AssertionError('test did not release its simulated reader')
        return {'path': str(root), 'total_bytes': 0, 'free_bytes': 1000, 'models': []}

    with tempfile.TemporaryDirectory() as temporary, patch('acprof.tui.models.store.disk_report', side_effect=scan):
        root = Path(temporary)
        app = AcprofTui(RunConfig.smoke('example/model'), settings_path=root / 'settings.json')
        async with app.run_test(size=(80, 24)) as pilot:
            try:
                app.query_one('#model-store', Input).value = str(root)
                app.preflight.open_model_store()
                await pilot.pause()
                assert (started.is_set())
                assert (await pilot.click('#store-close'))
                await pilot.pause()
                assert (acknowledged.is_set())
                assert isinstance(app.screen, ModelStoreScreen)
                assert (app.is_busy())
            finally:
                release.set()
                await app.workers.wait_for_complete()
            await pilot.pause()
            assert not isinstance(app.screen, ModelStoreScreen)
            assert not (app.is_busy())


def store_report(root, *, models=()):
    return {
        'path': str(root), 'total_bytes': 6_000_000, 'free_bytes': None,
        'capacity_bytes': None, 'models': list(models),
    }


async def open_store(app, pilot, root, language):
    app.query_one('#ui-language', Select).value = language
    await pilot.pause()
    app.query_one('#model-store', Input).value = str(root)
    app.preflight.open_model_store()
    await app.workers.wait_for_complete()
    await pilot.pause()
    assert isinstance(app.screen, ModelStoreScreen)
    return app.screen


@pytest.mark.parametrize('language', ['zh', 'en'])
async def test_readable_preview_and_scrollable_confirmation_preserve_approved_entries(language, tmp_path):
    models = [
        {'entry_id': f'{index:064x}', 'model_id': f'org/model-{index:02d}',
         'model_revision': f'{index:040x}', 'model_artifact_bytes': 2_000_000,
         'last_used': 946_728_000}
        for index in range(12)
    ]
    models[0]['model_id'] = 'org/[draft]-模型'
    missing_entry = 'f' * 64
    selected = [model['entry_id'] for model in models[:10]] + [missing_entry]
    report = store_report(tmp_path, models=models)
    untouched = deepcopy(report)
    preview = {'entries': selected, 'reclaimable_bytes': 1_250_000}
    after = store_report(tmp_path, models=models[10:])
    after['total_bytes'] = 4_750_000
    config = replace(RunConfig.smoke('example/model'), model_store_max='2GB')
    app = AcprofTui(config, settings_path=tmp_path / 'settings.json')
    with patch('acprof.tui.models.store.disk_report', side_effect=[report, after]), patch(
        'acprof.tui.models.store.prune_store',
        side_effect=[preview, {'entries': selected, 'reclaimable_bytes': 1_250_000},
                     {'entries': [], 'reclaimable_bytes': 0}],
    ) as prune:
        async with app.run_test(size=(80, 24)) as pilot:
            screen = await open_store(app, pilot, tmp_path, language)
            content = str(screen.query_one('#store-report', Static).content)
            assert ('缓存占用：6.00 MB' if language == 'zh' else 'Cache used: 6.00 MB') in content
            assert ('磁盘可用：未知' if language == 'zh' else 'Disk free: Unknown') in content
            assert ('容量上限：2.00 GB' if language == 'zh' else 'Capacity limit: 2.00 GB') in content
            assert ('预计可回收：1.25 MB' if language == 'zh' else 'Estimated reclaimable: 1.25 MB') in content
            assert 'org/[draft]-模型' in content
            assert models[0]['model_revision'] in content
            assert '2000-' in content
            assert ('模型文件大小包含共享权重，不能相加作为磁盘占用。' if language == 'zh'
                    else 'Model file sizes include shared weights; their sum is not disk usage.') in content
            assert all(key not in content for key in ('total_bytes', 'prune_preview', 'reclaimable_bytes', 'last_used'))
            assert not screen.query_one('#store-prune', Button).disabled
            app.save_screenshot(path=str(tmp_path), filename='store-summary.svg')
            scroll = screen.query_one('#store-scroll', VerticalScroll)
            scroll.focus()
            await pilot.press('end')
            await pilot.wait_for_scheduled_animations()
            assert scroll.scroll_y > 0
            app.save_screenshot(path=str(tmp_path), filename='store-summary-bottom.svg')
            assert await pilot.click('#store-prune')
            await pilot.pause()
            confirmation = app.screen
            assert isinstance(confirmation, ConfirmActionScreen)
            assert '1.25 MB' in confirmation.message
            assert 'org/[draft]-模型' in confirmation.message
            assert missing_entry in confirmation.message
            assert 'org/model-11' not in confirmation.message
            assert 'reclaimable_bytes' not in confirmation.message
            message = confirmation.query_one('#confirm-message')
            await pilot.resize_terminal(120, 30)
            await pilot.pause()
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            await pilot.press('tab')
            assert confirmation.focused is message
            await pilot.press('end')
            await pilot.wait_for_scheduled_animations()
            assert message.scroll_y > 0
            assert message.max_scroll_y > 0
            region = message.content_region
            lines = confirmation._compositor.render_strips()
            assert missing_entry[-20:] in ''.join(
                line.crop(region.x, region.right).text for line in lines[region.y:region.bottom])
            for identity in ('confirm-no', 'confirm-yes'):
                button = confirmation.query_one(f'#{identity}', Button)
                assert 0 <= button.region.y < button.region.bottom <= 24
                assert cell_len(str(button.label)) <= button.content_region.width
                assert app.get_widget_at(*button.region.center)[0] is button
            app.save_screenshot(path=str(tmp_path), filename='store-confirm-bottom.svg')
            await pilot.press('tab')
            assert confirmation.focused.id == 'confirm-no'
            await pilot.press('tab')
            assert confirmation.focused.id == 'confirm-yes'
            await pilot.press('escape')
            await pilot.pause()
            assert app.screen is screen
            assert app.is_busy()
            assert not any(call.kwargs.get('apply') for call in prune.call_args_list)
            assert await pilot.click('#store-prune')
            await pilot.pause()
            await pilot.press('enter', 'space')
            assert not any(call.kwargs.get('apply') for call in prune.call_args_list)
            assert await pilot.click('#confirm-yes')
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            applied = [call for call in prune.call_args_list if call.kwargs.get('apply')]
            assert len(applied) == 1
            assert applied[0].kwargs['approved_entries'] == set(selected)
            assert report == untouched
            assert screen.query_one('#store-prune', Button).disabled
            assert ('没有可清理的缓存。' if language == 'zh' else 'No cache can be pruned.') in str(
                screen.query_one('#store-report', Static).content)
            assert await pilot.click('#store-close')
            await pilot.pause()
            assert not app.is_busy()


@pytest.mark.parametrize('language', ['zh', 'en'])
@pytest.mark.parametrize('reclaimable', [0, 37])
async def test_empty_model_list_keeps_cache_usage_and_orphan_preview_visible(language, reclaimable, tmp_path):
    app = AcprofTui(RunConfig.smoke('example/model'), settings_path=tmp_path / 'settings.json')
    with patch('acprof.tui.models.store.disk_report', return_value=store_report(tmp_path)), patch(
        'acprof.tui.models.store.prune_store', return_value={'entries': [], 'reclaimable_bytes': reclaimable},
    ):
        async with app.run_test(size=(80, 24)) as pilot:
            screen = await open_store(app, pilot, tmp_path, language)
            content = str(screen.query_one('#store-report', Static).content)
            assert ('已登记模型：0' if language == 'zh' else 'Registered models: 0') in content
            assert ('容量上限：不限' if language == 'zh' else 'Capacity limit: Unlimited') in content
            assert '6.00 MB' in content
            assert ('没有已登记模型；缓存仍可能包含未完成下载或未引用文件。' if language == 'zh'
                    else 'No registered models; unfinished downloads or unreferenced files may still occupy space.') in content
            assert screen.query_one('#store-prune', Button).disabled is (reclaimable == 0)
            if reclaimable:
                assert '37 B' in content
                assert await pilot.click('#store-prune')
                await pilot.pause()
                assert ('仅清理未引用的缓存文件，不删除已登记模型。' if language == 'zh'
                        else 'Only unreferenced cache files will be pruned; registered models will remain.') in app.screen.message
                await pilot.press('escape')
            else:
                assert ('没有可清理的缓存。' if language == 'zh' else 'No cache can be pruned.') in content


async def test_failed_refresh_removes_stale_cleanup_preview_and_allows_retry(tmp_path):
    report = store_report(tmp_path)
    preview = {'entries': [], 'reclaimable_bytes': 37}
    app = AcprofTui(RunConfig.smoke('example/model'), settings_path=tmp_path / 'settings.json')
    with patch('acprof.tui.models.store.disk_report', side_effect=[report, OSError('cannot scan [cache]'), report]), patch(
        'acprof.tui.models.store.prune_store', side_effect=[preview, preview],
    ):
        async with app.run_test(size=(80, 24)) as pilot:
            screen = await open_store(app, pilot, tmp_path, 'en')
            assert await pilot.click('#store-refresh')
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert 'cannot scan [cache]' in str(screen.query_one('#store-status', Static).content)
            assert '37 B' not in str(screen.query_one('#store-report', Static).content)
            assert screen.query_one('#store-prune', Button).disabled
            assert not screen.query_one('#store-refresh', Button).disabled
            assert await pilot.click('#store-refresh')
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not screen.query_one('#store-prune', Button).disabled


@pytest.mark.parametrize('language', ['zh', 'en'])
@pytest.mark.parametrize(('capacity', 'zh', 'en'), [
    ('missing', '未知', 'Unknown'), (None, '不限', 'Unlimited'), (0, '0 B', '0 B'),
])
def test_summary_distinguishes_unknown_zero_and_unlimited_without_mutating_report(capacity, zh, en, language, tmp_path):
    report = store_report(tmp_path, models=[{
        'entry_id': 'entry', 'model_id': 'org/[literal]-模型', 'model_revision': 'revision',
        'model_artifact_bytes': None, 'last_used': None,
    }])
    report['total_bytes'] = None
    if capacity == 'missing':
        report.pop('capacity_bytes')
    else:
        report['capacity_bytes'] = capacity
    original = deepcopy(report)
    text = model_store_summary(report, None, lambda value: translate(value, language))
    if language == 'zh':
        assert f'容量上限：{zh}' in text
        assert '缓存占用：未知' in text
        assert '模型文件：未知' in text
        assert '最近使用（本机时间）：未知' in text
    else:
        assert f'Capacity limit: {en}' in text
        assert 'Cache used: Unknown' in text
        assert 'Model files: Unknown' in text
        assert 'Last used (host time): Unknown' in text
    assert 'org/[literal]-模型' in text
    assert report == original


async def test_invalid_capacity_is_visible_without_starting_cleanup(tmp_path):
    config = replace(RunConfig.smoke('example/model'), model_store_max='invalid-size')
    app = AcprofTui(config, settings_path=tmp_path / 'settings.json')
    with patch('acprof.tui.models.store.disk_report', return_value=store_report(tmp_path)), patch(
        'acprof.tui.models.store.prune_store',
    ) as prune:
        async with app.run_test(size=(80, 24)) as pilot:
            screen = await open_store(app, pilot, tmp_path, 'zh')
            assert '5GB' in str(screen.query_one('#store-status', Static).content)
            assert screen.query_one('#store-prune', Button).disabled
            assert not screen.query_one('#store-refresh', Button).disabled
            prune.assert_not_called()
            assert await pilot.click('#store-close')
            await pilot.pause()
            assert not app.is_busy()


@pytest.mark.parametrize('language', ['zh', 'en'])
@pytest.mark.parametrize('mode', ['target-met', 'partial-cleanup', 'unbounded', 'zero'])
async def test_confirmed_cleanup_rechecks_capacity_after_another_cleanup(mode, language, tmp_path, monkeypatch):
    monkeypatch.delenv('ACPROF_MODEL_STORE_MAX', raising=False)
    store = ModelStoreFixture(tmp_path / 'store')
    count = 1 if mode == 'target-met' else 2
    blobs = [store.blob(str(index), 500_000) for index in range(1, count + 1)]
    entries = [store.entry(index, [blob]) for index, blob in enumerate(blobs, start=1)]
    target = 500_000 + len((store.root / 'entries' / entries[-1] / PLAN_FILENAME).read_bytes())
    capacity = '' if mode == 'unbounded' else '0' if mode == 'zero' else str(target)
    app = AcprofTui(replace(RunConfig.smoke('example/test'), model_store_max=capacity),
                    settings_path=tmp_path / 'settings.json')
    external = None
    if mode in {'target-met', 'partial-cleanup'}:
        external_blob = store.blob('leased', 500_000)
        external = store.entry(99, [external_blob])
        plan = model_store.read_entry(external, store.root)
        manifest = {'model_store': {'entry_id': external, 'plan_sha256': plan['plan_sha256']}}
    with model_store.acquire_mount(manifest, store.root) if external else nullcontext() as lease:
        async with app.run_test(size=(80, 24)) as pilot:
            screen = await open_store(app, pilot, store.root, language)
            assert screen.preview['entries'] == entries
            assert await pilot.click('#store-prune')
            await pilot.pause()
            assert isinstance(app.screen, ConfirmActionScreen)
            if external:
                # Another consumer releases a model that was protected during preview.
                lease.close()
                released = await asyncio.to_thread(
                    model_store.prune_store, root=store.root, apply=True, approved_entries={external},
                )
                assert released['entries'] == [external]
                assert not external_blob.exists()
                assert all(blob.exists() for blob in blobs)
                current = sum(path.stat().st_size for path in store.root.rglob('*')
                              if path.is_file() and not path.is_symlink())
                assert (current <= target) is (mode == 'target-met')
            assert await pilot.click('#confirm-yes')
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.screen is screen
            remaining = {path.name for path in (store.root / 'entries').iterdir() if path.is_dir()}
            expected = {entries[-1]} if external else set()
            assert remaining == expected
            for entry, blob in zip(entries, blobs, strict=True):
                assert blob.exists() is (entry in expected)
                if entry in expected:
                    assert blob.read_bytes() == b'x' * 500_000
            assert screen.query_one('#store-prune', Button).disabled
            assert await pilot.click('#store-close')
            await pilot.pause()
            assert not app.is_busy()
