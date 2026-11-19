"""Source disclosure, partial failures and request scope in the real overlay."""
from dataclasses import replace
from unittest.mock import Mock

import pytest
from textual.widgets import Collapsible, Static

from acprof.experiments.configuration import RunConfig
from acprof.tui.run.diagnostics import PreflightCheck
from acprof.tui.run.progress import ProgressSnapshot
from tests.support.tui_fixtures import AcprofTui


@pytest.mark.parametrize(('language', 'size'), [('zh', (80, 24)), ('en', (120, 30))])
async def test_source_disclosure_preserves_choices_and_partial_evidence(tmp_path, monkeypatch, language, size):
    app = AcprofTui(replace(RunConfig.smoke('fixture/model'), runtime='docker', profiling_mode='full', gpus='on', repeat=3),
                    settings_path=tmp_path / 'settings.json')
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=size) as pilot:
        probe = Mock(side_effect=AssertionError('viewing a source must not probe'))
        monkeypatch.setattr(app.preflight, 'execute_quick_check', probe)
        app.preflight.environment_checks = (
            PreflightCheck('CPU', 'ok', 'cpu read', check_id='process_cpu'),
            PreflightCheck('Memory', 'ok', 'memory read', check_id='process_memory'),
            PreflightCheck('GPU memory', 'ok', 'memory read', check_id='gpu_memory'),
            PreflightCheck('GPU power', 'fail', 'unsupported power read', 'unsupported', check_id='gpu_power'),
        )
        app.preflight.refresh_preflight_state()
        panel = app.preflight.panel
        panel.show(pin=True, group='success')
        await pilot.pause()
        process = panel.query_one('#preflight-section-process', Collapsible)
        nvml = panel.query_one('#preflight-section-nvml', Collapsible)
        assert process.collapsed
        assert not nvml.collapsed
        assert nvml.title == 'NVIDIA NVML · ! 1 · ✓ 1'
        assert process.title == 'Process Runtime · ✓ 2'
        label = nvml.query_one('CollapsibleTitle').label
        assert label.plain == nvml.title
        assert {span.style: label.plain[span.start:span.end] for span in label.spans} == {
            '$warning': '! 1', '$success': '✓ 1',
        }
        assert panel.query_one('#preflight-signal-gpu_memory').row.status == 'ok'
        assert panel.query_one('#preflight-signal-gpu_power').row.status == 'warn'
        title = process.query_one('CollapsibleTitle')
        title.scroll_visible(immediate=True, force=True)
        await pilot.pause()
        assert await pilot.click(title)
        assert not process.collapsed
        assert 'Process' not in panel.query_one('#preflight-signal-process_cpu').render().plain
        title = nvml.query_one('CollapsibleTitle')
        title.scroll_visible(immediate=True, force=True)
        title.focus()
        await pilot.press('enter')
        assert nvml.collapsed
        app.preflight.environment_checks = tuple(replace(row, detail='updated') for row in app.preflight.environment_checks)
        app.preflight.refresh_preflight_state()
        await pilot.pause()
        assert not panel.query_one('#preflight-section-process', Collapsible).collapsed
        assert panel.query_one('#preflight-section-nvml', Collapsible).collapsed
        panel.show(pin=True, group='warning')
        await pilot.pause()
        assert panel.query_one('#preflight-section-nvml', Collapsible).collapsed
        title = panel.query_one('#preflight-section-nvml CollapsibleTitle')
        title.focus()
        await pilot.press('enter')
        assert not panel.query_one('#preflight-actual-note', Static).display
        for width, height in ((150, 45), (52, 24)):
            await pilot.resize_terminal(width, height)
            await pilot.pause()
            assert panel.region.right <= width
            assert panel.region.bottom <= height
            assert panel.query_one('#preflight-section-nvml').region.width > 0
        panel.show(pin=True, group='success')
        await pilot.pause()
        panel.select(panel.query_one('#preflight-signal-gpu_memory').row)
        panel.show(pin=True, group='warning')
        await pilot.pause()
        assert panel._selected_row is None
        signal = panel.query_one('#preflight-signal-gpu_power')
        signal.scroll_visible(immediate=True, force=True)
        await pilot.pause()
        assert await pilot.click(signal)
        assert panel._selected_row.check_id == 'gpu_power'
        note = panel.query_one('#preflight-actual-note', Static)
        assert note.display
        assert ('预检仅说明环境能力' if language == 'zh' else 'Preflight shows environment capabilities') in str(note.content)
        assert 'gpu_energy_total_j' not in str(note.content)
        app.execution.session.snapshot = ProgressSnapshot(measurement_active=True)
        panel.update_cached()
        await pilot.pause()
        assert panel.retry_button.disabled
        probe.assert_not_called()


@pytest.mark.parametrize(('language', 'size'), [('zh', (80, 24)), ('en', (120, 30))])
async def test_docker_and_cgroup_are_separate_collapsible_sources(tmp_path, monkeypatch, language, size):
    app = AcprofTui(replace(RunConfig.smoke('fixture/model'), runtime='docker', profiling_mode='full'),
                    settings_path=tmp_path / 'settings.json')
    app.configuration.preferences = replace(app.configuration.preferences, language=language)
    async with app.run_test(size=size) as pilot:
        probe = Mock(side_effect=AssertionError('viewing a source must not probe'))
        monkeypatch.setattr(app.preflight, 'execute_quick_check', probe)
        app.preflight.environment_checks = (
            PreflightCheck('Docker', 'ok', 'daemon ready', check_id='docker'),
            PreflightCheck('cgroup v2', 'ok', 'hierarchy ready', check_id='cgroup'),
            PreflightCheck('CPU', 'ok', 'cpu.stat readable', check_id='container_cpu'),
            PreflightCheck('Memory', 'ok', 'memory.current readable', check_id='container_memory'),
            PreflightCheck('cgroup metrics', 'fail', 'partially unavailable', 'partial', check_id='cgroup_metrics'),
        )
        app.preflight.refresh_preflight_state()
        panel = app.preflight.panel
        panel.show(pin=True, group='success')
        await pilot.pause()
        docker = panel.query_one('#preflight-section-docker', Collapsible)
        cgroup = panel.query_one('#preflight-section-cgroup', Collapsible)
        assert docker.collapsed
        assert not cgroup.collapsed
        assert docker.title == 'Docker · ✓ 1'
        assert cgroup.title == 'cgroup v2 · ! 1 · ✓ 3'
        assert docker.query_one('#preflight-signal-docker')
        assert cgroup.query_one('#preflight-signal-container_cpu')
        assert cgroup.query_one('#preflight-signal-cgroup_metrics')
        assert not docker.query('#preflight-signal-container_cpu')
        panel.show(pin=True, group='warning')
        await pilot.pause()
        assert not panel.query('#preflight-section-docker')
        assert panel.query_one('#preflight-section-cgroup', Collapsible)
        probe.assert_not_called()
