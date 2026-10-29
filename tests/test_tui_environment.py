"""Connection settings use an isolated project and never send notifications."""
import os
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest
from environment_fixtures import isolated_environment
from textual.widgets import Button, Checkbox, Input, Static, TabbedContent
from tui_fixtures import AcprofTui, checkbox_text

from acprof.experiment import RunConfig
from acprof.host.env_utils import load_project_env
from acprof.tui.views import ConfirmActionScreen


class TestTuiEnvironment:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        directory = tmp_path
        self.root = Path(str(directory))
        for context in (patch('acprof.tui.app.PROJECT_DIR', self.root),
                        patch.dict(os.environ, isolated_environment({'PATH': os.environ.get('PATH', '')}), clear=True)):
            context.start()
            self._request.addfinalizer(partial(context.stop))

    async def open_connections(self, app, pilot):
        # Keyboard focus scrolls the last settings section into view on short terminals.
        button = app.query_one('#open-environment-settings', Button)
        button.focus()
        await pilot.pause()
        assert (await pilot.click(button))
        await pilot.pause()

    @pytest.mark.parametrize(('language', 'theme', 'no_color', 'size'), [
        ('zh', 'acprof-graphite', False, (80, 24)),
        ('en', 'acprof-light', False, (120, 30)),
        ('zh', 'acprof-light', True, (150, 45)),
        ('en', 'acprof-graphite', True, (80, 24)),
    ])
    async def test_permission_marks_remain_distinct_in_themes_and_grayscale(
        self, language, theme, no_color, size, monkeypatch,
    ):
        if no_color:
            monkeypatch.setenv('NO_COLOR', '1')
        else:
            monkeypatch.delenv('NO_COLOR', raising=False)
        app = AcprofTui(RunConfig.smoke('demo/model'), settings_path=self.root / 'tui.json')
        app.configuration.preferences = replace(app.configuration.preferences, language=language, theme=theme)
        async with app.run_test(size=size) as pilot:
            await pilot.press('f2')
            await pilot.pause()
            await self.open_connections(app, pilot)
            screen = app.screen
            screen.query_one('#environment-tabs', TabbedContent).active = 'permissions-tab'
            await pilot.pause()
            perf = screen.query_one('#permission-perf', Checkbox)
            tcpdump = screen.query_one('#permission-tcpdump', Checkbox)
            assert checkbox_text(perf).count('✓') == 1
            assert checkbox_text(tcpdump).count('✓') == 1
            assert await pilot.click(perf)
            await pilot.pause()
            assert not perf.value
            assert tcpdump.value
            assert '✓' not in checkbox_text(perf)
            assert checkbox_text(tcpdump).count('✓') == 1
            app.save_screenshot(path=str(self.root), filename='permission-marks.svg')

            # NO_COLOR changes the final frame, not just a widget's theme styles.
            colors = {
                color.get_truecolor()
                for strip in screen._compositor.render_strips()
                for segment in strip if segment.style
                for color in (segment.style.color, segment.style.bgcolor) if color
            }
            assert colors
            assert all(len(set(rgb)) == 1 for rgb in colors) == no_color
            with patch('acprof.tui.environment.build_permission_plan') as build, patch(
                'acprof.tui.environment.execute_permission_plan',
            ) as install:
                build.return_value.commands = (('/usr/sbin/setcap', 'cap_net_raw=ep', '/usr/bin/tcpdump'),)
                assert await pilot.click('#configure-environment-permissions')
                await pilot.pause()
                build.assert_called_once_with(('tcpdump',))
                await pilot.press('escape')
                await pilot.pause()
                install.assert_not_called()

            perf.focus()
            await pilot.press('space')
            await pilot.pause()
            assert perf.value
            assert tcpdump.value
            assert checkbox_text(perf).count('✓') == 1
            assert checkbox_text(tcpdump).count('✓') == 1

    async def test_permission_review_can_cancel_and_failed_sudo_returns_to_form(self):
        app = AcprofTui(RunConfig.smoke('demo/model'), settings_path=self.root / 'tui.json')
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press('f2')
            await pilot.pause()
            await self.open_connections(app, pilot)
            screen = app.screen
            screen.query_one('#environment-tabs', TabbedContent).active = 'permissions-tab'
            await pilot.pause()
            with patch('acprof.tui.environment.build_permission_plan') as build, patch(
                'acprof.tui.environment.execute_permission_plan', side_effect=RuntimeError('sudo cancelled'),
            ) as install, patch('textual.app.App.suspend', return_value=nullcontext()):
                # The real plan contains ten commands, including long Ubuntu ELF paths.
                build.return_value.commands = (
                    ('/usr/sbin/setcap', 'cap_perfmon=ep',
                     '/usr/lib/linux-hwe-7.0-tools-7.0.0-28/perf'),
                ) * 10
                await pilot.click('#configure-environment-permissions')
                await pilot.pause()
                confirmation = app.screen
                assert isinstance(confirmation, ConfirmActionScreen)
                assert isinstance(confirmation, ConfirmActionScreen)
                assert ('cap_perfmon') in (confirmation.message)
                commands = confirmation.query_one('#confirm-message')
                assert (commands.max_scroll_y) > (0), 'The complete permission plan must be scrollable'
                commands.focus()
                await pilot.press('end')
                await pilot.wait_for_scheduled_animations()
                assert (commands.scroll_y) > (0)
                install.assert_not_called()
                await pilot.press('escape')
                await pilot.pause()
                install.assert_not_called()
                await pilot.click('#configure-environment-permissions')
                await pilot.pause()
                confirm = app.screen.query_one('#confirm-yes', Button)
                assert (confirm.region.bottom) <= (24)
                assert (app.get_widget_at(*confirm.region.center)[0]) is (confirm)
                await pilot.click('#confirm-yes')
                await pilot.pause()
                install.assert_called_once()
                assert (app.screen) is (screen)
                status = screen.query_one('#environment-status', Static).content
                assert isinstance(status, str)
                assert ('sudo cancelled') in (status)
                assert not (screen.query_one('#close-environment-settings', Button).disabled)
            await pilot.press('escape')
            await pilot.pause()
            assert not (app.is_busy())

    async def test_permission_check_uses_measurement_lock_and_ignores_unmounted_callback(self):
        app = AcprofTui(RunConfig.smoke('demo/model'), settings_path=self.root / 'tui.json')
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press('f2')
            await pilot.pause()
            await self.open_connections(app, pilot)
            screen = app.screen
            screen.query_one('#environment-tabs', TabbedContent).active = 'permissions-tab'
            await pilot.pause()
            with patch('acprof.tui.environment.MeasurementLock') as lock, patch(
                'acprof.tui.environment.quick_preflight', return_value=[],
            ) as preflight:
                assert await pilot.click('#check-environment-permissions')
                for _ in range(20):
                    await pilot.pause()
                    if not screen._working:
                        break
                lock.assert_called_once_with()
                assert lock.return_value.__enter__.called
                assert lock.return_value.__exit__.called
                preflight.assert_called_once()
            await screen.dismiss(False)
            await pilot.pause()
            assert app.screen is not screen
            screen._checked_permissions('late', 0)


    @pytest.mark.parametrize('language', ['zh', 'en'])
    @pytest.mark.parametrize('size', [(80, 24), (120, 30), (150, 45)])
    async def test_bilingual_dialog_actions_are_reachable_at_supported_sizes(self, language, size):
        app = AcprofTui(RunConfig.smoke('demo/model'), settings_path=self.root / 'tui.json')
        async with app.run_test(size=(150, 45)) as pilot:
            await pilot.resize_terminal(*size)
            app.configuration.preferences = replace(app.configuration.preferences, language=language)
            app.configuration.apply_ui_preferences()
            await pilot.press('f2')
            await pilot.pause()
            await self.open_connections(app, pilot)
            for widget_id in ('close-environment-settings', 'save-environment-settings'):
                widget = app.screen.query_one('#' + widget_id, Button)
                assert (widget.region.width)
                assert (widget.region.height)
                assert (widget.region.right) <= (size[0])
                assert (widget.region.bottom) <= (size[1])
            app.screen.query_one('#environment-tabs', TabbedContent).active = 'permissions-tab'
            await pilot.pause()
            # Check actual hit testing rather than just DOM presence.
            assert (await pilot.click('#permission-tcpdump'))
            assert not (app.screen.query_one('#permission-tcpdump', Checkbox).value)
            assert (await pilot.click('#close-environment-settings'))
            await pilot.pause()

    async def test_settings_can_save_masked_credentials_and_reopen_them(self):
        app = AcprofTui(RunConfig.smoke('demo/model'), settings_path=self.root / 'tui.json')
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press('f2')
            await pilot.pause()
            assert (len(app.query('#open-environment-settings'))) == (1), 'Settings must expose connections and system permissions'
            await self.open_connections(app, pilot)
            token = app.screen.query_one('#env-hf-token', Input)
            assert (token.password)
            show_secrets = app.screen.query_one('#show-environment-secrets', Checkbox)
            assert '✓' not in checkbox_text(show_secrets)
            assert 'X' not in checkbox_text(show_secrets)
            assert await pilot.click(show_secrets)
            await pilot.pause()
            assert checkbox_text(show_secrets).count('✓') == 1
            assert not token.password
            await pilot.press('space')
            await pilot.pause()
            assert '✓' not in checkbox_text(show_secrets)
            assert token.password
            token.focus()
            await pilot.press(*'hf_testonly')
            assert not app.screen.query('#env-hf-endpoint, #env-http-proxy, #env-https-proxy, #env-all-proxy, #env-no-proxy, #env-hf-download-mode')
            assert (await pilot.click('#save-environment-settings'))
            await pilot.pause()
            env = {}
            load_project_env(self.root, environ=env)
            assert (env['HF_TOKEN']) == ('hf_testonly')
            assert (os.environ['HF_TOKEN']) == ('hf_testonly')
            assert 'HTTP_PROXY' not in env
            assert 'HTTPS_PROXY' not in env
            assert not ((self.root / 'tui.json').exists())
            status = app.screen.query_one('#environment-status', Static).content
            assert isinstance(status, str)
            assert ('hf_testonly') not in (status)
            await pilot.press('escape')
            await pilot.pause()
            assert not (app.is_busy())
            await self.open_connections(app, pilot)
            assert (app.screen.query_one('#env-hf-token', Input).value) == ('hf_testonly')

    async def test_close_does_not_save_and_busy_run_blocks_configuration(self):
        app = AcprofTui(RunConfig.smoke('demo/model'), settings_path=self.root / 'tui.json')
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.press('f2')
            await pilot.pause()
            assert (len(app.query('#open-environment-settings'))) == (1)
            await self.open_connections(app, pilot)
            app.screen.query_one('#env-hf-token', Input).value = 'hf_unsaved'
            await pilot.press('escape')
            await pilot.pause()
            assert not ((self.root / '.env.local').exists())
            assert ('HF_TOKEN') not in (os.environ)
            app.process_kind = 'run'
            app.set_busy(True)
            assert (app.query_one('#open-environment-settings', Button).disabled)
            app.process_kind = ''
            app.set_busy(False)
