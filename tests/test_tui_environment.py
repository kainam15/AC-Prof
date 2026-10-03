"""Connection settings use an isolated project and never send notifications."""
import os
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Button, Checkbox, Input, Static, TabbedContent
from tui_fixtures import AcprofTui

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
                        patch.dict(os.environ, {'PATH': os.environ.get('PATH', '')}, clear=True)):
            context.start()
            self._request.addfinalizer(partial(context.stop))

    async def open_connections(self, app, pilot):
        # Keyboard focus scrolls the last settings section into view on short terminals.
        button = app.query_one('#open-environment-settings', Button)
        button.focus()
        await pilot.pause()
        assert (await pilot.click(button))
        await pilot.pause()

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
            assert not (app._is_busy())

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


    @pytest.mark.parametrize('language', ('zh', 'en'))
    @pytest.mark.parametrize('size', ((80, 24), (120, 30), (150, 45)))
    async def test_bilingual_dialog_actions_are_reachable_at_supported_sizes(self, language, size):
        app = AcprofTui(RunConfig.smoke('demo/model'), settings_path=self.root / 'tui.json')
        async with app.run_test(size=(150, 45)) as pilot:
            await pilot.resize_terminal(*size)
            app.ui_preferences = replace(app.ui_preferences, language=language)
            app._apply_ui_preferences()
            await pilot.press('f2')
            await pilot.pause()
            await self.open_connections(app, pilot)
            for widget_id in ('close-environment-settings', 'save-environment-settings'):
                widget = app.screen.query_one('#' + widget_id, Button)
                assert (widget.region.width and widget.region.height)
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
            token.focus()
            await pilot.press(*'hf_testonly')
            app.screen.query_one('#env-hf-endpoint', Input).value = 'https://hub.example'
            assert (await pilot.click('#save-environment-settings'))
            await pilot.pause()
            env = {}
            load_project_env(self.root, environ=env)
            assert (env['HF_TOKEN']) == ('hf_testonly')
            assert (os.environ['HF_TOKEN']) == ('hf_testonly')
            assert (env['HF_ENDPOINT']) == ('https://hub.example')
            assert not ((self.root / 'tui.json').exists())
            status = app.screen.query_one('#environment-status', Static).content
            assert isinstance(status, str)
            assert ('hf_testonly') not in (status)
            await pilot.press('escape')
            await pilot.pause()
            assert not (app._is_busy())
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
            app._process_kind = 'run'
            app._set_busy(True)
            assert (app.query_one('#open-environment-settings', Button).disabled)
            app._process_kind = ''
            app._set_busy(False)
