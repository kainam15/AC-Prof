"""Unrelated UI tests use a ready host without running startup hardware probes.

The startup suite imports the production app and exercises its real workers.
This fixture preserves the startup state machine and configuration invalidation.
"""

from contextlib import asynccontextmanager

from acprof.tui.app import AcprofTui as ProductionTui


class AcprofTui(ProductionTui):
    CSS_PATH = ProductionTui.CSS_PATH

    def _execute_quick_check(self, config, token):
        self._show_quick_check([], "", token)

    @asynccontextmanager
    async def run_test(self, *args, **kwargs):
        async with super().run_test(*args, **kwargs) as pilot:
            await pilot.pause()
            yield pilot
