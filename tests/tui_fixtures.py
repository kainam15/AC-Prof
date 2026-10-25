"""Unrelated UI tests use a ready host without running startup hardware probes.

The startup suite imports the production app and exercises its real workers.
This fixture preserves the startup state machine and configuration invalidation.
"""

from contextlib import asynccontextmanager
from unittest.mock import patch

from acprof.tui.app import AcprofTui as ProductionTui


class AcprofTui(ProductionTui):
    CSS_PATH = ProductionTui.CSS_PATH

    @asynccontextmanager
    async def run_test(self, *args, **kwargs):
        with patch.object(
            self.preflight, "execute_quick_check",
            side_effect=lambda config, token: self.preflight.show_quick_check([], "", token),
        ):
            async with super().run_test(*args, **kwargs) as pilot:
                await pilot.pause()
                yield pilot
