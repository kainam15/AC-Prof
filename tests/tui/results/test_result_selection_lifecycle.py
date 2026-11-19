"""Late result-selection messages must not outlive the mounted form."""

import pytest
from textual.widgets import Input

from tests.support.tui_fixtures import AcprofTui


@pytest.mark.parametrize("source_id", ["experiment-source"])
async def test_result_selection_message_is_safe_during_partial_unmount(tmp_path, source_id):
    class ClosingSelectionApp(AcprofTui):
        CSS_PATH = AcprofTui.CSS_PATH

        async def _close_all(self):
            # Textual stops the app before pruning widgets, then dispatches
            # App.Unmount. Replay a queued message inside that interval.
            source = self.query_one(f"#{source_id}", Input)
            with self.prevent(Input.Changed):
                source.value = str(tmp_path / "late.csv")
            await self.query_one("#experiment-source").remove()
            try:
                assert not self.is_running
                assert self.form_ready
                assert not self.ui_closing
                await self._dispatch_message(Input.Changed(source, source.value))
                assert self.results.source_path is None
                assert self.results.selection_timer is None
                assert not self.read_jobs
            finally:
                await super()._close_all()

    app = ClosingSelectionApp(settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(120, 30)):
        assert app.form_ready


@pytest.mark.parametrize("source_id", ["experiment-source"])
async def test_result_selection_message_does_not_restart_reads_after_quit(tmp_path, source_id):
    app = AcprofTui(settings_path=tmp_path / "settings.json")
    async with app.run_test(size=(80, 24)):
        source = app.query_one(f"#{source_id}", Input)
        with app.prevent(Input.Changed):
            source.value = str(tmp_path / "late.csv")
        app.action_request_quit()
        assert app.is_running
        assert app.ui_closing
        await app._dispatch_message(Input.Changed(source, source.value))
        assert app.results.source_path is None
        assert app.results.selection_timer is None
        assert not app.read_jobs
