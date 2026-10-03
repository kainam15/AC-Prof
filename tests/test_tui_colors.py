import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.segment import Segment
from rich.style import Style
from textual.strip import Strip
from tui_fixtures import AcprofTui

from acprof.cli.tui import main


class TestTuiColor:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        temporary = tmp_path
        self.settings_path = Path(str(temporary)) / "tui.json"

    def render_surface(self, app):
        # Inspect emitted terminal codes, not SVG export (which always uses RGB).
        surface = app.get_css_variables()["surface"]
        return Strip([Segment(" ", Style(bgcolor=surface))]).render(app.console)

    @pytest.mark.parametrize('colorterm', (None, '', 'truecolor', '24bit'))
    def test_missing_colorterm_keeps_the_same_rgb_surface_as_vscode(self, colorterm):
        with patch.dict(
            os.environ, {"TERM": "xterm-256color"}, clear=True,
        ):
            if colorterm is not None:
                os.environ["COLORTERM"] = colorterm
            original_environment = dict(os.environ)
            app = AcprofTui(settings_path=self.settings_path)
            assert ("\x1b[48;2;41;43;50m") in (self.render_surface(app))
            assert (dict(os.environ)) == (original_environment)

    def test_256_color_mode_uses_extended_palette_even_in_truecolor_terminal(self):
        with patch.dict(os.environ, {"TERM": "xterm-256color", "COLORTERM": "truecolor"}, clear=True):
            app = AcprofTui(settings_path=self.settings_path, color_system="256")
            assert ("\x1b[48;5;235m") in (self.render_surface(app))
            assert ("48;2;") not in (self.render_surface(app))

    @pytest.mark.parametrize('colorterm,expected', (('', '\x1b[48;5;235m'), ('truecolor', '\x1b[48;2;41;43;50m')))
    def test_auto_mode_retains_terminal_capability_detection(self, colorterm, expected):
        with patch.dict(
            os.environ, {"TERM": "xterm-256color", "COLORTERM": colorterm}, clear=True,
        ):
            app = AcprofTui(settings_path=self.settings_path, color_system="auto")
            assert (expected) in (self.render_surface(app))

    @pytest.mark.parametrize('mode,expected', (('truecolor', '\x1b[48;2;41;43;50m'), ('256', '\x1b[48;5;235m')))
    def test_cli_color_mode_applies_before_rendering(self, mode, expected):
        with patch.dict(
            os.environ, {"TERM": "xterm-256color"}, clear=True,
        ), patch("acprof.tui.app.default_settings_path", return_value=self.settings_path), patch(
            "acprof.tui.app.AcprofTui.run", autospec=True,
        ) as run:
            main(["--color-system", mode])
            run.assert_called_once()
            assert (expected) in (self.render_surface(run.call_args.args[0]))


@pytest.mark.parametrize('no_color', (False, True))
async def test_no_color_still_converts_the_rendered_frame_to_grayscale(no_color):
    environment = {"TERM": "xterm-256color"}
    if no_color:
        environment["NO_COLOR"] = "1"
    with tempfile.TemporaryDirectory() as directory, patch.dict(
        os.environ, environment, clear=True,
    ):
        app = AcprofTui(settings_path=Path(directory) / "tui.json")
        async with app.run_test(size=(105, 27)) as pilot:
            await pilot.pause()
            colors = {
                color.get_truecolor()
                for strip in app.screen._compositor.render_strips()
                for segment in strip if segment.style
                for color in (segment.style.color, segment.style.bgcolor) if color
            }
            assert (colors)
            assert (all(len(set(rgb)) == 1 for rgb in colors)) == (no_color)
