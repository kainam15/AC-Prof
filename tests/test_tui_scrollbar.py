import pytest
from rich.color import Color
from rich.console import Console
from rich.style import Style

from acprof.tui.scrollbar import SolidScrollBarRender


class TestSolidScrollBarRender:
    def render_cells(self, **kwargs):
        rendered = SolidScrollBarRender.render_bar(**kwargs)
        return [segment for segment in Console().render(rendered) if segment.text != "\n"]

    @pytest.mark.parametrize('back_hex,thumb_hex', (('#243746', '#77c5d5'), ('#edf3fa', '#3b5c95')))
    def test_thumb_is_contiguous_and_keeps_click_and_drag_targets(self, back_hex, thumb_hex):
        back, thumb = Color.parse(back_hex), Color.parse(thumb_hex)
        cells = self.render_cells(
            size=20, virtual_size=100, window_size=20,
            position=40, back_color=back, bar_color=thumb,
        )
        assert (len(cells)) == (20)
        actions = [cell.style.meta.get("@mouse.down") for cell in cells]
        assert (actions) == (["scroll_up"] * 8 + ["grab"] * 4 + ["scroll_down"] * 8)
        for cell, action in zip(cells, actions):
            assert (cell.text) == (" ")
            assert not (cell.style.reverse)
            assert (cell.style.bgcolor) == (thumb if action == "grab" else back)

    @pytest.mark.parametrize('position', (-100, 0, 0.125, 20, 40.5, 60, 80, 1000))
    def test_scroll_range_reaches_both_ends_and_clamps_overscroll(self, position):
        starts = []
        cells = self.render_cells(size=20, virtual_size=100, window_size=20, position=position)
        actions = [cell.style.meta.get("@mouse.down") for cell in cells]
        grabbed = [index for index, action in enumerate(actions) if action == "grab"]
        assert (len(cells)) == (20)
        assert (grabbed) == (list(range(grabbed[0], grabbed[-1] + 1)))
        starts.append(grabbed[0])
        if position <= 0:
            assert (grabbed[0]) == (0)
            assert ("scroll_up") not in (actions)
        if position >= 80:
            assert (grabbed[-1]) == (19)
            assert ("scroll_down") not in (actions)
        assert (starts) == (sorted(starts))

    @pytest.mark.parametrize('position_case', range(2), ids=['0', 'virtual - window'])
    @pytest.mark.parametrize('virtual,window', ((100000, 1), (100, 99)))
    @pytest.mark.parametrize('size', (1, 2, 10))
    def test_small_tracks_keep_a_draggable_thumb_and_scroll_direction(self, position_case, virtual, window, size):
        position = tuple((0, virtual - window))[position_case]
        cells = self.render_cells(size=size, virtual_size=virtual, window_size=window, position=position)
        actions = [cell.style.meta.get("@mouse.down") for cell in cells]
        assert ("grab") in (actions)
        assert (len(cells)) == (size)
        if size > 1:
            assert ("scroll_down" if position == 0 else "scroll_up") in (actions)

    @pytest.mark.parametrize('vertical', (True, False))
    def test_horizontal_and_vertical_thickness_and_renderer_integration(self, vertical):
        console = Console(width=12, height=12)
        width, height = (3, 12) if vertical else (12, 3)
        renderer = SolidScrollBarRender(
            virtual_size=60, window_size=20, position=20,
            thickness=3, vertical=vertical, style=Style(color="#ffbb66", bgcolor="#243746"),
        )
        lines = console.render_lines(renderer, console.options.update(width=width, height=height))
        assert (len(lines)) == (height)
        assert (all(sum(segment.cell_length for segment in line) == width for line in lines))
        colors = [[segment.style.bgcolor for segment in line for _ in segment.text] for line in lines]
        if vertical:
            assert (all(len(set(row)) == 1 for row in colors))
            assert (sum(row[0] == Color.parse("#ffbb66") for row in colors)) == (4)
        else:
            assert (all(row == colors[0] for row in colors))
            assert (colors[0].count(Color.parse("#ffbb66"))) == (4)

    @pytest.mark.parametrize('virtual,window', ((100, 100), (100, 101), (0, 0), (100, 0)))
    def test_non_scrollable_or_empty_track_has_no_interactive_thumb(self, virtual, window):
        cells = self.render_cells(size=12, virtual_size=virtual, window_size=window, position=200)
        assert (len(cells)) == (12)
        assert (all(not cell.style.meta for cell in cells))
        assert (all(cell.style.bgcolor == Color.parse("#555555") for cell in cells))
        assert (self.render_cells(size=0)) == ([])
        assert (self.render_cells(size=-1)) == ([])
        assert (self.render_cells(thickness=0)) == ([])
