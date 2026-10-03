"""通过实际表头字符、鼠标事件和滚动范围验证列宽拖动。"""
import pytest
from rich.cells import cell_len
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.widgets import Static

from acprof.tui.images import ImageTable
from acprof.tui.table import ResizableDataTable


class TableApp(App):
    table_type = ResizableDataTable
    CSS = """
    ResizableDataTable { width: 52; height: 10; margin: 1 2; padding: 1 2; border: solid $primary; }
    """

    def __init__(self):
        super().__init__()
        self.headers = []
        self.selected = []

    def compose(self) -> ComposeResult:
        yield self.table_type(id="table", fixed_columns=1, header_height=2, cursor_type="row")
        yield Static("outside", id="outside")

    def on_mount(self):
        table = self.query_one(ResizableDataTable)
        table.add_column("中文名称", width=10, key="name")
        table.add_column("Size", width=12, key="size")
        table.add_column("Description", width=30, key="description")
        for index in range(20):
            table.add_row("模型名称较长", str(index), "details " * 8, key=str(index))

    def on_data_table_header_selected(self, event):
        self.headers.append(event.column_key.value)

    def on_data_table_row_selected(self, event):
        self.selected.append(event.row_key.value)


def header_offset(table, boundary_index=0, y=0):
    text = table.render_line(y).text
    # 从渲染的分隔符定位，不使用被测控件的命中判断或元数据。
    boundaries = [cell_len(text[:index]) for index, char in enumerate(text) if char == "│"]
    inset = table.content_region.offset - table.region.offset
    return inset.x + boundaries[boundary_index], inset.y + y


async def drag(pilot, table, start, dx, dy=0, *, release_click=False):
    await pilot.mouse_down(table, offset=start)
    end = (table.region.x + start[0] + dx, table.region.y + start[1] + dy)
    await pilot.hover(offset=end)
    await pilot.mouse_up(offset=end)
    # Pilot.mouse_up 不像真实驱动那样生成释放后的 Click。
    if release_click:
        await pilot._post_mouse_events([events.Click], offset=end, button=1)
    await pilot.pause()


class TestResizableTable:
    async def test_real_header_click_preserves_native_action_dispatch(self):
        class HeaderActionTable(ResizableDataTable):
            header_actions = 0

            def action_header_click(self):
                self.header_actions += 1

        app = TableApp()
        app.table_type = HeaderActionTable
        async with app.run_test(size=(80, 24)) as pilot:
            table = app.query_one(HeaderActionTable)
            table.clear(columns=True)
            table.add_column(Text.from_markup("[@click=header_click]Action[/]"), width=10, key="name")
            await pilot.pause()
            inset = table.content_region.offset - table.region.offset
            assert (await pilot.click(table, offset=(inset.x + 2, inset.y)))
            assert (app.headers) == (["name"])
            assert (table.header_actions) == (1)

    @pytest.mark.parametrize('table_type_case', range(2), ids=['ResizableDataTable', 'ImageTable'])
    async def test_empty_header_click_does_not_exit_or_select(self, table_type_case):
        table_type = tuple((ResizableDataTable, ImageTable))[table_type_case]
        app = TableApp()
        app.table_type = table_type
        async with app.run_test(size=(80, 24)) as pilot:
            table = app.query_one(ResizableDataTable)
            table.clear(columns=True)
            await pilot.pause()
            inset = table.content_region.offset - table.region.offset
            for x, y in ((1, 0), (38, 0), (38, 1)):
                assert (await pilot.click(table, offset=(inset.x + x, inset.y + y)))
            assert (app.is_running)
            assert (app.mouse_captured) is None
            assert (app.headers) == ([])
            assert (app.selected) == ([])

    @pytest.mark.parametrize('table_type_case', range(2), ids=['ResizableDataTable', 'ImageTable'])
    async def test_header_filler_is_ignored_but_real_header_and_row_filler_work(self, table_type_case):
        table_type = tuple((ResizableDataTable, ImageTable))[table_type_case]
        app = TableApp()
        app.table_type = table_type
        async with app.run_test(size=(80, 24)) as pilot:
            table = app.query_one(ResizableDataTable)
            table.clear(columns=True)
            table.add_column("名称", width=10, key="name")
            table.add_row("第一个", key="first")
            table.add_row("第二个", key="second")
            await pilot.pause()
            inset = table.content_region.offset - table.region.offset
            await pilot.click(table, offset=(inset.x + 38, inset.y))
            assert (app.headers) == ([]), "表头右侧留白不能误选第 0 列"
            await pilot.click(table, offset=(inset.x + 2, inset.y))
            assert (app.headers) == (["name"])
            await pilot.click(table, offset=(inset.x + 38, inset.y + table.header_height + 1))
            assert (table.cursor_row) == (1), "行尾留白仍应定位到对应行"
            assert (app.selected) == ([]), "镜像行尾不能误触勾选"
            await pilot.press("enter")
            assert (app.selected) == (["second"])
            assert (app.mouse_captured) is None

    @pytest.mark.parametrize('table_type_case', range(2), ids=['ResizableDataTable', 'ImageTable'])
    async def test_queued_header_click_ignores_a_removed_column(self, table_type_case):
        table_type = tuple((ResizableDataTable, ImageTable))[table_type_case]
        app = TableApp()
        app.table_type = table_type
        async with app.run_test(size=(80, 24)) as pilot:
            table = app.query_one(ResizableDataTable)
            await pilot.pause()
            screen_x, screen_y = table.content_region.x + 14, table.content_region.y
            old_style = app.screen.get_style_at(screen_x, screen_y)
            assert (old_style.meta["column"]) == (1)
            assert not (old_style.meta.get("out_of_bounds"))
            table.clear(columns=True)
            table.add_column("新列", width=10, key="new")
            table.add_row("新数据", key="new-row")
            await pilot.pause()
            # 模拟渲染后、事件处理前列已被重建，保留实际旧表头的元数据。
            table.post_message(events.Click(
                table, 14, 0, 0, 0, 1, False, False, False,
                screen_x=screen_x, screen_y=screen_y, style=old_style,
            ))
            await pilot.pause()
            assert (app.is_running)
            assert (app.headers) == ([])
            inset = table.content_region.offset - table.region.offset
            await pilot.click(table, offset=(inset.x + 2, inset.y))
            assert (app.headers) == (["new"])

    @pytest.mark.parametrize('layout', ('wide', 'scrolled', 'single'))
    async def test_last_header_edge_cannot_resize_and_keeps_header_clicks(self, layout):
        app = TableApp()
        async with app.run_test(size=(80, 24)) as pilot:
            table = app.query_one(ResizableDataTable)
            if layout == "wide":
                table.styles.width = 72
            elif layout == "single":
                table.clear(columns=True)
                table.add_column("中文名称", width=10, key="name")
                table.add_row("模型名称")
            await pilot.pause()
            if layout == "scrolled":
                table.scroll_to(x=table.max_scroll_x, animate=False, force=True)
                await pilot.pause()
            columns = table.ordered_columns
            widths = tuple(column.get_render_width(table) for column in columns)
            edge = sum(widths) - 1 - table.scroll_offset.x
            inset = table.content_region.offset - table.region.offset
            start = (inset.x + edge, inset.y)
            await pilot.mouse_down(table, offset=start)
            assert (app.mouse_captured) is None, "末列右沿不能捕获鼠标调整宽度"
            await pilot.hover(table, offset=(start[0] - 4, start[1]))
            await pilot.mouse_up(table, offset=(start[0] - 4, start[1]))
            await pilot.pause()
            assert (tuple(column.get_render_width(table) for column in columns)) == (widths)
            assert (table.render_line(0).crop(edge, edge + 1).text) != ("│")
            if layout == "single":
                assert ("│") not in (table.render_line(0).text)
            await pilot.click(table, offset=start)
            assert (app.headers) == ([columns[-1].key.value])

    async def test_drag_updates_rendered_cells_and_scroll_extent(self):
        app = TableApp()
        async with app.run_test(size=(80, 24)) as pilot:
            table = app.query_one(ResizableDataTable)
            await pilot.pause()
            before = table.virtual_size.width
            scroll = table.max_scroll_x
            start = header_offset(table, y=1)
            await drag(pilot, table, start, 6, release_click=True)
            assert (table.columns["name"].width) == (16)
            assert (header_offset(table)[0]) == (start[0] + 6)
            assert (table.virtual_size.width) == (before + 6)
            assert (table.max_scroll_x) == (scroll + 6)
            assert ("模型名称较长") in (table.render_line(2).text)
            assert not (app.headers)
            assert not (app.selected)
            assert (app.mouse_captured) is None

    async def test_scrolled_header_and_fixed_column_use_visible_boundaries(self):
        app = TableApp()
        async with app.run_test(size=(80, 24)) as pilot:
            table = app.query_one(ResizableDataTable)
            await pilot.pause()
            fixed = header_offset(table)
            second = header_offset(table, 1)
            table.scroll_to(x=5, y=6, animate=False, force=True)
            await pilot.pause()
            assert (header_offset(table)) == (fixed)
            assert (header_offset(table, 1)[0]) == (second[0] - 5)
            await drag(pilot, table, header_offset(table, 1), 4)
            assert (table.columns["size"].width) == (16)
            assert (table.columns["name"].width) == (10)
            assert (table.scroll_offset.x) == (5)
            assert (table.scroll_offset.y) == (6)
            await drag(pilot, table, header_offset(table), -3)
            assert (table.columns["name"].width) == (7)
            assert (table.columns["size"].width) == (16)

    async def test_drag_outside_table_clamps_width_and_releases_capture(self):
        app = TableApp()
        async with app.run_test(size=(80, 24)) as pilot:
            table = app.query_one(ResizableDataTable)
            await pilot.pause()
            start = header_offset(table)
            await pilot.mouse_down(table, offset=start)
            assert (app.mouse_captured) is (table)
            await pilot.hover(offset=(0, 20))
            await pilot.mouse_up(offset=(0, 20))
            assert (table.columns["name"].width) == (1)
            assert (app.mouse_captured) is None
            await pilot.hover(table, offset=start)
            assert (table.columns["name"].width) == (1)
            await drag(pilot, table, header_offset(table), 8)
            assert (table.columns["name"].width) == (9)

    async def test_body_right_button_and_plain_header_clicks_keep_normal_behavior(self):
        app = TableApp()
        async with app.run_test(size=(80, 24)) as pilot:
            table = app.query_one(ResizableDataTable)
            await pilot.pause()
            start = header_offset(table)
            await drag(pilot, table, (start[0], start[1] + 2), 4)
            assert (table.columns["name"].width) == (10)
            await pilot.mouse_down(table, offset=start, button=3)
            assert (app.mouse_captured) is None
            await pilot.hover(table, offset=(start[0] + 4, start[1]))
            assert (table.columns["name"].width) == (10)
            await pilot.click(table, offset=start)
            assert not (app.headers), "边界点击不能排序"
            await pilot.click(table, offset=(start[0] - 5, start[1]))
            assert (app.headers) == (["name"])

    @pytest.mark.parametrize('interruption', ('disabled', 'locked', 'hidden', 'clear', 'capture_lost'))
    async def test_interrupted_drag_cannot_leave_mouse_captured(self, interruption):
        app = TableApp()
        async with app.run_test(size=(80, 24)) as pilot:
            table = app.query_one(ResizableDataTable)
            await pilot.pause()
            await pilot.mouse_down(table, offset=header_offset(table))
            assert (app.mouse_captured) is (table)
            if interruption == "disabled":
                table.disabled = True
            elif interruption == "locked":
                table.resize_enabled = False
            elif interruption == "hidden":
                table.display = False
            elif interruption == "clear":
                table.clear(columns=True)
            else:
                table.release_mouse()
            await pilot.pause()
            assert (app.mouse_captured) is None
            await pilot.mouse_up(offset=(70, 20))
            table.disabled = False
            table.resize_enabled = True
            table.display = True
            if interruption == "clear":
                table.add_column("中文名称", width=10, key="name")
                table.add_column("Size", width=12, key="size")
            await pilot.pause()
            await pilot.hover(table, offset=header_offset(table))
            assert (table.columns["name"].width) == (10)
            await drag(pilot, table, header_offset(table), 2)
            assert (table.columns["name"].width) == (12)

    async def test_auto_width_becomes_manual_and_survives_rebuild_by_key(self):
        app = TableApp()
        async with app.run_test(size=(80, 24)) as pilot:
            table = app.query_one(ResizableDataTable)
            table.clear(columns=True)
            table.add_column("中文名称", key="name")
            table.add_column("Size", width=4, key="size")
            table.add_row("模型名称较长", "1")
            await pilot.pause()
            await drag(pilot, table, header_offset(table), -4)
            assert (table.columns["name"].width) == (8)
            table.clear(columns=True)
            table.add_column("Name", key="name", width=20)
            table.add_column("Size", width=4, key="size")
            table.add_row("much longer content than the manually sized column", "1")
            await pilot.pause()
            assert (table.columns["name"].width) == (8)
            assert (table.virtual_size.width) == (16)
            assert (len(table.columns)) == (2)

    async def test_oversized_fixed_column_can_be_shrunk_after_terminal_resize(self):
        app = TableApp()
        async with app.run_test(size=(80, 24)) as pilot:
            table = app.query_one(ResizableDataTable)
            await pilot.pause()
            await drag(pilot, table, header_offset(table), 26)
            assert (table.columns["name"].width) == (36)
            table.styles.width = 30
            await pilot.resize_terminal(60, 24)
            await pilot.pause()
            assert ("│") in (table.render_line(0).text), "固定列超出视口时，仍需提供可抓取的边界"
            start = header_offset(table)
            await drag(pilot, table, start, -5)
            assert (table.columns["name"].width) < (table.scrollable_content_region.width)
            assert (header_offset(table)[0]) == (start[0] - 5)
            assert (app.mouse_captured) is None
