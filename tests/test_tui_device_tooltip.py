"""Device labels stay readable while hover preserves the recorded GPU identity."""
import json

import pytest
from catalog_fixtures import CatalogFixture
from textual.widgets import DataTable, Input, Tooltip

from acprof.tui.experiment_catalog import scan_experiments
from acprof.tui.experiment_picker import SearchPickerScreen, experiment_choice
from tests.test_tui_experiment_picker import PickerHarness

GPU_A = 'GPU-11111111-1111-1111-1111-111111111111'
GPU_B = 'GPU-22222222-2222-2222-2222-222222222222'


@pytest.fixture
def device_choices(request, tmp_path):
    fixture = CatalogFixture()
    fixture.build(request, tmp_path)
    for index, gpu in enumerate((
        {'uuid': GPU_A, 'name': 'NVIDIA GeForce RTX 4090'},
        {'uuid': GPU_B, 'name': 'NVIDIA GeForce RTX 4090'},
        {'uuid': 'GPU-legacy'},
        {},
    )):
        path = fixture.record(str(index), run_id=f'run-{index}', model=f'demo/model-{index}')
        if gpu:
            metadata_path = path / 'static_meta.json'
            metadata = json.loads(metadata_path.read_text())
            metadata['gpu_device'] = gpu
            metadata_path.write_text(json.dumps(metadata))
            state_path = path / 'run_state.json'
            state = json.loads(state_path.read_text())
            state['options']['gpus'] = 'on'
            state_path.write_text(json.dumps(state))
    catalog = scan_experiments([tmp_path])
    assert not catalog.warnings
    return tuple(experiment_choice(record) for record in sorted(catalog.records, key=lambda item: item.model_id))


def cell_offset(table, row, column):
    x = sum(item.get_render_width(table) for item in table.ordered_columns[:column])
    if column >= table.fixed_columns:
        x -= table.scroll_offset.x
    return (x + table.cell_padding, table.header_height + row - table.scroll_offset.y)


@pytest.mark.parametrize('size,language', (((80, 24), 'zh'), ((120, 30), 'en')))
async def test_device_model_and_hover_uuid_follow_rows_and_search(device_choices, size, language):
    app = PickerHarness(language)
    app.TOOLTIP_DELAY = 0.01
    screen = SearchPickerScreen('选择实验', ('模型', '日期', '设备', '状态', 'Run ID'),
        lambda cancelled: (device_choices, ()))
    async with app.run_test(size=size, tooltips=True) as pilot:
        await app.push_screen(screen)
        await app.workers.wait_for_complete()
        await pilot.pause()
        table = screen.query_one('#picker-table', DataTable)
        assert str(table.get_row_at(0)[2]) == 'NVIDIA GeForce RTX 4090'
        assert str(table.get_row_at(1)[2]) == 'NVIDIA GeForce RTX 4090'
        assert str(table.get_row_at(2)[2]) == 'GPU-legacy'
        assert str(table.get_row_at(3)[2]) == 'cpu Fixture CPU'
        tooltip = screen.query_one(Tooltip)
        for row, uuid in ((0, GPU_A), (1, GPU_B)):
            # Move around the native tooltip overlay before hovering another row.
            assert await pilot.hover(table, offset=cell_offset(table, row, 0))
            await pilot.pause()
            assert await pilot.hover(table, offset=cell_offset(table, row, 2))
            await pilot.pause(app.TOOLTIP_DELAY + 0.05)
            assert tooltip.display
            assert str(tooltip.content) == uuid
            assert screen.region.contains_region(tooltip.region)

        # Search must clear an old tooltip and match either the name or the UUID.
        search = screen.query_one('#picker-search', Input)
        search.value = GPU_B
        await pilot.pause()
        assert table.row_count == 1
        assert not tooltip.display
        assert screen.filtered[0].value.device == GPU_B
        assert await pilot.hover(table, offset=cell_offset(table, 0, 2))
        await pilot.pause(app.TOOLTIP_DELAY + 0.05)
        assert tooltip.display
        assert str(tooltip.content) == GPU_B

        # Header, another column, empty rows and leaving the table cannot retain a UUID.
        for target, offset in ((table, cell_offset(table, -1, 2)),
                               (table, cell_offset(table, 0, 0)),
                               (table, cell_offset(table, 2, 2)), (search, (1, 1))):
            assert await pilot.hover(target, offset=offset)
            await pilot.pause(app.TOOLTIP_DELAY + 0.05)
            assert not tooltip.display

        search.value = 'RTX 4090'
        await pilot.pause()
        assert table.row_count == 2
        await pilot.resize_terminal(80, 24)
        table.scroll_to(x=10, animate=False)
        await pilot.pause()
        assert await pilot.hover(table, offset=cell_offset(table, 1, 2))
        await pilot.pause(app.TOOLTIP_DELAY + 0.05)
        assert tooltip.display
        assert str(tooltip.content) == GPU_B
        search.value = 'no-such-device'
        await pilot.pause()
        assert table.row_count == 0
        assert not tooltip.display
