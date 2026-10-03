from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from acprof.plotting.metrics import plot_metric_overview


class TestOverviewLegend:
    @pytest.fixture(autouse=True)
    def _setup(self, request, tmp_path, monkeypatch):
        self._request = request
        yield
        plt.close("all")

    def _render(self, configs, *, columns=2, figure_note=None):
        df = pd.DataFrame([
            {
                "cpu_cores": cpu,
                "mem_cap_gb": mem,
                "gpu_mode": "on" if gpu_on else "off",
                "input_scale": scale,
                "latency_s": scale / (cpu + mem),
            }
            for cpu, mem, gpu_on in configs
            for scale in (64, 128)
        ])
        with patch.object(plt, "close"):
            plot_metric_overview(
                df,
                panels=(("latency_s", "Latency", "Seconds"),) * columns,
                rows=1,
                columns=columns,
                shared_y_groups=(),
                title="Overview legend regression",
                xlabel="input_scale",
                out_png=None,
                figure_note=figure_note,
            )
        figure = plt.gcf()
        figure.canvas.draw()
        return figure

    def _assert_header_fits(self, figure) -> None:
        renderer = figure.canvas.get_renderer()
        header_boxes = [
            artist.get_window_extent(renderer)
            for artist in [*figure.legends, *figure.texts]
        ]
        for index, box in enumerate(header_boxes):
            assert (box.x0) >= (0)
            assert (box.y0) >= (0)
            assert (box.x1) <= (figure.bbox.x1)
            assert (box.y1) <= (figure.bbox.y1)
            for other in header_boxes[index + 1:]:
                assert not (box.overlaps(other))
            for axis in figure.axes:
                assert not (box.overlaps(axis.get_tightbbox(renderer)))

    @pytest.mark.parametrize('modes', ((True, False), (True,), (False,)))
    def test_complete_grids_align_memory_rows_under_cpu_mode_headings(self, modes):
        configs = [
            (cpu, mem, gpu_on)
            for gpu_on in modes
            for cpu in (1, 2, 4, 8)
            for mem in (2, 4, 8, 16)
        ]
        figure = self._render(configs)
        renderer = figure.canvas.get_renderer()
        legend = figure.legends[0]
        texts = legend.get_texts()
        headings = [text for text in texts if "CPU" in text.get_text()]
        assert ([text.get_text() for text in headings]) == ([
                f"{'GPU+' if gpu_on else ''}CPU{cpu}"
                for gpu_on in modes
                for cpu in (1, 2, 4, 8)
            ])
        for mem in (2, 4, 8, 16):
            cells = [text for text in texts if text.get_text() == f"Mem{mem}"]
            assert (len(cells)) == (len(headings))
            assert (len({round(text.get_window_extent(renderer).y0, 3) for text in cells})) == (1)
        samples = [
            handle for handle, text in zip(legend.get_lines(), texts)
            if text.get_text().startswith("Mem")
        ]
        assert (len(samples)) == (len(configs))
        assert ([line.get_color() for line in samples]) == ([line.get_color() for line in figure.axes[0].lines])
        self._assert_header_fits(figure)
        plt.close(figure)

    def test_sparse_grid_keeps_missing_memory_slots_empty(self):
        figure = self._render([(1, 2, True), (1, 8, True), (4, 4, True)])
        texts = figure.legends[0].get_texts()
        assert ([text.get_text() for text in texts]) == (["GPU+CPU1", "Mem2", "", "Mem8", "GPU+CPU4", "", "Mem4", ""])
        assert (len(figure.axes[0].lines)) == (3)
        self._assert_header_fits(figure)

    def test_wrapped_groups_and_note_stay_above_the_panels(self):
        configs = [
            (cpu, mem, gpu_on)
            for gpu_on in (True, False)
            for cpu in (1, 2, 4, 8, 16)
            for mem in (2, 4, 8, 16, 32, 64)
        ]
        figure = self._render(
            configs,
            columns=1,
            figure_note="Measured CPU package idle: CPU-only 5.00 W\nGPU-enabled 8.00 W",
        )
        texts = figure.legends[0].get_texts()
        assert (sum(text.get_text().startswith("Mem") for text in texts)) == (len(configs))
        self._assert_header_fits(figure)
