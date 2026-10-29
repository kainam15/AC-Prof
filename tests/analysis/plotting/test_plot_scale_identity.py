"""Plot panels and labels preserve distinct recorded input scales."""
import numpy as np
import pandas as pd
import pytest

from acprof.plotting import diagnostics
from acprof.plotting.config import FEASIBILITY_STATE_INDEX


@pytest.fixture
def figures(monkeypatch):
    captured = []
    close = diagnostics.plt.close

    def record(figure=None):
        if hasattr(figure, "axes"):
            captured.append(figure)
        close(figure)

    monkeypatch.setattr(diagnostics.plt, "close", record)
    return captured


@pytest.mark.parametrize("scales", [(1., np.nextafter(1., 2.)), (1e-9, 2e-9), (64., 64.000064)])
def test_pareto_panels_do_not_mix_nearby_scales(scales, figures):
    rows = []
    for index, scale in enumerate(scales):
        for cpu in (1, 2):
            rows.append({"cpu_cores": cpu, "mem_cap_gb": 4, "gpu_mode": "off", "input_scale": scale,
                         "latency_app_p95_s": cpu + 2 * index,
                         "container_attributed_energy_eff_j": 6 - 2 * cpu + 3 * index})
    frame = pd.DataFrame(rows)
    before = frame.copy(deep=True)
    assert diagnostics.plot_latency_energy_pareto(frame, xlabel="scale", out_png=None)
    axes = figures[-1].axes
    assert len(axes) == 2
    for index, axis in enumerate(axes):
        points = [collection.get_offsets().tolist() for collection in axis.collections]
        assert points == [[[1 + 2 * index, 4 + 3 * index]], [[2 + 2 * index, 2 + 3 * index]]]
        assert float(axis.get_title().split(" = ")[1]) == scales[index]
    assert axes[0].get_title() != axes[1].get_title()
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("scales", [(1., np.nextafter(1., 2.)), (1e-9, 2e-9), (64., 64.000064)])
def test_feasibility_cells_do_not_borrow_nearby_success(scales, figures):
    frame = pd.DataFrame([
        {"cpu_cores": 2, "mem_cap_gb": 4, "gpu_mode": "off", "input_scale": scales[0],
         "status": "ok", "error": ""},
        {"cpu_cores": 2, "mem_cap_gb": 4, "gpu_mode": "off", "input_scale": scales[1],
         "status": "error", "error": "runtime failure"},
    ])
    before = frame.copy(deep=True)
    assert diagnostics.plot_resource_feasibility_heatmap(frame, xlabel="scale", out_png=None)
    axis = figures[-1].axes[0]
    assert axis.images[0].get_array().tolist() == [[FEASIBILITY_STATE_INDEX["ok"], FEASIBILITY_STATE_INDEX["error"]]]
    labels = [label.get_text() for label in axis.get_xticklabels()]
    assert [float(label) for label in labels] == list(scales)
    assert labels[0] != labels[1]
    pd.testing.assert_frame_equal(frame, before)
