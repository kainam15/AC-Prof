"""A plot refresh may replace a PNG name without changing any input it aliases."""
import json
import os
import stat
from pathlib import Path

import pytest
from matplotlib import rc_context
from matplotlib.figure import Figure
from PIL import Image

from acprof.cli.plot import main
from acprof.config import STATIC_META_SCHEMA_VERSION
from acprof.plotting.data import prepare_df
from acprof.plotting.metrics import plot_metric


@pytest.fixture
def process_result(tmp_path):
    source = tmp_path / "result_all.csv"
    source.write_text(
        "cpu_cores,mem_cap_gb,gpu_mode,input_scale,warmup,repeat_idx,status,latency_app_s\n"
        "2,8,off,64,0,0,ok,0.1\n"
    )
    (tmp_path / "static_meta.json").write_text(json.dumps({
        "schema_version": STATIC_META_SCHEMA_VERSION, "runtime_kind": "process",
    }))
    target = tmp_path / "process" / "input_scaling.png"
    target.parent.mkdir()
    return source, target


@pytest.mark.parametrize("link", ["hardlink", "symlink"])
def test_plot_replaces_only_the_png_entry_when_it_aliases_measurements(process_result, link):
    source, target = process_result
    if link == "hardlink":
        os.link(source, target)
    else:
        target.symlink_to(source)
    original = source.read_bytes()
    assert main([str(source)]) == 0
    assert source.read_bytes() == original
    assert not target.samefile(source)
    with Image.open(target) as image:
        assert image.format == "PNG"


def test_failed_render_preserves_the_previous_png_and_cleans_staging(process_result, monkeypatch):
    source, target = process_result
    target.write_bytes(b"prior plot")
    original = source.read_bytes()

    def interrupted(_figure, destination, **_kwargs):
        if isinstance(destination, (str, os.PathLike)):
            Path(destination).write_bytes(b"partial image")
        else:
            destination.write(b"partial image")
        raise OSError("renderer failed after writing")

    with monkeypatch.context() as failure:
        failure.setattr(Figure, "savefig", interrupted)
        with pytest.raises(OSError, match="renderer failed after writing"):
            main([str(source)])
    assert target.read_bytes() == b"prior plot"
    assert list(target.parent.iterdir()) == [target]
    assert source.read_bytes() == original
    target.chmod(0o640)
    assert main([str(source)]) == 0
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    with Image.open(target) as image:
        assert image.format == "PNG"


@pytest.mark.parametrize("name", ["latency.png", "latency"])
def test_metric_plot_keeps_explicit_dpi_and_filename_format(process_result, name):
    source, _ = process_result
    target = source.with_name(name)
    with rc_context({"savefig.dpi": 72, "savefig.format": "svg"}):
        plot_metric(prepare_df(str(source)), "latency_app_s", "Latency", "s", "scale", str(target))
    if target.suffix == ".png":
        with Image.open(target) as image:
            assert image.format == "PNG"
            assert image.info["dpi"] == pytest.approx((200, 200), abs=0.02)
    else:
        assert not target.exists()
        assert b"<svg" in target.with_suffix(".svg").read_bytes()
