"""Responsive classes change atomically without clobbering unrelated styles."""

from __future__ import annotations

from textual.geometry import Size

from acprof.experiments.configuration import RunConfig
from acprof.tui.app import AcprofTui


def test_responsive_classes_batch_changes_and_preserve_theme(tmp_path, monkeypatch) -> None:
    app = AcprofTui(RunConfig.smoke("demo/model"), settings_path=tmp_path / "settings.json")
    app.add_class("test-extra", update=False)
    observed: list[bool] = []
    monkeypatch.setattr(app, "update_node_styles", lambda animate=True: observed.append(animate))

    app.update_responsive_layout(Size(80, 24))
    assert len(observed) == 1
    assert all(app.has_class(name) for name in ("narrow", "compact-navigation", "short"))
    assert app.has_class("test-extra")

    app.update_responsive_layout(Size(80, 24))
    assert len(observed) == 1  # identical size has no stylesheet update

    app.update_responsive_layout(Size(120, 36))
    assert len(observed) == 2
    assert app.has_class("narrow")  # 120 columns minus 22 for the sidebar
    assert not app.has_class("compact-navigation")
    assert not app.has_class("short")
    assert app.has_class("test-extra")

    app.update_responsive_layout(Size(160, 48))
    assert len(observed) == 3
    assert not app.has_class("narrow")
    assert app.has_class("test-extra")
