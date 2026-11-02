"""Continuous visual Pilot acceptance; no spawned collectors or downloads."""

from dataclasses import replace
from unittest.mock import patch

import pytest
from textual.widgets import TabbedContent

from acprof.experiments.configuration import RunConfig
from tests.support.tui_fixtures import AcprofTui
from tests.tui.workflows.visual_journey import exercise_journey


@pytest.mark.parametrize(
    ("language", "size"),
    [
        ("zh", (120, 36)),
        ("zh", (80, 24)),
        ("en", (80, 24)),
    ],
)
async def test_complete_pilot_sequence_without_real_collection(tmp_path, language, size):
    steps = await exercise_journey(workspace=tmp_path, size=size, language=language)
    names = [row["id"] for row in steps]
    assert names[:5] == [
        "configuration",
        "launch-review",
        "monitor-started",
        "download-review",
        "download-progress",
    ]
    assert names.index("measurement") < names.index("measurement-config") < names.index("finished")
    assert names.index("finished") < names.index("result-overview")
    assert all(row["terminal"][0] <= size[0] for row in steps)
    assert next(row for row in steps if row["id"] == "launch-review")["screen"] == "RecoveryScreen"
    assert (
        next(row for row in steps if row["id"] == "download-review")["screen"]
        == "PreparationScreen"
    )
    assert next(row for row in steps if row["id"] == "result-overview")["page"] == "results-tab"


async def test_fresh_experiment_starts_without_a_confirmation_modal(tmp_path):
    # This is the actual behavior: only existing output invokes recovery review.
    config = replace(RunConfig.smoke("fixture/model"), output_dir=str(tmp_path / "fresh"))
    app = AcprofTui(config, settings_path=tmp_path / "fresh-settings.json")
    with patch.object(app.process, "execute_command") as execute:
        async with app.run_test(size=(80, 24)) as pilot:
            assert await pilot.click("#start-run")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.screen.id == "_default"
            assert app.query_one("#main-tabs", TabbedContent).active == "monitor-tab"
            execute.assert_called_once()


async def test_pilot_evidence_uses_noto_svg_and_keeps_ansi_for_png(tmp_path):
    frames = tmp_path / "frames"
    await exercise_journey(
        workspace=tmp_path / "workspace", size=(80, 24), language="zh", capture=frames
    )
    svg = (frames / "configuration.svg").read_text(encoding="utf-8")
    ansi = (frames / "configuration.ansi").read_bytes()
    assert 'font-family: "Noto Sans Mono CJK SC", monospace;' in svg
    assert "Fira Code" not in svg
    assert "cdnjs.cloudflare.com" not in svg
    assert b"\x1b[?25l" in ansi
    assert "实验配置".encode() in ansi


async def test_standalone_synthetic_capture_uses_its_own_measurement_lock(tmp_path, monkeypatch):
    """A concurrently running real collector must not block fake Pilot results."""
    from unittest.mock import patch

    from acprof.host import run_state
    from scripts import capture_tui_journey

    external_root = tmp_path / "external-collector"
    external_namespace = b"acprof.production-test-namespace"
    inspected = []

    async def fake_journey(*, workspace, size, language, capture):
        inspected.append(language)
        assert run_state.MEASUREMENT_LOCK_ROOT != external_root
        assert run_state.MEASUREMENT_LOCK_NAMESPACE != external_namespace
        assert run_state.MEASUREMENT_LOCK_NAMESPACE.startswith(b"acprof.pilot.")
        return []  # Only inspect lock isolation, not the full scene contract.

    monkeypatch.setattr(capture_tui_journey, "exercise_journey", fake_journey)
    with (
        patch.object(run_state, "MEASUREMENT_LOCK_ROOT", external_root),
        patch.object(run_state, "MEASUREMENT_LOCK_NAMESPACE", external_namespace),
    ):
        await capture_tui_journey.capture(tmp_path / "synthetic-capture")
        assert run_state.MEASUREMENT_LOCK_ROOT == external_root
        assert run_state.MEASUREMENT_LOCK_NAMESPACE == external_namespace
    assert inspected == ["zh", "zh", "en"]
