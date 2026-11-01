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
