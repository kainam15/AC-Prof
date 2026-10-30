"""Real multi-task selection overlay with pinned metadata, no model resolution."""

from __future__ import annotations

from acprof.experiments.configuration import RunConfig
from acprof.tui.app import AcprofTui
from acprof.tui.run.preparation import PreparationScreen
from tests.support.task_selection_fixtures import t5_profiles
from tests.visual_png.scenarios.common import freeze_for_screenshot


def initial_config() -> RunConfig:
    return RunConfig.smoke("google-t5/t5-small")


def mount(app: AcprofTui) -> None:
    profiles = t5_profiles()
    event = {
        "stage": "resolution",
        "status": "waiting",
        "request": {
            "id": 1,
            "kind": "review",
            "resolved": False,
            "fields": {"Model": "google-t5/t5-small", "Revision": "fixture-revision"},
            "questions": [
                {
                    "path": "tasks",
                    "kind": "multi",
                    "value": [],
                    "options": [item["task_id"] for item in profiles],
                    "profiles": profiles,
                }
            ],
        },
    }
    freeze_for_screenshot(app)
    app.push_screen(PreparationScreen(event))
