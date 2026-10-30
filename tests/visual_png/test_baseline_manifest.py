"""Guard curated baseline coverage and CI's selective automatic trigger."""

from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image

from scripts.compare_tui_png import BASELINES
from tests.visual_png.scenarios import SCENARIOS

REPO = Path(__file__).resolve().parents[2]


def test_all_scenes_have_git_managed_golden_images() -> None:
    paths = {path.stem: path for path in BASELINES.glob("*.png")}
    assert set(paths) == set(SCENARIOS)
    for scene, path in paths.items():
        with Image.open(path) as image:
            image.load()
            assert image.format == "PNG"
            assert image.width > SCENARIOS[scene].columns * 7
            assert image.height > SCENARIOS[scene].rows * 12
            assert len(image.getcolors(maxcolors=1_000_000) or []) > 50


def test_ci_selective_png_gate_remains_enabled() -> None:
    # BaseLoader preserves YAML "on"; PyYAML's YAML 1.1 loader interprets
    # it as boolean True, unlike the GitHub Actions workflow parser.
    workflow = yaml.load(
        (REPO / ".github" / "workflows" / "tui-png.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    triggers = workflow["on"]
    assert set(triggers) == {"push", "pull_request", "workflow_dispatch"}
    assert triggers["push"]["branches"] == ["main"]
    for trigger in ("push", "pull_request"):
        patterns = set(triggers[trigger]["paths"])
        assert "acprof/tui/**" in patterns
        assert "tests/visual_png/**" in patterns
        assert ".github/workflows/tui-png.yml" in patterns
    job = workflow["jobs"]["png-visual"]
    step = next(
        step for step in job["steps"] if step.get("name") == "Capture and check real TUI scenes"
    )
    assert step["env"]["ACPROF_PNG_VISUAL"] == "1"
