"""Capture one continuous, fully synthetic Pilot walk in several terminal sizes.

No VHS, Docker, GPU, network or real model execution. SVGs are per-run evidence,
not new golden baselines. The public report may embed these synthetic fixtures.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.tui.workflows.visual_journey import exercise_journey  # noqa: E402

DEFAULT = ROOT / "internal-testing/shared/tui-png/journey"
from scripts.tui_journey_contract import CHECKPOINTS, SCENARIOS  # noqa: E402


async def capture(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    manifest.unlink(missing_ok=True)
    scenarios = []
    for scenario, language, size in SCENARIOS:
        images = output / scenario
        if images.exists():
            shutil.rmtree(images)
        with tempfile.TemporaryDirectory(prefix="acprof-tui-synthetic-") as temporary:
            rows = await exercise_journey(
                workspace=Path(temporary), size=size, language=language, capture=images
            )
        assert {item["id"] for item in rows}.issubset(CHECKPOINTS)
        for row in rows:
            row["checkpoint"] = CHECKPOINTS[row["id"]]
            row["image"] = f"{scenario}/{row['image']}"
            if not (output / row["image"]).is_file():
                raise RuntimeError(f"Pilot did not write the expected SVG: {row['id']}")
        scenarios.append(
            {"id": scenario, "language": language, "terminal": list(size), "steps": rows}
        )
        print(f"{scenario}: {len(rows)} checkpoints, no real subprocess", flush=True)
    document = {
        "schema_version": 1,
        "fixture": "synthetic-pilot",
        "collection_executed": False,
        "scenarios": scenarios,
    }
    temporary_manifest = output / "manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(asyncio.run(capture(args.output.resolve())))


if __name__ == "__main__":
    main()
