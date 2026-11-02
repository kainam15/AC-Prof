"""Capture one continuous, fully synthetic Pilot walk in several terminal sizes.

No VHS, Docker, GPU, network or real model execution. SVGs are per-run evidence,
not new golden baselines. The public report may embed these synthetic fixtures.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.tui.workflows.visual_journey import exercise_journey  # noqa: E402

DEFAULT = ROOT / "internal-testing/shared/tui-png/journey"
from scripts.tui_journey_contract import CHECKPOINTS, SCENARIOS  # noqa: E402


async def capture(output: Path, *, png_preview: bool = False) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.json"
    manifest.unlink(missing_ok=True)
    scenarios = []
    for scenario, language, size in SCENARIOS:
        images = output / scenario
        if images.exists():
            shutil.rmtree(images)
        # Unlike pytest's autouse fixtures, this standalone capture command
        # otherwise shares the production measurement socket with real runs.
        # Isolate the *synthetic* Pilot's offline result reads as well.
        with tempfile.TemporaryDirectory(prefix="acprof-tui-synthetic-") as temporary:
            root = Path(temporary)
            namespace = b"acprof.pilot." + hashlib.sha256(str(root).encode()).hexdigest().encode()
            with (
                patch("acprof.host.run_state.MEASUREMENT_LOCK_ROOT", root),
                patch("acprof.host.run_state.MEASUREMENT_LOCK_NAMESPACE", namespace),
            ):
                rows = await exercise_journey(
                    workspace=root, size=size, language=language, capture=images
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
    if png_preview:
        from scripts.tui_pilot_png import capture_pilot_previews

        capture_pilot_previews(output, scenarios)
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
    parser.add_argument(
        "--png-preview",
        action="store_true",
        help="Render Noto CJK PNG previews with existing headless VHS tooling",
    )
    args = parser.parse_args()
    print(asyncio.run(capture(args.output.resolve(), png_preview=args.png_preview)))


if __name__ == "__main__":
    main()
