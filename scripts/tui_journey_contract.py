"""Small stdlib-only contract for public synthetic Pilot evidence.

Only snapshots created by the fixed test fixture may be displayed. Arbitrary
user paths and actual experiment captures are never accepted by the report.
"""

from __future__ import annotations

import json
from pathlib import Path

SCENARIOS = (
    ("zh-wide", "zh", (120, 36)),
    ("zh-small", "zh", (80, 24)),
    ("en-small", "en", (80, 24)),
)
CHECKPOINTS = {
    "configuration": "configure",
    "launch-review": "launch-review",
    "monitor-started": "prepare",
    "download-review": "download",
    "download-progress": "download",
    "measurement": "measurement",
    "measurement-config": "measurement",
    "finished": "finish",
    "result-overview": "results",
    "results-small": "results",
}


def load_manifest(folder: Path) -> dict | None:
    """Check every file, identity and fixture boundary before public embedding."""
    manifest = folder / "manifest.json"
    if not manifest.exists():
        return None
    if not manifest.is_file() or manifest.stat().st_size > 100_000:
        raise ValueError("invalid synthetic Pilot manifest size")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or set(data)
        != {
            "schema_version",
            "fixture",
            "collection_executed",
            "scenarios",
        }
        or data["schema_version"] != 1
        or data["fixture"] != "synthetic-pilot"
        or data["collection_executed"] is not False
        or not isinstance(data["scenarios"], list)
    ):
        raise ValueError("invalid synthetic Pilot manifest identity")
    expected = {scenario: (language, size) for scenario, language, size in SCENARIOS}
    if len(data["scenarios"]) != len(expected):
        raise ValueError("missing or duplicated synthetic Pilot scenario")
    seen = set()
    for scenario in data["scenarios"]:
        if not isinstance(scenario, dict) or set(scenario) != {
            "id",
            "language",
            "terminal",
            "steps",
        }:
            raise ValueError("invalid synthetic Pilot scenario")
        name = scenario["id"]
        if not isinstance(name, str) or name not in expected or name in seen:
            raise ValueError("unknown/duplicate synthetic Pilot scenario")
        seen.add(name)
        lang, size = expected[name]
        if scenario["language"] != lang or scenario["terminal"] != list(size):
            raise ValueError("synthetic Pilot language/terminal mismatch")
        steps = scenario["steps"]
        if not isinstance(steps, list) or not steps or len(steps) > len(CHECKPOINTS):
            raise ValueError("synthetic Pilot steps missing")
        keys = set()
        for row in steps:
            required = {
                "id",
                "checkpoint",
                "image",
                "action",
                "page",
                "screen",
                "focus",
                "buttons",
                "regions",
                "terminal",
            }
            if (
                not isinstance(row, dict)
                or not required.issubset(row)
                or set(row) - required not in (set(), {"preview"})
            ):
                raise ValueError("synthetic Pilot step schema mismatch")
            step = row["id"]
            if (
                not isinstance(step, str)
                or step not in CHECKPOINTS
                or step in keys
                or row["checkpoint"] != CHECKPOINTS[step]
            ):
                raise ValueError("unknown/duplicate synthetic Pilot step")
            keys.add(step)
            relative = f"{name}/{step}.svg"
            if row["image"] != relative:
                raise ValueError("unsafe synthetic Pilot path")
            path = folder / name / f"{step}.svg"
            if (
                not path.is_file()
                or path.is_symlink()
                or (folder / name).is_symlink()
                or not path.resolve().is_relative_to(folder.resolve())
                or path.stat().st_size > 2_000_000
                or b"<svg" not in path.read_bytes()[:3000]
            ):
                raise ValueError("invalid/missing synthetic Pilot image")
            if "preview" in row:
                preview = f"{name}/{step}.png"
                if row["preview"] != preview:
                    raise ValueError("unsafe synthetic Pilot preview path")
                image = folder / name / f"{step}.png"
                if (
                    not image.is_file()
                    or image.is_symlink()
                    or image.stat().st_size > 2_000_000
                    or not image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
                ):
                    raise ValueError("invalid/missing synthetic Pilot PNG preview")
            if (
                row["terminal"] not in (list(size), [80, 24])
                or row["page"] not in {"run-tab", "monitor-tab", "results-tab"}
                or row["screen"] not in {"CjkScreen", "EnvironmentHostScreen", "RecoveryScreen", "PreparationScreen"}
                or not isinstance(row["action"], str)
                or len(row["action"]) > 180
                or not isinstance(row["regions"], dict)
                or not isinstance(row["buttons"], list)
            ):
                raise ValueError("unsafe synthetic Pilot step metadata")
        if not {
            "configuration",
            "launch-review",
            "download-review",
            "download-progress",
            "measurement",
            "finished",
            "result-overview",
        }.issubset(keys):
            raise ValueError("incomplete synthetic Pilot workflow")
    return data
