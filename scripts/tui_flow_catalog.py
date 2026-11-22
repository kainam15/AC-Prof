"""Public-fixture-only scene index for TUI workflow and visual evidence.

This is developer tooling, not an AC-Prof runtime dependency. Never discover or
embed live run directories, logs, user settings or model credentials.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Collection
from pathlib import Path

from scripts.tui_journey_contract import CHECKPOINTS, load_manifest

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "tests" / "visual" / "flow_catalog.json"
SVG_DIR = Path("tests/visual/__snapshots__")
PNG_DIR = Path("tests/visual_png/__snapshots__")
TEST_DIR = Path("tests/tui")
SAFE_ID = re.compile(r"[a-z][a-z0-9-]*\Z")
GROUPS = {"main", "optional", "exception", "support"}


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(f"Invalid TUI flow catalog: {reason}")


def _string(value: object, label: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{label} must be nonempty text")
    return str(value)


def _source(
    root: Path, raw: object, prefix: Path, suffix: str, *, require_file: bool = True
) -> Path:
    name = _string(raw, "source")
    path = Path(name)
    _require(
        not path.is_absolute() and ".." not in path.parts and path.suffix == suffix,
        f"invalid source path {name!r}",
    )
    _require(path.parts[: len(prefix.parts)] == prefix.parts, f"source outside {prefix}: {name!r}")
    destination = (root / path).resolve()
    _require(
        destination.is_relative_to((root / prefix).resolve()), f"source escaped {prefix}: {name!r}"
    )
    if require_file:
        _require(destination.is_file(), f"missing source {name!r}")
    return destination


def load_catalog(*, source: Path = CATALOG, root: Path = ROOT, require_baselines: bool = True) -> dict:
    """Validate explicit references; unmatched new SVGs are discovered automatically."""
    data = json.loads(source.read_text(encoding="utf-8"))
    _require(
        isinstance(data, dict) and set(data) == {"schema_version", "pages", "checkpoints"},
        "expected schema_version, pages and checkpoints",
    )
    _require(data["schema_version"] == 1, "unsupported schema version")
    pages, checkpoints = data["pages"], data["checkpoints"]
    _require(isinstance(pages, list) and bool(pages), "pages must be nonempty")
    _require(isinstance(checkpoints, list) and bool(checkpoints), "checkpoints must be nonempty")
    page_ids: set[str] = set()
    for page in pages:
        _require(
            isinstance(page, dict)
            and {"id", "title", "purpose"}.issubset(page)
            and set(page) <= {"id", "title", "purpose", "tab_label"},
            "invalid page",
        )
        page_id = _string(page["id"], "page id")
        _require(
            bool(SAFE_ID.fullmatch(page_id)) and page_id not in page_ids,
            f"duplicate/invalid page: {page_id}",
        )
        page_ids.add(page_id)
        _string(page["title"], "page title")
        _string(page["purpose"], "page purpose")
        if "tab_label" in page:
            _string(page["tab_label"], "page tab label")
    _require(
        page_ids == {
            "run-tab", "monitor-tab", "results-tab", "images-tab", "settings-tab"
        },
        "page catalog must match actual TUI tabs",
    )

    seen_steps: set[str] = set()
    seen_png: set[str] = set()
    seen_svg: set[str] = set()
    for item in checkpoints:
        _require(
            isinstance(item, dict)
            and set(item) - {"pilot"}
            == {
                "id",
                "title",
                "page",
                "group",
                "trigger",
                "expected",
                "png",
                "svg",
                "tests",
            },
            "invalid checkpoint",
        )
        checkpoint = _string(item["id"], "checkpoint id")
        _require(
            bool(SAFE_ID.fullmatch(checkpoint)) and checkpoint not in seen_steps,
            f"duplicate/invalid checkpoint: {checkpoint}",
        )
        seen_steps.add(checkpoint)
        _require(item["page"] in page_ids, f"unknown page for {checkpoint}")
        _require(item["group"] in GROUPS, f"unknown group for {checkpoint}")
        for key in ("title", "trigger", "expected"):
            _string(item[key], f"{checkpoint}.{key}")
        for key in ("png", "svg", "tests"):
            _require(isinstance(item[key], list), f"{checkpoint}.{key} must be a list")
        _require(bool(item["tests"]), f"{checkpoint}: missing interaction evidence references")
        pilot = item.get("pilot", [])
        _require(
            isinstance(pilot, list)
            and all(isinstance(step, str) and step in CHECKPOINTS for step in pilot)
            and len(pilot) == len(set(pilot)),
            f"{checkpoint}: unknown or duplicate Pilot step",
        )
        for step in pilot:
            _require(
                CHECKPOINTS[step] == checkpoint or (checkpoint, step) == ("recover", "launch-review"),
                f"{checkpoint}: unrelated Pilot step {step}",
            )
        for png in item["png"]:
            name = _string(png, "PNG scene")
            _require(bool(SAFE_ID.fullmatch(name)), f"invalid PNG scene {name!r}")
            _require(name not in seen_png, f"PNG scene mapped multiple times: {name}")
            seen_png.add(name)
            _source(root, str(PNG_DIR / f"{name}.png"), PNG_DIR, ".png", require_file=require_baselines)
        for svg in item["svg"]:
            path = _source(root, svg, SVG_DIR, ".svg", require_file=require_baselines)
            relative = path.relative_to(root.resolve()).as_posix()
            _require(relative not in seen_svg, f"SVG mapped multiple times: {relative}")
            seen_svg.add(relative)
        for test in item["tests"]:
            _source(root, test, TEST_DIR, ".py")
    return data


def missing_visual_evidence(
    item: dict, *, captured: Collection[str], svg: Collection[str], png: Collection[str]
) -> list[str]:
    if not (item["svg"] or item["png"] or item.get("pilot")):
        return ["未登记视觉证据"]
    return (
        [step for step in item.get("pilot", []) if step not in captured]
        + [name for name in item["svg"] if name not in svg]
        + [name for name in item["png"] if name not in png]
    )


def check_visual_coverage(catalog: dict, manifest: dict | None, *, root: Path = ROOT) -> None:
    """CI requires the promised frames to actually exist in the validated capture."""
    captured = {
        step["id"]
        for scenario in (manifest or {}).get("scenarios", [])
        for step in scenario["steps"]
    }
    svg = set(svg_inventory(root=root))
    png = set(png_inventory(root=root))
    gaps = []
    for item in catalog["checkpoints"]:
        missing = missing_visual_evidence(item, captured=captured, svg=svg, png=png)
        if missing:
            gaps.append(f"{item['id']}: {', '.join(missing)}")
    if gaps:
        raise ValueError("TUI flow coverage gaps:\n" + "\n".join(gaps))


def svg_inventory(*, root: Path = ROOT) -> list[str]:
    """Discover Git-tracked SVGs only; never publish untracked local screenshots."""
    try:
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--", SVG_DIR.as_posix()],
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        tracked = None
    if tracked is not None and tracked.returncode == 0:
        return sorted(
            name
            for raw in tracked.stdout.split(bytes([0]))
            if (name := raw.decode("utf-8")) and name.endswith(".svg") and (root / name).is_file()
        )
    # In a Git-less source export, display only explicitly reviewed fixture paths.
    steps = load_catalog(root=root, require_baselines=False)["checkpoints"]
    return sorted({name for step in steps for name in step["svg"] if (root / name).is_file()})


def png_inventory(*, root: Path = ROOT) -> list[str]:
    directory = (root / PNG_DIR).resolve()
    return [path.stem for path in sorted(directory.glob("*.png")) if path.is_file()]


if __name__ == "__main__":
    catalog = load_catalog(require_baselines=False)
    check_visual_coverage(catalog, load_manifest(ROOT / "internal-testing/shared/tui-png/journey"))
    print(f"Visual evidence found for {len(catalog['checkpoints'])} TUI checkpoints")
