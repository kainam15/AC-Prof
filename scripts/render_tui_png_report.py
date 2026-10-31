"""Render a standalone HTML viewer for existing TUI PNG comparison evidence.

This script never captures screenshots, compares pixels, or changes PNG baselines.
It uses report.json as the sole source of comparison decisions and embeds images
so the HTML remains viewable offline after downloading a GitHub Actions artifact.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ACTUAL = ROOT / "internal-testing" / "tui-png"
BASELINES = ROOT / "tests" / "visual_png" / "__snapshots__"
SCENE_NAME = re.compile(r"[a-z0-9][a-z0-9-]*\Z")

STYLE = """
:root{color-scheme:dark;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
*{box-sizing:border-box}
body{margin:0;background:#15181e;color:#edf0f7}
main{max-width:1720px;margin:auto;padding:28px 20px 64px}
header{display:flex;justify-content:space-between;align-items:start;gap:16px;flex-wrap:wrap}
h1{font-size:1.6rem;margin:0;color:#d5a5ff}
.subtitle,.muted{color:#a9b3c2}
.subtitle{margin:8px 0 0}
.counts{display:flex;gap:10px;flex-wrap:wrap}
.count{border:1px solid #343d4a;border-radius:9px;padding:10px 15px;background:#20252d}
.count strong{font-size:1.3rem;display:block}
.count span{font-size:.85rem;color:#b5becc}
.filters{display:flex;flex-wrap:wrap;gap:8px;margin:24px 0 16px}
.filters button{border:1px solid #566070;border-radius:7px;padding:9px 14px;background:#232a33;color:#edf0f7;cursor:pointer}
.filters button[aria-pressed=true]{background:#633a83;border-color:#c78cff}
.scene{border:1px solid #37414e;background:#1d232b;border-radius:12px;margin:16px 0;overflow:hidden}
.scene[hidden]{display:none}
.scene-head{display:flex;align-items:center;flex-wrap:wrap;gap:11px;padding:16px 18px;border-bottom:1px solid #343d49}
.scene-head h2{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:1rem;margin:0;overflow-wrap:anywhere}
.status{padding:3px 10px;border-radius:999px;font-size:.78rem;font-weight:700}
.status.pass{color:#9af0b7;background:#16482b}
.status.fail{color:#ffaaa8;background:#56282d}
.metrics{margin-left:auto;color:#c1c9d5;font-size:.85rem}
.details{padding:0 18px;color:#f4c6a9;overflow-wrap:anywhere}
.comparison{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;padding:16px}
.frame{margin:0;min-width:0;background:#14181f;border:1px solid #36404d;border-radius:9px;overflow:hidden}
.frame figcaption{padding:9px 12px;color:#cbd5e1;font-size:.85rem;border-bottom:1px solid #303a45}
.frame img{display:block;width:100%;height:auto;object-fit:contain}
.placeholder{display:grid;place-items:center;min-height:140px;padding:16px;text-align:center;color:#a9b3c2}
.notice{border:1px solid #8c5c32;background:#372b24;padding:18px;border-radius:10px;margin:22px 0}
footer{color:#9aa5b6;font-size:.84rem;padding:20px 0}
@media(max-width:950px){.comparison{grid-template-columns:1fr}.metrics{margin-left:0}}
"""

SCRIPT = """
for (const button of document.querySelectorAll('[data-filter]')) {
  button.addEventListener('click', () => {
    const filter = button.getAttribute('data-filter');
    for (const other of document.querySelectorAll('[data-filter]')) {
      other.setAttribute('aria-pressed', String(other === button));
    }
    for (const card of document.querySelectorAll('.scene')) {
      card.hidden = filter !== 'all' && card.getAttribute('data-state') !== filter;
    }
  });
}
"""


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _image_data(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except (FileNotFoundError, OSError):
        return None
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _frame(label: str, path: Path | None, unavailable: str) -> str:
    image = _image_data(path) if path is not None else None
    contents = (
        f'<img loading="lazy" alt="{_escape(label)}" src="{image}">'
        if image
        else f'<div class="placeholder">{_escape(unavailable)}</div>'
    )
    return f'<figure class="frame"><figcaption>{_escape(label)}</figcaption>{contents}</figure>'


def _load_results(path: Path) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("report.json must contain an array")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("status"), str):
            raise ValueError("report.json contains an invalid result")
        scene = row.get("scene")
        if not isinstance(scene, str) or SCENE_NAME.fullmatch(scene) is None:
            raise ValueError(f"Invalid scene name in report.json: {scene!r}")
    return rows


def _card(row: dict[str, Any], *, actual: Path, baselines: Path) -> str:
    scene = row["scene"]
    status = row["status"]
    passed = status == "passed"
    changed = row.get("changed_pixels", "?")
    allowed = row.get("allowed_pixels", "?")
    comparison_completed = status in {"passed", "visual-regression", "size-mismatch"}
    dimensions = (
        f" · {_escape(row['width'])}×{_escape(row['height'])}"
        if row.get("width") and row.get("height")
        else ""
    )
    details = row.get("details")
    message = f'<p class="details">{_escape(details)}</p>' if details else ""
    expected = baselines / f"{scene}.png"
    if not expected.is_file():
        expected = actual / "diff" / "expected" / f"{scene}.png"
    frames = (
        _frame("Historical baseline", expected, "Baseline not available")
        + _frame("Current capture", actual / f"{scene}.png", "Capture not available")
        + _frame(
            "Highlighted differences",
            None if passed else actual / "diff" / f"{scene}.png",
            "No significant differences" if passed else "Difference image not available",
        )
    )
    state = "passed" if passed else "failed"
    badge = "PASS" if passed else "FAIL"
    metrics = (
        f"{_escape(changed)} changed pixels / {_escape(allowed)} allowed{dimensions}"
        if comparison_completed
        else "Not compared"
    )
    return (
        f'<section class="scene" data-state="{state}">'
        f'<div class="scene-head"><h2>{_escape(scene)}</h2>'
        f'<span class="status {"pass" if passed else "fail"}">{badge}: {_escape(status)}</span>'
        f'<span class="metrics">{metrics}</span>'
        f'</div>{message}<div class="comparison">{frames}</div></section>'
    )


def render_report(
    *, actual: Path = ACTUAL, baselines: Path = BASELINES, output: Path | None = None
) -> Path:
    """Write an HTML report even when screenshot comparison was never reached."""
    rows = _load_results(actual / "report.json")
    passed = sum(row["status"] == "passed" for row in rows or ())
    failed = len(rows) - passed if rows is not None else 0
    count = len(rows) if rows is not None else 0
    cards = "".join(_card(row, actual=actual, baselines=baselines) for row in rows or ())
    if rows is None:
        cards = (
            '<div class="notice" role="alert"><strong>Comparison not available.</strong> '
            "No report.json was produced. The test may have failed before the PNG "
            "comparison stage. Inspect the GitHub Actions logs; no pass/fail result "
            "can be inferred.</div>"
        )
    elif not rows:
        cards = '<div class="notice" role="alert">No scene results were recorded.</div>'
    filters = (
        f'<nav class="filters" aria-label="Filter scenes">'
        f'<button type="button" data-filter="all" aria-pressed="true">All ({count})</button>'
        f'<button type="button" data-filter="failed" aria-pressed="false">Failed ({failed})</button>'
        f'<button type="button" data-filter="passed" aria-pressed="false">Passed ({passed})</button>'
        f"</nav>"
        if rows
        else ""
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AC-Prof TUI PNG Visual Report</title>
<style>{STYLE}</style>
</head>
<body>
<main>
<header>
  <div><h1>AC-Prof · TUI PNG Visual Report</h1>
    <p class="subtitle">Real terminal PNG snapshots compared with reviewed golden images</p></div>
  <div class="counts" aria-label="Comparison summary">
    <div class="count"><strong>{count}</strong><span>Recorded scenes</span></div>
    <div class="count"><strong>{passed}</strong><span>Matched</span></div>
    <div class="count"><strong>{failed}</strong><span>Changed / unavailable</span></div>
  </div>
</header>
{filters}
{cards}
<footer>Comparison decisions come from report.json. Images are embedded for offline review.
The report does not approve or update baselines.</footer>
</main>
<script>{SCRIPT}</script>
</body>
</html>
"""
    destination = output if output is not None else actual / "report.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", type=Path, default=ACTUAL)
    parser.add_argument("--baselines", type=Path, default=BASELINES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(render_report(actual=args.actual, baselines=args.baselines, output=args.output))


if __name__ == "__main__":
    main()
