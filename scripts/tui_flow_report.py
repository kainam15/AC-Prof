"""Offline HTML sections joining TUI workflow evidence with existing PNG diffs.

All screenshots originate in checked-in visual fixtures or the fixed PNG capture
directory supplied by the caller. Never read live experiments or runtime logs.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any

from scripts.tui_flow_catalog import ROOT, load_catalog, svg_inventory
from scripts.tui_journey_report import JOURNEY_STYLE, render_journey

FLOW_STYLE = """
.view-tabs{display:flex;gap:9px;flex-wrap:wrap;margin:25px 0 18px}
.view-tabs button{background:#232a33;border:1px solid #566070;border-radius:7px;
  padding:10px 15px;color:#edf0f7;cursor:pointer;font:inherit}
.view-tabs button[aria-pressed=true]{background:#384f63;border-color:#82c9de}
.view-panel[hidden]{display:none!important}
.view-panel h2{font-size:1.3rem;margin:24px 0 10px}
.flow-note{color:#bec8d4;line-height:1.6;margin:10px 0 20px}
.flow-counts{display:flex;flex-wrap:wrap;gap:9px;margin:16px 0}
.flow-count{border:1px solid #384451;border-radius:8px;padding:8px 12px;background:#20262e}
.flow-list{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:14px}
.flow-card{border:1px solid #37414e;background:#1d232b;border-radius:11px;padding:17px;min-width:0}
.flow-card h3{font-size:1.05rem;margin:0 0 8px}
.flow-card p{margin:7px 0;line-height:1.6;overflow-wrap:anywhere}
.flow-card .field{color:#a9b3c2}
.flow-evidence{display:flex;gap:7px;flex-wrap:wrap;margin:13px 0}
.flow-chip{font-size:.79rem;border:1px solid #546171;border-radius:999px;padding:3px 9px;color:#d7e3f0}
.flow-chip.missing{color:#e4b898;border-color:#845943}
.flow-thumb{margin:12px 0 0;border:1px solid #343d49;border-radius:6px;overflow:hidden}
.flow-thumb img{display:block;max-width:100%;max-height:260px;margin:auto;object-fit:contain}
.flow-thumb figcaption{font-size:.77rem;background:#262d36;padding:6px 10px;color:#c5ced8}
.flow-card details{margin-top:12px;border-top:1px solid #343d49;padding-top:10px}
.flow-card summary,.svg-gallery summary{cursor:pointer;color:#bcdaee}
.flow-card ul{padding-left:20px;overflow-wrap:anywhere;font-size:.83rem}
.flow-section{margin-top:28px}
.flow-section > h3{font-size:1.05rem;margin:0 0 14px}
.coverage-table{width:100%;border-collapse:collapse;font-size:.9rem}
.coverage-table th,.coverage-table td{text-align:left;border-bottom:1px solid #37414e;
  padding:10px 9px;vertical-align:top}
.coverage-table th{color:#b9cbdf}
.coverage-table td{overflow-wrap:anywhere}
.coverage-wrap{overflow-x:auto}
.svg-gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,440px),1fr));gap:12px}
.svg-gallery details{border:1px solid #37414e;border-radius:8px;padding:12px;background:#1d232b;min-width:0}
.svg-gallery summary{overflow-wrap:anywhere;font-size:.85rem}
.svg-gallery img{display:block;width:100%;height:auto;max-height:420px;object-fit:contain;margin-top:12px}
@media(max-width:680px){.flow-list{grid-template-columns:1fr}
  .coverage-table{font-size:.8rem}.coverage-table th,.coverage-table td{padding:8px 5px}}
"""

FLOW_STYLE += JOURNEY_STYLE

FLOW_SCRIPT = """
for (const button of document.querySelectorAll('[data-report-view]')) {
  button.addEventListener('click', () => {
    const chosen = button.getAttribute('data-report-view');
    for (const other of document.querySelectorAll('[data-report-view]')) {
      other.setAttribute('aria-pressed', String(other === button));
    }
    for (const panel of document.querySelectorAll('.view-panel')) {
      panel.hidden = panel.getAttribute('data-view-panel') !== chosen;
    }
  });
}
"""


def _safe(value: object) -> str:
    return html.escape(str(value), quote=True)


def _data_image(path: Path, mime: str) -> str | None:
    try:
        # Baselines are checked-in fixture images; reject unexpectedly large files.
        if path.stat().st_size > 2 * 1024 * 1024:
            return None
        raw = path.read_bytes()
    except (FileNotFoundError, OSError):
        return None
    if not raw:
        return None
    if mime == "image/svg+xml" and b"<svg" not in raw[:3000]:
        return None
    if mime == "image/png" and not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _thumbnail(path: Path, label: str, mime: str) -> str:
    data = _data_image(path, mime)
    if data is None:
        return f'<p class="flow-note">{_safe(label)}：图片不可用</p>'
    return (
        f'<figure class="flow-thumb"><figcaption>{_safe(label)}</figcaption>'
        f'<img loading="lazy" alt="{_safe(label)}" src="{data}"></figure>'
    )


def _checkpoint_card(
    item: dict,
    *,
    page_name: str,
    root: Path,
    actual: Path,
    baselines: Path,
    png_status: dict[str, str],
    pilot_count: int,
) -> str:
    svg = item["svg"]
    png = item["png"]
    tests = item["tests"]
    view = (
        [_thumbnail(root / svg[0], "已审阅 SVG 基准（非本次截图）", "image/svg+xml")] if svg else []
    )
    if png:
        first = png[0]
        live = actual / f"{first}.png"
        path = live if live.is_file() else baselines / f"{first}.png"
        label = "本次 PNG 截图" if live.is_file() else "已审阅 PNG 基准（尚无本次截图）"
        view.insert(0, _thumbnail(path, label, "image/png"))
    png_statuses = [f"{scene}: {png_status.get(scene, 'not-compared')}" for scene in png]
    sources = "".join(
        f"<li><code>{_safe(name)}</code></li>" for name in [*png_statuses, *svg, *tests]
    )
    kind_label = {
        "main": "常规阶段",
        "optional": "按条件出现",
        "exception": "异常分支",
        "support": "独立页面",
    }[item["group"]]
    badges = (
        f'<span class="flow-chip">{_safe(kind_label)}</span>'
        f'<span class="flow-chip {"missing" if not png else ""}">PNG {len(png)}</span>'
        f'<span class="flow-chip {"missing" if not svg else ""}">SVG {len(svg)}</span>'
        f'<span class="flow-chip">测试文件 {len(tests)}（非运行结果）</span>'
        f'<span class="flow-chip">Pilot 截图 {pilot_count}（模拟）</span>'
    )
    return (
        f'<article class="flow-card" id="flow-{_safe(item["id"])}">'
        f"<h3>{_safe(item['title'])}</h3>"
        f'<div class="flow-evidence"><span class="flow-chip">{_safe(page_name)}</span>{badges}</div>'
        f'<p><span class="field">触发：</span>{_safe(item["trigger"])}</p>'
        f'<p><span class="field">预期：</span>{_safe(item["expected"])}</p>'
        + "".join(view)
        + f"<details><summary>全部证据索引（{len(png) + len(svg) + len(tests)} 项）</summary>"
        f"<ul>{sources}</ul></details></article>"
    )


def render_flow_report(
    *, actual: Path, baselines: Path, png_results: list[dict[str, Any]] | None, root: Path = ROOT
) -> tuple[str, str, str]:
    """Return workflow, coverage and SVG gallery sections, using only fixture paths."""
    catalog = load_catalog(root=root)
    pages = catalog["pages"]
    page_names = {page["id"]: page["title"] for page in pages}
    checkpoints = catalog["checkpoints"]
    status = {row["scene"]: row["status"] for row in png_results or ()}
    journey_html, pilot_counts = render_journey(actual)
    grouped = [
        (
            "采集主路径 · 条件性确认按实际顺序穿插",
            lambda s: s["group"] in {"main", "optional"} and s["id"] != "recover",
        ),
        ("恢复与新建实验", lambda s: s["id"] == "recover"),
        ("异常与退出分支", lambda s: s["group"] == "exception"),
        ("独立功能页面", lambda s: s["group"] == "support"),
    ]
    sections = []
    for title, included in grouped:
        cards = "".join(
            _checkpoint_card(
                item,
                page_name=page_names[item["page"]],
                root=root,
                actual=actual,
                baselines=baselines,
                png_status=status,
                pilot_count=pilot_counts.get(item["id"], 0),
            )
            for item in checkpoints
            if included(item)
        )
        sections.append(
            f'<section class="flow-section"><h3>{_safe(title)}</h3>'
            f'<div class="flow-list">{cards}</div></section>'
        )
    svg_all = svg_inventory(root=root)
    mapped = {image for step in checkpoints for image in step["svg"]}
    missing_visual = [step for step in checkpoints if not step["png"] and not step["svg"]]
    counts = (
        f'<div class="flow-counts">'
        f'<div class="flow-count">主页面：<strong>{len(pages)}</strong></div>'
        f'<div class="flow-count">流程检查点：<strong>{len(checkpoints)}</strong></div>'
        f'<div class="flow-count">SVG 基准：<strong>{len(svg_all)}</strong></div>'
        f'<div class="flow-count">无固定图片基准：<strong>{len(missing_visual)}</strong></div></div>'
    )
    workflow = (
        "<h2>采集流程与界面证据</h2>"
        '<p class="flow-note">流程按常见路径排列；条件性步骤可能跳过，异常步骤不是必经流程。'
        "截图只证明该固定状态曾被记录，测试文件索引不代表本次执行通过。"
        "这里展示的是脱敏测试夹具，不含真实采集日志或用户配置。</p>"
        + counts
        + journey_html
        + "".join(sections)
    )
    page_rows = []
    for page in pages:
        steps = [s for s in checkpoints if s["page"] == page["id"]]
        num_png = sum(len(s["png"]) for s in steps)
        num_svg = sum(len(s["svg"]) for s in steps)
        gaps = sum(not s["png"] and not s["svg"] for s in steps)
        page_rows.append(
            f'<tr><td><strong>{_safe(page["title"])}</strong><br><span class="muted">{_safe(page["purpose"])}</span></td>'
            f"<td>{len(steps)}</td><td>{num_png}</td><td>{num_svg}</td><td>{gaps}</td></tr>"
        )
    step_rows = []
    for item in checkpoints:
        screenshots = len(item["png"]) + len(item["svg"])
        missing = "待补视觉证据" if not screenshots else "已有固定截图"
        step_rows.append(
            f"<tr><td>{_safe(item['title'])}</td><td>{_safe(item['page'])}</td>"
            f"<td>{len(item['png'])}</td><td>{len(item['svg'])}</td>"
            f"<td>{pilot_counts.get(item['id'], 0)}</td>"
            f"<td>{len(item['tests'])}</td><td>{missing}</td></tr>"
        )
    coverage = (
        "<h2>覆盖矩阵</h2>"
        '<p class="flow-note">下表统计已登记的截图和测试源码，不是本次通过率。'
        "PNG 测试通过与否只以“视觉对比”页的 report.json 为准；SVG 通过情况应查看独立 SVG CI。"
        "连续 Pilot 图像由测试夹具生成，不能证明真实采集。</p>"
        '<h3>主页面</h3><div class="coverage-wrap"><table class="coverage-table">'
        "<thead><tr><th>页面</th><th>检查点</th><th>PNG</th><th>SVG</th><th>无截图</th></tr></thead><tbody>"
        + "".join(page_rows)
        + "</tbody></table></div>"
        '<h3>流程检查点</h3><div class="coverage-wrap"><table class="coverage-table">'
        "<thead><tr><th>检查点</th><th>所在页面</th><th>PNG</th><th>SVG</th><th>Pilot 帧</th><th>测试文件</th><th>固定截图</th></tr></thead><tbody>"
        + "".join(step_rows)
        + "</tbody></table></div>"
        + (
            f'<p class="flow-note">另有 {len(set(svg_all) - mapped)} 张未关联的 SVG，已自动列入图库。</p>'
            if set(svg_all) - mapped
            else ""
        )
    )
    names = {svg: step["title"] for step in checkpoints for svg in step["svg"]}
    gallery = []
    for path in svg_all:
        gallery.append(
            f"<details><summary>{_safe(names.get(path, '未关联'))} · {_safe(Path(path).name)}</summary>"
            + _thumbnail(root / path, "SVG 历史基准", "image/svg+xml")
            + "</details>"
        )
    svg_gallery = (
        f"<h2>SVG 基准图库（{len(svg_all)} 张）</h2>"
        '<p class="flow-note">所有已登记的 SVG 自动收录；它们不参与本次 PNG 像素差异比较，'
        "也不等同于本次 SVG CI 通过。</p>"
        '<div class="svg-gallery">' + "".join(gallery) + "</div>"
    )
    return workflow, coverage, svg_gallery
