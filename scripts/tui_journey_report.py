"""Render the synthetic continuous Pilot trace within the existing offline report."""

from __future__ import annotations

import base64
import html
from pathlib import Path

from scripts.tui_journey_contract import load_manifest

JOURNEY_STYLE = """
.pilot-section{margin:24px 0 28px}
.pilot-section h3{margin:0 0 10px}
.pilot-note{font-size:.85rem;color:#bbd2df;line-height:1.6}
.pilot-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,430px),1fr));gap:12px}
.pilot-step{border:1px solid #334657;background:#1b2731;border-radius:9px;padding:13px;min-width:0}
.pilot-step h4{margin:0 0 5px;font-size:.98rem}
.pilot-step p{font-size:.82rem;margin:5px 0;overflow-wrap:anywhere}
.pilot-alternatives{margin-top:15px;border:1px solid #384451;border-radius:8px;padding:12px}
.pilot-alternatives summary{cursor:pointer;color:#bce1ee}
.pilot-evidence{font-size:.82rem;color:#b2d8e8}
"""


def _safe(value: object) -> str:
    return html.escape(str(value), quote=True)


def render_journey(actual: Path) -> tuple[str, dict[str, int]]:
    """Use fixed scenario identities; never import live-run data or arbitrary SVGs."""
    folder = actual / "journey"
    manifest = load_manifest(folder)
    if manifest is None:
        return (
            '<section class="pilot-section"><h3>连续 Pilot 走查</h3>'
            '<p class="pilot-note">本次没有连续走查证据；现有静态截图不能证明页面切换已验证。</p>'
            "</section>",
            {},
        )
    counts: dict[str, int] = {}
    sections = [
        '<section class="pilot-section"><h3>连续 Pilot 走查 · 同一实例逐步操作</h3>'
        '<p class="pilot-note">真实 Pilot 点击、页面切换与模拟事件。模型准备、下载、'
        "正式测量和成功归属均由测试夹具模拟，绝非真实采集验收。</p>"
    ]
    for scenario in manifest["scenarios"]:
        cards: list[str] = []
        for number, step in enumerate(scenario["steps"], 1):
            counts[step["checkpoint"]] = counts.get(step["checkpoint"], 0) + 1
            path = folder / step["image"]
            preview = step.get("preview")
            if preview:
                path = folder / preview
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            mime = "image/png" if preview else "image/svg+xml"
            caption = (
                "合成 Pilot PNG（Noto CJK）· 非 Golden 基准"
                if preview
                else "合成 Pilot SVG 诊断 · 非 Golden 基准"
            )
            image = (
                '<figure class="flow-thumb">'
                f"<figcaption>{caption}</figcaption>"
                f'<img loading="lazy" alt="{_safe(step["id"])}" '
                f'src="data:{mime};base64,{data}"></figure>'
            )
            cards.append(
                f'<article class="pilot-step" id="pilot-step-{_safe(scenario["id"])}-{_safe(step["id"])}">'
                f"<h4>{number:02d}. {_safe(step['id'])}</h4>"
                f"<p>{_safe(step['action'])}</p>"
                f'<p class="pilot-evidence">{_safe(step["page"])} · {_safe(step["screen"])}'
                f" · focus: {_safe(step['focus'] or 'none')}"
                f" · {step['terminal'][0]}×{step['terminal'][1]}</p>"
                f"{image}</article>"
            )
        if scenario["id"] == "zh-wide":
            sections.append('<div class="pilot-grid">' + "".join(cards) + "</div>")
        else:
            sections.append(
                f'<details class="pilot-alternatives"><summary>{_safe(scenario["id"])}'
                f" · {len(cards)} 步 · 小终端 / 多语言验证</summary>"
                '<div class="pilot-grid">' + "".join(cards) + "</div></details>"
            )
    sections.append("</section>")
    return "".join(sections), counts
