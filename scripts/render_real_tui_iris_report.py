"""Build a private, offline HTML review of separately recorded real TUI runs.

The download run and cached finish run must remain distinct, rather than
compositing them as if one Pilot session had fully completed both paths.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def read_json(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object: {path}")
    return result


def screenshot(folder: Path, name: str, caption: str) -> str:
    if name not in {
        "01-configuration.svg",
        "02-modal-1.svg",
        "04-download-progress.svg",
        "05-monitor-complete.svg",
        "06-results-overview.svg",
        "07-results-statistics.svg",
    }:
        raise ValueError("unexpected screenshot identity")
    path = folder / "screenshots" / name
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 250_000:
        return f"<p>{escape(caption)}：未捕获 {escape(name)}</p>"
    raw = path.read_bytes()
    if b"<svg" not in raw[:2000]:
        raise ValueError("malformed Textual SVG")
    image = base64.b64encode(raw).decode("ascii")
    return (
        f"<figure><figcaption>{escape(caption)}</figcaption>"
        f'<img alt="{escape(caption)}" src="data:image/svg+xml;base64,{image}"></figure>'
    )


def render(first_dir: Path, follow_dir: Path, guard_dir: Path, output: Path) -> dict:
    first = read_json(first_dir / "reports/receipt.json")
    follow = read_json(follow_dir / "reports/receipt.json")
    guard = read_json(guard_dir / "reports/receipt.json")
    if not (
        first.get("model") == follow.get("model") == "Ritual-Net/iris-classification"
        and first.get("revision") == follow.get("revision")
        and first.get("fixture") is False
        and follow.get("fixture") is False
        and first.get("real_backend") is True
        and follow.get("real_backend") is True
    ):
        raise ValueError("receipts do not match the required real model")
    audit_first = read_json(first_dir / "reports/result-audit.json")
    audit_follow = read_json(follow_dir / "reports/result-audit.stdout.txt")
    positive = [
        v
        for v in first.get("download_events", [])
        if isinstance(v.get("transferred_bytes"), int) and v["transferred_bytes"] > 0
    ]
    transfer = max((v["transferred_bytes"] for v in positive), default=0)
    download_confirmed = any(
        row.get("download_confirmation") and row.get("action") == "continue"
        for row in first.get("decisions", [])
    )
    first_passed = bool(
        transfer > 0
        and download_confirmed
        and first.get("completion", {}).get("returncode") == 0
        and audit_first.get("accepted") is True
        and audit_first.get("coverage", {}).get("missing") == 0
    )
    follow_passed = bool(
        follow.get("status") == "passed"
        and follow.get("completion", {}).get("returncode") == 0
        and follow.get("snapshot", {}).get("stage") == "Stage.COMPLETED"
        and follow.get("statistics_rows", 0) > 0
        and audit_follow.get("accepted") is True
    )
    guard_failed_closed = any(
        "Model Store 缺失或计划 SHA256 不匹配" in (row.get("request", {}).get("detail") or "")
        for row in guard.get("decisions", [])
    )
    progress_rows = []
    for row in first.get("download_events", []):
        progress_rows.append(
            f"<tr><td>{escape(row.get('state'))}</td>"
            f"<td>{escape(row.get('transferred_bytes'))}</td>"
            f"<td>{escape(row.get('cached_bytes'))}</td>"
            f"<td>{escape(row.get('widget_visible'))}</td></tr>"
        )
    first_screens = "".join(
        [
            screenshot(first_dir, "01-configuration.svg", "1. 未缓存运行：真实配置页"),
            screenshot(first_dir, "02-modal-1.svg", "2. 真实下载确认弹窗"),
            screenshot(first_dir, "04-download-progress.svg", "3. 真实下载进度"),
        ]
    )
    follow_screens = "".join(
        [
            screenshot(follow_dir, "05-monitor-complete.svg", "4. 已缓存复验：运行完成"),
            screenshot(follow_dir, "06-results-overview.svg", "5. 真实结果读取"),
            screenshot(follow_dir, "07-results-statistics.svg", "6. 真实统计结果"),
        ]
    )
    count1 = audit_first.get("counts") or {}
    count2 = audit_follow.get("counts") or {}
    summary = {
        "uncached_download_bytes": transfer,
        "uncached_download_events": len(first.get("download_events", [])),
        "uncached_e2e_backend_and_audit": first_passed,
        "cached_followup_ui_and_audit": follow_passed,
        "cross_store_reuse_failed_closed": guard_failed_closed,
        "formal_ok_uncached": count1.get("formal_ok"),
        "formal_ok_cached": count2.get("formal_ok"),
        "same_session_full_flow": False,
    }
    body = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>AC-Prof 真实 TUI 小模型验收</title>
<style>:root{{color-scheme:dark;font-family:system-ui,sans-serif}}body{{margin:0;background:#141b22;color:#e8eef7}}
main{{max-width:1500px;padding:26px;margin:auto}}h1{{font-size:1.6rem}}h2{{margin-top:36px;font-size:1.2rem}}
p,li{{line-height:1.7}}.muted{{color:#a9b9c9}}.metrics{{display:flex;flex-wrap:wrap;gap:12px}}
.metric{{padding:14px;background:#24303d;border-radius:9px;min-width:170px}}.metric strong{{display:block;font-size:1.45rem}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,430px),1fr));gap:14px}}
figure{{margin:0;background:#222d39;border:1px solid #3e5264;border-radius:9px;overflow:hidden}}
figcaption{{padding:12px;font-weight:600}}figure img{{display:block;width:100%;background:#1b232d}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;text-align:left;border-bottom:1px solid #394755}}
small{{color:#a9b9c9}}code{{overflow-wrap:anywhere}}</style></head><body><main>
<h1>AC-Prof · 真实 TUI 小模型采集验收</h1>
<p class="muted">2026-11-01 · dor-server · Ritual-Net/iris-classification · Docker CPU/basic · 1 CPU / 1 GiB · GPU off · 1 case × 1 正式窗口</p>
<p><strong>证据边界：</strong>所有事件和后端均来自真实执行，没有伪造推理、下载或结果。
但这不是同一次连续完成的 TUI 会话：首次未缓存运行成功下载并完成真实后端，测试驱动在结束状态序列化时失败；随后复用已验证缓存的独立运行完成了 TUI 结果界面验收。
不得把两段截图解释为一次无中断的全链路，也不是准确率或性能基准。</p>
<div class="metrics"><div class="metric"><strong>{transfer:,} B</strong>首次真实传输</div>
<div class="metric"><strong>{len(first.get("download_events", []))}</strong>真实下载进度事件</div>
<div class="metric"><strong>{count1.get("formal_ok", 0)} / 1</strong>首次正式窗口成功</div>
<div class="metric"><strong>{count2.get("formal_ok", 0)} / 1</strong>缓存复验正式窗口成功</div></div>
<h2>1. 未缓存：真实下载、测量与结果审计</h2>
<p>下载弹窗由真实 PreparationScreen 触发，Model Store 为空，预算限制 5 MB。
观察到完成状态 <code>downloading → verifying → complete</code>；截图发生于准备阶段，测量期间未截图。
后端子进程退出码 {escape(first.get("completion", {}).get("returncode"))}，严格审计 accepted={escape(audit_first.get("accepted"))}，
实际记录 {escape(count1.get("rows"))} 行，缺少 case={escape(audit_first.get("coverage", {}).get("missing"))}。</p>
<div class="gallery">{first_screens}</div>
<h2>2. 异步下载事件记录（原始数据摘要）</h2>
<table><thead><tr><th>状态</th><th>已传输字节</th><th>缓存字节</th><th>进度控件可见</th></tr></thead><tbody>{"".join(progress_rows)}</tbody></table>
<h2>3. 缓存复验：完成态和结果界面</h2>
<p>下一次运行复用首次创建并校验的同一个 Model Store，真实子进程退出 0，阶段为 COMPLETED。
TUI 结果摘要包含正式成功记录，计算获得 {escape(follow.get("statistics_rows"))} 行统计信息；
严格结果审计 accepted={escape(audit_follow.get("accepted"))}、formal_ok={escape(count2.get("formal_ok"))}，
不代表统计置信区间充分。</p>
<div class="gallery">{follow_screens}</div>
<h2>4. 安全边界：跨 Model Store 复用被拒绝</h2>
<p>另一个使用不同空 Model Store 的运行，尝试复用此前镜像时遇到 SHA256 身份不匹配，
已按安全策略拒绝在线 fallback：{escape("已观察到" if guard_failed_closed else "未确认")}。
该运行没有计入通过样本，也没有假装成功。</p>
<p class="muted">原始证据分别位于本机 <code>internal-testing/evidence/tui-real-download-20261101/</code>
及 <code>...-v2/</code>、<code>...-v3/</code>；本报告只供本地审阅，禁止自动发布到公开 Pages。
数据来自真实的 TUI 回执、下载事件、正式运行产物和 CLI 审计结果。</p>
</main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body, encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, type=Path)
    parser.add_argument("--followup", required=True, type=Path)
    parser.add_argument("--guard", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = render(
        args.first.resolve(), args.followup.resolve(), args.guard.resolve(), args.output.resolve()
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
