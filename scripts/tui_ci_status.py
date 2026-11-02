"""Produce and display same-run TUI visual CI evidence, without inventing success.

Only a completed GitHub Actions step outcome *and* its corresponding validated
artifact can pass. This duplicates the SVG suite in the visual workflow on the
same commit; it does not claim to read the independent main CI's conclusion.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.tui_journey_contract import load_manifest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACTUAL = ROOT / "internal-testing/shared/tui-png"
SCHEMA_VERSION = 1
CHECKS = ("svg", "png", "pilot")
OUTCOMES = {"success", "failure", "cancelled", "skipped"}
STATUSES = {"passed", "failed", "not_run"}
SHA = re.compile(r"[0-9a-f]{40}\Z")
REPOSITORY = re.compile(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+\Z")
RUN_ID = re.compile(r"[1-9][0-9]{0,19}\Z")

STYLE = """
.ci-review{margin:22px 0 18px}
.ci-review h2{font-size:1.12rem;margin:0 0 10px}
.ci-meta{color:#aebdcd;font-size:.87rem;line-height:1.55;overflow-wrap:anywhere}
.ci-meta a{color:#b7dff3;text-decoration:underline;text-underline-offset:2px}
.ci-items{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:14px 0}
.ci-item{border:1px solid #3a4958;border-radius:9px;background:#202730;padding:14px;min-width:0}
.ci-item strong{display:block;margin:7px 0;font-size:1.1rem}
.ci-label{font-weight:600;font-size:.95rem}
.ci-item small{color:#afbbcb;font-size:.81rem;line-height:1.5;overflow-wrap:anywhere}
.ci-state{display:inline-block;border:1px solid #5c6c7c;border-radius:999px;padding:2px 10px;
 font-size:.78rem;letter-spacing:.01em}
.ci-state.passed{color:#a9efc0;border-color:#387d56}
.ci-state.failed{color:#ffc3ba;border-color:#b16b66}
.ci-state.not_run{color:#e2d4ad;border-color:#8a805d}
@media(max-width:790px){.ci-items{grid-template-columns:1fr}}
"""


def _check_file(path: Path, max_bytes: int) -> bool:
    return path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= max_bytes


def _svg_results(actual: Path) -> tuple[int, str]:
    source = actual / "svg-tests.xml"
    if not _check_file(source, 2_000_000):
        return 0, "缺少本次 SVG 测试 JUnit 结果"
    try:
        root = ET.fromstring(source.read_bytes())
        suites = (
            [root] if root.tag == "testsuite" else list(root) if root.tag == "testsuites" else []
        )
        if not suites or any(suite.tag != "testsuite" for suite in suites):
            raise ValueError("invalid JUnit root")
        counts = {
            key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
    except (ValueError, ET.ParseError, OSError) as exc:
        return 0, f"SVG JUnit 数据无效：{type(exc).__name__}"
    count = counts["tests"]
    if count < 1 or count == counts["skipped"] or counts["failures"] or counts["errors"]:
        return 0, (
            f"SVG JUnit：{count} 项、失败 {counts['failures']}、"
            f"错误 {counts['errors']}、跳过 {counts['skipped']}"
        )
    return count - counts[
        "skipped"
    ], f"通过 {count - counts['skipped']} 项，跳过 {counts['skipped']} 项"


def _png_results(actual: Path, root: Path) -> tuple[int, str]:
    source = actual / "report.json"
    if not _check_file(source, 500_000):
        return 0, "缺少本次 PNG 像素比较结果"
    try:
        rows = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("PNG report must be a list")
        expected = {path.stem for path in (root / "tests/visual_png/__snapshots__").glob("*.png")}
        names = [
            row["scene"]
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("scene"), str)
        ]
        if (
            not expected
            or len(rows) != len(expected)
            or set(names) != expected
            or len(names) != len(set(names))
        ):
            raise ValueError("incomplete or duplicated PNG scenes")
        failed = [row["scene"] for row in rows if row.get("status") != "passed"]
        if failed:
            return 0, f"PNG 对比未通过：{', '.join(sorted(failed))[:100]}"
        for name in expected:
            image = actual / f"{name}.png"
            if (
                not _check_file(image, 12_000_000)
                or not image.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
            ):
                raise ValueError("missing/invalid current PNG image")
    except (ValueError, KeyError, TypeError, json.JSONDecodeError, OSError) as exc:
        return 0, f"PNG 比较证据无效：{type(exc).__name__}"
    return len(expected), f"{len(expected)} 张当前 PNG 与 Golden 一致"


def _pilot_results(actual: Path) -> tuple[int, str]:
    try:
        manifest = load_manifest(actual / "journey")
        if not manifest:
            return 0, "缺少本次连续 Pilot 走查证据"
        scenarios = manifest["scenarios"]
        count = sum(len(scene["steps"]) for scene in scenarios)
        if not scenarios or count < len(scenarios):
            raise ValueError("empty Pilot manifest")
    except (ValueError, OSError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return 0, f"Pilot 证据无效：{type(exc).__name__}"
    return count, f"{len(scenarios)} 种终端条件 · {count} 帧连续交互证据"


def record_status(
    *,
    actual: Path,
    root: Path,
    outcomes: Mapping[str, str],
    commit: str,
    repository: str,
    run_id: str,
    run_attempt: str,
    event: str,
) -> dict[str, Any]:
    """Only complete GitHub context and same-run evidence may produce a pass."""
    if (
        not SHA.fullmatch(commit)
        or not REPOSITORY.fullmatch(repository)
        or not RUN_ID.fullmatch(run_id)
    ):
        raise ValueError("missing/invalid GitHub run identity")
    if not RUN_ID.fullmatch(run_attempt) or event not in {
        "push",
        "pull_request",
        "workflow_dispatch",
    }:
        raise ValueError("invalid GitHub run attempt or event")
    if set(outcomes) != set(CHECKS) or any(value not in OUTCOMES for value in outcomes.values()):
        raise ValueError("invalid GitHub step outcome")
    validators = {
        "svg": lambda: _svg_results(actual),
        "png": lambda: _png_results(actual, root),
        "pilot": lambda: _pilot_results(actual),
    }
    checks = {}
    for name in CHECKS:
        outcome = outcomes[name]
        if outcome == "skipped":
            checks[name] = {"status": "not_run", "count": 0, "detail": "步骤未执行"}
        elif outcome in {"failure", "cancelled"}:
            checks[name] = {
                "status": "failed",
                "count": 0,
                "detail": "GitHub Actions 测试失败或取消",
            }
        else:
            count, detail = validators[name]()
            checks[name] = {
                "status": "passed" if count > 0 else "failed",
                "count": count,
                "detail": detail,
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "origin": "same-run-tui-visual-ci",
        "commit": commit,
        "repository": repository,
        "run_id": run_id,
        "run_attempt": int(run_attempt),
        "event": event,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "checks": checks,
    }


def _valid_status(document: object) -> bool:
    if (
        not isinstance(document, dict)
        or set(document)
        != {
            "schema_version",
            "origin",
            "commit",
            "repository",
            "run_id",
            "run_attempt",
            "event",
            "generated_at",
            "checks",
        }
        or document["schema_version"] != SCHEMA_VERSION
        or document["origin"] != "same-run-tui-visual-ci"
    ):
        return False
    if not (
        isinstance(document["commit"], str)
        and SHA.fullmatch(document["commit"])
        and isinstance(document["repository"], str)
        and REPOSITORY.fullmatch(document["repository"])
        and isinstance(document["run_id"], str)
        and RUN_ID.fullmatch(document["run_id"])
        and type(document["run_attempt"]) is int
        and 1 <= document["run_attempt"] <= 1000
        and isinstance(document["event"], str)
        and document["event"] in {"push", "pull_request", "workflow_dispatch"}
    ):
        return False
    try:
        stamp = datetime.fromisoformat(document["generated_at"])
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            return False
    except (TypeError, ValueError):
        return False
    checks = document["checks"]
    if not isinstance(checks, dict) or set(checks) != set(CHECKS):
        return False
    return all(
        isinstance(entry, dict)
        and set(entry) == {"status", "count", "detail"}
        and isinstance(entry["status"], str)
        and entry["status"] in STATUSES
        and type(entry["count"]) is int
        and 0 <= entry["count"] <= 10000
        and isinstance(entry["detail"], str)
        and len(entry["detail"]) <= 220
        and (entry["count"] > 0 if entry["status"] == "passed" else entry["count"] == 0)
        for entry in checks.values()
    )


def load_status(path: Path, *, environ: Mapping[str, str] | None = None) -> dict | None:
    """Never trust missing, stale, malformed or cross-commit status as a pass."""
    if not _check_file(path, 20_000):
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not _valid_status(data):
        return None
    environment = os.environ if environ is None else environ
    if environment.get("GITHUB_ACTIONS") == "true" and (
        data["commit"] != environment.get("GITHUB_SHA")
        or data["run_id"] != environment.get("GITHUB_RUN_ID")
        or data["repository"] != environment.get("GITHUB_REPOSITORY")
    ):
        return None
    return data


def render_status(document: dict | None) -> str:
    def safe(item: object) -> str:
        return html.escape(str(item), quote=True)

    names = {"svg": "SVG 快照", "png": "PNG 视觉回归", "pilot": "Pilot 连续走查"}
    labels = {"passed": "通过", "failed": "失败", "not_run": "未运行"}
    checks = (
        document["checks"]
        if document is not None
        else {
            name: {"status": "not_run", "count": 0, "detail": "没有可验证的本次 CI 结果"}
            for name in CHECKS
        }
    )
    if document is not None:
        run_url = (
            f"https://github.com/{document['repository']}/actions/runs/"
            f"{document['run_id']}/attempts/{document['run_attempt']}"
        )
        meta = (
            f'<p class="ci-meta">Commit: <code>{safe(document["commit"][:12])}</code>'
            f" · {safe(document['generated_at'])} · "
            f'<a href="{safe(run_url)}" rel="noopener noreferrer" target="_blank">GitHub Actions 本次运行</a></p>'
        )
    else:
        meta = '<p class="ci-meta">未找到与本次 CI 对应的状态记录。已有截图不代表本次测试通过。</p>'
    cards = "".join(
        f'<article class="ci-item" data-ci-check="{name}" data-ci-status="{checks[name]["status"]}">'
        f'<div class="ci-label">{safe(names[name])}</div>'
        f'<strong><span class="ci-state {checks[name]["status"]}">{labels[checks[name]["status"]]}</span></strong>'
        f"<small>{safe(checks[name]['detail'])}</small></article>"
        for name in CHECKS
    )
    passed = sum(item["status"] == "passed" for item in checks.values())
    return (
        f'<section class="ci-review" aria-label="同一次 TUI CI 验收状态">'
        f"<h2>本次 TUI CI 验收 · {passed}/3 项通过</h2>{meta}"
        f'<div class="ci-items">{cards}</div>'
        '<p class="ci-meta">三个结果均来自同一提交的 TUI 视觉工作流；SVG 在此复跑测试，'
        "不代表另一条独立主 CI 的结论。未运行、测试失败或证据缺失均不会显示“通过”。</p></section>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ACTUAL / "ci-status.json")
    parser.add_argument("--actual", type=Path, default=DEFAULT_ACTUAL)
    for name in CHECKS:
        parser.add_argument(f"--{name}-outcome", required=True, choices=sorted(OUTCOMES))
    args = parser.parse_args()
    data = record_status(
        actual=args.actual,
        root=ROOT,
        outcomes={name: getattr(args, f"{name}_outcome") for name in CHECKS},
        commit=os.environ.get("GITHUB_SHA", ""),
        repository=os.environ.get("GITHUB_REPOSITORY", ""),
        run_id=os.environ.get("GITHUB_RUN_ID", ""),
        run_attempt=os.environ.get("GITHUB_RUN_ATTEMPT", ""),
        event=os.environ.get("GITHUB_EVENT_NAME", ""),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {name: check["status"] for name, check in data["checks"].items()}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
