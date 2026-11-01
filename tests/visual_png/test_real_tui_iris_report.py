"""Real TUI reports must label separate runs and reject fake model identities."""

from __future__ import annotations

import json

import pytest

from scripts.render_real_tui_iris_report import render, screenshot


def write_receipt(root, *, cached=False):
    report = root / "reports"
    report.mkdir(parents=True)
    record = {
        "model": "Ritual-Net/iris-classification",
        "revision": "rev-123",
        "fixture": False,
        "real_backend": True,
        "completion": {"returncode": 0},
    }
    if cached:
        record.update(status="passed", snapshot={"stage": "Stage.COMPLETED"}, statistics_rows=3)
    else:
        record.update(
            decisions=[{"download_confirmation": True, "action": "continue"}],
            download_events=[
                {
                    "state": "downloading",
                    "transferred_bytes": 12,
                    "cached_bytes": 0,
                    "widget_visible": True,
                }
            ],
        )
    (report / "receipt.json").write_text(json.dumps(record), encoding="utf-8")
    (report / ("result-audit.stdout.txt" if cached else "result-audit.json")).write_text(
        json.dumps({"accepted": True, "coverage": {"missing": 0}, "counts": {"formal_ok": 1}}),
        encoding="utf-8",
    )
    return record


def test_report_distinguishes_real_download_from_cached_followup(tmp_path):
    first, followup, guard = (tmp_path / name for name in ("first", "cached", "guard"))
    write_receipt(first)
    write_receipt(followup, cached=True)
    (guard / "reports").mkdir(parents=True)
    (guard / "reports/receipt.json").write_text(
        json.dumps(
            {
                "decisions": [
                    {
                        "request": {"detail": "Model Store 缺失或计划 SHA256 不匹配"},
                    }
                ]
            }
        )
    )
    target = tmp_path / "output.html"
    result = render(first, followup, guard, target)
    assert result == {
        "uncached_download_bytes": 12,
        "uncached_download_events": 1,
        "uncached_e2e_backend_and_audit": True,
        "cached_followup_ui_and_audit": True,
        "cross_store_reuse_failed_closed": True,
        "formal_ok_uncached": 1,
        "formal_ok_cached": 1,
        "same_session_full_flow": False,
    }
    assert json.loads(target.with_suffix(".json").read_text()) == result
    document = target.read_text()
    assert "不是同一次连续完成的 TUI 会话" in document
    assert "未捕获" in document
    assert "真实下载确认弹窗" in document
    assert "严格审计 accepted=True" in document


def test_report_rejects_misidentified_model_and_unexpected_screenshot(tmp_path):
    first, followup, guard = (tmp_path / name for name in ("first", "cached", "guard"))
    record = write_receipt(first)
    write_receipt(followup, cached=True)
    (guard / "reports").mkdir(parents=True)
    (guard / "reports/receipt.json").write_text(json.dumps({"decisions": []}))
    record["model"] = "unrelated/model"
    (first / "reports/receipt.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="real model"):
        render(first, followup, guard, tmp_path / "output.html")
    with pytest.raises(ValueError, match="screenshot identity"):
        screenshot(first, "../../private.svg", "bad")
