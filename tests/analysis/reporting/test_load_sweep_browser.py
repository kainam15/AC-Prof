"""Opt-in offline browser checks of actual Plotly traces, evidence and controls."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from acprof.host.measurement.load import LoadConfig
from acprof.plotting.load_sweep import write_load_sweep_report
from tests.analysis.reporting.test_load_dataset_report import (
    make_dataset_sweep,
    saved_dataset_report,
)
from tests.support.load_sweep_fixtures import change_manifest, change_point, make_sweep

__all__ = ["saved_dataset_report"]

pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    os.environ.get("ACPROF_BROWSER_TESTS") != "1", reason="opt-in headless browser test",
)]


@pytest.fixture(name="browser_page")
def fixture_browser_page(request):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=os.environ.get("ACPROF_BROWSER_EXECUTABLE"),
                                             headless=True, args=["--no-sandbox", "--disable-gpu"])
        context = browser.new_context(offline=True, viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        errors, remote = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda event: remote.append(event.url) if event.url.startswith(("http:", "https:")) else None)
        artifact_dir = os.environ.get("ACPROF_BROWSER_ARTIFACT_DIR")

        def open_report(root):
            if artifact_dir:
                destination = Path(artifact_dir) / request.node.name
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copytree(root, destination / "source")
                root = destination / "source"
                (destination / "browser.json").write_text(json.dumps({
                    "chromium": browser.version, "offline": True,
                    "executable": os.environ.get("ACPROF_BROWSER_EXECUTABLE"),
                }))
            output = root.parent / "sweep.html"
            write_load_sweep_report(root, output)
            page.goto(output.as_uri())
            settled(page)
            return output

        try:
            yield page, open_report
            assert errors == []
            assert remote == []
        finally:
            if artifact_dir:
                capture_dir = Path(artifact_dir) / request.node.name
                capture_dir.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(capture_dir / "page.png"), full_page=True)
                (capture_dir / "events.json").write_text(json.dumps({"errors": errors, "remote_requests": remote}))
            context.close()
            browser.close()


def settled(page):
    page.wait_for_function("document.documentElement.dataset.reportReady === 'true'")


def traces(page):
    return page.evaluate("document.getElementById('chart').data.map(t => ({name:t.name,x:t.x,y:t.y,mode:t.mode,connectgaps:t.connectgaps}))")


def test_metric_controls_and_narrow_layout_remain_offline(browser_page, tmp_path):
    page, open_report = browser_page
    root = make_sweep(tmp_path / "source")
    open_report(root)
    assert traces(page)[0]["y"] == [1, 2, 3]
    assert page.locator("#points tr").count() == 3
    page.select_option("#metric", "output_tokens_per_s")
    settled(page)
    assert traces(page)[0]["y"] == [3, 6, 9]
    page.select_option("#metric", "ttft_p99_s")
    settled(page)
    assert traces(page)[0]["y"] == pytest.approx([0.1, 0.2, 0.3])
    page.select_option("#axis", "rate")
    settled(page)
    assert traces(page) == []
    assert "暂无" in page.locator("#chart").inner_text()
    page.select_option("#axis", "concurrency")
    page.set_viewport_size({"width": 390, "height": 844})
    settled(page)
    page.wait_for_function("document.documentElement.scrollWidth <= innerWidth")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert page.locator("#metric").is_visible()


def test_gaps_changed_conditions_and_partial_points_are_not_connected(browser_page, tmp_path):
    page, open_report = browser_page
    root = make_sweep(tmp_path / "source", [LoadConfig(scenario="concurrent", concurrency=c, requests=4) for c in range(1, 6)])
    change_point(root, 2, lambda point: point["identity"]["request_body"].update(temperature=0.8))
    (root / "points/0003/load.json").unlink()
    change_point(root, 4, lambda point: point.update(status="failed"))
    change_manifest(root, lambda manifest: (manifest.update(status="failed", successful=False),
                                            manifest["points"][3].update(status="failed", successful=False)))
    open_report(root)
    chart = traces(page)
    normal = [trace for trace in chart if trace["mode"] == "lines+markers"]
    assert normal[0]["x"] == [1, 2, 3, 4, 5]
    assert normal[0]["y"] == [1, None, None, None, 5]
    assert normal[1]["y"] == [None, 2, None, None, None]
    assert all(trace["connectgaps"] is False for trace in normal)
    partial = [trace for trace in chart if trace["mode"] == "markers"]
    assert partial[0]["x"] == [4]
    assert partial[0]["y"] == [4]
    assert page.locator('#points tr[data-evidence="missing"]').count() == 1
    assert page.locator('#points tr[data-evidence="partial"]').count() == 1
    second = page.locator("#condition option").nth(2).get_attribute("value")
    page.select_option("#condition", second)
    settled(page)
    assert page.locator("#points tr").count() == 2  # selected group plus unknown/missing evidence


def test_rate_curves_fix_concurrency_and_xss_text_stays_literal(browser_page, tmp_path):
    page, open_report = browser_page
    root = make_sweep(tmp_path / "source", [LoadConfig(scenario="arrival-rate", concurrency=c, rate=r, requests=4)
                                           for c in (1, 2) for r in (20, 10)])
    hostile = "</script><script>window.injected=true</script>__STYLE__"
    for index in range(1, 5):
        change_point(root, index, lambda point: point["service_identity"]["observed"].update(version=hostile))
    output = open_report(root)
    assert page.evaluate("window.injected === undefined")
    conditions_text = page.locator("#conditions").text_content()
    assert conditions_text is not None
    assert hostile in conditions_text
    chart = traces(page)
    assert [trace["x"] for trace in chart] == [[10, 20], [10, 20]]
    assert [trace["y"] for trace in chart] == [[2, 1], [4, 3]]
    assert all("concurrency=" in trace["name"] for trace in chart)
    page.select_option("#axis", "concurrency")
    settled(page)
    assert [trace["y"] for trace in traces(page)] == [[1, 3], [2, 4]]
    assert all("rate=" in trace["name"] for trace in traces(page))
    href = page.locator("#points a").first.get_attribute("href")
    assert isinstance(href, str)
    assert (output.parent / href).resolve().is_file()


def test_dataset_sources_are_visible_safe_and_narrow(browser_page, tmp_path, dataset_report):
    page, open_report = browser_page
    hostile = "</script><script>window.injected=true</script>.jsonl"
    root = make_dataset_sweep(tmp_path / "source", dataset_report, "source", source_path=hostile)
    open_report(root)
    assert page.locator("#points summary").count() == 3
    source = page.locator("#points details").nth(1)
    source.locator("summary").click()
    assert hostile in source.locator("pre").inner_text()
    assert "a" * 64 in source.locator("pre").inner_text()
    assert source.locator("pre").evaluate("element => element.clientHeight <= 200")
    assert page.evaluate("window.injected === undefined")
    assert page.locator("#condition option").count() == 2  # identical inputs despite different source paths
    page.set_viewport_size({"width": 390, "height": 844})
    settled(page)
    page.wait_for_function("document.documentElement.scrollWidth <= innerWidth")
    assert source.locator("summary").is_visible()
