import io
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.host.network_preflight import format_summary, preflight, runtime_sources
from acprof.host.runtime_images import PROJECT_ROOT
from acprof.network_policy import (
    DownloadPolicyError,
    DownloadSource,
    enforce_download_budget,
    require_source_transition,
    summarize_downloads,
)
from acprof.preparation_events import parse_event
from acprof.runtime_profiles import PROFILES


@pytest.mark.parametrize("download_bytes", [890_152_505, None])
def test_download_review_keeps_technical_details_out_of_main_summary(tmp_path, capsys, download_bytes):
    task = SimpleNamespace(model_id="demo/model", model_revision="a" * 40)
    plan = {"selected_bytes": 890_152_505, "endpoint": "https://hf-mirror.com"}
    disk = {"total_bytes": 0, "free_bytes": 41_370_132_480,
            "reclaimable_bytes": 0, "remaining_bytes": 40_479_979_975}
    reply = '{"id": 1, "action": "answer", "answers": {"下载计划": "按此计划下载"}}\n'
    with (
        patch.dict("os.environ", {"ACPROF_INTERACTIVE_PREPARATION": "1"}, clear=True),
        patch("sys.stdin", io.StringIO(reply)),
        patch("acprof.host.model_store.model_sources", return_value=[
            DownloadSource("model", plan["endpoint"], download_bytes),
        ]),
        patch("acprof.host.model_store.require_space", return_value=disk),
        patch("acprof.host.network_preflight.runtime_sources", return_value=([], {"platform_local": True})),
        patch("acprof.host.static_metadata._docker_storage_metadata", return_value={}),
    ):
        preflight(task, PROFILES["nlp-cpu"], tmp_path, plan, root=tmp_path / "cache")
    events = [event for line in capsys.readouterr().out.splitlines() if (event := parse_event(line))]
    request = events[-1]["request"]
    assert not request.get("detail")
    assert request["fields"]["预计下载"] == (
        "890,152,505 B (0.890 GB)" if download_bytes is not None else None
    )
    assert request["fields"]["可用空间"] == "41,370,132,480 B (41.370 GB)"
    assert request["fields"]["下载源"] == "https://hf-mirror.com"
    assert "expected_download_bytes:" in request["summary"]
    assert "Docker storage:" in request["summary"]
    assert "upstream routing is not verified." in request["summary"]


def test_download_confirmation_contains_budget_and_cache_path():
    report = summarize_downloads([DownloadSource("model", "https://hf-mirror.com", 100)])
    report.update(max_download_bytes=5_000_000_000, model_store_path="/custom/cache")
    summary = format_summary(report)
    assert ("5.000 GB") in (summary)
    assert ("/custom/cache") in (summary)
    assert ("100 B") in (summary)

def test_route_totals_do_not_treat_unknown_as_zero():
    report = summarize_downloads([DownloadSource("model", "https://hf-mirror.com", 100),
                                 DownloadSource("oci", "https://ghcr.io/repo", 50),
                                 DownloadSource("python", "https://files.pythonhosted.org/a.whl", None)])
    assert (report["direct_download_bytes"]) == (100)
    assert (report["proxy_download_bytes"]) is None
    assert (report["expected_download_bytes"]) is None
    with pytest.raises(DownloadPolicyError):
        enforce_download_budget(report, "5GB")

def test_domestic_to_proxy_fallback_requires_explicit_consent():
    with pytest.raises(DownloadPolicyError):
        require_source_transition("https://docker.m.daocloud.io/repo", "https://ghcr.io/repo", allow_proxy=False)

def test_local_runtime_hit_does_not_probe_registry_or_packages():
    profile = PROFILES["nlp-cpu"]
    with patch.dict("os.environ", {"ACPROF_RUNTIME_IMAGE_SOURCE": "auto"}, clear=True):
        sources, state = runtime_sources(profile, PROJECT_ROOT,
            inspect=lambda _: {"image_id": "sha256:" + "a" * 64},
            size_probe=lambda _: pytest.fail("local hit fetched artifact metadata"),
            manifest_probe=lambda _: pytest.fail("local hit queried registry"))
    assert (sources) == ([])
    assert (state["platform_local"])
    assert (state["environment_local"])
    assert not (state["will_attempt_ghcr_pull"])

def test_local_miss_reports_ghcr_and_compressed_upper_bound():
    with patch.dict("os.environ", {"ACPROF_RUNTIME_IMAGE_SOURCE": "auto"}, clear=True):
        sources, state = runtime_sources(PROFILES["nlp-cpu"], PROJECT_ROOT, inspect=lambda _: None,
            manifest_probe=lambda _: {"bytes": 100, "image_id": "sha256:" + "a" * 64})
    assert (state["will_attempt_ghcr_pull"])
    assert (summarize_downloads(sources)["expected_download_bytes"]) == (200)
