import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from runtime_fixture import copy_dependency_tree

from acprof.artifacts import MAX_JSON_ARTIFACT_BYTES
from acprof.dependency_locks import read_python_lock
from acprof.host.network_preflight import preflight, runtime_sources
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
def test_download_confirmation_preserves_raw_bytes(tmp_path, capsys, download_bytes):
    task = SimpleNamespace(model_id="demo/model", model_revision="a" * 40)
    plan = {"selected_bytes": 890_152_505, "endpoint": "https://hf-mirror.com"}
    disk = {"total_bytes": 0, "free_bytes": 41_370_132_480,
            "reclaimable_bytes": 0, "remaining_bytes": 40_479_979_975}
    reply = '{"id": 1, "action": "confirm"}\n'
    with (
        patch.dict("os.environ", {"ACPROF_INTERACTIVE_PREPARATION": "1"}, clear=True),
        patch("sys.stdin", io.StringIO(reply)),
        patch("acprof.host.model_store.model_sources", return_value=[
            DownloadSource("model", plan["endpoint"], download_bytes),
        ]),
        patch("acprof.host.model_store.require_space", return_value=disk),
        patch("acprof.host.model_store.probe_model_download", return_value=[]),
        patch("acprof.host.network_preflight.runtime_sources", return_value=([], {"platform_local": True})),
        patch("acprof.host.static_metadata._docker_storage_metadata", return_value={}),
    ):
        report = preflight(task, PROFILES["nlp-cpu"], tmp_path, plan, root=tmp_path / "cache")
    lines = capsys.readouterr().out.splitlines()
    events = [event for line in lines if (event := parse_event(line))]
    request = events[-1]["request"]
    assert request["resolved"] is True
    assert request["questions"] == []
    assert "sources" not in request["download_report"]
    assert request["download_report"]["expected_download_bytes"] == download_bytes
    assert request["download_report"]["disk"]["free_bytes"] == 41_370_132_480
    assert report["expected_download_bytes"] == download_bytes
    assert report["disk"]["free_bytes"] == 41_370_132_480
    assert report["model"]["total_bytes"] == 890_152_505
    assert report["model"]["endpoint"] == "https://hf-mirror.com"
    logged = next(json.loads(line.split(" ", 1)[1]) for line in lines if line.startswith("[network-preflight] "))
    assert logged == report


def test_route_totals_do_not_treat_unknown_as_zero():
    report = summarize_downloads([DownloadSource("model", "https://hf-mirror.com", 100),
                                 DownloadSource("oci", "https://ghcr.io/repo", 50),
                                 DownloadSource("python", "https://files.pythonhosted.org/a.whl", None)])
    assert "direct_download_bytes" not in report
    assert "proxy_download_bytes" not in report
    assert report["public_egress"] == "externally-managed"
    assert (report["expected_download_bytes"]) is None
    with pytest.raises(DownloadPolicyError):
        enforce_download_budget(report, "5GB")

def test_endpoint_fallback_does_not_infer_or_gate_proxy_routes():
    require_source_transition("https://docker.m.daocloud.io/repo", "https://ghcr.io/repo")

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


def test_runtime_sources_uses_valid_recorded_artifact_sizes_without_network(tmp_path):
    copy_dependency_tree(tmp_path)
    profile = PROFILES["nlp-cpu"]
    lock = tmp_path / profile.environment.requirements_lock
    records = read_python_lock(lock)
    metadata = lock.with_suffix(".artifacts.json")
    metadata.write_text(json.dumps({"schema_version": 1, "artifacts": [
        {"url": entry["url"], "sha256": entry["sha256"], "size": 123} for entry in records
    ]}))
    inspections = iter(({"image_id": "sha256:" + "a" * 64}, None))
    with patch.dict("os.environ", {"ACPROF_RUNTIME_IMAGE_SOURCE": "build"}, clear=True):
        sources, _ = runtime_sources(
            profile, tmp_path, inspect=lambda _: next(inspections),
            size_probe=lambda _: pytest.fail("recorded Python artifact size used network"),
            manifest_probe=lambda _: pytest.fail("local platform/build-only path queried registry"),
        )
    assert sources
    assert all(source.estimated_bytes == 123 for source in sources)


def test_runtime_sources_rejects_nonfinite_artifact_metadata(tmp_path):
    copy_dependency_tree(tmp_path)
    profile = PROFILES["nlp-cpu"]
    metadata = (tmp_path / profile.environment.requirements_lock).with_suffix(".artifacts.json")
    metadata.write_text('{"schema_version": 1, "artifacts": [], "corrupt_metric": NaN}')
    with patch.dict("os.environ", {"ACPROF_RUNTIME_IMAGE_SOURCE": "build"}, clear=True):
        with pytest.raises(ValueError, match="invalid dependency artifact metadata JSON|non-finite"):
            runtime_sources(profile, tmp_path, inspect=lambda _: None, manifest_probe=lambda _: None)


def test_runtime_sources_bounds_artifact_metadata_read(tmp_path):
    copy_dependency_tree(tmp_path)
    profile = PROFILES["nlp-cpu"]
    metadata = (tmp_path / profile.environment.requirements_lock).with_suffix(".artifacts.json")
    metadata.write_text(json.dumps({"schema_version": 1, "artifacts": [],
                                    "padding": "x" * MAX_JSON_ARTIFACT_BYTES}))
    assert metadata.stat().st_size > MAX_JSON_ARTIFACT_BYTES
    with patch.dict("os.environ", {"ACPROF_RUNTIME_IMAGE_SOURCE": "build"}, clear=True):
        with pytest.raises(ValueError, match="4 MiB read limit"):
            runtime_sources(profile, tmp_path, inspect=lambda _: None, manifest_probe=lambda _: None)

def test_local_miss_reports_ghcr_and_compressed_upper_bound():
    with patch.dict("os.environ", {"ACPROF_RUNTIME_IMAGE_SOURCE": "auto"}, clear=True):
        sources, state = runtime_sources(PROFILES["nlp-cpu"], PROJECT_ROOT, inspect=lambda _: None,
            manifest_probe=lambda _: {"bytes": 100, "image_id": "sha256:" + "a" * 64})
    assert (state["will_attempt_ghcr_pull"])
    assert (summarize_downloads(sources)["expected_download_bytes"]) == (200)
