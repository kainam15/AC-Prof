import hashlib
import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acprof.artifacts import MAX_JSON_ARTIFACT_BYTES
from acprof.events.preparation import parse_event
from acprof.host.images.runtime import PROJECT_ROOT
from acprof.host.model_store.store import store_lock
from acprof.host.preparation.network_preflight import preflight, registry_manifest, runtime_sources
from acprof.network_policy import (
    DownloadPolicyError,
    DownloadSource,
    enforce_download_budget,
    require_source_transition,
    summarize_downloads,
)
from acprof.runtime.dependency_locks import read_python_lock
from acprof.runtime.profiles import PROFILES
from tests.support.runtime_fixture import copy_dependency_tree


@pytest.mark.parametrize("download_bytes", [890_152_505, None])
def test_download_confirmation_preserves_raw_bytes(tmp_path, capsys, download_bytes):
    task = SimpleNamespace(model_id="demo/model", model_revision="a" * 40,
        task_family="nlp", runtime_backend="transformers_pipeline", model_adapter="family-default")
    plan = {"selected_bytes": 890_152_505, "endpoint": "https://hf-mirror.com"}
    disk = {"total_bytes": 0, "free_bytes": 41_370_132_480,
            "reclaimable_bytes": 0, "remaining_bytes": 40_479_979_975}
    reply = '{"id": 1, "action": "confirm"}\n'
    with (
        patch.dict("os.environ", {"ACPROF_INTERACTIVE_PREPARATION": "1"}, clear=True),
        patch("sys.stdin", io.StringIO(reply)),
        patch("acprof.host.model_store.store.model_sources", return_value=[
            DownloadSource("model", plan["endpoint"], download_bytes),
        ]),
        patch("acprof.host.model_store.store.require_space", return_value=disk),
        patch("acprof.host.model_store.store.probe_model_download", return_value=[]),
        patch("acprof.host.preparation.network_preflight.runtime_sources", return_value=([], {"platform_local": True})),
        patch("acprof.host.metadata.machine.docker_storage_metadata", return_value={}),
    ):
        report = preflight(task, PROFILES["nlp-cpu"], tmp_path, plan, root=tmp_path / "cache")
    lines = capsys.readouterr().out.splitlines()
    events = [event for line in lines if (event := parse_event(line))]
    request = events[-1]["request"]
    assert request["resolved"] is True
    assert request["questions"] == []
    assert request["download_report"]["sources"][0]["category"] == "model"
    assert request["download_report"]["expected_download_bytes"] == download_bytes
    assert request["download_report"]["disk"]["free_bytes"] == 41_370_132_480
    assert report["expected_download_bytes"] == download_bytes
    assert report["disk"]["free_bytes"] == 41_370_132_480
    assert report["model"]["total_bytes"] == 890_152_505
    assert report["model"]["endpoint"] == "https://hf-mirror.com"
    logged = next(json.loads(line.split(" ", 1)[1]) for line in lines if line.startswith("[network-preflight] "))
    assert logged == report


@pytest.mark.parametrize("cache_state", ["missing", "empty", "populated"])
def test_preflight_disk_and_model_estimates_use_the_same_cache_snapshot(tmp_path, cache_state):
    cache_root = tmp_path / "cache"
    if cache_state != "missing":
        cache_root.mkdir()
    if cache_state == "populated":
        (cache_root / "existing-payload.bin").write_bytes(b"cached")
    task = SimpleNamespace(model_id="demo/model", model_revision="a" * 40,
        task_family="nlp", runtime_backend="transformers_pipeline", model_adapter="family-default")
    plan = {"selected_bytes": 2, "endpoint": "https://huggingface.co"}
    missing = DownloadSource("model", plan["endpoint"], 2, cached_bytes=0, cache_status="miss")
    hit = DownloadSource("model", plan["endpoint"], 0, cached_bytes=2, cache_status="hit")

    def disk_for(sources, _root):
        required = sum(source.estimated_bytes for source in sources)
        return {"total_bytes": 2, "free_bytes": 1_000_000, "required_bytes": required,
                "remaining_bytes": 1_000_000 - required, "reclaimable_bytes": 0}

    with (
        patch.dict("os.environ", {}, clear=True),
        patch("acprof.host.model_store.store.model_sources", side_effect=([missing], [hit])),
        patch("acprof.host.model_store.store.require_space", side_effect=disk_for),
        patch("acprof.host.model_store.store.probe_model_download", return_value=[]),
        patch("acprof.host.model_store.store.store_lock", wraps=store_lock) as lock,
    ):
        report = preflight(task, None, tmp_path, plan, root=cache_root,
                           runtime_plan=([], {"kind": "process"}))

    assert report["expected_download_bytes"] == 0
    assert report["model"]["cached_bytes"] == 2
    assert report["disk"]["required_bytes"] == report["expected_download_bytes"]
    assert lock.call_count == (1 if cache_state == "populated" else 0)
    if cache_state == "missing":
        assert not cache_root.exists()
    elif cache_state == "empty":
        assert list(cache_root.iterdir()) == []
    else:
        assert (cache_root / ".lock").is_file()


def test_model_only_budget_allows_cached_model_with_unknown_oci_size(tmp_path):
    """Regression for hardware run #2: GHCR manifest may have no size."""
    task = SimpleNamespace(model_id="demo/model", model_revision="a" * 40,
        task_family="nlp", runtime_backend="transformers_pipeline", model_adapter="family-default")
    plan = {"selected_bytes": 989_138, "endpoint": "https://hf-mirror.com"}
    model = DownloadSource("model", plan["endpoint"], 0, cached_bytes=989_138, cache_status="hit")
    oci = DownloadSource("oci", "https://ghcr.io/demo/runtime", None, cache_status="miss")
    settings = {"ACPROF_MODEL_MAX_DOWNLOAD": "4GB", "ACPROF_MAX_DOWNLOAD": ""}
    with (
        patch.dict("os.environ", settings, clear=True),
        patch("acprof.host.model_store.store.model_sources", return_value=[model]),
        patch("acprof.host.model_store.store.require_space", return_value={"free_bytes": 35_000_000_000}),
        patch("acprof.host.model_store.store.probe_model_download", return_value=[]),
        patch("acprof.host.preparation.network_preflight.runtime_sources", return_value=(
            [oci], {"platform_local": False, "will_attempt_ghcr_pull": True},
        )),
        patch("acprof.host.metadata.machine.docker_storage_metadata", return_value={}),
    ):
        report = preflight(task, PROFILES["nlp-cpu"], tmp_path, plan, root=tmp_path / "cache")

    assert report["expected_download_bytes"] is None  # OCI is still unknown, not zero.
    assert report["max_download_bytes"] is None
    assert report["model_max_download_bytes"] == 4_000_000_000
    assert report["runtime"]["will_attempt_ghcr_pull"]
    with pytest.raises(DownloadPolicyError, match="未知大小"):
        enforce_download_budget(report, "4GB")


@pytest.mark.parametrize("model_bytes", [None, 4_000_000_001])
def test_model_only_budget_blocks_unknown_or_oversized_model_before_oci(tmp_path, model_bytes):
    task = SimpleNamespace(model_id="demo/model", model_revision="a" * 40,
        task_family="nlp", runtime_backend="transformers_pipeline", model_adapter="family-default")
    plan = {"selected_bytes": model_bytes, "endpoint": "https://hf-mirror.com"}
    with (
        patch.dict("os.environ", {"ACPROF_MODEL_MAX_DOWNLOAD": "4GB", "ACPROF_MAX_DOWNLOAD": ""}, clear=True),
        patch("acprof.host.model_store.store.model_sources", return_value=[
            DownloadSource("model", plan["endpoint"], model_bytes),
        ]),
        patch("acprof.host.model_store.store.require_space", return_value={"free_bytes": 35_000_000_000}),
        patch("acprof.host.model_store.store.probe_model_download") as probe,
        patch("acprof.host.preparation.network_preflight.runtime_sources") as runtime,
        pytest.raises(DownloadPolicyError, match="ACPROF_MODEL_MAX_DOWNLOAD"),
    ):
        preflight(task, PROFILES["nlp-cpu"], tmp_path, plan, root=tmp_path / "cache")
    probe.assert_not_called()
    runtime.assert_not_called()


def test_total_budget_remains_strict_with_unknown_oci_size(tmp_path):
    task = SimpleNamespace(model_id="demo/model", model_revision="a" * 40,
        task_family="nlp", runtime_backend="transformers_pipeline", model_adapter="family-default")
    plan = {"selected_bytes": 2, "endpoint": "https://hf-mirror.com"}
    with (
        patch.dict("os.environ", {"ACPROF_MODEL_MAX_DOWNLOAD": "4GB", "ACPROF_MAX_DOWNLOAD": "4GB"}, clear=True),
        patch("acprof.host.model_store.store.model_sources", return_value=[
            DownloadSource("model", plan["endpoint"], 0, cached_bytes=2),
        ]),
        patch("acprof.host.model_store.store.require_space", return_value={"free_bytes": 35_000_000_000}),
        patch("acprof.host.model_store.store.probe_model_download", return_value=[]),
        patch("acprof.host.preparation.network_preflight.runtime_sources", return_value=(
            [DownloadSource("oci", "https://ghcr.io/demo/runtime", None)],
            {"platform_local": False},
        )),
        patch("acprof.host.metadata.machine.docker_storage_metadata", return_value={}),
        pytest.raises(DownloadPolicyError, match="未知大小"),
    ):
        preflight(task, PROFILES["nlp-cpu"], tmp_path, plan, root=tmp_path / "cache")

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
            manifest_probe=lambda ref: {"bytes": 64, "image_id": "sha256:" + "b" * 64}
            if "docker/dockerfile:" in ref else pytest.fail("unexpected base/runtime registry access"),
        )
    assert sources
    assert all(source.estimated_bytes == 123 for source in sources if source.category == "python")


def test_runtime_sources_preserves_recorded_zero_size_without_network(tmp_path):
    copy_dependency_tree(tmp_path)
    profile = PROFILES["nlp-cpu"]
    lock = tmp_path / profile.environment.requirements_lock
    records = read_python_lock(lock)
    metadata = lock.with_suffix(".artifacts.json")
    metadata.write_text(json.dumps({"schema_version": 1, "artifacts": [
        {"url": entry["url"], "sha256": entry["sha256"], "size": 0} for entry in records
    ]}))
    inspections = iter(({"image_id": "sha256:" + "a" * 64}, None))
    with patch.dict("os.environ", {"ACPROF_RUNTIME_IMAGE_SOURCE": "build"}, clear=True):
        sources, _ = runtime_sources(
            profile, tmp_path, inspect=lambda _: next(inspections),
            size_probe=lambda _: pytest.fail("recorded zero-byte artifact size used network"),
            manifest_probe=lambda ref: {"bytes": 64, "image_id": "sha256:" + "b" * 64}
            if "docker/dockerfile:" in ref else pytest.fail("unexpected base/runtime registry access"),
        )
    assert sources
    assert all(source.estimated_bytes == 0 for source in sources if source.category == "python")


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

@pytest.mark.parametrize(
    ("layer_size", "config_digest"),
    [
        (True, "sha256:" + "a" * 64),
        (-1, "sha256:" + "a" * 64),
        (1, "not-a-digest"),
    ],
)
def test_registry_manifest_rejects_invalid_budget_metadata(layer_size, config_digest):
    payload = {"layers": [{"size": layer_size}], "config": {"size": 10, "digest": config_digest}}
    result = SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")
    with patch("acprof.host.preparation.network_preflight.host_command.run_command", return_value=result):
        assert registry_manifest("ghcr.io/example/runtime:tag") is None


def test_local_miss_reports_ghcr_and_compressed_upper_bound():
    with patch.dict("os.environ", {"ACPROF_RUNTIME_IMAGE_SOURCE": "auto"}, clear=True):
        sources, state = runtime_sources(PROFILES["nlp-cpu"], PROJECT_ROOT, inspect=lambda _: None,
            manifest_probe=lambda _: {"bytes": 100, "image_id": "sha256:" + "a" * 64})
    assert (state["will_attempt_ghcr_pull"])
    assert (summarize_downloads(sources)["expected_download_bytes"]) == (200)


def test_local_build_preflights_frontend_and_base_from_selected_mirror():
    references = []

    def manifest(reference):
        references.append(reference)
        return {"bytes": 4096, "image_id": "sha256:" + "a" * 64}

    with patch.dict("os.environ", {"ACPROF_RUNTIME_IMAGE_SOURCE": "build",
                                   "ACPROF_DOCKER_HUB_MIRROR": "mirror.example/docker.io"}, clear=True):
        sources, state = runtime_sources(PROFILES["nlp-cpu"], PROJECT_ROOT,
                                         inspect=lambda _: None, manifest_probe=manifest)
    frontend = [source for source in sources if source.stage == "buildkit-frontend"]
    assert len(frontend) == 1
    assert frontend[0].estimated_bytes == 4096
    assert frontend[0].cache_status == "unknown"
    assert all(ref.startswith("mirror.example/docker.io/") and "@sha256:" in ref for ref in references)
    assert len(references) == 2
    assert state["will_attempt_ghcr_pull"] is False


def test_auto_preflight_discloses_conditional_build_sources():
    with patch.dict("os.environ", {"ACPROF_RUNTIME_IMAGE_SOURCE": "auto",
                                   "ACPROF_DOCKER_HUB_MIRROR": "mirror.example/docker.io",
                                   "ACPROF_PYPI_ARTIFACT_MIRROR": "tencent"}, clear=True):
        _, state = runtime_sources(PROFILES["nlp-cpu"], PROJECT_ROOT, inspect=lambda _: None,
                                   manifest_probe=lambda _: None)
    fallback = state["alternatives"]["local_build_sources"]
    assert any("mirror.example/docker.io/docker/dockerfile" in entry for entry in fallback)
    assert any("mirrors.cloud.tencent.com" in entry for entry in fallback)
    assert not any("files.pythonhosted.org" in entry for entry in fallback)


def test_registry_manifest_reads_only_selected_platform_with_buildx_and_verifies_digest():
    manifest = json.dumps({"schemaVersion": 2, "layers": [{"size": 100}],
                           "config": {"size": 25, "digest": "sha256:" + "a" * 64}})
    digest = "sha256:" + hashlib.sha256(manifest.encode()).hexdigest()
    index = json.dumps({"schemaVersion": 2, "manifests": [
        {"digest": digest, "platform": {"os": "linux", "architecture": "amd64"}},
        {"digest": "sha256:" + "b" * 64, "platform": {"os": "linux", "architecture": "arm64"}},
    ]})
    reference = "mirror.example/docker/dockerfile:1@sha256:" + hashlib.sha256(index.encode()).hexdigest()
    calls = []

    def inspect(command, **kwargs):
        assert command[:4] == ["docker", "buildx", "imagetools", "inspect"]
        assert command[-1] == "--raw"
        assert 0 < kwargs["timeout"] <= 20
        calls.append(command[4])
        return SimpleNamespace(returncode=0, stdout=index if len(calls) == 1 else manifest, stderr="")

    evidence = []
    with patch("acprof.host.preparation.network_preflight.host_command.run_command", side_effect=inspect):
        assert registry_manifest(reference, evidence=evidence) == {"bytes": 125, "image_id": "sha256:" + "a" * 64}
    assert calls == [reference, "mirror.example/docker/dockerfile:1@" + digest]
    assert evidence[0]["status"] == "passed"
    assert evidence[0]["network_context"] == "host-docker-cli"


def test_registry_manifest_rejects_content_changed_by_mirror():
    response = SimpleNamespace(returncode=0, stdout=json.dumps({"layers": [],
        "config": {"size": 1, "digest": "sha256:" + "a" * 64}}), stderr="")
    evidence = []
    with patch("acprof.host.preparation.network_preflight.host_command.run_command", return_value=response):
        assert registry_manifest("mirror.example/image@sha256:" + "b" * 64, evidence=evidence) is None
    assert evidence[0]["reason"] == "integrity"


def test_registry_manifest_reports_arm_only_image_without_layer_download():
    document = {"schemaVersion": 2, "manifests": [
        {"digest": "sha256:" + "a" * 64, "platform": {"os": "linux", "architecture": "arm64"}},
    ]}
    evidence = []
    with patch("acprof.system.command.run_command", return_value=SimpleNamespace(
        returncode=0, stdout=json.dumps(document), stderr="",
    )) as command:
        assert registry_manifest("ghcr.io/example/runtime:tag", evidence=evidence) is None
    command.assert_called_once()
    assert evidence[0]["reason"] == "platform"
