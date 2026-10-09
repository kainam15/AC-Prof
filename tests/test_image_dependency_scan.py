"""Historical image package provenance, scanning and persistent cache regressions."""
from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import patch

import pytest

from acprof.host.image_dependencies import resolve_image_dependencies
from acprof.host.image_dependency_scan import _docker_read, read_snapshot
from acprof.host.image_management import DockerConnection, ImageInventory, ManagedImage

BASE = "sha256:" + "1" * 64
RUNTIME = "sha256:" + "2" * 64
WEIGHTS = "sha256:" + "3" * 64
CONNECTION = DockerConnection(("--context", "local-test"), "local-test")


@pytest.fixture(autouse=True)
def cache_in_tempdir(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))


def base() -> ManagedImage:
    return ManagedImage(BASE, ("acprof-platform-cpu:old",), 200, "", "base",
                        acprof=True, platform_id="cpu", platform_key="old-base",
                        dependency_stage="base")


def runtime() -> ManagedImage:
    return ManagedImage(RUNTIME, ("acprof-runtime-env:old",), 300, "", "runtime",
                        acprof=True, platform_id="cpu", platform_key="old-base",
                        environment_id="old-environment", environment_key="old-runtime",
                        dependency_stage="runtime", parent_id=BASE, parent_source="metadata",
                        ancestor_ids=(BASE,))


def image_manifest(item, python, system):
    result = {"schema_version": 1, "platform_id": item.platform_id,
              "platform_build_fingerprint": item.platform_key,
              "packages": python, "system_packages": system}
    if item.kind == "runtime":
        result.update(environment_id=item.environment_id,
                      environment_build_fingerprint=item.environment_key)
    return json.dumps(result)


def test_historical_manifest_is_tied_to_image_and_not_current_lock():
    node = base()
    with patch("acprof.host.image_dependency_scan._docker_read",
               return_value=image_manifest(node, {"torch": "2.5.1"}, {"base-files:amd64": "1"})) as read:
        result = resolve_image_dependencies(ImageInventory(CONNECTION, "daemon-a", (node,)), BASE)
        resolved = result.images[0]
        assert resolved.dependency_source == "historical-manifest"
        assert resolved.dependency_scope == "full"
        assert resolved.python_dependencies == (("torch", "2.5.1"),)
        assert resolved.system_dependencies == (("base-files:amd64", "1"),)
        assert read.call_count == 1
    # Subsequent queries must never call Docker when immutable image ID/daemon matches.
    with patch("acprof.host.image_dependency_scan._docker_read", side_effect=AssertionError("unexpected Docker")):
        again = resolve_image_dependencies(ImageInventory(CONNECTION, "daemon-a", (node,)), BASE)
    assert again.images[0].dependency_source == "historical-manifest"
    # Cache is not shared across different daemon identities.
    with patch("acprof.host.image_dependency_scan._docker_read", side_effect=RuntimeError("unavailable")):
        other = resolve_image_dependencies(ImageInventory(CONNECTION, "daemon-b", (node,)), BASE)
    assert other.images[0].dependency_source == "unknown"


def test_missing_manifest_falls_back_to_live_scan_and_reuses_result():
    node = base()
    responses = [RuntimeError("no manifest"), json.dumps({"python": {"torch": "1.9"},
                                                           "system": {"libc6:amd64": "2.1"}})]
    with patch("acprof.host.image_dependency_scan._docker_read", side_effect=responses) as read:
        snapshot, error = read_snapshot(CONNECTION, "daemon", node)
    assert not error
    assert snapshot.source == "scan"
    assert snapshot.python == (("torch", "1.9"),)
    assert read.call_count == 2
    with patch("acprof.host.image_dependency_scan._docker_read", side_effect=AssertionError("uncached")):
        assert read_snapshot(CONNECTION, "daemon", node)[0] == snapshot


def test_identity_mismatch_does_not_claim_historical_manifest_as_verified():
    node = base()
    wrong = json.loads(image_manifest(node, {"torch": "999"}, {}))
    wrong["platform_build_fingerprint"] = "not-this-image"
    with patch("acprof.host.image_dependency_scan._docker_read",
               side_effect=[json.dumps(wrong), RuntimeError("scan failed")]):
        snapshot, note = read_snapshot(CONNECTION, "daemon", node)
    assert snapshot is None
    assert "构建记录不可用" in note and "实际扫描失败" in note


def test_unrecognized_build_history_does_not_execute_image_code():
    node = replace(base(), dependency_stage="")
    with patch("acprof.host.image_dependency_scan._docker_read", side_effect=AssertionError("not allowed")):
        snapshot, note = read_snapshot(CONNECTION, "daemon", node)
    assert snapshot is None and "构建步骤" in note


def test_runtime_delta_is_computed_from_historical_parent_not_current_lock():
    parent, child = base(), runtime()
    data = {BASE: image_manifest(parent, {"torch": "1", "numpy": "1"}, {"libc6:amd64": "2"}),
            RUNTIME: image_manifest(child, {"torch": "1", "numpy": "2", "transformers": "4"},
                                    {"libc6:amd64": "2"})}
    with patch("acprof.host.image_dependency_scan._docker_read",
               side_effect=lambda conn, key, entry, args: data[key]):
        resolved = resolve_image_dependencies(ImageInventory(CONNECTION, "daemon", (parent, child)), RUNTIME)
    changed = resolved.images[1]
    assert changed.dependency_source == "historical-manifest"
    assert changed.dependency_scope == "delta"
    assert changed.python_dependencies == (("numpy", "2"), ("transformers", "4"))
    assert changed.system_dependencies == ()
    # Rehydration is metadata-only after TUI periodic refresh.
    with patch("acprof.host.image_dependency_scan._docker_read", side_effect=AssertionError("unexpected Docker")):
        cached = resolve_image_dependencies(ImageInventory(CONNECTION, "daemon", (parent, child)),
                                            RUNTIME, cached_only=True)
    assert cached.images[1].python_dependencies == changed.python_dependencies


def test_missing_parent_never_calls_complete_snapshot_new_packages():
    node = replace(runtime(), ancestor_ids=(), parent_id="", parent_source="missing")
    with patch("acprof.host.image_dependency_scan._docker_read",
               return_value=image_manifest(node, {"torch": "1"}, {"libc6:amd64": "2"})):
        resolved = resolve_image_dependencies(ImageInventory(CONNECTION, "daemon", (node,)),
                                              RUNTIME).images[0]
    assert resolved.dependency_scope == "full"
    assert "完整安装集合" in resolved.dependency_note


def test_inherited_model_requires_parent_evidence_and_matching_history():
    parent, child = base(), runtime()
    weights = ManagedImage(WEIGHTS, ("acprof-weights-foo",), 400, "", "weights",
                           acprof=True, dependency_stage="weights", parent_id=RUNTIME,
                           parent_source="metadata", ancestor_ids=(BASE, RUNTIME),
                           platform_id="cpu", platform_key="old-base",
                           environment_id="old-environment", environment_key="old-runtime")
    data = {BASE: image_manifest(parent, {"torch": "1"}, {}),
            RUNTIME: image_manifest(child, {"torch": "1", "requests": "2"}, {})}
    with patch("acprof.host.image_dependency_scan._docker_read",
               side_effect=lambda conn, key, entry, args: data[key]):
        resolved = resolve_image_dependencies(ImageInventory(CONNECTION, "daemon", (parent, child, weights)), WEIGHTS)
    assert resolved.images[2].dependency_source == "inherited-historical"
    assert replace(weights, dependency_stage="").dependency_stage == ""
    without_stage = replace(weights, dependency_stage="")
    with patch("acprof.host.image_dependency_scan._docker_read", side_effect=AssertionError("cached")):
        result = resolve_image_dependencies(ImageInventory(CONNECTION, "daemon", (parent, child, without_stage)),
                                            WEIGHTS, cached_only=True)
    assert result.images[2].dependency_source == "unknown"




def test_periodic_inventory_refresh_restores_cached_history_without_docker():
    from acprof.host.image_dependencies import describe_dependencies

    node = base()
    with patch("acprof.host.image_dependency_scan._docker_read",
               return_value=image_manifest(node, {"torch": "1"}, {"libc6:amd64": "2"})):
        resolve_image_dependencies(ImageInventory(CONNECTION, "daemon", (node,)), BASE)
    with patch("acprof.host.image_dependency_scan._docker_read", side_effect=AssertionError("refresh ran Docker")):
        refreshed = describe_dependencies(ImageInventory(CONNECTION, "daemon", (node,)))
    assert refreshed.images[0].dependency_source == "historical-manifest"
    assert refreshed.images[0].python_dependencies == (("torch", "1"),)


def test_cached_history_cannot_override_changed_build_history():
    node = base()
    with patch("acprof.host.image_dependency_scan._docker_read",
               return_value=image_manifest(node, {"torch": "1"}, {})):
        resolve_image_dependencies(ImageInventory(CONNECTION, "daemon", (node,)), BASE)
    suspect = replace(node, dependency_stage="")
    with patch("acprof.host.image_dependency_scan._docker_read", side_effect=AssertionError("unexpected Docker")):
        result = resolve_image_dependencies(ImageInventory(CONNECTION, "daemon", (suspect,)),
                                            BASE, cached_only=True)
    assert result.images[0].dependency_source == "unknown"

def test_docker_scan_enforces_restricted_runtime():
    with patch("acprof.host.image_dependency_scan.run_docker_command", return_value="{}") as run:
        _docker_read(CONNECTION, BASE, "python", ["-I", "-S", "-B", "-c", "print(1)"])
    argv = run.call_args.args[0]
    for arg in ("--network", "none", "--read-only", "--cap-drop", "ALL", "no-new-privileges",
                "--pids-limit", "64", "--pull", "never", "--user", "65534:65534"):
        assert arg in argv
    assert argv[-6:] == [BASE, "-I", "-S", "-B", "-c", "print(1)"]
    assert argv[argv.index("--entrypoint") + 1] == "python"
