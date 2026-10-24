"""Interrupted and concurrent publications retain complete result snapshots."""
import csv
import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from acprof import result_layers as layers
from acprof.result_csv import KEY_FIELDS, ResultValidationError

FIELDS = [*KEY_FIELDS, "latency_p50_s", "gpu_energy_total_j",
          "container_mem_usage_peak_bytes", "packet_total_wire_bytes_per_request",
          "gpu_kernel_launch_count_per_request_ncu"]


def _rows(value):
    return [{
        "cpu_cores": "2", "mem_cap_gb": "4", "gpu_mode": "on",
        "input_scale": "64", "warmup": "0", "repeat_idx": "0",
        **dict.fromkeys(FIELDS[len(KEY_FIELDS):], str(value)),
    }]


def _publish(root, value):
    return layers.publish_result_rows(FIELDS, _rows(value), root)


def _capture_files(root, manifest):
    return {record["path"]: (root / record["path"]).read_bytes()
            for record in manifest["layers"].values()}


@pytest.mark.parametrize("layer", ["summary", "performance", "resources", "energy", "network", "ncu"])
@pytest.mark.parametrize("after_write", [False, True])
def test_failed_layer_write_keeps_prior_snapshot(tmp_path, monkeypatch, layer, after_write):
    old_manifest = _publish(tmp_path, 1)
    old_result = layers.read_result_layers(tmp_path)
    old_files = _capture_files(tmp_path, old_manifest)
    write = layers.atomic_write

    def fail(destination, writer, **kwargs):
        if destination.name == Path(layers.LAYER_FILES[layer]).name:
            if after_write:
                write(destination, writer, **kwargs)
            assert layers.read_result_layers(tmp_path) == old_result
            raise OSError("injected layer write failure")
        return write(destination, writer, **kwargs)

    monkeypatch.setattr(layers, "atomic_write", fail)
    with pytest.raises(OSError, match="injected layer write"):
        _publish(tmp_path, 2)
    assert layers.read_result_layers(tmp_path) == old_result
    assert _capture_files(tmp_path, old_manifest) == old_files
    assert not list((tmp_path / layers.GENERATIONS_DIR).glob(".pending-*"))


@pytest.mark.parametrize("stage", ["generation_sync", "generation_rename", "parent_sync", "manifest_write"])
def test_failed_publication_boundary_keeps_prior_snapshot(tmp_path, monkeypatch, stage):
    old_manifest = _publish(tmp_path, 1)
    old_result = layers.read_result_layers(tmp_path)
    old_files = _capture_files(tmp_path, old_manifest)

    def fail(*args, **kwargs):
        raise OSError(f"injected {stage}")

    if stage == "generation_rename":
        monkeypatch.setattr(layers.os, "rename", fail)
    elif stage == "manifest_write":
        monkeypatch.setattr(layers, "atomic_write_json", fail)
    else:
        sync = layers.sync_directory

        def fail_sync(directory):
            if ((stage == "generation_sync" and directory.name.startswith(".pending-"))
                    or (stage == "parent_sync" and directory.name == layers.GENERATIONS_DIR)):
                fail()
            return sync(directory)

        monkeypatch.setattr(layers, "sync_directory", fail_sync)
    with pytest.raises(OSError, match="injected"):
        _publish(tmp_path, 2)
    assert layers.read_result_layers(tmp_path) == old_result
    assert _capture_files(tmp_path, old_manifest) == old_files


def test_failure_after_manifest_replacement_leaves_complete_new_snapshot(tmp_path, monkeypatch):
    old_manifest = _publish(tmp_path, 1)
    old_files = _capture_files(tmp_path, old_manifest)
    write = layers.atomic_write_json

    def fail_after_replace(*args, **kwargs):
        write(*args, **kwargs)
        raise OSError("injected post-replacement directory sync failure")

    monkeypatch.setattr(layers, "atomic_write_json", fail_after_replace)
    with pytest.raises(OSError, match="post-replacement"):
        _publish(tmp_path, 2)
    assert layers.read_result_layers(tmp_path)[1] == _rows(2)
    assert _capture_files(tmp_path, old_manifest) == old_files


def test_identical_publication_is_idempotent_without_rewriting(tmp_path, monkeypatch):
    manifest = _publish(tmp_path, 1)
    paths = [tmp_path / layers.MANIFEST_NAME, *layers.result_layer_paths(tmp_path, manifest).values()]
    before = [(path.stat().st_ino, path.stat().st_mtime_ns) for path in paths]

    def unexpected_write(*args, **kwargs):
        raise AssertionError("identical result must not rewrite immutable files or its manifest")

    monkeypatch.setattr(layers, "atomic_write", unexpected_write)
    monkeypatch.setattr(layers, "atomic_write_json", unexpected_write)
    assert _publish(tmp_path, 1) == manifest
    assert [(path.stat().st_ino, path.stat().st_mtime_ns) for path in paths] == before


def test_corrupted_generation_is_never_silently_repaired(tmp_path):
    manifest = _publish(tmp_path, 1)
    damaged = tmp_path / manifest["layers"]["energy"]["path"]
    damaged.write_text("corrupt\n")
    with pytest.raises(ResultValidationError, match="immutable result layer"):
        _publish(tmp_path, 1)
    assert damaged.read_text() == "corrupt\n"
    with pytest.raises(ResultValidationError, match="modified energy"):
        layers.read_result_layers(tmp_path)


def test_concurrent_different_publications_never_mix_layers(tmp_path, monkeypatch):
    _publish(tmp_path, 0)
    gate = Barrier(2)
    write = layers.atomic_write_json

    def publish_after_both_generations_exist(*args, **kwargs):
        gate.wait(timeout=10)
        write(*args, **kwargs)
        # A concurrent complete winner is allowed; a mixture is never allowed.
        assert layers.read_result_layers(tmp_path)[1] in [_rows(1), _rows(2)]

    monkeypatch.setattr(layers, "atomic_write_json", publish_after_both_generations_exist)
    with ThreadPoolExecutor(max_workers=2) as pool:
        manifests = list(pool.map(lambda value: _publish(tmp_path, value), (1, 2)))
    assert manifests[0]["generation"] != manifests[1]["generation"]
    current = json.loads((tmp_path / layers.MANIFEST_NAME).read_text())
    assert current in manifests
    assert layers.read_result_layers(tmp_path)[1] in [_rows(1), _rows(2)]
    for manifest in manifests:
        for path, raw in _capture_files(tmp_path, manifest).items():
            assert hashlib.sha256(raw).hexdigest() == next(
                record["sha256"] for record in manifest["layers"].values() if record["path"] == path)


def test_concurrent_identical_generation_is_installed_once(tmp_path, monkeypatch):
    gate = Barrier(2)
    rename = os.rename

    def simultaneous_rename(*args, **kwargs):
        gate.wait(timeout=10)
        return rename(*args, **kwargs)

    monkeypatch.setattr(layers.os, "rename", simultaneous_rename)
    with ThreadPoolExecutor(max_workers=2) as pool:
        manifests = list(pool.map(lambda _: _publish(tmp_path, 1), (0, 1)))
    assert manifests[0] == manifests[1]
    assert layers.read_result_layers(tmp_path)[1] == _rows(1)
    assert [path.name for path in (tmp_path / layers.GENERATIONS_DIR).iterdir()] == [manifests[0]["generation"]]


@pytest.mark.parametrize("part", ["manifest", "generation", "profiling", "energy"])
def test_reader_rejects_symlink_files_and_ancestors(tmp_path, part):
    manifest = _publish(tmp_path, 1)
    generation = tmp_path / layers.GENERATIONS_DIR / manifest["generation"]
    target = {"manifest": tmp_path / layers.MANIFEST_NAME,
              "generation": generation,
              "profiling": generation / "profiling",
              "energy": generation / "energy.csv"}[part]
    renamed = tmp_path / "moved"
    target.rename(renamed)
    target.symlink_to(renamed, target_is_directory=renamed.is_dir())
    with pytest.raises(ResultValidationError, match="symlink"):
        layers.read_result_layers(tmp_path)


def test_publisher_rejects_symlink_generation_root(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "results"
    root.mkdir()
    (root / layers.GENERATIONS_DIR).symlink_to(outside, target_is_directory=True)
    with pytest.raises(ResultValidationError, match="symlink"):
        _publish(root, 1)
    assert not list(outside.iterdir())


@pytest.mark.parametrize("bad_path", ["../escape.csv", "/tmp/escape.csv", "result_generations/../energy.csv"])
def test_manifest_paths_cannot_escape_or_alias_generation(tmp_path, bad_path):
    manifest = _publish(tmp_path, 1)
    manifest["layers"]["energy"]["path"] = bad_path
    (tmp_path / layers.MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(ResultValidationError, match="energy layer schema"):
        layers.read_result_layers(tmp_path)


def test_wide_export_cannot_write_anywhere_in_generations(tmp_path):
    manifest = _publish(tmp_path, 1)
    destinations = [tmp_path / manifest["layers"]["energy"]["path"],
                    tmp_path / layers.GENERATIONS_DIR / "new.csv",
                    tmp_path / layers.LAYER_FILES["energy"]]
    for destination in destinations:
        with pytest.raises(ResultValidationError, match="overwrite a result layer"):
            layers.export_result_layers(tmp_path, destination)


def _write_legacy_snapshot(root):
    assignments = {"summary": [], "performance": ["latency_p50_s"],
                   "resources": ["container_mem_usage_peak_bytes"],
                   "energy": ["gpu_energy_total_j"],
                   "network": ["packet_total_wire_bytes_per_request"],
                   "ncu": ["gpu_kernel_launch_count_per_request_ncu"]}
    manifest = {"schema_version": 1, "wide_fields": FIELDS, "row_count": 1, "layers": {}}
    for name, columns in assignments.items():
        fields = [*KEY_FIELDS, *columns]
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_rows(1))
        raw = stream.getvalue().encode()
        path = root / layers.LAYER_FILES[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        manifest["layers"][name] = {"path": layers.LAYER_FILES[name], "fields": fields,
                                   "row_count": 1, "sha256": hashlib.sha256(raw).hexdigest()}
    (root / layers.MANIFEST_NAME).write_text(json.dumps(manifest))
    return manifest


def test_legacy_snapshot_remains_readable_and_upgrade_keeps_original_files(tmp_path):
    legacy = _write_legacy_snapshot(tmp_path)
    originals = _capture_files(tmp_path, legacy)
    assert layers.read_result_layers(tmp_path)[1] == _rows(1)
    current = _publish(tmp_path, 2)
    assert current["schema_version"] == 2
    assert layers.read_result_layers(tmp_path)[1] == _rows(2)
    assert _capture_files(tmp_path, legacy) == originals


def test_legacy_source_identity_check_is_preserved(tmp_path):
    manifest = _write_legacy_snapshot(tmp_path)
    source = tmp_path / "original.csv"
    source.write_bytes(b"old source")
    manifest.update(source_name=source.name, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (tmp_path / layers.MANIFEST_NAME).write_text(json.dumps(manifest))
    source.write_bytes(b"new source")
    with pytest.raises(ResultValidationError, match="source CSV has changed"):
        layers.read_result_layers(tmp_path)
