"""Layered readers release join containers while preserving independent wide rows."""
from __future__ import annotations

import hashlib
import json
import shutil
import tracemalloc

import pytest

from acprof.results.layout import ArtifactLayout
from acprof.artifacts import file_sha256
from acprof.results.csv import KEY_FIELDS
from acprof.results.layers import (
    LAYER_FILES,
    layer_fields,
    publish_result_rows,
    read_result_layers,
)
from acprof.result_store.protocol import csv_bytes, digest, json_bytes, save_commit


def row_store(root, version, *, count=1000, width=128, cell="0"):
    """Build readable snapshots, without claiming a complete writer replay archive."""
    fields = [*KEY_FIELDS, "environment_class", "gpu_kernel_launch_count_per_request_ncu",
              *(f"extension_{index}" for index in range(width))]
    rows = []
    for index in range(count):
        row = dict.fromkeys(fields, cell)
        row.update(cpu_cores="2", mem_cap_gb="8", gpu_mode="off", input_scale="1", warmup="0",
                   repeat_idx=str(index), environment_class="native_linux",
                   gpu_kernel_launch_count_per_request_ncu="4" if index == 0 else "nan")
        rows.append(row)
    manifest = publish_result_rows(fields, rows, root)
    if version == 2:
        return manifest, rows
    for layer, info in manifest["layers"].items():
        destination = root / LAYER_FILES[layer]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / info["path"], destination)
        info["path"] = LAYER_FILES[layer]
    manifest.pop("generation")
    manifest["schema_version"] = version
    if version == 3:
        layout = ArtifactLayout(root)
        assigned = layer_fields(manifest["wide_fields"])
        for layer, fields in assigned.items():
            path = root / LAYER_FILES[layer]
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(csv_bytes(fields, header=True))
                manifest["layers"][layer] = {"path": LAYER_FILES[layer], "fields": fields,
                                             "row_count": 0, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            manifest["layers"][layer].update(committed_bytes=path.stat().st_size, revision=0)
        manifest.update(run_id="row-assembly-fixture", sequence=0, state="complete",
                        schema_sha256=digest(json_bytes(assigned)))
        layout.result_store_path("commits").mkdir(parents=True)
        layout.result_store_path("lock").touch()
        manifest = save_commit(layout, None, manifest, [], transaction="fixture")
    (root / "result_layers.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, rows


@pytest.mark.parametrize("version", [1, 2, 3])
def test_wide_result_does_not_retain_a_second_complete_row_mapping(tmp_path, version):
    manifest, expected = row_store(tmp_path, version)
    source = (tmp_path / "result_layers.json").read_bytes()
    tracemalloc.start()
    try:
        fields, rows = read_result_layers(tmp_path)
        retained, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert set(fields) == set(manifest["wide_fields"])
    assert rows == expected
    assert all(list(row) == fields for row in rows)
    assert (tmp_path / "result_layers.json").read_bytes() == source
    assert peak - retained < 2 * 1024 * 1024, f"temporary row assembly: {peak - retained:,} bytes"


@pytest.mark.parametrize("version", [1, 2, 3])
def test_returned_rows_keep_missing_profiler_values_order_and_independence(tmp_path, version):
    _, expected = row_store(tmp_path, version, count=3, width=4, cell='诊断,"quoted"\n第二行')
    before = {str(path.relative_to(tmp_path)): file_sha256(path) for path in tmp_path.rglob("*") if path.is_file()}
    fields, first = read_result_layers(tmp_path)
    second_fields, second = read_result_layers(tmp_path)
    assert first == second == expected
    assert fields == second_fields
    assert all(list(row) == fields for row in first)
    assert [row["gpu_kernel_launch_count_per_request_ncu"] for row in first] == ["4", "nan", "nan"]
    first[0]["extension_0"] = "edited local row"
    fields[0] = "edited local fields"
    assert first[1] == expected[1]
    assert second == expected
    assert read_result_layers(tmp_path) == (second_fields, expected)
    assert {str(path.relative_to(tmp_path)): file_sha256(path) for path in tmp_path.rglob("*") if path.is_file()} == before
