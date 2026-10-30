"""Statistics CSV exports retain format and publish only a complete document."""
import csv
import gc
import json
import tracemalloc
from copy import deepcopy

import pytest

from acprof.analysis.export import export_analysis


def test_large_export_does_not_retain_all_flattened_rows(tmp_path):
    payload = {"schema_version": 1, "groups": [
        {"metric": "latency_app_s", "configuration": {"cpu": index % 8, "gpu": "off"},
         "mean": index / 1000, "details": {"values": list(range(32)), "note": "验证记录" * 100}}
        for index in range(4000)
    ]}
    gc.collect()
    tracemalloc.start()
    try:
        export_analysis(payload, tmp_path / "large.csv", format="csv")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Exclude the already-owned input. Only a header and current flattened row
    # are needed; leave ample room for CSV and atomic-publication buffering.
    assert peak < 2 * 1024 * 1024, f"export retained {peak} bytes beyond its input"
    with (tmp_path / "large.csv").open(encoding="utf-8", newline="") as stream:
        assert sum(1 for _ in csv.DictReader(stream)) == 4000


def test_late_columns_nested_lists_and_identity_keep_their_original_order(tmp_path):
    payload = {"schema_version": 1, "kind": "experiment_comparison", "seed": 17,
               "source_sha256": {"first.json": "a" * 64}, "status": "unknown",
               "quality_reasons": ["未验证"], "groups": [
        {"metric": "latency_app_s", "limits": {"cpu": 2}, "values": [None, True, "中文,\"引号\"\n行"],
         "mean": None, "empty": {}, "tuple": (1, 2)},
        {"metric": "latency_s", "limits": {"memory": 4}, "values": [], "late": "末列\n行"},
    ]}
    original = deepcopy(payload)
    output = tmp_path / "statistics.csv"
    export_analysis(payload, output, format="csv")
    with output.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert reader.fieldnames == ["analysis.schema_version", "analysis.kind", "analysis.seed",
                                "analysis.source_sha256.first.json", "analysis.status", "analysis.quality_reasons",
                                "metric", "limits.cpu", "values", "mean", "tuple", "limits.memory", "late"]
    assert rows[0]["analysis.seed"] == rows[1]["analysis.seed"] == "17"
    assert rows[0]["analysis.source_sha256.first.json"] == "a" * 64
    assert rows[0]["analysis.quality_reasons"] == '["未验证"]'
    assert rows[0]["values"] == '[null, true, "中文,\\\"引号\\\"\\n行"]'
    assert rows[0]["mean"] == rows[0]["limits.memory"] == rows[0]["late"] == ""
    assert rows[0]["tuple"] == "(1, 2)"
    assert rows[1]["limits.cpu"] == ""
    assert rows[1]["limits.memory"] == "4"
    assert rows[1]["values"] == "[]"
    assert rows[1]["late"] == "末列\n行"
    assert output.read_bytes().endswith(b"\n")
    assert payload == original


@pytest.mark.parametrize(("group", "expected"), [({"x.y": 1, "x": {"y": 2}}, "2"),
                                               ({"x": {"y": 1}, "x.y": 2}, "2")])
def test_flattened_collisions_keep_last_value_and_first_column(tmp_path, group, expected):
    payload = {"seed": 17, "groups": [{**group, "analysis.seed": 99, "after": True}]}
    output = tmp_path / "collisions.csv"
    export_analysis(payload, output, format="csv")
    with output.open(newline="") as stream:
        reader = csv.DictReader(stream)
        row, = reader
    assert reader.fieldnames == ["analysis.seed", "x.y", "after"]
    assert row == {"analysis.seed": "99", "x.y": expected, "after": "True"}


@pytest.mark.parametrize("groups", [None, {}, [1], [{}, None], [], [{}], [{"empty": {}}]])
def test_invalid_or_columnless_groups_never_publish(tmp_path, groups):
    with pytest.raises(ValueError, match="statistics.*(groups|rows)"):
        export_analysis({"groups": groups}, tmp_path / "invalid.csv", format="csv")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), object()])
def test_late_list_serialization_error_discards_partial_output(tmp_path, bad):
    payload = {"groups": [{"value": [1]}, {"value": [bad]}]}
    with pytest.raises((ValueError, TypeError)):
        export_analysis(payload, tmp_path / "invalid.csv", format="csv")
    assert list(tmp_path.iterdir()) == []


def test_lists_are_serialized_once_and_existing_output_is_rejected_before_work(tmp_path, monkeypatch):
    payload = {"quality_reasons": ["unknown"], "groups": [{"value": [i]} for i in range(5)]}
    calls = []
    original = json.dumps

    def observe(value, **kwargs):
        calls.append(value)
        return original(value, **kwargs)

    monkeypatch.setattr(json, "dumps", observe)
    output = tmp_path / "statistics.csv"
    export_analysis(payload, output, format="csv")
    assert calls == [["unknown"], [0], [1], [2], [3], [4]]
    calls.clear()
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        export_analysis(payload, output, format="csv")
    assert not calls
    assert output.read_bytes() == before


def test_competing_destination_survives_streaming_publication(tmp_path, monkeypatch):
    from acprof import artifacts
    output = tmp_path / "statistics.csv"
    original = artifacts.os.link

    def compete(source, destination):
        output.write_bytes(b"competing result\n")
        return original(source, destination)

    monkeypatch.setattr(artifacts.os, "link", compete)
    with pytest.raises(FileExistsError):
        export_analysis({"groups": [{"value": i} for i in range(100)]}, output, format="csv")
    assert output.read_bytes() == b"competing result\n"
    assert list(tmp_path.iterdir()) == [output]
