"""Dataset report contracts reject ambiguous mappings and preserve input grouping."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from acprof.analysis.load_report import analyze_load_report
from acprof.plotting.load_report import write_load_report
from tests.execution.operations.test_streaming_load import streaming_server
from tests.execution.operations.test_streaming_request_dataset import write_dataset


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@pytest.fixture(scope="module", name="dataset_report")
def saved_dataset_report(tmp_path_factory):
    from contextlib import nullcontext

    from acprof.cli.main import main

    root = tmp_path_factory.mktemp("dataset-report")
    source, output = root / "input.jsonl", root / "output"
    write_dataset(source)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("acprof.cli.streaming_load.MeasurementLock", nullcontext)
        patch.setattr("acprof.cli.streaming_load.host_identity", lambda _root: {"hostname": "fixture"})
        with streaming_server() as (origin, _ports, _paths, _bodies, _ready):
            assert main(["load", "--base-url", origin, "--protocol", "openai-chat", "--request-jsonl", str(source),
                         "--requests", "2", "--warmup", "1", "--output-dir", str(output)]) == 0
    return json.loads((output / "load.json").read_text())


def rehash_inputs(payload):
    dataset = payload["identity"]["request_dataset"]
    dataset["sha256"] = digest(dataset["bodies"])
    for name in ("warmup", "result"):
        phase = payload.get(name)
        if phase is not None:
            phase["protocol"]["request_inputs_sha256"] = digest(dataset["bodies"][:phase["protocol"]["requests"]])
            phase["protocol_sha256"] = digest(phase["protocol"])
    payload["identity_sha256"] = digest(payload["identity"])


def test_dataset_report_retains_full_selected_prefix_and_phase_binding(dataset_report):
    data = analyze_load_report(copy.deepcopy(dataset_report))
    assert data["version"] == 3
    assert data["capacity"]["counts"]["offered"] == 2
    assert len(data["experiment"]["identity"]["request_dataset"]["bodies"]) == 2
    assert dataset_report["warmup"]["protocol"]["request_inputs_sha256"] != dataset_report["result"]["protocol"]["request_inputs_sha256"]


@pytest.mark.parametrize(("path", "value"), [
    (("identity", "request_dataset", "sha256"), "0" * 64),
    (("identity", "request_dataset", "selection"), "cycle"),
    (("identity", "request_dataset", "source"), None),
    (("identity", "request_dataset", "source", "sha256"), "invalid"),
    (("identity", "request_dataset", "source", "rows"), True),
    (("identity", "request_dataset", "source", "rows"), 1),
    (("identity", "request_dataset", "source", "lines"), [1, 1]),
    (("identity", "request_dataset", "bodies", 0, "model"), "other"),
    (("identity", "request_dataset", "bodies", 0, "stream"), False),
    (("identity", "warmup_requests"), True),
    (("identity", "warmup_scenario"), "concurrent"),
    (("identity", "adapter", "endpoint"), "/v1/completions"),
    (("result", "requests", 1, "input_index"), 0),
    (("result", "requests", 1, "input_index"), True),
    (("result", "requests", 1, "request_id"), "other:1"),
    (("result", "protocol", "request_inputs_sha256"), "0" * 64),
    (("result", "protocol_sha256"), "0" * 64),
    (("result", "protocol", "requests"), 1),
    (("warmup", "requests", 0, "input_index"), 1),
    (("warmup", "protocol", "adapter", "endpoint"), "/v1/completions"),
], ids=["input-digest", "selection", "source", "source-sha", "boolean-count", "source-count", "source-lines",
        "body-model", "body-stream", "warmup-count", "warmup-mode", "endpoint", "duplicate-index", "boolean-index",
        "request-id", "phase-input-digest", "phase-protocol-digest", "phase-count", "warmup-index", "phase-adapter"])
def test_invalid_dataset_report_is_not_published(dataset_report, tmp_path, path, value):
    payload = copy.deepcopy(dataset_report)
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    if path[:3] == ("identity", "request_dataset", "bodies"):
        rehash_inputs(payload)
    payload["identity_sha256"] = digest(payload["identity"])
    source, output = tmp_path / "load.json", tmp_path / "invalid.html"
    source.write_text(json.dumps(payload))
    before = source.read_bytes()
    with pytest.raises(ValueError, match="dataset|stream|model"):
        write_load_report(source, output)
    assert source.read_bytes() == before
    assert not output.exists()


@pytest.mark.parametrize("change", ["reorder-input", "reorder-records", "missing-record", "downgrade", "ambiguous", "warmup-missing"])
def test_rehashed_identity_cannot_hide_dataset_mapping_conflicts(dataset_report, change):
    payload = copy.deepcopy(dataset_report)
    if change == "reorder-input":
        dataset = payload["identity"]["request_dataset"]
        dataset["bodies"].reverse()
        dataset["sha256"] = digest(dataset["bodies"])
    elif change == "reorder-records":
        payload["result"]["requests"].reverse()
    elif change == "missing-record":
        payload["result"]["requests"].pop()
    elif change == "downgrade":
        payload["schema_version"] = payload["result"]["schema_version"] = 2
    elif change == "ambiguous":
        payload["identity"]["request_body"] = payload["identity"]["request_dataset"]["bodies"][0]
    else:
        del payload["warmup"]
    payload["identity_sha256"] = digest(payload["identity"])
    with pytest.raises(ValueError, match="dataset"):
        analyze_load_report(payload)


def test_dataset_html_escapes_source_paths(dataset_report, tmp_path):
    payload = copy.deepcopy(dataset_report)
    payload["identity"]["request_dataset"]["source"]["path"] = "<script>dataset</script>.jsonl"
    payload["identity_sha256"] = digest(payload["identity"])
    source = tmp_path / "load.json"
    source.write_text(json.dumps(payload))
    html = write_load_report(source, tmp_path / "safe.html").read_text()
    assert "<script>dataset</script>" not in html
    assert "&lt;script&gt;dataset&lt;/script&gt;.jsonl" in html


@pytest.mark.parametrize("version", [4, 3.0, True, "3"])
def test_dataset_reader_requires_exact_supported_schema(dataset_report, version):
    payload = copy.deepcopy(dataset_report)
    payload["schema_version"] = version
    with pytest.raises(ValueError, match="schema"):
        analyze_load_report(payload)


def make_dataset_sweep(root, dataset_report, change, source_path="/different/location/requests.jsonl"):
    points = []
    for index in (1, 2, 3):
        payload = copy.deepcopy(dataset_report)
        payload["run_id"] = f"point-{index}"
        config = payload["identity"]["load_config"]
        config.update(scenario="concurrent", concurrency=index)
        payload["result"]["protocol"].update(config)
        payload["result"]["capacity"]["client_concurrency"] = index
        for slot, record in enumerate(payload["result"]["requests"]):
            record["request_id"] = f"point-{index}:{slot}"
        if index == 2:
            dataset = payload["identity"]["request_dataset"]
            dataset["source"].update(path=source_path, sha256="a" * 64)
            if change == "prompt":
                dataset["bodies"][0]["messages"][0]["content"] = "different input"
            elif change == "order":
                dataset["bodies"].reverse()
        rehash_inputs(payload)
        relative = f"points/{index:04d}/load.json"
        destination = root / relative
        destination.parent.mkdir(parents=True)
        destination.write_text(json.dumps(payload))
        points.append({"point_id": f"{index:04d}", "report_path": relative, "load_config": config,
                       "status": "complete", "successful": True, "run_id": payload["run_id"]})
    (root / "sweep.json").write_text(json.dumps({
        "schema_version": 1, "kind": "stream_load_sweep", "run_id": "sweep-fixture", "status": "complete",
        "successful": True, "execution_order": "listed_concurrency_then_rate", "points": points,
    }))
    return root


@pytest.mark.parametrize("change", ["source", "prompt", "order"])
def test_sweep_groups_actual_input_order_and_preserves_each_source(dataset_report, tmp_path, change):
    from acprof.analysis.load_sweep import read_load_sweep

    root = make_dataset_sweep(tmp_path / "sweep", dataset_report, change)
    data = read_load_sweep(root).payload
    assert data["valid_points"] == 3
    assert len(data["conditions"]) == (1 if change == "source" else 2)
    assert data["points"][0]["condition_id"] == data["points"][2]["condition_id"]
    assert data["points"][1]["request_dataset"]["source"]["path"] == "/different/location/requests.jsonl"
    assert all("bodies" not in point["request_dataset"] for point in data["points"])
