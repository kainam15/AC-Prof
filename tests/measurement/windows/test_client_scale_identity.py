"""Scales with the same display label still have independent requests and evidence."""
import csv
import hashlib
import json
from types import SimpleNamespace

import pytest

from acprof.host.client import ClientRunner
from acprof.host.measurement.config import ClientConfig
from acprof.packet.sniff_parse_pcap import extract_group_id_from_request_lines
from acprof.results.layout import case_sidecar
from tests.measurement.windows.test_window_execution import Collector


@pytest.mark.parametrize("labels", [
    pytest.param(("same", "same"), id="repeated-label"),
    pytest.param((None, None), id="rounded-default"),
    pytest.param(("seq:one", "seq,two"), id="packet-separators"),
])
def test_client_preserves_each_scale_through_request_and_window_publication(tmp_path, monkeypatch, labels):
    scales = [1_000_000, 1_000_001, 65, 64]
    entries = [{"input_scale": scale, "payload": {"scale": scale}} for scale in scales]
    reserved = f"scale-{float(scales[0])!r}"
    for entry, label in zip(entries, [*labels, reserved, "seq64"], strict=True):
        if label is not None:
            entry["scale_label"] = label
    plan = tmp_path / "input_scale_plan.json"
    plan.write_text(json.dumps({"schema_version": 2, "entries": entries}))
    path = tmp_path / "case.csv"
    records = tmp_path / "records"
    telemetry = tmp_path / "telemetry"
    monkeypatch.setattr("acprof.host.preflight.require_collection_host", lambda **_: None)
    monkeypatch.setattr("acprof.host.client.requests.get", lambda *_, **__: SimpleNamespace(status_code=200, text="ok"))
    monkeypatch.setattr(ClientRunner, "_monitor_module", staticmethod(lambda _: None))
    client = ClientRunner(ClientConfig(
        out_csv=str(path), input_scale_plan_file=str(plan), profiling_mode="basic",
        run_id="run", case_id="1c_4g_off", case_attempt_id="a" * 32, case_name="case",
        window_records_dir=str(records), telemetry_dir=str(telemetry), cpu_cores=1, mem_cap_gb=4,
        warmup=1, repeat=1, repeat_in_window=1, idle_seconds=0, idle_cooldown_seconds=0, use_mips=False,
    ))
    client.resource_usage_mod = SimpleNamespace(ResourceUsageMonitor=lambda **_: Collector([]))
    requests = []

    def post(_url, **options):
        scale = options["json"]["scale"]
        requests.append((scale, options["headers"]["X-Req-Id"]))
        body = json.dumps({"effective_input_scale": scale})
        return SimpleNamespace(status_code=200, text=body, content=body.encode())

    monkeypatch.setattr("acprof.host.client.requests.post", post)
    client.main()

    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    journals = [json.loads(line) for line in case_sidecar(path, "requests").read_text().splitlines()]
    groups = [json.loads(line)["sniff_group_id"] for line in case_sidecar(path, "sniff_groups").read_text().splitlines()]
    assert len(groups) == len(set(groups)) == len(scales) * 2
    assert groups == [row["window_id"] for row in rows]
    assert groups == [journal["sniff_group_id"] for journal in journals]
    assert groups[-4:] == [f"case_{reserved}_w0", f"case_{reserved}_r0", "case_seq64_w0", "case_seq64_r0"]
    receipts = {}
    for item in records.glob("*.json"):
        receipt = json.loads(item.read_bytes())
        receipts[receipt["identity"]["window_id"]] = receipt
    assert set(receipts) == set(groups)
    for index, (row, journal, group, request) in enumerate(zip(rows, journals, groups, requests, strict=True)):
        scale = scales[index // 2]
        assert request == (scale, f"{group}:0")
        assert extract_group_id_from_request_lines(f"X-Req-Id: {request[1]}") == group
        assert float(row["input_scale"]) == journal["input_scale"] == scale
        assert int(row["warmup"]) == journal["warmup"] == (1 - index % 2)
        receipt = receipts[group]
        assert receipt["request"] == journal
        evidence = (records / receipt["telemetry"]["path"]).read_bytes()
        assert hashlib.sha256(evidence).hexdigest() == receipt["telemetry"]["sha256"]
        payload = json.loads(evidence)
        assert payload["identity"]["window_id"] == group
        assert payload["context"]["input_scale"] == scale
        assert payload["context"]["scale_label"] == entries[index // 2].get("scale_label", f"scale{scale:g}")
