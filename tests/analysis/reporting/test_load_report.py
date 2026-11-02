"""Single-point load report compatibility, integrity and exclusive publication."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from acprof.analysis.streaming_capacity import build_capacity
from acprof.cli.main import main
from acprof.plotting.load_report import read_load_report, write_load_report


def request():
    return {
        "status": "ok", "sent_s": 0.0, "completed_s": 0.3,
        "input_tokens": 2, "output_tokens": 3,
        "input_token_count_source": "server_usage", "output_token_count_source": "server_usage",
        "input_token_count_scope": "server_prompt_tokens", "token_count_scope": "server_generated_tokens",
        "ttft_s": 0.1, "tpot_s": 0.02, "e2e_s": 0.3,
    }


def experiment(version=2):
    thresholds = {"ttft_s": 0.1, "tpot_s": 0.02, "e2e_s": 0.3}
    identity = {
        "model": "fixture", "model_revision": "declared-revision", "load_config": {"concurrency": 2},
    }
    result = {
        "schema_version": version, "kind": "stream_load_result", "successful": True,
        "duration_s": 2.0, "success_rate": 1.0, "successful_requests_per_s": 0.5,
        "protocol": {"concurrency": 2},
        "counts": {"offered": 1, "sent": 1, "succeeded": 1, "failed": 0, "dropped": 0, "cancelled": 0},
        "requests": [request()],
        "streaming": {
            name: {"mean": value, "p50": value, "p90": value, "p95": value, "p99": value, "sample_count": 1}
            for name, value in (("ttft_s", 0.1), ("tpot_s", 0.02), ("e2e_s", 0.3))
        },
    }
    payload = {
        "schema_version": version, "kind": "stream_load_experiment", "run_id": "run",
        "status": "complete", "successful": True, "identity": identity, "result": result,
    }
    if version == 2:
        identity["slo_thresholds_s"] = thresholds
        result["capacity"] = build_capacity(result["requests"], result["duration_s"], 2, thresholds)
        payload["service_identity"] = {
            "schema_version": 1,
            "declared": {"backend": "vllm", "version": "0.11.2", "model": "fixture", "revision": "declared-revision"},
            "observed": {"backend": "vllm", "version": "0.11.2", "model": "fixture", "revision": None},
            "probe": {"kind": "vllm", "status": "available", "errors": {}},
            "matches": {"backend": True, "version": True, "model": True, "revision": None},
            "mismatches": [],
        }
    payload["identity_sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return payload


def save_experiment(tmp_path: Path, version=2) -> Path:
    directory = tmp_path / f"v{version}"
    directory.mkdir()
    path = directory / "load.json"
    path.write_text(json.dumps(experiment(version)))
    return path


def test_v2_report_recomputes_capacity_and_renders_single_point_metrics(tmp_path):
    source = save_experiment(tmp_path)
    output = tmp_path / "capacity.html"
    assert write_load_report(source, output) == output
    html = output.read_text()
    assert "One offered load point only" in html
    assert "not a capacity curve or a maximum-capacity estimate" in html
    assert '<th scope="row">goodput_requests_per_s</th><td>0.5</td>' in html
    assert '<th scope="row">input</th><td>2</td><td>1</td><td>2</td><td>1</td><td>1</td><td>1</td><td>1</td>' in html
    assert '<th scope="row">revision</th><td>declared-revision</td><td>unavailable</td>' in html


def test_v1_report_keeps_new_capacity_and_token_values_unavailable(tmp_path):
    source = save_experiment(tmp_path, version=1)
    output = tmp_path / "v1.html"
    write_load_report(source, output)
    html = output.read_text()
    assert '<th scope="row">offered</th><td>1</td>' in html
    assert '<th scope="row">goodput_requests_per_s</th><td>unavailable</td>' in html
    assert "<section><h2>SLO goodput</h2><p>unavailable</p></section>" in html
    assert (
        '<th scope="row">input</th>' + "<td>unavailable</td>" * 7
    ) in html


@pytest.mark.parametrize("source_kind", ["file", "directory"])
def test_report_cli_dispatches_one_load_file_or_directory(tmp_path, source_kind):
    source = save_experiment(tmp_path)
    selected = source if source_kind == "file" else source.parent
    output = tmp_path / f"{source_kind}.html"
    assert main(["report", str(selected), "--output", str(output)]) == 0
    assert output.is_file()
    assert "single-point load capacity" in output.read_text()


@pytest.mark.parametrize("options", [
    ["--baseline", "run"], ["--comparison-purpose", "same-hardware"],
])
def test_load_report_rejects_comparison_options(tmp_path, options):
    source = save_experiment(tmp_path)
    output = tmp_path / "rejected.html"
    with pytest.raises(SystemExit) as error:
        main(["report", str(source), "--output", str(output), *options])
    assert error.value.code == 1
    assert not output.exists()


def test_load_report_rejects_multiple_input_combination(tmp_path):
    source = save_experiment(tmp_path)
    other = tmp_path / "other.csv"
    other.write_text("not,a,formal,result\n")
    output = tmp_path / "combined.html"
    with pytest.raises(SystemExit) as error:
        main(["report", str(source), str(other), "--output", str(output)])
    assert error.value.code == 1
    assert not output.exists()


@pytest.mark.parametrize("tamper", ["capacity", "requests", "identity"])
def test_v2_report_rejects_inconsistent_or_tampered_evidence(tmp_path, tamper):
    source = save_experiment(tmp_path)
    payload = json.loads(source.read_text())
    if tamper == "capacity":
        payload["result"]["capacity"]["goodput_requests_per_s"] = 99
    elif tamper == "requests":
        payload["result"]["requests"][0]["e2e_s"] = 9
    else:
        payload["identity"]["slo_thresholds_s"]["e2e_s"] = 9
    source.write_text(json.dumps(payload))
    output = tmp_path / "tampered.html"
    with pytest.raises(ValueError, match="inconsistent|identity_sha256"):
        write_load_report(source, output)
    assert not output.exists()


def test_load_report_never_overwrites_existing_output(tmp_path):
    source = save_experiment(tmp_path)
    output = tmp_path / "existing.html"
    output.write_text("prior evidence")
    with pytest.raises(ValueError, match="new file"):
        write_load_report(source, output)
    assert output.read_text() == "prior evidence"


def test_nonstream_load_is_not_treated_as_a_capacity_report(tmp_path):
    source = save_experiment(tmp_path)
    payload = experiment(1)
    payload["kind"] = "nonstream_load_experiment"
    source.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="stream_load_experiment"):
        read_load_report(source)
