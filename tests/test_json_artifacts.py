"""Historical UTF-8 artifacts remain readable without weakening JSON validation."""
import csv
import json

import pytest

from acprof.analysis.model import load_analysis
from acprof.artifacts import MAX_JSON_ARTIFACT_BYTES, loads_finite_json, read_json_object


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_json_artifact_accepts_utf8_without_changing_bytes(tmp_path, encoding):
    path = tmp_path / "metadata.json"
    payload = {"model_name": "示例/model", "scale": 1.25, "nested": {"enabled": True}}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding=encoding)
    original = path.read_bytes()
    assert read_json_object(path) == payload
    assert path.read_bytes() == original


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_bom_json_artifact_still_rejects_nonfinite_numbers(tmp_path, number):
    path = tmp_path / "metadata.json"
    path.write_text('{"nested": {"metric": ' + number + '}}', encoding="utf-8-sig")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="non-finite"):
        read_json_object(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("content", ["[]", "null", "42", "true"])
def test_bom_json_artifact_still_requires_an_object(tmp_path, content):
    path = tmp_path / "metadata.json"
    path.write_text(content, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="must be an object"):
        read_json_object(path)


def test_bom_bytes_count_toward_the_artifact_size_limit(tmp_path):
    path = tmp_path / "metadata.json"
    content = b'\xef\xbb\xbf{}' + b' ' * (MAX_JSON_ARTIFACT_BYTES - 5)
    path.write_bytes(content)
    assert read_json_object(path) == {}
    path.write_bytes(content + b' ')
    with pytest.raises(ValueError, match="4 MiB"):
        read_json_object(path)


@pytest.mark.parametrize("content", [b'{}\xef\xbb\xbf', b'\xef\xbb\xbf\xef\xbb\xbf{}', b'\xff{}'])
def test_artifact_reader_does_not_strip_embedded_or_repeated_bom(tmp_path, content):
    path = tmp_path / "metadata.json"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="invalid .*JSON"):
        read_json_object(path)


def test_wire_json_decoder_remains_strict_about_bom():
    with pytest.raises(ValueError):
        loads_finite_json('\ufeff{}')


def test_analysis_preserves_bom_metadata_and_recorded_run_identity(tmp_path):
    csv_path = tmp_path / "result_all.csv"
    row = {"cpu_cores": "2", "mem_cap_gb": "4", "gpu_mode": "off", "input_scale": "32",
           "repeat_idx": "0", "warmup": "0", "status": "ok", "repeat_in_window": "2",
           "latency_app_p95_s": ".04", "cpu_energy_total_j": "3"}
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    meta = tmp_path / "static_meta.json"
    meta.write_text(json.dumps({"model_name": "example/model", "runtime_backend": "torch"}), encoding="utf-8-sig")
    state = tmp_path / "run_state.json"
    state.write_text(json.dumps({"run_id": "recorded-bom-run", "status": "complete"}), encoding="utf-8-sig")
    before = {path: path.read_bytes() for path in (csv_path, meta, state)}

    result = load_analysis([csv_path])

    assert result.configs[0]["run_id"] == "recorded-bom-run"
    assert result.configs[0]["model"] == "example/model"
    assert result.configs[0]["metrics"]["latency_app_p95_s"]["value"] == .04
    assert result.sources[0]["run_state"] == "complete"
    assert {path: path.read_bytes() for path in before} == before
