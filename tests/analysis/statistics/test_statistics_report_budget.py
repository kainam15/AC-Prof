"""Duplicate lookup must read reports that its own writer can publish."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from acprof import artifacts
from acprof.analysis import statistics_report
from acprof.cli import stats


def save(directory, report):
    return statistics_report.save_unique_statistics_report(directory, report)


def test_real_large_report_is_reused_without_another_file(tmp_path):
    report = {"schema_version": 1, "groups": [
        {"metric": "latency_app_s", "mean": index, "reason": "证据" * 512} for index in range(1400)
    ]}
    first, reused = save(tmp_path, report)
    assert not reused
    assert first.stat().st_size > artifacts.MAX_JSON_ARTIFACT_BYTES
    original = hashlib.sha256(first.read_bytes()).hexdigest()
    assert save(tmp_path, report) == (first, True)
    assert list(tmp_path.iterdir()) == [first]
    assert hashlib.sha256(first.read_bytes()).hexdigest() == original
    with pytest.raises(ValueError, match="4 MiB read limit"):
        artifacts.read_json_object(first)


@pytest.mark.parametrize("encoding", ["utf8", "bom", "compact", "reordered"])
def test_report_budget_counts_published_utf8_bytes(tmp_path, monkeypatch, encoding):
    monkeypatch.setattr(artifacts, "MAX_JSON_ARTIFACT_BYTES", 128)
    report = {"schema_version": 1, "kind": "window_statistics", "groups": [
        {"reason": "中文记录", "mean": index, "observations": [0, 1, 2]} for index in range(15)
    ]}
    first, _ = save(tmp_path, report)
    if encoding == "bom":
        first.write_bytes(b"\xef\xbb\xbf" + first.read_bytes())
    elif encoding == "compact":
        first.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    elif encoding == "reordered":
        first.write_text(json.dumps(dict(reversed(list(report.items()))), ensure_ascii=False, indent=2), encoding="utf-8")
    original = first.read_bytes()
    assert len(original) > artifacts.MAX_JSON_ARTIFACT_BYTES
    assert save(tmp_path, report) == (first, True)
    assert first.read_bytes() == original
    assert list(tmp_path.iterdir()) == [first]


def test_read_budget_matches_pretty_output_instead_of_compact_json(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "MAX_JSON_ARTIFACT_BYTES", 256)
    report = {"groups": [{"value": 1} for _ in range(15)]}
    assert len(json.dumps(report)) < artifacts.MAX_JSON_ARTIFACT_BYTES
    first, _ = save(tmp_path, report)
    assert first.stat().st_size > artifacts.MAX_JSON_ARTIFACT_BYTES
    assert save(tmp_path, report) == (first, True)


def test_each_candidate_has_a_fresh_bounded_read_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "MAX_JSON_ARTIFACT_BYTES", 256)
    report = {"groups": [{"reason": "证据" * 200, "mean": 1}]}
    good = tmp_path / "window-statistics-z-matching.json"
    artifacts.atomic_write_json(good, report)
    assert good.stat().st_size > artifacts.MAX_JSON_ARTIFACT_BYTES
    # A candidate that grows past the current report's size must not exhaust
    # the next candidate's allowance or be consumed in full.
    bad = tmp_path / "window-statistics-a-oversized.json"
    bad.write_bytes(good.read_bytes() + b" " * 100000)
    original = Path.open
    received = {good: 0, bad: 0}
    closed = []

    class Reader:
        def __init__(self, path, *args, **kwargs):
            self.path = path
            self.stream = original(path, *args, **kwargs)

        def __enter__(self):
            return self

        def read(self, size):
            data = self.stream.read(size)
            received[self.path] += len(data)
            return data

        def __exit__(self, *_exc):
            self.stream.close()
            closed.append(self.path)

    def observe(path, *args, **kwargs):
        if path in received and args and args[0] == "rb":
            return Reader(path, *args, **kwargs)
        return original(path, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", observe)
        assert save(tmp_path, report) == (good, True)
    # The optional UTF-8 BOM and one overflow probe are the only extra bytes.
    assert received[bad] <= max(256, good.stat().st_size + 3) + 1
    assert received[bad] < bad.stat().st_size
    assert received[good] == good.stat().st_size
    assert closed == [bad, good]


@pytest.mark.parametrize("change", [True, 1.0, "1"])
def test_large_lookup_preserves_numeric_and_type_distinctions(tmp_path, monkeypatch, change):
    monkeypatch.setattr(artifacts, "MAX_JSON_ARTIFACT_BYTES", 128)
    report = {"groups": [{"mean": 1, "detail": "record" * 100}]}
    first, _ = save(tmp_path, report)
    altered = {"groups": [{**report["groups"][0], "mean": change}]}
    second, reused = save(tmp_path, altered)
    assert not reused
    assert second != first
    assert save(tmp_path, report) == (first, True)
    assert save(tmp_path, altered) == (second, True)


def test_cli_receipt_reuses_the_complete_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(artifacts, "MAX_JSON_ARTIFACT_BYTES", 128)
    source = tmp_path / "source.csv"
    source.write_text("cpu_cores,mem_cap_gb,gpu_mode,input_scale,warmup,repeat_idx,status,latency_app_s\n"
                      "2,8,off,64,0,0,ok,0.01\n2,8,off,64,0,1,ok,0.02\n", encoding="utf-8")
    before = source.read_bytes()
    output = tmp_path / "reports"
    arguments = [str(source), "--metric", "latency_app_s", "--resamples", "10", "--output-dir", str(output)]
    assert stats.main(arguments) == 0
    first = json.loads(capsys.readouterr().out.removeprefix("ACPROF_STATS "))
    assert stats.main(arguments) == 0
    second = json.loads(capsys.readouterr().out.removeprefix("ACPROF_STATS "))
    assert not first["reused"]
    assert second == {**first, "reused": True}
    assert len(list(output.iterdir())) == 1
    assert source.read_bytes() == before


def test_concurrent_writers_publish_one_report_above_metadata_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "MAX_JSON_ARTIFACT_BYTES", 128)
    report = {"groups": [{"mean": 1, "detail": "record" * 1000}]}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(save, tmp_path, report) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]
    assert results[0][0] == results[1][0]
    assert sorted(reused for _, reused in results) == [False, True]
    assert len(list(tmp_path.iterdir())) == 1
