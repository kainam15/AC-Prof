"""Result cleanup must never remove evidence before a completed merge."""
from pathlib import Path

from acprof.artifact_layout import ArtifactLayout
from acprof.host.result_cleanup import cleanup_intermediate_results


def test_flat_case_cleanup_preserves_final_csv(tmp_path: Path):
    intermediate = tmp_path / "result_case_1c_4g_off.csv"
    intermediate.write_text("status\nok\n")
    sniff = tmp_path / (intermediate.name + ".sniff_groups.jsonl")
    sniff.write_text("{}\n")
    latency = tmp_path / "lat_case_1c_4g_off.json"
    latency.write_text("{}\n")
    capture = tmp_path / "sniff_case_1c_4g_off.pcap"
    capture.write_bytes(b"pcap")
    merged = tmp_path / "result_all.csv"
    merged.write_text("status\nok\n")

    cleanup_intermediate_results([str(intermediate)], str(tmp_path), str(merged))

    assert merged.is_file()
    assert not any(path.exists() for path in (intermediate, sniff, latency, capture))


def test_missing_final_result_keeps_intermediate(tmp_path: Path):
    intermediate = tmp_path / "result_case_1c_4g_off.csv"
    intermediate.write_text("status\nok\n")

    cleanup_intermediate_results(
        [str(intermediate)], str(tmp_path), str(tmp_path / "result_all.csv")
    )

    assert intermediate.is_file()


def test_v2_cleanup_retains_request_evidence(tmp_path: Path):
    layout = ArtifactLayout.for_new_run(tmp_path)
    layout.initialize()
    case = layout.case("org/model", 1, 4, "off")
    case.csv.parent.mkdir(parents=True, exist_ok=True)
    case.csv.write_text("status\nok\n")
    case.requests.write_text('{"request_id":"sample-1"}\n')
    case.pcap.write_bytes(b"pcap")
    case.latency.write_text("{}\n")
    merged = layout.result_csv
    merged.parent.mkdir(parents=True, exist_ok=True)
    merged.write_text("status\nok\n")

    cleanup_intermediate_results([str(case.csv)], str(tmp_path), str(merged))

    assert merged.is_file()
    assert case.retained_requests.read_text() == '{"request_id":"sample-1"}\n'
    assert not any(path.exists() for path in (case.csv, case.requests, case.pcap, case.latency))
