"""Checkpoint real committed window evidence and retry only its projections."""
import csv
import hashlib
import json
import os
import struct
from pathlib import Path

import pytest

from acprof.artifact_layout import ArtifactLayout
from acprof.host.measurement import checkpoint as checkpoints
from acprof.host.measurement.publication import append_request_window
from acprof.result_csv import KEY_FIELDS
from acprof.telemetry import window_evidence
from acprof.window_records import (
    WindowPublicationContext,
    commit_window_record,
    prepare_window_telemetry,
)

RUN_ID = "run-checkpoint-test"
ATTEMPT = "a" * 32
EXPECTED = {("2", "4", "off", "64", "0", "0")}


def _processing(*, capture=True):
    return {
        "capture_available": capture, "case_incomplete": False,
        "completed_rows_before_failure": 1, "incomplete_case_reason": "",
        "require_packet_latency": capture, "profiling_mode": "full" if capture else "basic",
        "gpu": "off", "server_port": 8000,
    }


def _write_csv(case, row):
    with case.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        start = stream.tell()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())
        return {"start_byte": start, "end_byte": stream.tell()}


def _fixture(root, *, attempt=ATTEMPT, capture=True, receipt_case_id=None, note=None):
    if (root / "result_manifest.json").exists():
        layout = ArtifactLayout.discover(root)
    else:
        layout = ArtifactLayout.for_new_run(root)
        layout.initialize()
    case = layout.case("example/model", 2, 4, "off")
    case.csv.parent.mkdir(parents=True, exist_ok=True)
    window_id = "2c_4g_off_scale64_r0"
    context = WindowPublicationContext(
        RUN_ID, receipt_case_id or case.case_id, attempt,
        str(case.window_records(attempt)), str(case.telemetry(attempt)),
    )
    row = {**dict(zip(KEY_FIELDS, ("2", "4", "off", "64", "0", "0"), strict=True)),
           "environment_class": "native_linux", "status": "ok", "error": "",
           "latency_app_s": "0.150000"}
    if note is not None:
        row["note"] = note
    with case.requests.open("w") as stream:
        request = append_request_window(stream, sniff_group_id=window_id, input_scale=64.0,
                                        warmup=0, repeat_idx=0, latencies=[0.1, 0.2])
    sidecar = case.sidecar("sniff_groups")
    with sidecar.open("w") as stream:
        stream.write(json.dumps({"sniff_group_id": window_id}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        sidecar_end = stream.tell()
    csv_range = _write_csv(case, row)
    evidence = window_evidence(
        identity=context.identity(window_id), context={"sampling_policy": "burst"},
        collectors={"resource": {"status": "unavailable", "reason": "fixture_without_hardware"}},
    )
    telemetry = prepare_window_telemetry(context, window_id, evidence)
    record = commit_window_record(
        context, window_id=window_id, row=row, request_record=request,
        telemetry_reference=telemetry,
        projections={
            "csv": {"path": os.path.relpath(case.csv, context.record_path(window_id).parent), **csv_range},
            "sniff_groups": {
                "path": os.path.relpath(sidecar, context.record_path(window_id).parent),
                "start_byte": 0, "end_byte": sidecar_end,
            },
        },
    )
    if capture:
        # A valid empty PCAP header: this test checks durable evidence, not parsing.
        case.pcap.write_bytes(struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
    return case, context, record, _processing(capture=capture)


def _publish(case, processing, attempt=ATTEMPT, *, expected=EXPECTED):
    return checkpoints.publish_measurement_checkpoint(
        case, run_id=RUN_ID, attempt_id=attempt, processing=processing, expected=expected,
    )


def _read(case, reference, attempt=ATTEMPT, *, expected=EXPECTED):
    return checkpoints.read_measurement_checkpoint(
        case, reference, run_id=RUN_ID, attempt_id=attempt, expected=expected,
    )


def _rewrite_checkpoint(case, reference, payload):
    path = case.layout.root / reference["path"]
    path.write_text(json.dumps(payload))
    raw = path.read_bytes()
    return {"path": reference["path"], "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}


def _partial_fixture(root, *, reason="request timeout"):
    case, context, _, processing = _fixture(root)
    failed_group = "2c_4g_off_scale64_r1"
    diagnostic = f"{reason}: request did not complete"
    # A terminal request can fail after earlier requests in its window finished.
    # The request journal and partial telemetry survive, but no receipt commits it.
    with case.requests.open("a") as stream:
        append_request_window(
            stream, sniff_group_id=failed_group, input_scale=64.0,
            warmup=0, repeat_idx=1, latencies=[0.05], error=diagnostic,
            failed_request_id=f"{failed_group}:1",
        )
    partial = prepare_window_telemetry(
        context, failed_group,
        window_evidence(
            identity=context.identity(failed_group),
            context={"completed_requests": 1, "uncommitted_error": diagnostic},
            collectors={"resource": {"status": "unavailable", "reason": "fixture_without_hardware"}},
        ),
    )
    with case.csv.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    row.update(repeat_idx="1", status="error", error=diagnostic, latency_app_s="nan")
    with case.csv.open("a", newline="") as stream:
        csv.DictWriter(stream, fieldnames=list(row)).writerow(row)
        stream.flush()
        os.fsync(stream.fileno())
    with case.sidecar("sniff_groups").open("a") as stream:
        stream.write('{"sniff_group_id": ""}\n')
        stream.flush()
        os.fsync(stream.fileno())
    processing.update(case_incomplete=True, incomplete_case_reason=reason)
    expected = EXPECTED | {("2", "4", "off", "64", "0", "1")}
    return case, context, partial, processing, expected


@pytest.mark.parametrize("reason", ["request timeout", "runtime OOM"])
def test_checkpoint_preserves_success_and_uncommitted_terminal_request(tmp_path, reason):
    case, context, partial, processing, expected = _partial_fixture(tmp_path, reason=reason)
    original = {"csv": case.csv.read_bytes(), "groups": case.sidecar("sniff_groups").read_bytes(),
                "requests": case.requests.read_bytes()}
    failed_record = context.record_path("2c_4g_off_scale64_r1")
    partial_path = failed_record.parent / partial["path"]
    partial_bytes = partial_path.read_bytes()
    reference = _publish(case, processing, expected=expected)
    payload = _read(case, reference, expected=expected)
    assert payload["processing"] == processing
    assert len(payload["window_records"]) == 1
    assert payload["window_records"][0]["path"] == (
        context.record_path("2c_4g_off_scale64_r0").relative_to(tmp_path).as_posix()
    )
    assert not failed_record.exists()
    requests = [json.loads(line) for line in original["requests"].splitlines()]
    assert requests[0]["status"] == "ok"
    assert requests[1]["status"] == "error"
    assert requests[1]["latency_app_s"] == [0.05]
    assert requests[1]["failed_request_id"] == "2c_4g_off_scale64_r1:1"
    for target in (case.csv, case.sidecar("sniff_groups"), case.requests):
        target.write_text("interrupted postprocessing\n")
    checkpoints.restore_measurement_projection(case, payload)
    for role, target in (("csv", case.csv), ("groups", case.sidecar("sniff_groups")),
                         ("requests", case.requests)):
        assert target.read_bytes() == original[role]
    assert partial_path.read_bytes() == partial_bytes
    assert not failed_record.exists()
    assert _read(case, reference, expected=expected) == payload


@pytest.mark.parametrize("entrypoint", ["publish", "read"])
@pytest.mark.parametrize("invalid", ["wrong_key", "non_error", "duplicate", "missing_diagnostic", "blank_diagnostic"])
def test_checkpoint_rejects_invalid_uncommitted_terminal_request(tmp_path, entrypoint, invalid):
    case, _, _, processing, expected = _partial_fixture(tmp_path)
    if entrypoint == "read":
        reference = _publish(case, processing, expected=expected)
        payload = _read(case, reference, expected=expected)
        path = tmp_path / payload["files"]["requests"]["path"]
    else:
        path = case.requests
    requests = [json.loads(line) for line in path.read_text().splitlines()]
    terminal = requests[-1]
    if invalid == "wrong_key":
        terminal["repeat_idx"] = 99
    elif invalid == "non_error":
        terminal["status"] = "ok"
    elif invalid == "duplicate":
        requests.append(dict(terminal))
    elif invalid == "missing_diagnostic":
        terminal.pop("error")
    else:
        terminal["error"] = "   "
    path.write_text("".join(json.dumps(request) + "\n" for request in requests))
    error = (
        "Request journal disagrees with committed window receipts"
        if invalid == "duplicate"
        else "Uncommitted request does not match a planned failed window"
    )
    if entrypoint == "read":
        raw = path.read_bytes()
        payload["files"]["requests"].update(sha256=hashlib.sha256(raw).hexdigest(), size=len(raw))
        reference = _rewrite_checkpoint(case, reference, payload)
        with pytest.raises(ValueError, match=error):
            _read(case, reference, expected=expected)
    else:
        with pytest.raises(ValueError, match=error):
            _publish(case, processing, expected=expected)
        assert not case.attempt_directory("measurements", ATTEMPT).exists()


def test_uncommitted_terminal_request_requires_incomplete_case(tmp_path):
    case, _, _, processing, expected = _partial_fixture(tmp_path)
    processing.update(case_incomplete=False, incomplete_case_reason="")
    with pytest.raises(ValueError, match="Uncommitted request does not match a planned failed window"):
        _publish(case, processing, expected=expected)
    assert not case.attempt_directory("measurements", ATTEMPT).exists()


def test_failed_terminal_group_cannot_be_published_as_committed_sidecar(tmp_path):
    case, _, _, processing, expected = _partial_fixture(tmp_path)
    sidecar = case.sidecar("sniff_groups")
    groups = [json.loads(line) for line in sidecar.read_text().splitlines()]
    groups[-1]["sniff_group_id"] = "2c_4g_off_scale64_r1"
    sidecar.write_text("".join(json.dumps(group) + "\n" for group in groups))
    with pytest.raises(ValueError, match="Sniff group"):
        _publish(case, processing, expected=expected)
    assert not case.attempt_directory("measurements", ATTEMPT).exists()


def test_checkpoint_roundtrip_and_projection_restore_preserve_raw_evidence(tmp_path):
    case, context, _, processing = _fixture(tmp_path)
    original = {"csv": case.csv.read_bytes(), "groups": case.sidecar("sniff_groups").read_bytes(),
                "requests": case.requests.read_bytes(), "pcap": case.pcap.read_bytes()}
    reference = _publish(case, processing)
    payload = _read(case, reference)
    assert payload["case_id"] == "2c_4g_off"
    assert payload["case_attempt_id"] == ATTEMPT
    assert payload["processing"] == processing
    assert set(payload["files"]) == set(original)
    assert len(payload["window_records"]) == 1
    assert context.record_path("2c_4g_off_scale64_r0").is_file()
    for name, entry in payload["files"].items():
        assert (tmp_path / entry["path"]).read_bytes() == original[name]
    for destination in (case.csv, case.sidecar("sniff_groups"), case.requests):
        destination.write_text("incomplete postprocessing projection\n")
    # Changed mutable projections do not invalidate the immutable measured input.
    verified = _read(case, reference)
    checkpoints.restore_measurement_projection(case, verified)
    assert case.csv.read_bytes() == original["csv"]
    assert case.sidecar("sniff_groups").read_bytes() == original["groups"]
    assert case.requests.read_bytes() == original["requests"]
    assert _read(case, reference) == payload


def test_retry_cannot_overwrite_a_checkpoint_for_the_same_attempt(tmp_path):
    case, _, _, processing = _fixture(tmp_path)
    reference = _publish(case, processing)
    before = (tmp_path / reference["path"]).read_bytes()
    with pytest.raises(ValueError, match="already has a checkpoint"):
        _publish(case, processing)
    assert (tmp_path / reference["path"]).read_bytes() == before
    assert _read(case, reference)["case_attempt_id"] == ATTEMPT


@pytest.mark.parametrize("failure", ["copy", "manifest", "staging_sync", "rename"])
def test_failed_new_attempt_publication_keeps_previous_evidence(tmp_path, monkeypatch, failure):
    original_case, _, _, processing = _fixture(tmp_path)
    original_reference = _publish(original_case, processing)
    original_payload = _read(original_case, original_reference)
    attempt = "b" * 32
    case, _, _, processing = _fixture(tmp_path, attempt=attempt)
    destination = case.attempt_directory("measurements", attempt)

    def fail(*args, **kwargs):
        raise OSError(f"injected {failure}")

    if failure == "copy":
        monkeypatch.setattr(checkpoints.shutil, "copyfile", fail)
    elif failure == "manifest":
        monkeypatch.setattr(checkpoints, "atomic_write_json", fail)
    elif failure == "rename":
        monkeypatch.setattr(checkpoints.os, "rename", fail)
    else:
        sync = checkpoints.sync_directory

        def fail_staging_sync(directory):
            if directory.name.startswith(".checkpoint-"):
                fail()
            return sync(directory)

        monkeypatch.setattr(checkpoints, "sync_directory", fail_staging_sync)
    with pytest.raises(OSError, match="injected"):
        _publish(case, processing, attempt)
    assert not destination.exists()
    assert not list(destination.parent.glob(".checkpoint-*"))
    assert _read(original_case, original_reference) == original_payload


def test_copied_measurements_are_revalidated_before_checkpoint_is_visible(tmp_path, monkeypatch):
    case, _, _, processing = _fixture(tmp_path)
    copy = checkpoints.shutil.copyfile

    def corrupt_csv_copy(source, destination, *args, **kwargs):
        result = copy(source, destination, *args, **kwargs)
        if Path(destination).name == "measurement.csv":
            path = Path(destination)
            path.write_bytes(path.read_bytes().replace(b"0.150000", b"0.250000"))
        return result

    monkeypatch.setattr(checkpoints.shutil, "copyfile", corrupt_csv_copy)
    with pytest.raises(ValueError, match="receipt|CSV|measurement"):
        _publish(case, processing)
    assert not case.attempt_directory("measurements", ATTEMPT).exists()


def test_post_rename_sync_failure_retains_a_complete_checkpoint(tmp_path, monkeypatch):
    case, _, _, processing = _fixture(tmp_path)
    destination = case.attempt_directory("measurements", ATTEMPT)
    sync = checkpoints.sync_directory

    def fail_parent_sync(directory):
        if directory == destination.parent:
            raise OSError("injected post-rename sync failure")
        return sync(directory)

    monkeypatch.setattr(checkpoints, "sync_directory", fail_parent_sync)
    with pytest.raises(OSError, match="post-rename"):
        _publish(case, processing)
    path = destination / "checkpoint.json"
    raw = path.read_bytes()
    reference = {"path": path.relative_to(tmp_path).as_posix(),
                 "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
    assert _read(case, reference)["case_attempt_id"] == ATTEMPT


@pytest.mark.parametrize("name", ["groups", "requests"])
def test_mismatching_journal_projection_prevents_publication(tmp_path, name):
    case, _, _, processing = _fixture(tmp_path)
    path = case.sidecar("sniff_groups") if name == "groups" else case.requests
    payload = json.loads(path.read_text())
    if name == "groups":
        payload["sniff_group_id"] = "a-different-window"
    else:
        payload["latency_app_s"] = [9.0, 9.0]
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="disagrees"):
        _publish(case, processing)
    assert not case.attempt_directory("measurements", ATTEMPT).exists()


@pytest.mark.parametrize("name", ["pcap", "groups", "requests"])
def test_missing_required_reprocessing_input_is_not_a_complete_checkpoint(tmp_path, name):
    case, _, _, processing = _fixture(tmp_path)
    {"pcap": case.pcap, "groups": case.sidecar("sniff_groups"), "requests": case.requests}[name].unlink()
    with pytest.raises(ValueError, match="[Mm]issing|[Rr]equired|[Dd]isappeared"):
        _publish(case, processing)
    assert not case.attempt_directory("measurements", ATTEMPT).exists()


def test_basic_measurement_does_not_require_a_pcap(tmp_path):
    case, _, _, processing = _fixture(tmp_path, capture=False)
    reference = _publish(case, processing)
    payload = _read(case, reference)
    assert set(payload["files"]) == {"csv", "groups", "requests"}
    assert payload["processing"]["capture_available"] is False


@pytest.mark.parametrize("changed", ["csv", "groups", "requests", "pcap", "receipt", "telemetry", "checkpoint"])
def test_changed_checkpoint_evidence_is_rejected_before_restore(tmp_path, changed):
    case, context, record, processing = _fixture(tmp_path)
    reference = _publish(case, processing)
    payload = _read(case, reference)
    record_path = context.record_path("2c_4g_off_scale64_r0")
    if changed in payload["files"]:
        target = tmp_path / payload["files"][changed]["path"]
    else:
        target = {"receipt": record_path,
                  "telemetry": record_path.parent / record["telemetry"]["path"],
                  "checkpoint": tmp_path / reference["path"]}[changed]
    target.write_bytes(target.read_bytes() + b" ")
    before = case.csv.read_bytes()
    with pytest.raises(ValueError, match="changed|hash mismatch|invalid"):
        _read(case, reference)
    assert case.csv.read_bytes() == before


@pytest.mark.parametrize("alias", ["file", "directory"])
def test_checkpoint_refuses_symlink_evidence_and_ancestors(tmp_path, alias):
    case, _, _, processing = _fixture(tmp_path)
    reference = _publish(case, processing)
    payload = _read(case, reference)
    target = (tmp_path / payload["files"]["csv"]["path"] if alias == "file"
              else case.attempt_directory("measurements", ATTEMPT))
    moved = tmp_path / "aliased-evidence"
    target.rename(moved)
    target.symlink_to(moved, target_is_directory=moved.is_dir())
    with pytest.raises(ValueError, match="symlink"):
        _read(case, reference)


def test_restore_rechecks_cached_projection_parent_before_writing(tmp_path):
    root = tmp_path / "experiment"
    case, _, _, processing = _fixture(root)
    reference = _publish(case, processing)
    checkpoint = _read(case, reference)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "result.csv"
    sentinel.write_bytes(b"another experiment must not be overwritten\n")
    original = sentinel.read_bytes()
    # The CaseArtifacts object predates the directory alias, as it would in a
    # caller that validates a checkpoint before starting projection recovery.
    work = case.csv.parent
    work.rename(tmp_path / "preserved-work")
    work.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes|symlink"):
        checkpoints.restore_measurement_projection(case, checkpoint)
    assert sentinel.read_bytes() == original
    assert list(outside.iterdir()) == [sentinel]
    assert _read(case, reference) == checkpoint


@pytest.mark.parametrize("relative", ["../outside.csv", "/tmp/outside.csv"])
def test_checkpoint_rejects_escaping_file_reference(tmp_path, relative):
    case, _, _, processing = _fixture(tmp_path)
    reference = _publish(case, processing)
    payload = _read(case, reference)
    payload["files"]["csv"]["path"] = relative
    reference = _rewrite_checkpoint(case, reference, payload)
    with pytest.raises(ValueError, match="inside|escapes"):
        _read(case, reference)


def test_checkpoint_cannot_swap_the_roles_of_valid_files(tmp_path):
    case, _, _, processing = _fixture(tmp_path)
    reference = _publish(case, processing)
    payload = _read(case, reference)
    payload["files"]["groups"], payload["files"]["requests"] = (
        payload["files"]["requests"], payload["files"]["groups"],
    )
    reference = _rewrite_checkpoint(case, reference, payload)
    with pytest.raises(ValueError, match="file|path|role|name"):
        _read(case, reference)


def test_checkpoint_refuses_other_case_attempt_and_run_identity(tmp_path):
    case, _, _, processing = _fixture(tmp_path)
    reference = _publish(case, processing)
    with pytest.raises(ValueError, match="another case attempt"):
        _read(case, reference, "b" * 32)
    other_case = case.layout.case("example/model", 4, 4, "off")
    with pytest.raises(ValueError, match="another case attempt"):
        _read(other_case, reference)
    with pytest.raises(ValueError, match="identity mismatch"):
        checkpoints.read_measurement_checkpoint(
            case, reference, run_id="another-run", attempt_id=ATTEMPT, expected=EXPECTED,
        )


def test_receipt_must_name_the_same_case_as_the_checkpoint(tmp_path):
    case, _, _, processing = _fixture(tmp_path, receipt_case_id="4c_4g_off")
    with pytest.raises(ValueError, match="case"):
        _publish(case, processing)


def test_removing_a_receipt_reference_cannot_hide_existing_committed_windows(tmp_path):
    case, _, _, processing = _fixture(tmp_path)
    reference = _publish(case, processing)
    payload = _read(case, reference)
    payload["window_records"] = []
    reference = _rewrite_checkpoint(case, reference, payload)
    with pytest.raises(ValueError, match="receipt set changed"):
        _read(case, reference)


@pytest.mark.parametrize(("field", "value", "error"), [
    ("capture_available", "true", "Invalid processing flag: capture_available"),
    ("completed_rows_before_failure", -1, "Invalid completed measurement count"),
    ("server_port", 0, "Invalid packet parser port"),
    ("profiling_mode", "unknown", "profiling.mode"),
])
def test_processing_contract_is_checked_before_publication(tmp_path, field, value, error):
    case, _, _, processing = _fixture(tmp_path)
    processing[field] = value
    with pytest.raises(ValueError, match=error):
        _publish(case, processing)
    assert not case.attempt_directory("measurements", ATTEMPT).exists()
