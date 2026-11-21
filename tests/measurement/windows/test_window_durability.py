"""Published directory entries must be durable before reuse or adoption."""
import errno
import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from acprof.host.measurement.window_recovery import restore_case_windows
from acprof.results.csv import measurement_key
from acprof.results.windows import (
    WindowPublicationContext,
    commit_window_record,
    prepare_window_telemetry,
)
from acprof.telemetry import window_evidence
from tests.results.storage.test_incremental_results import initialized


class ParentSync:
    """Fail the actual directory descriptor only after its renamed child is visible."""

    def __init__(self, monkeypatch, visible_child):
        self.blocked = True
        self.calls = 0
        fsync = os.fsync

        def sync(fd):
            descriptor = os.fstat(fd)
            if stat.S_ISDIR(descriptor.st_mode) and visible_child.exists() and os.path.samestat(
                descriptor, visible_child.parent.stat()
            ):
                self.calls += 1
                if self.blocked:
                    raise OSError("injected parent fsync failure")
            return fsync(fd)

        monkeypatch.setattr(os, "fsync", sync)


@pytest.fixture
def window(tmp_path):
    layout = initialized(tmp_path)
    case = layout.case("fixture/model", 1, 4, "off")
    attempt = "a" * 32
    context = WindowPublicationContext(json.loads(layout.result_csv.read_bytes())["run_id"], case.case_id,
                                       attempt, str(case.window_records(attempt)), str(case.telemetry(attempt)))
    window_id = "window-1"
    row = {"cpu_cores": "1", "mem_cap_gb": "4", "gpu_mode": "off", "input_scale": "1",
           "warmup": "0", "repeat_idx": "0", "status": "ok", "error": "", "latency_app_s": "0.1",
           "environment_class": "native_linux", **context.identity(window_id)}
    return case, context, window_id, row


def publish_telemetry(context, window_id):
    evidence = window_evidence(identity=context.identity(window_id), context={}, collectors={})
    return prepare_window_telemetry(context, window_id, evidence)


def publish_receipt(context, window_id, row, telemetry, sequence):
    return commit_window_record(context, window_id=window_id, row=row, telemetry_reference=telemetry,
                                request_record={"record": {"sniff_group_id": window_id}},
                                projections={}, sequence=sequence)


@pytest.mark.parametrize("artifact", ["telemetry", "v1-receipt", "v2-receipt"])
@pytest.mark.parametrize("identical", [False, True])
def test_competing_first_publishers_keep_one_immutable_artifact(window, monkeypatch, artifact, identical):
    case, context, window_id, row = window
    if artifact == "telemetry":
        target = case.telemetry(context.case_attempt_id) / context.record_path(window_id).name
        conflict = "refusing to replace different window evidence"

        def publish(value):
            evidence = window_evidence(identity=context.identity(window_id), context={"value": value}, collectors={})
            prepare_window_telemetry(context, window_id, evidence)
    else:
        target = context.record_path(window_id)
        telemetry = publish_telemetry(context, window_id)
        conflict = "refusing to replace a committed window record"

        def publish(value):
            publish_receipt(context, window_id, {**row, "latency_app_s": str(value)}, telemetry,
                            1 if artifact == "v2-receipt" else None)

    ready = Barrier(2)
    installed = []

    def rendezvous(operation):
        def install(source, destination, *args, **kwargs):
            if Path(destination) != target:
                return operation(source, destination, *args, **kwargs)
            ready.wait(timeout=5)
            result = operation(source, destination, *args, **kwargs)
            installed.append(source)
            return result
        return install

    monkeypatch.setattr(os, "link", rendezvous(os.link))
    monkeypatch.setattr(os, "replace", rendezvous(os.replace))
    winners = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        calls = [(value, pool.submit(publish, value)) for value in (1, 1 if identical else 2)]
        for value, future in calls:
            if future.exception() is None:
                winners.append(value)
            else:
                with pytest.raises(ValueError, match=conflict):
                    future.result()
    content = json.loads(target.read_bytes())
    saved = content["context"]["value"] if artifact == "telemetry" else int(content["row"]["latency_app_s"])
    assert winners == [saved] * (2 if identical else 1)
    assert len(installed) == 1
    assert list(target.parent.glob("*.tmp")) == []


def test_unsupported_window_link_fails_without_installing_or_leaving_a_temporary(window, monkeypatch):
    case, context, window_id, _ = window
    target = case.telemetry(context.case_attempt_id) / context.record_path(window_id).name
    unsupported = OSError(errno.EOPNOTSUPP, "hard links unsupported")

    def fail_link(source, destination):
        assert Path(destination) == target
        raise unsupported

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(OSError, match="hard links unsupported") as caught:
        publish_telemetry(context, window_id)
    assert caught.value is unsupported
    assert list(target.parent.iterdir()) == []


def test_first_publication_propagates_a_file_exists_error_after_install(window, monkeypatch):
    case, context, window_id, _ = window
    target = case.telemetry(context.case_attempt_id) / context.record_path(window_id).name
    failure = FileExistsError("fixture parent fsync error")
    fsync = os.fsync

    def fail_once(fd):
        if target.exists() and os.path.samestat(os.fstat(fd), target.parent.stat()):
            monkeypatch.setattr(os, "fsync", fsync)
            raise failure
        return fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_once)
    with pytest.raises(FileExistsError) as caught:
        publish_telemetry(context, window_id)
    assert caught.value is failure
    visible = target.read_bytes()
    assert publish_telemetry(context, window_id)
    assert target.read_bytes() == visible


@pytest.mark.parametrize("artifact", ["telemetry", "v1-receipt", "v2-receipt"])
def test_identical_window_retry_requires_successful_directory_sync(window, monkeypatch, artifact):
    case, context, window_id, row = window
    receipt = context.record_path(window_id)
    if artifact == "telemetry":
        target = case.telemetry(context.case_attempt_id) / receipt.name

        def publish():
            return publish_telemetry(context, window_id)
    else:
        target = receipt
        telemetry = publish_telemetry(context, window_id)

        def publish():
            return publish_receipt(context, window_id, row, telemetry, 1 if artifact == "v2-receipt" else None)

    sync = ParentSync(monkeypatch, target)
    with pytest.raises(OSError, match="parent fsync failure"):
        publish()
    installed, inode = target.read_bytes(), target.stat().st_ino
    with pytest.raises(OSError, match="parent fsync failure"):
        publish()
    sync.blocked = False
    assert publish()
    if artifact != "telemetry":
        with pytest.raises(ValueError, match="refusing to replace a committed window record"):
            publish_receipt(context, window_id, {**row, "latency_app_s": "2"}, telemetry,
                            1 if artifact == "v2-receipt" else None)
    assert sync.calls == 3
    assert target.read_bytes() == installed
    assert target.stat().st_ino == inode


@pytest.mark.parametrize("sequence", [None, 1], ids=["v1", "v2"])
def test_orphan_receipt_must_be_durable_before_restore(window, monkeypatch, sequence):
    case, context, window_id, row = window
    receipt = context.record_path(window_id)
    telemetry = publish_telemetry(context, window_id)
    sync = ParentSync(monkeypatch, receipt)
    with pytest.raises(OSError, match="parent fsync failure"):
        publish_receipt(context, window_id, row, telemetry, sequence)
    head = case.layout.result_csv.read_bytes()
    original = receipt.read_bytes()
    expected = {measurement_key(row)}
    projections = (case.csv, case.requests, case.sidecar("sniff_groups"))
    for path in projections:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"old projection")
    with pytest.raises(OSError, match="parent fsync failure"):
        restore_case_windows(case, run_id=context.run_id, expected=expected)
    assert case.layout.result_csv.read_bytes() == head
    assert all(path.read_bytes() == b"old projection" for path in projections)
    sync.blocked = False
    telemetry_sync = ParentSync(monkeypatch, case.telemetry(context.case_attempt_id) / receipt.name)
    with pytest.raises(OSError, match="parent fsync failure"):
        restore_case_windows(case, run_id=context.run_id, expected=expected)
    assert case.layout.result_csv.read_bytes() == head
    assert all(path.read_bytes() == b"old projection" for path in projections)
    telemetry_sync.blocked = False
    assert restore_case_windows(case, run_id=context.run_id, expected=expected) == 1
    committed = case.layout.result_csv.read_bytes()
    assert json.loads(committed)["row_count"] == 1
    assert receipt.read_bytes() == original
    sync.blocked = telemetry_sync.blocked = True
    assert restore_case_windows(case, run_id=context.run_id, expected=expected) == 1
    assert case.layout.result_csv.read_bytes() == committed
    assert sync.calls == 3
    assert telemetry_sync.calls == 2


def test_recovery_syncs_each_orphan_attempt_directory_once(window, monkeypatch):
    case, context, _, template = window
    expected = set()
    for index in range(2):
        window_id = f"repeat-{index}"
        row = {**template, **context.identity(window_id), "repeat_idx": str(index)}
        telemetry = publish_telemetry(context, window_id)
        publish_receipt(context, window_id, row, telemetry, index + 1)
        expected.add(measurement_key(row))
    receipt = context.record_path("repeat-0")
    windows_dir = ParentSync(monkeypatch, receipt)
    telemetry_dir = ParentSync(monkeypatch, case.telemetry(context.case_attempt_id) / receipt.name)
    windows_dir.blocked = telemetry_dir.blocked = False
    assert restore_case_windows(case, run_id=context.run_id, expected=expected) == 2
    assert windows_dir.calls == telemetry_dir.calls == 1
