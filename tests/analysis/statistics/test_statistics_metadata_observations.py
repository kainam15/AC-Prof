"""Recheck only the selected source and metadata observed by this statistics operation."""
from concurrent.futures import CancelledError
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from acprof import artifacts
from acprof.results.layout import MANIFEST_NAME, ArtifactLayout
from acprof.results.metadata import ResultMetadataSnapshot


def intercept_reads(monkeypatch, path, read):
    original = Path.open

    @contextmanager
    def monitored(self, *args, **kwargs):
        with original(self, *args, **kwargs) as stream:
            yield SimpleNamespace(read=lambda size: read(stream, size)) if self == path else stream

    monkeypatch.setattr(Path, "open", monitored)


@pytest.fixture
def result(tmp_path):
    source = tmp_path / "result_all.csv"
    source.touch()
    return source, ResultMetadataSnapshot(source)


def test_verify_parsed_bytes_preserves_identical_replacement_and_rejects_reformatting(result):
    source, metadata = result
    path = source.with_name("run_state.json")
    original = b'\xef\xbb\xbf{"run_id":"a"}\r\n'
    path.write_bytes(original)
    metadata.read(path.name)
    replacement = path.with_suffix(".next")
    replacement.write_bytes(original)
    replacement.replace(path)
    metadata.verify_current(source)
    path.write_bytes(b'{"run_id": "a"}')
    with pytest.raises(ValueError, match=path.name):
        metadata.verify_current(source)


@pytest.mark.parametrize("initially_present", [True, False])
def test_verify_presence_checks_only_observed_names(result, initially_present):
    source, metadata = result
    path = source.with_name("quality_checks.json")
    if initially_present:
        path.write_bytes(b"{}")
    assert metadata.exists(path.name) == initially_present
    source.with_name("unobserved.json").touch()
    metadata.verify_current(source)
    if initially_present:
        path.unlink()
    else:
        path.touch()
    with pytest.raises(ValueError, match=path.name):
        metadata.verify_current(source)


@pytest.mark.parametrize("failure", ["malformed", "oversized", "denied"])
def test_stable_read_failure_is_preserved_until_original_read_recovers(result, failure, monkeypatch):
    source, metadata = result
    path = source.with_name("quality_checks.json")
    path.write_bytes(b"{}")
    if failure == "malformed":
        path.write_bytes(b"{")
    elif failure == "oversized":
        monkeypatch.setattr(artifacts, "MAX_JSON_ARTIFACT_BYTES", 32)
        path.write_bytes(b" " * 32 + b"{}")
    with monkeypatch.context() as errors:
        if failure == "denied":
            original = Path.open

            def unavailable(self, *args, **kwargs):
                if self == path:
                    raise PermissionError(13, "permission denied", str(path))
                return original(self, *args, **kwargs)

            errors.setattr(Path, "open", unavailable)
        with pytest.raises((ValueError, PermissionError)):
            metadata.read(path.name, label="quality evidence")
        assert metadata.exists(path.name)
        metadata.verify_current(source)
    path.write_bytes(b"{}")
    with pytest.raises(ValueError, match=path.name):
        metadata.verify_current(source)


def test_failed_presence_probe_stays_distinct_from_absence(result, monkeypatch):
    source, metadata = result
    path = source.with_name("quality_checks.json")
    original = Path.lstat

    def denied(self):
        if self == path:
            raise PermissionError(13, "permission denied", str(path))
        return original(self)

    with monkeypatch.context() as denied_access:
        denied_access.setattr(Path, "lstat", denied)
        with pytest.raises(PermissionError, match="permission denied"):
            metadata.exists(path.name)
        metadata.verify_current(source)
    with pytest.raises(ValueError, match=path.name):
        metadata.verify_current(source)


def test_replacing_selected_alias_or_layout_invalidates_metadata_source(result):
    source, metadata = result
    alias = source.with_name("explicit.csv")
    alias.symlink_to(source)
    selected = ResultMetadataSnapshot(alias)
    selected.verify_current(source)
    other = source.with_name("other.csv")
    other.touch()
    alias.unlink()
    alias.symlink_to(other)
    with pytest.raises(ValueError, match=f"different path: {alias}"):
        selected.verify_current(alias)
    alias.unlink()
    alias.symlink_to(alias.name)
    with pytest.raises((ValueError, OSError), match=alias.name):
        selected.verify_current(alias)

    # Same resolved file and experiment root, but this generation routes sidecars elsewhere.
    new_run = ArtifactLayout.for_new_run(source.parent / "v2")
    new_run.initialize()
    (new_run.root / MANIFEST_NAME).replace(source.parent / MANIFEST_NAME)
    with pytest.raises(ValueError, match="layout changed"):
        metadata.verify_current(source)


def test_verification_errors_report_each_changed_observation_in_read_order(result):
    source, metadata = result
    state = source.with_name("run_state.json")
    state.write_bytes(b"{}")
    metadata.read(state.name)
    quality = source.with_name("quality_checks.json")
    assert not metadata.exists(quality.name)
    state.write_bytes(b"{}\n")
    quality.touch()

    errors = list(metadata.verification_errors(source))
    assert len(errors) == 2
    assert state.name in str(errors[0])
    assert quality.name in str(errors[1])
    with pytest.raises(ValueError, match=state.name):
        metadata.verify_current(source)


@pytest.mark.parametrize("valid", [True, False], ids=["compare-bytes", "retry-failed-parse"])
def test_cancel_during_live_metadata_read_does_not_finish_the_file(result, monkeypatch, valid):
    source, metadata = result
    path = source.with_name("quality_checks.json")
    payload = b" " * (512 * 1024) + (b"{}" if valid else b"{")
    path.write_bytes(payload)
    if valid:
        metadata.read(path.name)
    else:
        with pytest.raises(ValueError, match="invalid quality evidence JSON"):
            metadata.read(path.name, label="quality evidence")
    cancelled = Event()
    reads = []

    def read(stream, size):
        assert not cancelled.is_set(), "continued reading a cancelled document"
        data = stream.read(size)
        reads.append(len(data))
        cancelled.set()
        return data

    with monkeypatch.context() as patch:
        intercept_reads(patch, path, read)
        with pytest.raises(CancelledError):
            metadata.verify_current(source, cancelled=cancelled.is_set)
    assert len(reads) == 1
    assert 0 < reads[0] < len(payload)
    metadata.verify_current(source)


def test_metadata_recheck_allows_short_reads_and_only_probes_one_extra_byte(result, monkeypatch):
    source, metadata = result
    path = source.with_name("run_state.json")
    original = b'{"status":"complete"}'
    path.write_bytes(original)
    metadata.read(path.name)
    received = 0

    def read(stream, size):
        nonlocal received
        block = stream.read(min(3, size))
        received += len(block)
        return block

    with monkeypatch.context() as patch:
        intercept_reads(patch, path, read)
        metadata.verify_current(source)
    assert received == len(original)
    received = 0
    path.write_bytes(original + b" " * 100_000)
    with monkeypatch.context() as patch:
        intercept_reads(patch, path, read)
        with pytest.raises(ValueError, match=path.name):
            metadata.verify_current(source)
    assert received == len(original) + 1
