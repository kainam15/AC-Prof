"""Recheck only the selected source and metadata observed by this statistics operation."""
from pathlib import Path

import pytest

from acprof import artifacts
from acprof.results.layout import MANIFEST_NAME, ArtifactLayout
from acprof.results.metadata import ResultMetadataSnapshot


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
    original = Path.stat

    def denied(self, **kwargs):
        if self == path and kwargs.get("follow_symlinks", True):
            raise PermissionError(13, "permission denied", str(path))
        return original(self, **kwargs)

    with monkeypatch.context() as denied_access:
        denied_access.setattr(Path, "stat", denied)
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
