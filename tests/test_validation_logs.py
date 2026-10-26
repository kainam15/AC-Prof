"""Publication errors must not leak captures or replace an earlier artifact."""
import subprocess

import pytest

from acprof.host import validation_logs


def test_publication_failure_closes_capture_files_and_preserves_existing_log(tmp_path, monkeypatch):
    path = tmp_path / "validation.log"
    path.write_text("previous attempt")
    log = validation_logs.ValidationLog(path)
    streams = []

    def run(command, **kwargs):
        for channel in ("stdout", "stderr"):
            streams.append(kwargs[channel])
            kwargs[channel].write(channel + " new output")
        return subprocess.CompletedProcess(command, 0)

    log.run(["fixture"], run, timeout=1)

    def fail_copy(source, destination, **kwargs):
        destination.write("partial publication")
        raise OSError("disk full")

    monkeypatch.setattr(validation_logs.shutil, "copyfileobj", fail_copy)
    with pytest.raises(OSError, match="disk full"):
        log.close()
    assert all(stream.closed for stream in streams)
    assert path.read_text() == "previous attempt"
    assert list(tmp_path.iterdir()) == [path]
