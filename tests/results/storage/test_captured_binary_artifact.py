"""An unpublished capture has no path; its installed file is immutable and durable."""
import errno
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from acprof import artifacts


def test_capture_claims_its_inode_when_linkable_and_never_reopens_input(tmp_path):
    destination = tmp_path / "digest.csv"
    native = False
    if hasattr(os, "O_TMPFILE"):
        try:
            fd = os.open(tmp_path, os.O_TMPFILE | os.O_RDWR, 0o600)
            try:
                os.link(f"/proc/self/fd/{fd}", tmp_path / "probe", src_dir_fd=fd, follow_symlinks=True)
                native = True
                (tmp_path / "probe").unlink()
            finally:
                os.close(fd)
        except NotImplementedError:
            native = False  # This Python build cannot request linkat with these flags.
        except OSError as exc:
            if exc.errno not in {errno.EISDIR, errno.EINVAL, errno.EOPNOTSUPP, errno.ENOENT,
                                 errno.ENOSYS, errno.EPERM, errno.EACCES, errno.EXDEV}:
                raise
    with artifacts.capture_binary_artifact(tmp_path) as capture:
        capture.stream.write(b"exact source bytes")
        source_stat = os.fstat(capture.stream.fileno())
        assert source_stat.st_nlink == 0
        assert list(tmp_path.iterdir()) == []
        capture.publish(destination, verify_existing=lambda: pytest.fail("unexpected collision"))
        assert capture.stream.closed
    assert destination.read_bytes() == b"exact source bytes"
    assert os.path.samestat(destination.stat(), source_stat) == native
    assert stat.S_IMODE(destination.stat().st_mode) == 0o644


@pytest.mark.parametrize("unavailable", ["open", "old-kernel", "proc", "no-linkat", "cross-device"])
def test_capture_falls_back_to_exclusive_local_copy(tmp_path, monkeypatch, unavailable):
    destination = tmp_path / "digest.csv"
    real_open, real_link = os.open, os.link
    copies = []

    def open_file(path, flags, *args, **kwargs):
        if (unavailable in {"open", "old-kernel"} and hasattr(os, "O_TMPFILE")
                and flags & os.O_TMPFILE == os.O_TMPFILE and not flags & os.O_EXCL):
            raise OSError(errno.ENOENT if unavailable == "old-kernel" else errno.EOPNOTSUPP, "no linkable anonymous files")
        return real_open(path, flags, *args, **kwargs)

    def link(src, dst, **kwargs):
        if str(src).startswith("/proc/"):
            if unavailable == "no-linkat":
                raise NotImplementedError("interpreter does not support linkat flags")
            raise OSError(errno.EXDEV if unavailable == "cross-device" else errno.ENOENT, "no native link")
        copies.append(Path(src).parent)
        return real_link(src, dst, **kwargs)

    monkeypatch.setattr(os, "open", open_file)
    monkeypatch.setattr(os, "link", link)
    with artifacts.capture_binary_artifact(tmp_path) as capture:
        data = b"captured" * 100_000
        capture.stream.write(data)
        read = capture.stream.read
        lengths = []

        def bounded_read(size=-1):
            lengths.append(size)
            assert 0 < size < len(data)
            return read(size)

        monkeypatch.setattr(capture.stream, "read", bounded_read)
        capture.publish(destination, verify_existing=lambda: pytest.fail("unexpected collision"))
    assert destination.read_bytes() == data
    assert copies == [destination.parent]
    assert len(lengths) > 1
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("race", [False, True], ids=["existing", "racing"])
@pytest.mark.parametrize("copy", [False, True], ids=["native", "fallback"])
def test_capture_preserves_conflicting_winner(tmp_path, monkeypatch, race, copy):
    destination = tmp_path / "winner"
    real_link = os.link
    verified = []

    def link(src, dst, **kwargs):
        if copy and str(src).startswith("/proc/"):
            raise OSError(errno.EXDEV, "other device")
        destination.write_bytes(b"winner")
        return real_link(src, dst, **kwargs)

    def verify():
        verified.append(destination.read_bytes())
        raise ValueError("different archive")

    if race:
        monkeypatch.setattr(os, "link", link)
    else:
        destination.write_bytes(b"winner")
    with artifacts.capture_binary_artifact(tmp_path) as capture:
        capture.stream.write(b"loser")
        with pytest.raises(ValueError, match="different archive"):
            capture.publish(destination, verify_existing=verify)
    assert verified == [b"winner"]
    assert destination.read_bytes() == b"winner"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize(("operation", "code"), [("fchmod", errno.EEXIST), ("fsync", errno.EIO),
                                          ("link", errno.ENOSPC), ("open", errno.EIO)])
def test_capture_preserves_publication_errors(tmp_path, monkeypatch, operation, code):
    destination = tmp_path / "unpublished"

    def fail(*args, **kwargs):
        raise OSError(code, "injected operation failure")

    def publish():
        if operation == "open":
            monkeypatch.setattr(os, operation, fail)
        with artifacts.capture_binary_artifact(tmp_path) as capture:
            capture.stream.write(b"data")
            monkeypatch.setattr(os, operation, fail)
            capture.publish(destination, verify_existing=lambda: pytest.fail("unexpected verification"))
    with pytest.raises(OSError, match="injected operation failure") as failure:
        publish()
    assert failure.value.errno == code
    assert list(tmp_path.iterdir()) == []


def test_capture_retries_visible_file_directory_sync(tmp_path, monkeypatch):
    destination = tmp_path / "archive"
    real_sync = os.fsync
    verified = []

    def fail_directory(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EIO, "directory fsync")
        return real_sync(fd)

    def publish():
        with artifacts.capture_binary_artifact(tmp_path) as capture:
            capture.stream.write(b"payload")
            capture.publish(destination, verify_existing=lambda: verified.append(destination.read_bytes()))

    with monkeypatch.context() as fault:
        fault.setattr(os, "fsync", fail_directory)
        for _ in range(2):
            with pytest.raises(OSError, match="directory fsync"):
                publish()
    publish()
    assert verified == [b"payload"] * 2


def test_capture_close_failure_is_not_a_competing_claim(tmp_path, monkeypatch):
    destination = tmp_path / "archive"
    error = OSError(errno.EEXIST, "capture close")
    with artifacts.capture_binary_artifact(tmp_path) as capture:
        capture.stream.write(b"valid")

        def fail_close():
            raise error

        monkeypatch.setattr(capture.stream, "close", fail_close)
        with pytest.raises(OSError, match="capture close") as failure:
            capture.publish(destination, verify_existing=lambda: pytest.fail("unexpected verification"))
        assert failure.value is error
    assert destination.read_bytes() == b"valid"


def test_process_exit_during_capture_leaves_no_stage_name(tmp_path):
    program = """
import os
import sys
from pathlib import Path
from acprof.artifacts import capture_binary_artifact

with capture_binary_artifact(Path(sys.argv[1])) as capture:
    capture.stream.write(b'partial result')
    os._exit(17)
"""
    child = subprocess.run([sys.executable, "-c", program, str(tmp_path)], check=False)
    assert child.returncode == 17
    assert list(tmp_path.iterdir()) == []


def test_capture_missing_directory_preserves_open_error(tmp_path, monkeypatch):
    missing = tmp_path / "missing"
    error = FileNotFoundError(errno.ENOENT, "capture directory missing")

    def fail(*args, **kwargs):
        raise error

    def forbid_fallback(*args, **kwargs):
        pytest.fail("a missing directory is not an unsupported capture flag")

    monkeypatch.setattr(os, "O_TMPFILE", getattr(os, "O_TMPFILE", 1), raising=False)
    monkeypatch.setattr(os, "open", fail)
    monkeypatch.setattr(artifacts.tempfile, "TemporaryFile", forbid_fallback)
    with pytest.raises(FileNotFoundError, match="capture directory missing") as failure, artifacts.capture_binary_artifact(missing):
        pytest.fail("opened missing directory")
    assert failure.value is error
