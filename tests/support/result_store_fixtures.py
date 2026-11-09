"""Filesystem assertions and real interrupted result-store transactions."""
import fcntl

import pytest

from acprof.result_store import writer as writer_module
from acprof.result_store.writer import ResultWriter
from acprof.results.layout import ArtifactLayout


def assert_writer_lease_available(layout: ArtifactLayout) -> None:
    """Check the real lease without replaying potentially damaged evidence."""
    with layout.result_store_path("writer.lock").open("rb") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def leave_pending_append(layout, source, monkeypatch):
    """Fail through _commit after an actual append has flushed and synced."""
    with ResultWriter(layout.root) as writer:
        append = writer._append_bytes

        def fail_after_sync(layer, raw):
            append(layer, raw)
            raise OSError("fixture append interrupted after sync")

        with monkeypatch.context() as guard:
            guard.setattr(writer, "_append_bytes", fail_after_sync)
            with pytest.raises(OSError, match="append interrupted after sync"):
                writer.append_window(source)
    assert layout.result_store_path("pending.json").is_file()


def leave_pending_replacement(layout, monkeypatch):
    """Persist a complete replacement transaction before changing its CSVs."""
    with ResultWriter(layout.root) as writer:
        def fail_before_replacement(*_args):
            raise OSError("fixture interrupted before module replacement")

        with monkeypatch.context() as guard:
            guard.setattr(writer_module, "write_bytes", fail_before_replacement)
            with pytest.raises(OSError, match="before module replacement"):
                writer.update_rows([{**row, "latency_app_s": "9"} for row in writer.rows.values()],
                                   reason="fixture replacement")
    assert layout.result_store_path("pending.json").is_file()
