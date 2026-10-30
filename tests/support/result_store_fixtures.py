"""Filesystem assertions for result-store resource ownership."""
import fcntl

from acprof.results.layout import ArtifactLayout


def assert_writer_lease_available(layout: ArtifactLayout) -> None:
    """Check the real lease without replaying potentially damaged evidence."""
    with layout.result_store_path("writer.lock").open("rb") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
