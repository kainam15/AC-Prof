"""Stage a comparison's exports and HTML window inputs before claiming their names."""

import os
import shutil
import stat
import tempfile
from contextlib import ExitStack
from functools import partial
from pathlib import Path

from acprof.artifacts import create_durable_directory, sync_directory
from acprof.results.outputs import validate_comparison_destinations
from acprof.system.cleanup import cleanup_on_exit


def _discard_link(staged: Path, destination: Path) -> None:
    try:
        current = destination.lstat()
    except FileNotFoundError:
        return
    # The private link pins the inode until every rollback callback has finished.
    if os.path.samestat(current, staged.lstat()):
        destination.unlink(missing_ok=True)
        sync_directory(destination.parent)


class ComparisonPublication(ExitStack):
    """Keep private output directories through publication and any ownership checks."""

    def __init__(self) -> None:
        super().__init__()
        self._outputs: list[tuple[Path, Path]] = []

    def stage(self, destination: Path) -> Path:
        # A missing parent might itself be the name of a yet undiscovered sidecar.
        parent = destination.parent.resolve()
        while True:
            try:
                parent.stat()
                break
            except FileNotFoundError:
                parent = parent.parent
        directory = Path(tempfile.mkdtemp(prefix=".acprof-compare-", dir=parent))
        self.enter_context(cleanup_on_exit(
            partial(shutil.rmtree, directory), description=f"comparison staging directory {directory}"))
        staged = directory / destination.name
        self._outputs.append((staged, destination))
        return staged

    def publish(self) -> None:
        sidecars = [(file, destination.with_name(file.name))
                    for staged, destination in self._outputs
                    for file in sorted(staged.parent.iterdir()) if file != staged]
        files = [*sidecars, *self._outputs]
        validate_comparison_destinations(destination for _, destination in files)
        for staged, _ in files:
            if not stat.S_ISREG(staged.lstat().st_mode):
                raise ValueError(f"comparison output must be a regular file: {staged}")

        with ExitStack() as rollback:
            for staged, destination in files:
                create_durable_directory(destination.parent)
                # Register before link: interruption can arrive after the name is visible.
                rollback.enter_context(cleanup_on_exit(
                    partial(_discard_link, staged, destination),
                    description=f"comparison output {destination}", on_error_only=True))
                os.link(staged, destination)
                if destination != files[-1][1]:
                    sync_directory(destination.parent)
        # Keep the complete, visible bundle if the last directory sync fails.
        if files:
            sync_directory(files[-1][1].parent)
