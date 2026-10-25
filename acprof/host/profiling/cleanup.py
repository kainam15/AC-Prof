"""Discard raw profiles only when publication can state their absence truthfully."""

import os
import shutil
import uuid
from pathlib import Path

from acprof.artifacts import replace_file_durably


def discard_profile_artifacts(profile_root: str, plan_path: str) -> None:
    """Invalidate old claims before a possibly partial deletion; preserve their bytes."""
    plan = Path(plan_path)
    invalidated = None
    if plan.exists():
        invalidated = plan.with_name(f"{plan.name}.invalidated-{uuid.uuid4().hex}")
        replace_file_durably(plan, invalidated)
    try:
        if os.path.lexists(profile_root):
            shutil.rmtree(profile_root)
        if os.path.lexists(profile_root):
            raise OSError("raw profile directory remains after removal")
    except OSError as exc:
        raise OSError(
            f"raw profile cleanup failed; deletion may be partial: {profile_root}; "
            f"prior plan (not valid evidence): {invalidated or 'none'}. "
            "No new profile plan was published."
        ) from exc
