"""Application failures translated to exit codes only by CLI adapters."""


class CollectionFailed(RuntimeError):
    """Collection or required evidence did not complete successfully."""
