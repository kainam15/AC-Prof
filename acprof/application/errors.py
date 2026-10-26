"""Application failures translated to exit codes only by CLI adapters."""


class CollectionFailed(RuntimeError):
    """Collection or required evidence did not complete successfully."""

    def __init__(self, message: str, *, task_collection_path: str | None = None):
        super().__init__(message)
        self.task_collection_path = task_collection_path
