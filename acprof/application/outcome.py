"""Result returned by one application run."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RunOutcome:
    status: str
    final_csv: str | None = None
    completed_cases: int = 0
    task_collection_path: str | None = None
