"""Bounded in-memory transport for existing stats/compare CLI stdout.

JSON here is a process protocol, not an experiment artifact. Parsing happens in
our process worker, never on the UI thread, and no partial result is displayed.
"""
from __future__ import annotations

import json
from pathlib import Path

from acprof.tui.reports import ReportView, report_view

MAX_ANALYSIS_BYTES = 32 * 1024 * 1024


class AnalysisOutput:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.size = 0

    def append(self, text: str) -> None:
        self.size += len(text.encode("utf-8"))
        if self.size > MAX_ANALYSIS_BYTES:
            raise ValueError("Analysis output exceeds 32 MiB; select fewer experiments or metrics")
        self.parts.append(text)

    def finish(self, kind: str, returncode: int, source: Path) -> ReportView:
        text = "".join(self.parts)
        self.parts.clear()
        if returncode not in ({0, 1} if kind == "compare" else {0}):
            raise ValueError(text[-4000:].strip() or f"Analysis exited with status {returncode}")
        try:
            payload = json.loads(text)
        except (ValueError, UnicodeError) as exc:
            raise ValueError("Analysis returned no valid JSON payload: " + text[-2000:].strip()) from exc
        if not isinstance(payload, dict):
            raise ValueError("Analysis payload must be an object")
        if kind == "stats":
            if payload.get("resampling_unit") != "csv_request_window":
                raise ValueError("Expected window statistics from the statistics process")
        elif kind == "compare":
            if payload.get("kind") != "independent_experiment_comparison":
                raise ValueError("Expected independent experiment comparison")
            expected_code = 0 if payload.get("status") == "compatible" else 1
            if returncode != expected_code:
                raise ValueError("Comparison exit status disagrees with its payload")
        else:
            raise ValueError(f"Unsupported analysis process: {kind}")
        return report_view(source, payload)
