"""Schema v1 evidence, with one record per item across all pytest phases."""

import importlib.metadata
import platform
import time
from collections import Counter
from datetime import datetime, timezone

from acprof.artifacts import atomic_write_json
from acprof.testing.sharding import suite_digest

OUTCOMES = {"passed": "passed", "failed": "failed", "error": "errors", "skipped": "skipped",
            "expected_failure": "expected_failures", "unexpected_success": "unexpected_successes"}
PRIORITY = {"passed": 0, "skipped": 1, "expected_failure": 2, "failed": 3,
            "unexpected_success": 4, "error": 5}


class Evidence:
    def __init__(self, config):
        self.config = config
        self.started = time.perf_counter()
        self.records = {}
        self.collected = []
        self.selected = []
        self.discovery_counts = {}
        self.collection_errors = []
        self.collection_skips = []

    def record(self, report):
        record = self.records.setdefault(report.nodeid, {
            "id": report.nodeid, "outcome": "passed", "reason": "", "duration_s": 0.0,
        })
        record["duration_s"] += report.duration
        reason = ""
        if hasattr(report, "wasxfail"):
            outcome = "expected_failure" if report.skipped else "unexpected_success"
            reason = report.wasxfail or outcome.replace("_", " ")
        elif report.failed and str(report.longrepr).startswith("[XPASS(strict)]"):
            outcome, reason = "unexpected_success", str(report.longrepr)
        elif report.failed:
            outcome = getattr(report, "acprof_outcome", "error")
            reason = report.longreprtext
        elif report.skipped:
            outcome = "skipped"
            reason = (str(report.longrepr[2]).removeprefix("Skipped: ")
                      if isinstance(report.longrepr, tuple) else report.longreprtext)
        else:
            outcome = "passed"
        if PRIORITY[outcome] >= PRIORITY[record["outcome"]]:
            record["outcome"] = outcome
        if reason:
            record["reason"] = "\n".join(filter(None, (record["reason"], reason)))

    def finish(self, session, exitstatus):
        config = self.config
        records = [self.records[key] for key in sorted(self.records)]
        totals = Counter(OUTCOMES[row["outcome"]] for row in records)
        no_skips = config.getoption("require_no_skips")
        successful = (exitstatus == 0 and bool(records) and len(records) == len(self.selected)
                      and not self.collection_errors and all(self.discovery_counts.values())
                      and not any(totals[key] for key in ("failed", "errors", "unexpected_successes"))
                      and not (no_skips and (totals["skipped"] or totals["expected_failures"]
                                            or self.collection_skips)))
        if not config.option.collectonly and not successful and exitstatus in (0, 5):
            session.exitstatus = 1
        report_path = config.getoption("report")
        if report_path:
            atomic_write_json(report_path, {
                "schema_version": 1, "successful": successful,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_s": time.perf_counter() - self.started,
                "python": platform.python_version(), "platform": platform.platform(),
                "packages": dict(sorted((dist.metadata["Name"], dist.version)
                                        for dist in importlib.metadata.distributions()
                                        if dist.metadata["Name"])),
                "require_no_skips": no_skips,
                "patterns": config.getoption("pattern") or ["test_*.py"],
                "discovery_counts": self.discovery_counts,
                "shard": {"index": config.getoption("shard_index"),
                          "count": config.getoption("shard_count"),
                          "discovered": len(self.collected), "selected": len(self.selected),
                          "suite_sha256": suite_digest(self.collected)},
                "counts": {"run": len(records), **{key: totals[key] for key in OUTCOMES.values()}},
                "tests": records,
                "collected_ids": self.collected,
                "collection_errors": self.collection_errors,
                "collection_skips": self.collection_skips,
            })
