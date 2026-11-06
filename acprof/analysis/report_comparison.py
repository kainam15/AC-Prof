"""Adapt strict independent comparisons to an offline report without new estimators."""
from __future__ import annotations

from itertools import permutations
from pathlib import Path

from acprof.analysis.independent_comparison import ExperimentComparisonSession
from acprof.analysis.model import AnalysisModel, verify_analysis_sources
from acprof.analysis.statistics_report import DEFAULT_STATISTICS_METRICS
from acprof.results.csv import ResultValidationError, verify_result_source
from acprof.results.layout import expand_result_sources
from acprof.results.metadata import ResultMetadataSnapshot
from acprof.results.metrics import METRICS, VIEW_METRICS


def comparison_sources(paths) -> list[str]:
    result = []
    for path in expand_result_sources(paths):
        snapshot = ResultMetadataSnapshot(path)
        result.append(str((snapshot.layout.result_csv if path.is_dir() else path).resolve()))
    return result


def verify_independent_report(report: dict) -> None:
    if report.get("kind") != "independent_experiment_comparison" or not report.get("source_sha256"):
        raise ValueError("independent comparison requires source fingerprints")
    for path, expected in report["source_sha256"].items():
        verify_result_source(Path(path), expected)


def _comparison_entry(model, left, right, *, options, purpose, session, report=None):
    sources = {source["path"]: source for source in model.sources}
    if not left or not right or any(path not in sources for path in [*left, *right]):
        raise ValueError("comparison groups must select sources included in this report")
    reason = ""
    ids = [sources[path]["run_id"] for path in [*left, *right]]
    if "unknown" in ids:
        reason = "missing_recorded_run_identity"
    elif len(set(ids)) != len(ids):
        reason = "duplicate_independent_run"
    if purpose == "resource-scaling" and not reason:
        for path in [*left, *right]:
            cases = [(config["device"], config["input_scale"]) for config in model.configs if config["source"] == path]
            if len(cases) != len(set(cases)):
                reason = "resource_scaling_requires_single_configuration_per_device_input"
    if report is None and not reason:
        try:
            report = session.compare(left, right, purpose=purpose, **options)
        except ResultValidationError as error:
            reason = f"invalid_measurement_identity: {error}"
    if report is not None:
        verify_independent_report(report)
        report = {**report, "engine": "python", "origin": "computed", "rng": "python_random_mt19937"}
    def label(paths):
        names = [str(sources[path]["metadata"].get("model_name", "unknown")) + " / " + sources[path]["run_id"]
                 for path in paths]
        return " + ".join(names)
    return {"left": left, "right": right, "left_label": label(left), "right_label": label(right),
            "report": report, "reason": reason,
            "display_only": report is None or report["status"] != "compatible"}


def build_experiment_comparisons(model: AnalysisModel, *, groups=None, independent_report=None,
                                 purpose="same-hardware", statistics_options=None) -> list[dict]:
    """Explicit groups are replicates; positional sources are compared one run at a time."""
    verify_analysis_sources(model)
    if independent_report is not None:
        verify_independent_report(independent_report)
        groups = tuple([run["source"] for run in independent_report["experiments"][side]] for side in ("left", "right"))
    options = dict(statistics_options or {})
    options.pop("block_size", None)  # Independent runs never inherit window block sampling.
    if "metrics" not in options:
        available = {name for entry in model.raw_rows for name in entry["values"]}
        options["metrics"] = [name for name in VIEW_METRICS if name in available and name in METRICS
                              and METRICS[name].window in {"request_window", "matched_control"}] or list(DEFAULT_STATISTICS_METRICS)
    pairs = ([tuple(comparison_sources(side) for side in groups)] if groups else
             [([a["path"]], [b["path"]]) for a, b in permutations(model.sources, 2)])
    session = ExperimentComparisonSession()
    result = [_comparison_entry(model, left, right, options=options, purpose=purpose, session=session, report=independent_report)
              for left, right in pairs]
    verify_analysis_sources(model)
    return result
