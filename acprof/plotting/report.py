"""Render charts, window statistics and evidence as a self-contained HTML artifact."""
from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from contextlib import ExitStack
from functools import partial
from html import escape
from importlib.resources import files
from pathlib import Path
from typing import TextIO

from acprof.analysis.conditions import ALLOWED_RESOURCE_DIMENSIONS, PURPOSES
from acprof.analysis.model import (
    DEFAULT_MATRIX,
    SUMMARY_FIELDS,
    AnalysisModel,
    verify_analysis_sources,
)
from acprof.analysis.report_comparison import (
    build_experiment_comparisons,
    verify_independent_report,
)
from acprof.analysis.report_statistics import build_report_statistics, write_window_inputs
from acprof.analysis.resource_comparison import REPORT_PURPOSES, build_baseline_comparisons
from acprof.artifacts import atomic_write
from acprof.quality import QUALITY_FIELDS
from acprof.results.metrics import ANALYSIS_METRICS, VIEW_METRICS
from acprof.system.cleanup import cleanup_on_exit


def report_payload(model: AnalysisModel, baseline: str | None = None, *, comparison_purpose="within-experiment",
                   statistics_options=None, comparison_groups=None, independent_report=None) -> dict:
    if comparison_purpose not in REPORT_PURPOSES:
        raise ValueError(f"comparison purpose must be one of {REPORT_PURPOSES}")
    selected = None
    if baseline:
        matches = [c for c in model.configs if baseline in {c["config_id"], c["run_id"]}]
        if len(matches) != 1:
            raise ValueError("baseline 必须匹配唯一 config_id；run_id 含多个配置时请在报告中选择")
        selected = matches[0]["config_id"]
    return {"kind": "acprof_analysis_bundle", "schema_version": 1,
            "analysis_contract": {"version": 1, "statistics_engine": "python",
                "filter": "status=ok and warmup=0 and result_origin!=inferred_not_measured",
                "aggregation": "acprof.analysis.model", "comparison": "acprof.analysis.conditions"},
            "configs": model.configs, "baseline": selected,
            "window_statistics": build_report_statistics(model, **(statistics_options or {})),
            "baseline_comparisons": build_baseline_comparisons(model),
            "experiment_comparisons": build_experiment_comparisons(
                model, groups=comparison_groups, independent_report=independent_report,
                purpose="same-hardware" if comparison_purpose == "within-experiment" else comparison_purpose,
                statistics_options=statistics_options),
            "comparison_purpose": comparison_purpose,
            "comparison_policies": {purpose: {"allowed_resource_dimensions":
                ALLOWED_RESOURCE_DIMENSIONS if purpose == "resource-scaling" else []} for purpose in PURPOSES},
            "registry": {name: ANALYSIS_METRICS[name].to_dict() for name in VIEW_METRICS},
            "default_matrix": DEFAULT_MATRIX, "summary_fields": SUMMARY_FIELDS,
            "overview": [{"source": source["path"], "identity": {
                key: source["metadata"].get(key, "unknown") for key in (
                    "model_name", "model_revision", "runtime_backend", "pipeline_tag", "input_scale_type",
                    "cpu", "cpu_model", "gpu", "gpu_name", "platform_fingerprint")}}
                for source in model.sources],
            "sources": [{k: s[k] for k in ("path", "sha256", "run_id", "run_state", "display_name",
                        "metadata", "metadata_sha256", "run_status", "measurement_status", *QUALITY_FIELDS)}
                        for s in model.sources]}


_JSON_LIST_BATCH_SIZE = 64


def _write_report_json(stream: TextIO, payload: dict[str, object]) -> None:
    """Encode bounded list batches without assembling the full document in memory."""
    encoder = json.JSONEncoder(ensure_ascii=True, allow_nan=False)

    def append(item: object) -> None:
        encoded = encoder.encode(item)
        stream.write(encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))

    stream.write("{")
    for index, (key, value) in enumerate(payload.items()):
        if index:
            stream.write(encoder.item_separator)
        append(key)
        stream.write(encoder.key_separator)
        if isinstance(value, list):
            stream.write("[")
            for start in range(0, len(value), _JSON_LIST_BATCH_SIZE):
                if start:
                    stream.write(encoder.item_separator)
                batch = encoder.encode(value[start:start + _JSON_LIST_BATCH_SIZE])
                stream.write(batch[1:-1].replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))
            stream.write("]")
        else:
            append(value)
    stream.write("}")


def write_report(model: AnalysisModel, output: str | Path, *, baseline: str | None = None,
                 comparison_purpose="within-experiment", statistics_options=None,
                 comparison_groups=None, independent_report=None) -> Path:
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError(f"output must be a new file: {output}")
    payload = report_payload(model, baseline, comparison_purpose=comparison_purpose,
                             statistics_options=statistics_options, comparison_groups=comparison_groups,
                             independent_report=independent_report)
    # Imported only after analysis and destination validation, never by collectors or --help.
    from plotly.offline import get_plotlyjs
    resources = files("acprof.plotting")
    template = resources.joinpath("report.html").read_text(encoding="utf-8")
    replacements = {
        "__PLOTLY__": get_plotlyjs().replace("</script", "<\\/script"),
        "__VIEW_MODEL__": resources.joinpath("view_model.js").read_text(encoding="utf-8"),
        "__APP__": resources.joinpath("report.js").read_text(encoding="utf-8"),
        "__EXPERIMENTS__": resources.joinpath("report_experiments.js").read_text(encoding="utf-8"),
        "__EXPORT__": resources.joinpath("report_export.js").read_text(encoding="utf-8"),
        "__STYLE__": resources.joinpath("report.css").read_text(encoding="utf-8"),
        "__LICENSE__": escape(resources.joinpath("plotly-LICENSE.txt").read_text(encoding="utf-8")),
    }
    # Optional template tokens also keep minimal/custom template consumers working.
    for token, asset in {"__IMPORT__": "report_import.js", "__WORKSPACE__": "report_workspace.js",
                         "__I18N__": "report_i18n.js", "__PAPA__": "vendor/papaparse.min.js",
                         "__SHA256__": "vendor/sha256.js", "__STATISTICS_RNG__": "statistics_rng.js",
                         "__STATISTICS_ENGINE__": "statistics_engine.js",
                         "__STATISTICS_CONTROLS__": "report_statistics.js"}.items():
        if token in template:
            replacements[token] = resources.joinpath(asset).read_text(encoding="utf-8")
    if "__IMPORT_LICENSE__" in template:
        replacements["__IMPORT_LICENSE__"] = escape("\n\n".join(
            resources.joinpath("vendor", name).read_text(encoding="utf-8")
            for name in ("papaparse-LICENSE.txt", "sha256-LICENSE.txt")))

    def write(stream: TextIO) -> None:
        # Only scan the original template: asset and user-data tokens are literal.
        # Write fragments directly, avoiding a second full page in memory.
        position = 0
        for match in re.finditer("__DATA__|" + "|".join(replacements), template):
            stream.write(template[position:match.start()])
            if match.group() == "__DATA__":
                _write_report_json(stream, payload)
            else:
                stream.write(replacements[match.group()])
            position = match.end()
        stream.write(template[position:])
        verify_analysis_sources(model)
        for comparison in payload.get("experiment_comparisons", []):
            if comparison["report"] is not None:
                verify_independent_report(comparison["report"])

    identities: dict[Path, tuple[int, int]] = {}

    def publish(path: Path, writer: Callable[[TextIO], None]) -> None:
        def tracked_write(stream: TextIO) -> None:
            opened = os.fstat(stream.fileno())
            identities[path] = (opened.st_dev, opened.st_ino)
            writer(stream)
        atomic_write(path, tracked_write, overwrite=False)

    def is_published(path: Path) -> bool:
        identity = identities.get(path)
        if identity is None:
            return False
        try:
            current = path.lstat()
        except FileNotFoundError:
            return False
        return (current.st_dev, current.st_ino) == identity

    def discard_input(path: Path) -> None:
        # Preserve inputs once our HTML is visible, even if its directory sync fails.
        if not is_published(output) and is_published(path):
            path.unlink(missing_ok=True)

    with ExitStack() as cleanup:
        for index, report in enumerate(payload.get("window_statistics", [])):
            inputs = report.get("recalculation", {})
            if inputs.get("status") == "external_required":
                sidecar = output.with_name(f"{output.stem}.windows-{index + 1}.json")
                if sidecar.exists() or sidecar.is_symlink():
                    raise ValueError(f"output must be a new file: {sidecar}")
                # Register before publication: a link can succeed before fsync raises.
                cleanup.enter_context(cleanup_on_exit(
                    partial(discard_input, sidecar), description=f"report window inputs {sidecar}",
                    on_error_only=True,
                ))
                publish(sidecar, lambda stream: write_window_inputs(model, report, stream))
                inputs["file"] = sidecar.name
        publish(output, write)
    return output
