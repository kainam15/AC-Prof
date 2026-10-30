"""Render the three exploratory views as an offline, self-contained HTML artifact."""
from __future__ import annotations

import json
import re
from html import escape
from importlib.resources import files
from pathlib import Path
from typing import TextIO

from acprof.analysis.conditions import ALLOWED_RESOURCE_DIMENSIONS, PURPOSES
from acprof.analysis.model import DEFAULT_MATRIX, SUMMARY_FIELDS, AnalysisModel
from acprof.artifacts import atomic_write
from acprof.quality import QUALITY_FIELDS
from acprof.results.metrics import ANALYSIS_METRICS, VIEW_METRICS


def report_payload(model: AnalysisModel, baseline: str | None = None, *, comparison_purpose="same-hardware") -> dict:
    if comparison_purpose not in PURPOSES:
        raise ValueError(f"comparison purpose must be one of {PURPOSES}")
    selected = None
    if baseline:
        matches = [c for c in model.configs if baseline in {c["config_id"], c["run_id"]}]
        if len(matches) != 1:
            raise ValueError("baseline 必须匹配唯一 config_id；run_id 含多个配置时请在报告中选择")
        selected = matches[0]["config_id"]
    return {"schema_version": 1, "configs": model.configs, "baseline": selected,
            "comparison_purpose": comparison_purpose,
            "comparison_policies": {purpose: {"allowed_resource_dimensions":
                ALLOWED_RESOURCE_DIMENSIONS if purpose == "resource-scaling" else []} for purpose in PURPOSES},
            "registry": {name: ANALYSIS_METRICS[name].to_dict() for name in VIEW_METRICS},
            "default_matrix": DEFAULT_MATRIX, "summary_fields": SUMMARY_FIELDS,
            "sources": [{k: s[k] for k in ("path", "sha256", "run_id", "run_state",
                        "run_status", "measurement_status", *QUALITY_FIELDS)}
                        for s in model.sources]}


def _write_report_json(stream: TextIO, payload: dict[str, object]) -> None:
    """Encode growing top-level lists one entry at a time with the C encoder."""
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
            for entry_index, entry in enumerate(value):
                if entry_index:
                    stream.write(encoder.item_separator)
                append(entry)
            stream.write("]")
        else:
            append(value)
    stream.write("}")


def write_report(model: AnalysisModel, output: str | Path, *, baseline: str | None = None,
                 comparison_purpose="same-hardware") -> Path:
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError(f"output must be a new file: {output}")
    payload = report_payload(model, baseline, comparison_purpose=comparison_purpose)
    # Imported only after analysis and destination validation, never by collectors or --help.
    from plotly.offline import get_plotlyjs
    resources = files("acprof.plotting")
    replacements = {
        "__PLOTLY__": get_plotlyjs().replace("</script", "<\\/script"),
        "__VIEW_MODEL__": resources.joinpath("view_model.js").read_text(encoding="utf-8"),
        "__APP__": resources.joinpath("report.js").read_text(encoding="utf-8"),
        "__STYLE__": resources.joinpath("report.css").read_text(encoding="utf-8"),
        "__LICENSE__": escape(resources.joinpath("plotly-LICENSE.txt").read_text(encoding="utf-8")),
    }
    template = resources.joinpath("report.html").read_text(encoding="utf-8")

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

    atomic_write(output, write, overwrite=False)
    return output
