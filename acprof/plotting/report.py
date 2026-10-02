"""Render the three exploratory views as an offline, self-contained HTML artifact."""
from __future__ import annotations

import json
import re
from html import escape
from importlib.resources import files
from pathlib import Path
from typing import TextIO

from acprof.analysis.model import DEFAULT_MATRIX, SUMMARY_FIELDS, AnalysisModel
from acprof.artifacts import atomic_write
from acprof.metric_registry import ANALYSIS_METRICS, VIEW_METRICS


def report_payload(model: AnalysisModel, baseline: str | None = None) -> dict:
    selected = None
    if baseline:
        matches = [c for c in model.configs if baseline in {c["config_id"], c["run_id"]}]
        if len(matches) != 1:
            raise ValueError("baseline 必须匹配唯一 config_id；run_id 含多个配置时请在报告中选择")
        selected = matches[0]["config_id"]
    return {"schema_version": 1, "configs": model.configs, "baseline": selected,
            "registry": {name: ANALYSIS_METRICS[name].to_dict() for name in VIEW_METRICS},
            "default_matrix": DEFAULT_MATRIX, "summary_fields": SUMMARY_FIELDS,
            "sources": [{k: s[k] for k in ("path", "sha256", "run_id", "run_state")}
                        for s in model.sources]}


def write_report(model: AnalysisModel, output: str | Path, *, baseline: str | None = None) -> Path:
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError(f"output must be a new file: {output}")
    payload = report_payload(model, baseline)
    # Imported only after analysis and destination validation, never by collectors or --help.
    from plotly.offline import get_plotlyjs
    resources = files("acprof.plotting")
    replacements = {
        "__DATA__": json.dumps(payload, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"),
        "__PLOTLY__": get_plotlyjs().replace("</script", "<\\/script"),
        "__VIEW_MODEL__": resources.joinpath("view_model.js").read_text(encoding="utf-8"),
        "__APP__": resources.joinpath("report.js").read_text(encoding="utf-8"),
        "__STYLE__": resources.joinpath("report.css").read_text(encoding="utf-8"),
        "__LICENSE__": escape(resources.joinpath("plotly-LICENSE.txt").read_text(encoding="utf-8")),
    }
    # Substitute only template tokens, never strings inside user-controlled JSON.
    def substitute(match: re.Match[str]) -> str:
        return replacements[match.group()]

    html = re.sub("|".join(replacements), substitute,
                  resources.joinpath("report.html").read_text(encoding="utf-8"))

    def write(stream: TextIO) -> None:
        stream.write(html)

    atomic_write(output, write)
    return output
