"""Self-contained offline curves for validated streaming load sweep evidence."""
from __future__ import annotations

import json
import os
import re
from html import escape
from importlib.resources import files
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import quote

from acprof.analysis.load_sweep import read_load_sweep, verify_load_sweep_sources
from acprof.artifacts import atomic_write


def write_load_sweep_report(source: str | Path, output: str | Path) -> dict[str, Any]:
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError(f"output must be a new file: {output}")
    analysis = read_load_sweep(source)
    payload = analysis.payload
    if output.resolve() in {analysis.root / point["report_path"] for point in payload["points"]}:
        raise ValueError("output must not replace a planned point report, including a missing or unrun point")
    for point in payload["points"]:
        if point["evidence_status"] in {"valid", "partial"}:
            relative = os.path.relpath(analysis.root / point["report_path"], output.resolve().parent)
            point["report_href"] = quote("./" + relative, safe="/")
    payload["sources"] = [{"path": path.relative_to(analysis.root).as_posix(), "sha256": digest}
                          for path, digest in analysis.sources.items()]
    encoded = json.dumps(payload, ensure_ascii=True, allow_nan=False)
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    from plotly.offline import get_plotlyjs
    resources = files("acprof.plotting")
    replacements = {
        "__PLOTLY__": get_plotlyjs().replace("</script", "<\\/script"),
        "__APP__": resources.joinpath("load_sweep.js").read_text(encoding="utf-8"),
        "__STYLE__": resources.joinpath("load_sweep.css").read_text(encoding="utf-8"),
        "__LICENSE__": escape(resources.joinpath("plotly-LICENSE.txt").read_text(encoding="utf-8")),
        "__DATA__": encoded,
    }
    template = resources.joinpath("load_sweep.html").read_text(encoding="utf-8")

    def write(stream: TextIO) -> None:
        position = 0
        for match in re.finditer("|".join(replacements), template):
            stream.write(template[position:match.start()])
            stream.write(replacements[match.group()])
            position = match.end()
        stream.write(template[position:])
        verify_load_sweep_sources(analysis)

    atomic_write(output, write, overwrite=False)
    return {"kind": "stream_load_sweep_capacity", **{key: payload[key] for key in (
        "point_count", "valid_points", "partial_points", "unavailable_points",
    )}}
