"""CJK-safe diagnostic SVG font selection for synthetic Pilot captures only.

Rich's SVG exporter hardcodes Fira Code and a remote webfont. Do not alter
Textual's production screenshot machinery or committed SVG Golden snapshots.
"""

from __future__ import annotations

import re

CJK_FONT_FAMILY = "Noto Sans Mono CJK SC"
_FONT_FACE = re.compile(r"(?m)^    @font-face \{\n.*?^    \}\n", re.DOTALL)
_MATRIX_FONT = "font-family: Fira Code, monospace;"


def use_cjk_svg_font(svg: str) -> str:
    """Drop unreachable CDN fonts and prefer Noto for test-only diagnostics."""
    svg, face_count = _FONT_FACE.subn("", svg)
    if face_count != 2 or svg.count(_MATRIX_FONT) != 1:
        raise ValueError("Rich SVG font template changed; cannot apply Pilot CJK override")
    return svg.replace(_MATRIX_FONT, f'font-family: "{CJK_FONT_FAMILY}", monospace;')
