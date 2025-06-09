from __future__ import annotations
import json
from pathlib import Path
import plotly.express as px
import plotly.colors as pc
from typing import Dict

# ------------------------------------------------------------------ #
def component_palette(n: int) -> list[str]:
    return pc.sample_colorscale("Viridis", [i/(n-1) for i in range(n)])

def distinct_palette(n: int) -> list[str]:
    return px.colors.qualitative.Alphabet[:n]

# ------------------------------------------------------------------ #
def load_cancer_colors(json_path: str | Path | None) -> Dict[str, str]:
    """Load user-defined cancer‐type color map.  Empty dict if file missing."""
    if not json_path:
        return {}
    p = Path(json_path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())
