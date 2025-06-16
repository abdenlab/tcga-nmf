import json
import glob
import re
from pathlib import Path
from typing import Final, List, Tuple, Dict

import numpy as np
import pandas as pd

from nmf_vis.sort_utils import get_sample_order
from nmf_vis.color_utils import load_cancer_colors, component_palette


cache: Final[dict[Path, pd.DataFrame]] = {}


def load_cfg(path: str | Path = "config.json") -> dict:
    return json.load(open(path, "r"))


def _get_dataframe(filepath: Path) -> pd.DataFrame:
    if filepath not in cache:
        cache[filepath] = pd.read_csv(filepath)
    return cache[filepath]


def _get_prepared_data(
    filepath: Path,
    sample_id_column: str = "sample_id",
    selection: list[int] | None = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Get H matrix, sample IDs, and cancer types, ready for visualization.
    """

    df = _get_dataframe(filepath)

    if selection is not None:
        df = df.iloc[selection]

    component_columns = [
        c
        for c in df.columns
        if c != sample_id_column and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not component_columns:
        raise ValueError(f"No numeric component columns found in {filepath}")

    sample_ids = df[sample_id_column].tolist()
    H = df[component_columns].values

    cancer_types = [i[:4] for i in sample_ids]

    return H, sample_ids, cancer_types


def _load_component_colors(path, n_components, component_order):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            f"Comp_{i}": color
            for i, color in zip(component_order, component_palette(n_components))
        }


def load_all_data(cfg_path, sort_method):
    """Loads and prepares all data needed for the visualization."""
    cfg = load_cfg(cfg_path)
    H, sample_ids, cancer_types = _get_prepared_data(
        Path(cfg.get("DEFAULT_CSV_FILENAME", "data/all_H_component_contributions.csv"))
    )
    n_samples, n_components = H.shape

    component_order = np.argsort(-H.sum(axis=0))
    H_ord = H[:, component_order]

    sample_order = get_sample_order(sort_method, H, sample_ids, cancer_types, cfg_path)

    H_sorted = H_ord[sample_order]
    x_labels_short = np.array(
        [label[:4] for label in np.array(sample_ids)[sample_order]]
    )

    component_colors = _load_component_colors(
        cfg.get("JSON_FILENAME_COMPONENT_COLORS", "nmf_component_color_map.json"),
        n_components,
        component_order,
    )
    cancer_color_map = load_cancer_colors(cfg.get("JSON_FILENAME_CANCER_TYPE_COLORS"))

    umap_df = pd.read_parquet(cfg.get("UMAP_FILENAME"))

    # These would be loaded similarly from the other JSON files
    organ_systems = []
    organ_system_colors = {}
    embryonic_layers = []
    embryonic_layer_colors = {}

    return (
        H,
        sample_ids,
        cancer_types,
        component_colors,
        cancer_color_map,
        organ_systems,
        organ_system_colors,
        embryonic_layers,
        embryonic_layer_colors,
        H_sorted,
        x_labels_short,
        component_order,
        umap_df,
    )
