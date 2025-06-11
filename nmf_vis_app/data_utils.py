import json
import glob
import re
from pathlib import Path
from typing import Final, List, Tuple, Dict

import numpy as np
import pandas as pd

from helpers.sort_utils import get_sample_order
from helpers.color_utils import load_cancer_colors, component_palette


cache: Final[dict[Path, pd.DataFrame]] = {}


def load_cfg(path: str | Path = "config.json") -> dict:
    return json.load(open(path, "r"))


def discover_nmf_k_files(cfg: dict) -> List[Dict[str, str]]:
    """
    Discover NMF files with different K values in the comps folder.

    Returns a list of dictionaries with K-value info:
    [{"filename": "file.csv", "k_value": "26", "path": "/full/path/file.csv"}, ...]
    """
    data_dir = cfg.get("NMF_DATA_DIRECTORY", "comps")
    file_pattern = cfg.get("NMF_FILE_PATTERN", "*.csv")

    pattern = str(Path(data_dir) / file_pattern)

    k_files = []
    for file_path in glob.glob(pattern):
        path = Path(file_path)
        k_match = re.search(r"_[kK](\d+)\.csv$", path.name)
        if k_match:
            k_value = k_match.group(1)
            k_files.append(
                {
                    "filename": path.name,
                    "k_value": int(k_value),
                    "path": str(path),
                    "display_name": f"K = {k_value}",
                }
            )

    k_files.sort(key=lambda x: int(x["k_value"]))
    return k_files


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
        Path(cfg.get("DEFAULT_CSV_FILENAME", "comps/all_H_component_contributions.csv"))
    )
    n_samples, n_components = H.shape

    component_order = np.argsort(-H.sum(axis=0))
    H_ord = H[:, component_order]

    sample_order = get_sample_order(sort_method, H, sample_ids, cancer_types, cfg_path)

    H_sorted = H_ord[sample_order]
    x_labels_short = np.array(
        [label[:4] for label in np.array(sample_ids)[sample_order]]
    )

    with open("config.json", "r") as f:
        cfg = json.load(f)

    component_colors = _load_component_colors(
        cfg.get("JSON_FILENAME_COMPONENT_COLORS", "nmf_component_color_map.json"),
        n_components,
        component_order,
    )
    cancer_color_map = load_cancer_colors(cfg.get("JSON_FILENAME_CANCER_TYPE_COLORS"))

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
    )
