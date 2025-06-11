from __future__ import annotations
import json
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import glob
import re
from typing import List, Tuple, Dict


# ------------------------------------------------------------------ #
def load_cfg(path: str | Path = "config.json") -> dict:
    """Load configuration from JSON file."""
    return json.loads(Path(path).read_text())


# ------------------------------------------------------------------ #
def _load_from_csv(csv_path: str | Path) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(csv_path, sep=None, engine="python")  # auto-detect delimiter
    sample_ids = df.iloc[:, 0].astype(str).tolist()
    H = df.iloc[:, 1:].to_numpy(float)
    return H, sample_ids


# ------------------------------------------------------------------ #
def _load_from_npy(
    h_path: str | Path, parquet_path: str | Path
) -> Tuple[np.ndarray, List[str]]:
    H = np.load(h_path)
    ids = sorted(pl.scan_parquet(parquet_path).collect_schema().names()[6:])
    if H.shape[0] != len(ids):
        raise ValueError("Sample count mismatch between H and parquet IDs.")
    return H, ids


# ------------------------------------------------------------------ #
def discover_nmf_k_files(cfg_path: str | Path = "config.json") -> List[Dict[str, str]]:
    """
    Discover NMF files with different K values in the comps folder.

    Returns a list of dictionaries with K-value info:
    [{"filename": "file.csv", "k_value": "26", "path": "/full/path/file.csv"}, ...]
    """
    cfg = load_cfg(cfg_path)
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
                    "k_value": k_value,
                    "path": str(path),
                    "display_name": f"K = {k_value}",
                }
            )

    k_files.sort(key=lambda x: int(x["k_value"]))
    return k_files


# ------------------------------------------------------------------ #
def get_prepared_data(
    cfg_path: str | Path = "config.json", k_filename: str = None
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Get H matrix, sample IDs, and cancer types, ready for visualization.

    Parameters
    ----------
    cfg_path : str | Path
        Path to configuration file
    k_filename : str, optional
        Specific K-value CSV filename to load. If None, uses the default from config.

    Returns
    -------
    H : np.ndarray
        Component values matrix
    sample_ids : list[str]
        Sample identifiers
    cancer_types : list[str]
    """
    cfg = load_cfg(cfg_path)

    if k_filename is None:
        default_csv_full_path = cfg.get("DEFAULT_CSV_FILENAME")
        if default_csv_full_path:
            k_filename = Path(default_csv_full_path).name
        else:
            k_files = discover_nmf_k_files(cfg_path)
            if k_files:
                k_filename = k_files[0]["filename"]
            else:
                raise ValueError(
                    "No K-value CSV files found and no default CSV specified in config.json"
                )

    data_dir = cfg.get("NMF_DATA_DIRECTORY", "comps")
    csv_path = Path(data_dir) / k_filename

    # print(f"Loading data from: {csv_path}") # Removed debug print

    df = pd.read_csv(csv_path)

    sample_id_col = cfg.get("SAMPLE_ID_COLUMN", "sample_id")

    comp_cols = [
        col
        for col in df.columns
        if col != sample_id_col and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not comp_cols:
        raise ValueError(f"No numeric component columns found in {csv_path}")

    sample_ids = df[sample_id_col].tolist()
    H = df[comp_cols].values

    cancer_types = [sid[:4] for sid in sample_ids]

    return H, sample_ids, cancer_types
