from __future__ import annotations
import json, numpy as np, pandas as pd, polars as pl
from pathlib import Path
from typing import Tuple, List

# ------------------------------------------------------------------ #
def load_cfg(path: str | Path = "config.json") -> dict:
    return json.loads(Path(path).read_text())

# ------------------------------------------------------------------ #
def _load_from_csv(csv_path: str | Path) -> Tuple[np.ndarray, List[str]]:
    df = pd.read_csv(csv_path, sep=None, engine="python")   # auto-detect delimiter
    sample_ids = df.iloc[:, 0].astype(str).tolist()
    H          = df.iloc[:, 1:].to_numpy(float)
    return H, sample_ids

# ------------------------------------------------------------------ #
def _load_from_npy(h_path: str | Path,
                   parquet_path: str | Path) -> Tuple[np.ndarray, List[str]]:
    H = np.load(h_path)
    ids = sorted(pl.scan_parquet(parquet_path).collect_schema().names()[6:])
    if H.shape[0] != len(ids):
        raise ValueError("Sample count mismatch between H and parquet IDs.")
    return H, ids

# ------------------------------------------------------------------ #
def get_prepared_data(cfg_path: str | Path = "config.json"):
    cfg = load_cfg(cfg_path)

    if csv_path := cfg.get("CSV_H_MATRIX_PATH"):
        H, sample_ids = _load_from_csv(csv_path)
    else:
        H, sample_ids = _load_from_npy(cfg["NMF_H_MATRIX_PATH"],
                                       cfg["TCGA_ZSCORES_PATH"])

    cancer_types = [sid.split("-")[0] for sid in sample_ids]
    return H, sample_ids, cancer_types
