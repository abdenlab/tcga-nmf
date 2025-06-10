import json
import numpy as np
import pandas as pd
from pathlib import Path
from helpers.data_loader import get_prepared_data
from helpers.sort_utils import get_sample_order
from helpers.color_utils import load_cancer_colors, component_palette

def _load_component_colors(path, n_comps, comp_order):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {f"Comp_{i}": color for i, color in zip(comp_order, component_palette(n_comps))}

def load_all_data(cfg_path, sort_method):
    """Loads and prepares all data needed for the visualization."""
    H, sample_ids, cancer_types = get_prepared_data(cfg_path)
    n_samples, n_comps = H.shape

    comp_order = np.argsort(-H.sum(axis=0))
    H_ord = H[:, comp_order]
    
    samp_order = get_sample_order(sort_method, H, sample_ids, cancer_types, cfg_path)
    
    H_sorted = H_ord[samp_order]
    x_labels_short = np.array([label[:4] for label in np.array(sample_ids)[samp_order]])
    
    with open("config.json", 'r') as f:
        cfg = json.load(f)
        
    comp_colors = _load_component_colors(
        cfg.get("JSON_FILENAME_COMPONENT_COLORS", "nmf_component_color_map.json"),
        n_comps,
        comp_order
    )
    cancer_color_map = load_cancer_colors(cfg.get("JSON_FILENAME_CANCER_TYPE_COLORS"))
    
    # These would be loaded similarly from the other JSON files
    organ_systems = [] 
    organ_system_colors = {}
    embryonic_layers = []
    embryonic_layer_colors = {}
    
    return (
        H, sample_ids, cancer_types, comp_colors, cancer_color_map,
        organ_systems, organ_system_colors, embryonic_layers, embryonic_layer_colors,
        H_sorted, x_labels_short, comp_order
    )