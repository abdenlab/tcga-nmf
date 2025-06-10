"""
NMF Heatmap plotting utilities - handles all Plotly figure creation and styling.
"""
from __future__ import annotations
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import numpy as np
import json
from helpers.data_loader import get_prepared_data, load_cfg 
from helpers.sort_utils import get_sample_order 
from helpers.color_utils import (
    component_palette,
    distinct_palette,
    load_cancer_colors
) 


def create_nmf_heatmap_figure(cfg_path: str | Path = "config.json", 
                             sort_method: str = "component",
                             k_filename: str = None, 
                             # FIX: Added the missing parameters to the function signature
                             all_sample_ids: list[str] = None, 
                             current_heatmap_samp_order_indices: np.ndarray = None, 
                             selected_sample_ids: list[str] = None) -> go.Figure: 
    """
    Create a complete NMF heatmap figure with component and cancer type strips.
    
    Parameters
    ----------
    cfg_path : str or Path
        Path to configuration file
    sort_method : str
        Sorting method to use: "component", "alphabetical", "cancer_type", or "organ_system"
    k_filename : str, optional
        Specific K-value CSV filename to load. If None, uses the default from config.
    all_sample_ids : list[str]
        List of all sample IDs in their original, unsorted order from the loaded K-file. Used for highlighting.
    current_heatmap_samp_order_indices : np.ndarray
        NumPy array of indices indicating the current sort order of samples on the heatmap.
    selected_sample_ids : list[str], optional
        List of sample IDs to highlight on the heatmap.
        
    Returns
    -------
    go.Figure
        Plotly figure with NMF heatmap and annotation strips
    """
    # --- load data ------------------------------------------------------ #
    H, sample_ids_from_file, cancer_types = get_prepared_data(cfg_path, k_filename=k_filename) 
    n_samples, n_comps = H.shape

    # --- ordering ------------------------------------------------------- #
    comp_order = np.argsort(-H.sum(axis=0))
    H_ord = H[:, comp_order]
    
    samp_order_indices = get_sample_order(sort_method, H, sample_ids_from_file, cancer_types, cfg_path) 
    
    H_sorted = H_ord[samp_order_indices]
    x_labels = np.array(sample_ids_from_file)[samp_order_indices]
    
    x_labels_short = np.array([label[:4] for label in x_labels])
    
    # --- color preparation ---------------------------------------------- #
    cfg = load_cfg(cfg_path) 
    
    component_color_file = cfg.get("JSON_FILENAME_COMPONENT_COLORS", "nmf_component_color_map.json") 
    print(f"Loading component colors from: {component_color_file}")
    
    comp_colors = _load_component_colors(component_color_file, n_comps, comp_order)
    
    user_cancer_colors = load_cancer_colors(cfg.get("JSON_FILENAME_CANCER_TYPE_COLORS")) 
    uniq_cancers = sorted(set(cancer_types))
    auto_cancer_palette = distinct_palette(len(uniq_cancers)) 
    cancer_color_map = {
        ct: user_cancer_colors.get(ct, auto_cancer_palette[i])
        for i, ct in enumerate(uniq_cancers)
    }
    
    organ_system_file = cfg.get("JSON_FILENAME_ORGAN_SYSTEM", "tissue_source_tcga.json") 
    organ_system_data = _load_organ_system_data(organ_system_file)

    embryonic_layer_file = cfg.get("JSON_FILENAME_EMBRYONIC_LAYER", "emb.json") 
    embryonic_layer_data = _load_organ_system_data(embryonic_layer_file)

    cancer_codes = x_labels_short
    organ_systems, organ_system_colors = _map_cancer_codes_to_organ_systems(cancer_codes, organ_system_data)
    embryonic_layers, embryonic_layer_colors = _map_cancer_codes_to_organ_systems(cancer_codes, embryonic_layer_data)

    # --- create figure -------------------------------------------------- #
    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=True,
        row_heights=[0.45, 0.20, 0.06, 0.06, 0.06, 0.06],
        vertical_spacing=0.04,
        subplot_titles=("NMF Component Activities", "Proportional NMF Activity", "Dominant Component", 
                        "Cancer Type", "Organ System", "Embryonic Layer")
    )

    # 1) Main heatmap - This is the first trace, used for selection
    fig.add_trace( 
        go.Heatmap(
            z=H_sorted.T,
            colorscale="Turbo",
            colorbar=dict(title="Activity", x=1.02),
            showscale=False,
            hovertemplate="Sample: %{x}<br>Component: %{y}<br>Activity: %{z}<extra></extra>",
        ),
        row=1, col=1
    )
    
    # 2) Stacked bar chart showing proportional NMF activity
    _add_proportional_bar_chart(fig, H_sorted, comp_colors, comp_order)
    
    # 3) Component strip with legend
    _add_component_strip_with_legend(fig, H_ord, samp_order_indices, comp_colors, n_comps, comp_order)
    
    # 4) Cancer type strip with legend
    _add_cancer_strip_with_legend(fig, cancer_types, samp_order_indices, uniq_cancers, cancer_color_map)
    
    # 5) Organ system strip with legend
    _add_annotation_strip_with_legend(fig, organ_systems, organ_system_colors, 
                                      "Organ System", row=5, col=1)
    
    # 6) Embryonic layer strip with legend
    _add_annotation_strip_with_legend(fig, embryonic_layers, embryonic_layer_colors, 
                                      "Embryonic Layer", row=6, col=1)

    # NEW: Add highlighting layer if selected_sample_ids are provided
    if selected_sample_ids:
        _add_highlight_layer(fig, all_sample_ids, current_heatmap_samp_order_indices, selected_sample_ids, n_comps)
    
    # 7) Layout and styling
    _configure_layout(fig, n_comps, n_samples, comp_order, x_labels, x_labels_short, 
                      n_annotation_strips=2)
    
    return fig


def _load_organ_system_data(json_filename: str) -> dict:
    """Load organ system or embryonic layer groupings from JSON file."""
    try:
        with open(json_filename, 'r') as f:
            data = json.load(f)
        return data["organ_system_groupings"]
    except Exception as e:
        print(f"Error loading grouping data from {json_filename}: {e}")
        return []


def _map_cancer_codes_to_organ_systems(cancer_codes, grouping_data):
    """Map cancer codes to their groups and colors."""
    code_to_group = {}
    code_to_color = {}
    
    for group in grouping_data:
        group_name = group["group_name"]
        color = group["color"]
        for code in group["cancer_codes"]:
            short_code = code[:4]
            code_to_group[short_code] = group_name
            code_to_color[short_code] = color
    
    groups = []
    group_colors = []
    
    for code in cancer_codes:
        groups.append(code_to_group.get(code, "Unknown"))
        group_colors.append(code_to_color.get(code, "#CCCCCC"))  # Gray for unknown
    
    return groups, group_colors


def _load_component_colors(json_filename: str, n_comps: int, comp_order: np.ndarray) -> list:
    """Load component colors from JSON file or generate fallback colors."""
    try:
        if json_filename:
            with open(json_filename, 'r') as f:
                color_map = json.load(f)
                
            print(f"Loaded component colors: {color_map}")
            ordered_colors = []
            auto_colors = component_palette(n_comps) 
            
            component_color_mapping = {}
            
            for i in comp_order:
                comp_name = f"Comp_{i}"
                alt_name1 = f"Component {i+1}"
                alt_name2 = f"Comp {i+1}"
                alt_name3 = str(i+1)
                
                color = (color_map.get(comp_name) or 
                         color_map.get(alt_name1) or 
                         color_map.get(alt_name2) or 
                         color_map.get(alt_name3))
                
                if color:
                    ordered_colors.append(color)
                    matched_key = next(k for k in [comp_name, alt_name1, alt_name2, alt_name3] 
                                      if k in color_map and color_map[k] == color)
                    component_color_mapping[f"Component {i+1}"] = {'color': color, 'matched_key': matched_key}
                else:
                    print(f"Warning: No color found for component {i}, using fallback")
                    fallback_color = auto_colors[len(ordered_colors) % len(auto_colors)]
                    ordered_colors.append(fallback_color)
                    component_color_mapping[f"Component {i+1}"] = {'color': fallback_color, 'fallback': True}
            
            print(f"Component color mapping: {component_color_mapping}")
            return ordered_colors
    except Exception as e:
        print(f"Error loading component colors: {e}. Using default palette.")
        import traceback
        traceback.print_exc()
    
    return component_palette(n_comps) 


def _add_main_heatmap(fig: go.Figure, H_sorted: np.ndarray, _: np.ndarray) -> None:
    pass 


def _add_proportional_bar_chart(fig: go.Figure, H_sorted: np.ndarray, comp_colors: list, comp_order: np.ndarray) -> None:
    _, _ = H_sorted.shape  
    
    H_proportional = H_sorted / H_sorted.sum(axis=1, keepdims=True)
    
    for i, comp_idx in enumerate(comp_order):
        color = comp_colors[i]
        
        comp_name = f"Comp {comp_idx+1}"
        hover_template = f"Sample: %{{x}}<br>Component: {comp_name}<br>Proportion: %{{y:.2f}}<extra></extra>"
        
        fig.add_trace(
            go.Bar(
                y=H_proportional[:, i],
                name=comp_name,
                marker_color=color,
                hovertemplate=hover_template
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        barmode='stack',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_yaxes(
        title="Proportion", 
        range=[0, 1],
        row=2, col=1
    )


def _add_component_strip_with_legend(fig: go.Figure, H_ord: np.ndarray, samp_order: np.ndarray, 
                                    comp_colors: list, n_comps: int, comp_order: np.ndarray) -> None:
    """Add component strip with interactive legend."""
    winning_comp_indices = np.argmax(H_ord[samp_order], axis=1)
    winning_comp_numbers = np.array([comp_order[i] + 1 for i in winning_comp_indices])
    
    idx_to_color = {i: comp_colors[i] for i in range(n_comps)}
    
    comp_scale = []
    for i in range(n_comps):
        if i == 0:
            comp_scale.append((0, idx_to_color[i]))
        else:
            comp_scale.append(((i-0.5)/(n_comps-1), idx_to_color[i-1]))
            comp_scale.append(((i)/(n_comps-1), idx_to_color[i]))
    
    if n_comps > 1:
        comp_scale.append((1, idx_to_color[n_comps-1]))
    
    fig.add_trace(
        go.Heatmap(
            z=[winning_comp_indices], 
            colorscale=comp_scale,
            showscale=False,
            hovertemplate="Sample: %{x}<br>Dominant Component: Comp %{customdata}<extra></extra>",
            customdata=[winning_comp_numbers],
            showlegend=False
        ), 
        row=3, col=1
    )
    
    for i, comp_idx in enumerate(comp_order):
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=10, color=comp_colors[i], symbol="square"),
                name=f"Comp {comp_idx+1}",
                legendgroup="Components",
                showlegend=True
            )
        )


def _add_cancer_strip_with_legend(fig: go.Figure, cancer_types: list, samp_order: np.ndarray,
                                  uniq_cancers: list, cancer_color_map: dict) -> None:
    """Add cancer type strip with interactive legend."""
    cancer_to_idx = {ct: i for i, ct in enumerate(uniq_cancers)}
    cancer_idx_arr = [cancer_to_idx[ct] for ct in np.array(cancer_types)[samp_order]]
    
    if len(uniq_cancers) == 1:
        cancer_scale = [(0, cancer_color_map[uniq_cancers[0]]), (1, cancer_color_map[uniq_cancers[0]])]
    else:
        cancer_scale = [(i/(len(uniq_cancers)-1), cancer_color_map[ct]) for i, ct in enumerate(uniq_cancers)]

    fig.add_trace(
        go.Heatmap(
            z=[cancer_idx_arr], 
            colorscale=cancer_scale,
            showscale=False,
            hovertemplate="Sample: %{x}<br>Cancer Type: %{customdata}<extra></extra>",
            customdata=[np.array(cancer_types)[samp_order]],
            showlegend=False
        ), 
        row=4, col=1
    )
    
    for cancer in uniq_cancers:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=10, color=cancer_color_map[cancer], symbol="square"),
                name=cancer,
                legendgroup="Cancer Types",
                showlegend=True
            )
        )


def _add_annotation_strip_with_legend(fig: go.Figure, group_names: list, group_colors: list, 
                                      label: str, row: int, col: int) -> None:
    """Add annotation strip (organ system/embryonic layer) with interactive legend."""
    unique_groups = sorted(list(set(group_names)))
    group_to_idx = {group: i for i, group in enumerate(unique_groups)}
    
    group_idx_arr = [group_to_idx[group] for group in group_names]
    
    unique_colors = {}
    for group_name, color in zip(group_names, group_colors):
        if group_name not in unique_colors:
            unique_colors[group_name] = color
            
    n_groups = len(unique_groups)
    if n_groups == 1:
        group_scale = [(0, unique_colors[unique_groups[0]]), (1, unique_colors[unique_groups[0]])]
    else:
        group_scale = [(i/(n_groups-1), unique_colors[ug]) for i, ug in enumerate(unique_groups)]
    
    fig.add_trace(
        go.Heatmap(
            z=[group_idx_arr], 
            colorscale=group_scale,
            showscale=False,
            hovertemplate=f"Sample: %{{x}}<br>{label}: %{{customdata}}<extra></extra>",
            customdata=[group_names],
            showlegend=False
        ), 
        row=row, col=col
    )
    
    for group in unique_groups:
        color = unique_colors.get(group, "#CCCCCC") 
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=10, color=color, symbol="square"),
                name=group,
                legendgroup=label,
                showlegend=True
            )
        )


def _add_highlight_layer(fig: go.Figure, all_sample_ids_original_order: list[str], 
                         current_heatmap_samp_order_indices: np.ndarray, 
                         selected_sample_ids: list[str], n_comps: int) -> None:
    """
    Adds a translucent highlight layer over selected samples on the main heatmap.
    """
    if not selected_sample_ids:
        return

    is_selected_in_original_order = np.array([sid in selected_sample_ids for sid in all_sample_ids_original_order])
    
    is_selected_on_heatmap = is_selected_in_original_order[current_heatmap_samp_order_indices]

    highlight_mask = np.zeros((n_comps, len(current_heatmap_samp_order_indices)))
    for i, is_selected in enumerate(is_selected_on_heatmap):
        if is_selected:
            highlight_mask[:, i] = 1 

    fig.add_trace(
        go.Heatmap(
            z=highlight_mask,
            colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(255,255,0,0.3)']], 
            showscale=False,
            hoverinfo='skip'
        ),
        row=1, col=1
    )


def _configure_layout(fig: go.Figure, n_comps: int, n_samples: int, 
                      comp_order: np.ndarray, _: np.ndarray,
                      x_labels_short: np.ndarray, n_annotation_strips: int = 2) -> None:
    """Configure axes, layout, and styling with horizontal legends above the heatmap."""
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(n_comps)),
        ticktext=[f"Comp {i+1}" for i in comp_order],
        row=1, col=1
    )
    
    total_rows = n_annotation_strips + 4
    for r in range(1, total_rows):
        fig.update_xaxes(showticklabels=False, row=r, col=1)
    
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(n_samples)),
        ticktext=x_labels_short,
        tickangle=90,
        row=total_rows, col=1
    )
    
    for r in range(2, total_rows + 1):
        fig.update_yaxes(showticklabels=False, row=r, col=1)
    
    legend_groups_info = {
        "Components": {"y": 1.22, "x": 0.0,  "title": "NMF Components"},
        "Cancer Types": {"y": 1.22, "x": 0.22, "title": "Cancer Types"},
        "Organ System": {"y": 1.22, "x": 0.55, "title": "Organ System"},
        "Embryonic Layer": {"y": 1.22, "x": 0.8, "title": "Embryonic Layer"}
    }
    
    legend_config = {}
    
    for _, trace in enumerate(fig.data):
        if trace.legendgroup in legend_groups_info:
            group_name = trace.legendgroup
            legend_id = f"legend{list(legend_groups_info.keys()).index(group_name) + 1}"
            trace.legend = legend_id
            
            if legend_id not in legend_config:
                info = legend_groups_info[group_name]
                legend_config[legend_id] = dict(
                    yanchor="top",
                    y=info["y"],
                    xanchor="left",
                    x=info["x"],
                    orientation="h",
                    font=dict(size=9),
                    title=dict(text=info["title"], font=dict(size=11)),
                    bgcolor="rgba(255,255,255,0)",
                )

    fig.update_layout(
        height=max(900, n_comps * 25 + 300),
        width=1400,
        autosize=True,  
        margin=dict(l=80, r=80, t=200, b=100), 
        title=dict(
            text="NMF Component Activities Across Samples",
            y=0.99,
            x=0.5,
            xanchor="center",
            yanchor="top"
        ),
        barmode="stack",
        hovermode="x unified",
        selectdirection="h",
        dragmode="select", 
        **legend_config  
    )


def create_empty_placeholder_figure(message="Error loading NMF data") -> go.Figure:
    """Create an empty placeholder figure for error cases."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, 
        xref="paper", yref="paper",
        x=0.5, y=0.5, xanchor='center', yanchor='middle',
        showarrow=False,
        font=dict(size=16, color="red")
    )
    fig.update_layout(
        title="NMF Heatmap Widget - Error",
        height=400,
        margin=dict(l=60, r=60, t=40, b=40)
    )
    return fig