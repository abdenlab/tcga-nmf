import jscatter
import numpy as np
import pandas as pd

from nmf_vis_app.umap_utils import create_umap_visualization
from nmf_vis_app.data_utils import load_all_data


def create_bar_chart(h_sorted, comp_colors, comp_order):
    """Creates stacked bar charts colored by component."""
    n_samples, n_comps = h_sorted.shape
    h_proportional = h_sorted / h_sorted.sum(axis=1, keepdims=True)

    all_bars = []

    # Create a separate scatter for each component (for proper coloring)
    for i in range(n_comps):
        comp_idx = comp_order[i]

        # Access color using the key format from the JSON
        color_key = f"Comp_{comp_idx}"

        # Use the color from comp_colors dictionary
        if color_key in comp_colors:
            color = comp_colors[color_key]
        else:
            # Log missing color but use a default
            # print(f"Warning: No color found for {color_key}")
            color = "#000000"  # Default to black if missing

        # Calculate position for stacked bars
        if i == 0:
            y_pos = h_proportional[:, i] / 2
            heights = h_proportional[:, i]
        else:
            prev_sum = np.sum(h_proportional[:, :i], axis=1)
            heights = h_proportional[:, i]
            y_pos = prev_sum + heights / 2

        df = pd.DataFrame(
            {
                "x": np.arange(n_samples),
                "y": y_pos,
                "height": heights,
                "component": [f"Comp {comp_idx + 1}"] * n_samples,
            }
        )

        scatter = jscatter.Scatter(
            data=df, x="x", y="y", color=[color] * len(df), height=150
        )

        scatter.options(
            {
                "regl_scatterplot_options": {
                    "shape": "rectangle",
                    "size": {"width": 0.9, "height": "height"},
                    "showLegend": True,
                    "xAxis": {"showTickLabels": False},
                    "yAxis": {"domain": [0, 1], "showTickLabels": True},
                }
            }
        )

        all_bars.append(scatter)

    return all_bars


def create_annotation_strip(data, colormap):
    """Creates an annotation strip (1D heatmap) using jupyter_scatter."""
    # Create a DataFrame with the data
    df = pd.DataFrame(
        {"x": np.arange(len(data)), "y": np.zeros(len(data)), "category": data}
    )

    scatter = jscatter.Scatter(
        data=df,
        x="x",
        y="y",
        color_by="category",
        color_map=colormap,
        height=30,
        show_legend=False,
    )

    # Use options instead of x_axis/y_axis
    scatter.options(
        {
            "regl_scatterplot_options": {
                "xAxis": {"showTickLabels": False},
                "yAxis": {"showTickLabels": False},
            }
        }
    )

    return scatter


def link_views(plots):
    """Links the pan and zoom of multiple jupyter_scatter plots."""
    # Update this if the linking mechanism has changed too
    for i in range(len(plots) - 1):
        plots[i].options({"regl_scatterplot_options": {"linkedViews": [plots[i + 1]]}})


def create_scatterplot(cfg_path="config.json", sort_method="component"):
    """Creates the complete NMF visualization using Plotly and UMAP."""

    (
        h_matrix,
        sample_ids,
        cancer_types,
        comp_colors,
        cancer_color_map,
        organ_systems,
        organ_system_colors,
        embryonic_layers,
        embryonic_layer_colors,
        h_sorted,
        x_labels_short,
        comp_order,
        umap_df,
    ) = load_all_data(cfg_path, sort_method)

    # Create UMAP visualization with lasso selection enabled
    umap_scatter, umap_df = create_umap_visualization(
        umap_df, h_matrix, sample_ids, cancer_types, cancer_color_map
    )

    return umap_scatter
