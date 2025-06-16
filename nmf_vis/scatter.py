import jscatter
import numpy as np
import pandas as pd
import seaborn as sns

from nmf_vis.data_utils import load_all_data


def create_umap_visualization(
    umap_df, h_matrix, sample_ids, cancer_types, cancer_color_map
):
    """Creates a UMAP visualization of the NMF components using jscatter."""
    # Ensure all cancer types have colors
    unique_cancer_types = sorted(list(set(cancer_types)))
    if not all(ct in cancer_color_map for ct in unique_cancer_types):
        color_palette = sns.color_palette("turbo", len(unique_cancer_types))
        # Convert RGB tuples to hex colors
        color_palette = [
            f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
            for r, g, b in color_palette
        ]
        missing_types = [ct for ct in unique_cancer_types if ct not in cancer_color_map]
        for i, ct in enumerate(missing_types):
            cancer_color_map[ct] = color_palette[i % len(color_palette)]

    # Create the scatter plot with lasso selection enabled
    scatter_plot = jscatter.Scatter(
        data=umap_df,
        x="UMAP-1",
        y="UMAP-2",
        color_by="Cancer Type",
        color_map=cancer_color_map,
        height=600,
        width=600,
        lasso_callback=True,  # This is important for selection to work
        selection_mode="lasso",  # Explicitly set lasso mode
    )

    scatter_plot.tooltip(enable=True, properties=["Sample ID", "Cancer Type"])
    scatter_plot.size(default=5)
    scatter_plot.options(
        {
            "aspectRatio": 1.0,
            "regl_scatterplot_options": {
                "showLegend": True,
                "xAxis": {"showGrid": True},
                "yAxis": {"showGrid": True},
                "title": "UMAP of NMF Components",
            },
        }
    )

    # Make sure to add sample_ids to the DataFrame
    umap_df["Sample ID"] = sample_ids

    return scatter_plot, umap_df


def create_scatterplot(cfg_path="conf/config.json", sort_method="component"):
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
