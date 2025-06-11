import umap
import numpy as np
import pandas as pd
import jscatter
import seaborn as sns
import warnings
from ipywidgets import Output
from IPython.display import clear_output, display


def perform_umap_reduction(data, n_components=2, random_state=42):
    """Performs UMAP dimensionality reduction on the input data."""
    umap_result = None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        umap_result = umap.UMAP(
            n_components=n_components, random_state=random_state
        ).fit_transform(data)
        if w and issubclass(w[-1].category, UserWarning):
            print(f"UMAP Warning: {str(w[-1].message)}")

    return umap_result


def create_umap_visualization(h_matrix, sample_ids, cancer_types, cancer_color_map):
    """Creates a UMAP visualization of the NMF components using jscatter."""
    # Perform UMAP dimensionality reduction
    umap_result = perform_umap_reduction(h_matrix)

    # Create DataFrame with UMAP results and metadata
    umap_df = pd.DataFrame(
        {
            "Sample ID": sample_ids,
            "Cancer Type": cancer_types,
            "UMAP-1": umap_result[:, 0],
            "UMAP-2": umap_result[:, 1],
        }
    )

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
        width=800,
        lasso_callback=True,  # This is important for selection to work
        selection_mode="lasso",  # Explicitly set lasso mode
    )

    scatter_plot.tooltip(properties=["Sample ID", "Cancer Type"]).size(
        default=5
    ).options(
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
