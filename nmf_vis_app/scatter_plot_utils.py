import jscatter
import numpy as np
import pandas as pd
from ipywidgets import Output, VBox, HBox
from IPython.display import clear_output, display
import plotly.graph_objects as go
from plotly.graph_objs import FigureWidget  # Add this import
from .umap_utils import create_umap_visualization
from .data_utils import load_all_data


def create_main_heatmap(h_sorted, x_labels_short, comp_order, sample_ids=None):
    """Creates the main NMF heatmap using Plotly FigureWidget."""
    from widget.nmf_plotting import create_nmf_heatmap_figure

    # Create output widget
    output = Output()

    with output:
        # Get the Plotly figure from the existing implementation
        fig = create_nmf_heatmap_figure("config.json", "component")
        display(fig)

    return output


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
            print(f"Warning: No color found for {color_key}")
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


def create_nmf_scatter_visualization(cfg_path="config.json", sort_method="component"):
    """Creates the complete NMF visualization using Plotly and UMAP."""
    from widget.nmf_plotting import create_nmf_heatmap_figure

    # Load all data as before
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
    ) = load_all_data(cfg_path, sort_method)

    # Create plotly heatmap - wrap in Output widget
    heatmap_output = Output()

    with heatmap_output:
        # Convert the Figure to a FigureWidget to enable selection events
        plotly_fig = FigureWidget(create_nmf_heatmap_figure(cfg_path, sort_method))

        # Enhance heatmap selection responsiveness
        plotly_fig.update_layout(
            dragmode="select",  # Enable box selection by default
            selectdirection="h",  # Horizontal selection is most useful for samples
            hovermode="closest",  # More precise hover information
            uirevision="same",  # Preserve UI state between updates
        )

        # Make heatmap more responsive to selections
        for i, trace in enumerate(plotly_fig.data):
            if trace.type == "heatmap":
                # Skip selection properties for heatmaps
                pass
            elif trace.type == "scatter":
                # Full selection styling for scatter plots
                plotly_fig.data[i].update(
                    selectedpoints=True,
                    selected=dict(marker=dict(opacity=1.0, size=10)),
                    unselected=dict(marker=dict(opacity=0.3)),
                )
            elif trace.type == "bar":
                # Bar charts only support color and opacity in selection
                plotly_fig.data[i].update(
                    selectedpoints=True,
                    selected=dict(marker=dict(opacity=1.0)),
                    unselected=dict(marker=dict(opacity=0.3)),
                )

        # Add selection callback for responsiveness
        plotly_fig.update_traces(
            customdata=sample_ids,
            hovertemplate="Sample: %{customdata}<br>Component: %{y}<br>Value: %{z:.3f}<extra></extra>",
        )

        display(plotly_fig)

    # Create UMAP visualization with lasso selection enabled
    umap_scatter, umap_df = create_umap_visualization(
        h_matrix, sample_ids, cancer_types, cancer_color_map
    )

    # Enable and configure lasso selection
    umap_scatter.options(
        {
            "regl_scatterplot_options": {
                "selection": {
                    "type": "lasso",
                    "enable": True,
                    "width": 2,
                    "color": "rgba(50, 50, 50, 0.8)",
                    "brushFilter": True,
                }
            }
        }
    )

    umap_scatter.tooltip(
        properties=["Sample ID", "Cancer Type", "Emb Origin", "Organ System"]
    ).size(default=5).options(
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

    # Create data table output for showing selection details
    data_table_output = Output()
    with data_table_output:
        print("Select points on either visualization to see details")

    # Global variables for selection state
    selected_sample_ids = []
    original_umap_df = umap_df.copy()

    # Update function for synchronizing selection between visualizations
    def update_selection(source, new_selection):
        nonlocal selected_sample_ids

        # Skip if selection hasn't changed
        if sorted(new_selection) == sorted(selected_sample_ids):
            return

        selected_sample_ids = new_selection

        # Update data table
        with data_table_output:
            clear_output(wait=True)
            if selected_sample_ids:
                selected_df = original_umap_df[
                    original_umap_df["Sample ID"].isin(selected_sample_ids)
                ]
                display(
                    selected_df[["Sample ID", "Cancer Type"]].style.hide(axis="index")
                )
            else:
                print("No samples selected")

        # Update UMAP if selection comes from heatmap
        if source == "heatmap":
            if selected_sample_ids:
                # Filter UMAP to show only selected samples
                filtered_df = original_umap_df[
                    original_umap_df["Sample ID"].isin(selected_sample_ids)
                ]
                umap_scatter.data = filtered_df

                # Get indices of selected samples to adjust zoom
                selected_indices = [
                    i for i, sid in enumerate(sample_ids) if sid in selected_sample_ids
                ]

                if selected_indices:
                    # Adjust plotly view to focus on selection
                    min_idx = max(0, min(selected_indices) - 3)  # Add padding
                    max_idx = min(len(sample_ids) - 1, max(selected_indices) + 3)

                    # Update x-axis range for all relevant subplots
                    for i in range(1, 7):  # For xaxis1 through xaxis6
                        axis_key = f"xaxis{i}" if i > 1 else "xaxis"
                        plotly_fig.layout[axis_key].range = [min_idx, max_idx]
            else:
                # Reset to all samples
                umap_scatter.data = original_umap_df

                # Reset axes to auto-range
                for i in range(1, 7):
                    axis_key = f"xaxis{i}" if i > 1 else "xaxis"
                    if axis_key in plotly_fig.layout:
                        plotly_fig.layout[axis_key].autorange = True
        elif source == "umap":
            # When selection comes from UMAP, highlight those points in the heatmap
            if selected_sample_ids:
                # Find indices in the heatmap that match selected sample IDs
                selected_indices = [
                    i for i, sid in enumerate(sample_ids) if sid in selected_sample_ids
                ]

                if selected_indices:
                    # Update selected points in all traces
                    for trace in plotly_fig.data:
                        if hasattr(trace, "selectedpoints"):
                            trace.selectedpoints = selected_indices

                    # Adjust plotly view to focus on selection
                    min_idx = max(0, min(selected_indices) - 3)  # Add padding
                    max_idx = min(len(sample_ids) - 1, max(selected_indices) + 3)

                    # Update x-axis range for all relevant subplots to focus on selection
                    for i in range(1, 7):
                        axis_key = f"xaxis{i}" if i > 1 else "xaxis"
                        if axis_key in plotly_fig.layout:
                            plotly_fig.layout[axis_key].range = [min_idx, max_idx]
                else:
                    # Clear selection if no valid indices
                    for trace in plotly_fig.data:
                        if hasattr(trace, "selectedpoints"):
                            trace.selectedpoints = None
            else:
                # Reset when no samples are selected
                for trace in plotly_fig.data:
                    if hasattr(trace, "selectedpoints"):
                        trace.selectedpoints = None

                # Reset axes to auto-range
                for i in range(1, 7):
                    axis_key = f"xaxis{i}" if i > 1 else "xaxis"
                    if axis_key in plotly_fig.layout:
                        plotly_fig.layout[axis_key].autorange = True

    # Connect selection callbacks
    # For Plotly heatmap
    def on_plotly_selection(trace, points, selector):
        if hasattr(points, "point_inds") and points.point_inds:
            # Get the correct customdata from points
            selection = []
            for ind in points.point_inds:
                # Make sure to handle customdata correctly
                if hasattr(points, "customdata") and points.customdata:
                    if ind < len(points.customdata):
                        selection.append(points.customdata[ind])
                elif hasattr(trace, "customdata") and trace.customdata:
                    if ind < len(trace.customdata):
                        selection.append(trace.customdata[ind])
        else:
            selection = []
        update_selection("heatmap", selection)

    # Apply the callback to each trace that supports selection:
    for trace in plotly_fig.data:
        if hasattr(trace, "on_selection"):
            trace.on_selection(on_plotly_selection)

    # For UMAP scatter
    def on_umap_selection(change):
        selection_indices = change.new
        if selection_indices is not None and len(selection_indices) > 0:
            # Get sample IDs from current scatter data
            try:
                current_df = umap_scatter.data
                # Check if we need to get the DataFrame from a callable
                if callable(current_df):
                    current_df = current_df()

                # Ensure selection indices are within bounds
                valid_indices = [i for i in selection_indices if i < len(current_df)]
                if valid_indices:
                    selected_samples = current_df.iloc[valid_indices][
                        "Sample ID"
                    ].tolist()
                    # Call update_selection with the selected samples
                    update_selection("umap", selected_samples)
                else:
                    update_selection("umap", [])
            except Exception as e:
                print(f"Error processing UMAP selection: {e}")
                update_selection("umap", [])
        else:
            update_selection("umap", [])

    umap_scatter.widget.observe(on_umap_selection, names=["selection"])

    # Create layout with heatmap on top, UMAP and data table side by side below
    main_layout = VBox([heatmap_output, HBox([umap_scatter.widget, data_table_output])])

    return main_layout
