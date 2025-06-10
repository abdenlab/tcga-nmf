import jscatter
import numpy as np
import pandas as pd
from ipywidgets import VBox, HBox, Dropdown
from .data_utils import load_all_data

def create_main_heatmap(h_sorted, x_labels_short, comp_order, sample_ids=None):
    """Creates the main NMF heatmap using Plotly FigureWidget."""
    from widget.nmf_plotting import create_nmf_heatmap_figure
    from IPython.display import display
    from ipywidgets import Output
    
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
            y_pos = prev_sum + heights/2
        
        df = pd.DataFrame({
            'x': np.arange(n_samples),
            'y': y_pos,
            'height': heights,
            'component': [f"Comp {comp_idx+1}"] * n_samples
        })
        
        scatter = jscatter.Scatter(
            data=df,
            x='x',
            y='y',
            color=[color] * len(df),
            height=150
        )
        
        scatter.options({
            'regl_scatterplot_options': {
                'shape': 'rectangle',
                'size': {'width': 0.9, 'height': 'height'},
                'showLegend': True,
                'xAxis': {'showTickLabels': False},
                'yAxis': {'domain': [0, 1], 'showTickLabels': True}
            }
        })
        
        all_bars.append(scatter)
    
    return all_bars

def create_annotation_strip(data, colormap):
    """Creates an annotation strip (1D heatmap) using jupyter_scatter."""
    # Create a DataFrame with the data
    df = pd.DataFrame({
        'x': np.arange(len(data)),
        'y': np.zeros(len(data)),
        'category': data
    })
    
    scatter = jscatter.Scatter(
        data=df,
        x='x',
        y='y',
        color_by='category',
        color_map=colormap,
        height=30,
        show_legend=False
    )
    
    # Use options instead of x_axis/y_axis
    scatter.options({
        'regl_scatterplot_options': {
            'xAxis': {'showTickLabels': False},
            'yAxis': {'showTickLabels': False}
        }
    })
    
    return scatter

def link_views(plots):
    """Links the pan and zoom of multiple jupyter_scatter plots."""
    # Update this if the linking mechanism has changed too
    for i in range(len(plots) - 1):
        plots[i].options({
            'regl_scatterplot_options': {
                'linkedViews': [plots[i+1]]
            }
        })

def create_nmf_scatter_visualization(cfg_path="config.json", sort_method="component"):
    """Creates the complete NMF visualization using both Plotly and jupyter_scatter."""
    from widget.nmf_plotting import create_nmf_heatmap_figure
    from ipywidgets import Output
    from IPython.display import display
    
    # Load all data as before
    (h_matrix, sample_ids, cancer_types, comp_colors, cancer_color_map,
     organ_systems, organ_system_colors, embryonic_layers, 
     embryonic_layer_colors, h_sorted, x_labels_short, comp_order) = load_all_data(cfg_path, sort_method)
    
    # Create plotly heatmap - wrap in Output widget
    output = Output()
    with output:
        plotly_fig = create_nmf_heatmap_figure(cfg_path, sort_method)
        display(plotly_fig)
    
    # Create bar charts and annotation strips as before
    bar_charts = create_bar_chart(h_sorted, comp_colors, comp_order)
    dominant_comp_indices = np.argmax(h_sorted, axis=1)
    dominant_comps = [f"Comp_{comp_order[idx]}" for idx in dominant_comp_indices]
    comp_strip_color_map = {comp_name: comp_colors[comp_name] 
                           for comp_name in comp_colors 
                           if comp_name in dominant_comps}
    comp_strip = create_annotation_strip(dominant_comps, comp_strip_color_map)
    valid_cancer_types = [ct for ct in cancer_types if ct in cancer_color_map]
    cancer_strip = create_annotation_strip(valid_cancer_types, cancer_color_map)
    
    # Combine with plotly figure (in output widget) first, then other charts
    widget_plots = [output] + [plot.widget for plot in bar_charts + [comp_strip, cancer_strip]]
    
    return VBox(widget_plots)