"""
NMF Heatmap Widget - anywidget wrapper for interactive NMF visualization.
"""
from __future__ import annotations
import anywidget
import plotly.graph_objects as go
from pathlib import Path
import numpy as np 
import traceback
import ipywidgets as widgets
import json
from traitlets import List, observe 
# FIX: Explicitly import ALL functions from nmf_plotting and helpers modules
# that are directly called within NMFHeatmapWidget methods.
from .nmf_plotting import create_nmf_heatmap_figure, create_empty_placeholder_figure
from helpers.data_loader import discover_nmf_k_files, get_prepared_data, load_cfg 
from helpers.sort_utils import get_sample_order # FIX: Import get_sample_order


# The load_cfg from helpers.data_loader is now directly imported and used.
# No need for a local definition or aliasing unless there's a conflict.


class NMFHeatmapWidget(anywidget.AnyWidget):
    """Plotly + anywidget interactive heat-map of NMF sample activities."""
    
    _esm = """
        function render({ model, el }) {
            // Simple container for the Plotly figure
            el.innerHTML = '<div class="nmf-widget-container"></div>';
        }
        
        export default { render };
    """

    selected_sample_ids_from_heatmap = List([], help="List of sample IDs selected from the NMF heatmap.").tag(sync=True)


    def __init__(self, cfg_path: str | Path = "config.json"):
        super().__init__()
    
        self._cfg_path = cfg_path
        self._current_sort = "component"
        self._current_k_file = None 
        self._highlighted_sample_ids = [] 
        
        self._all_original_sample_ids_for_current_k = [] 
        self._current_heatmap_sample_order_indices = np.array([]) 

        # FIX: Initialize widget_layout and its main components as placeholders AT THE VERY START
        # This prevents AttributeError if errors occur early in __init__
        self.figure = go.FigureWidget(create_empty_placeholder_figure("Initializing...")) 
        self.k_dropdown = widgets.Dropdown(options=[("Loading...", "")], value="", description='NMF K Value:', disabled=True) 
        self.sort_dropdown = widgets.Dropdown(options=[("Loading...", "")], value="", description='Sort by:', disabled=True)
        self.control_box = widgets.HBox([self.k_dropdown, self.sort_dropdown])
        self.widget_layout = widgets.VBox([self.control_box, self.figure]) 


        try:
            cfg = load_cfg(cfg_path) 
            self.k_files = discover_nmf_k_files(cfg_path) 
            
            if not self.k_files:
                default_csv_full_path = cfg.get("DEFAULT_CSV_FILENAME") 
                if default_csv_full_path:
                    self._current_k_file = Path(default_csv_full_path).name 
                else:
                    self.figure = go.FigureWidget(create_empty_placeholder_figure("No NMF data files found in specified directory. Please check config.json and data/comps folder."))
                    self.widget_layout.children = [self.figure] # No controls if no data
                    print("NMFHeatmapWidget: Initialization complete with no data.")
                    return 
            else:
                self._current_k_file = self.k_files[0]["filename"] 
            
            # Initial figure creation - this will also populate _all_original_sample_ids_for_current_k and _current_heatmap_sample_order_indices
            self._recreate_figure() 

            # Update controls with actual options and values after successful data load
            k_options = [(k["display_name"], k["filename"]) for k in self.k_files]
            
            # Ensure the value assigned to dropdown is always one of its options
            if self._current_k_file not in [opt[1] for opt in k_options]:
                # Fallback to the first option if the current default isn't found
                self._current_k_file = k_options[0][1] if k_options else ""

            self.k_dropdown.options = k_options
            self.k_dropdown.value = self._current_k_file
            self.k_dropdown.disabled = False
            self.k_dropdown.observe(self._on_k_change, names='value')
            
            self.sort_dropdown.options = [
                ('Sort by Component', 'component'), 
                ('Sort Alphabetically', 'alphabetical'),
                ('Sort by Cancer Type', 'cancer_type'), 
                ('Sort by Organ System', 'organ_system')
            ]
            self.sort_dropdown.value = self._current_sort
            self.sort_dropdown.disabled = False
            self.sort_dropdown.observe(self._on_sort_change, names='value')

            self.control_box.children = [self.k_dropdown, self.sort_dropdown] 
            self.widget_layout.children = [self.control_box, self.figure] 
            
        except Exception as e:
            print(f"Error initializing NMFHeatmapWidget: {e}")
            traceback.print_exc()
            self.figure = go.FigureWidget(create_empty_placeholder_figure(f"Initialization Error: {e}"))
            # Ensure widget_layout displays the error figure even if controls failed
            self.widget_layout.children = [self.figure] 
            
        print("NMFHeatmapWidget: Initialization complete.")


    def _create_sort_controls(self):
        # This function is mostly for initial creation; options are updated in __init__ now.
        return widgets.Dropdown(
            options=[("Loading...", "")], value="", description='Sort by:', disabled=True,
            style={'description_width': 'initial'}, layout=widgets.Layout(width='200px')
        )
    
    def _on_k_change(self, change):
        new_k_file = change['new']
        if new_k_file != self._current_k_file:
            self._current_k_file = new_k_file
            k_value_display = next((k_file["k_value"] for k_file in self.k_files if k_file["filename"] == new_k_file), "Unknown")
            print(f"NMFHeatmapWidget: Changed to K={k_value_display} ({new_k_file})")
            self.selected_sample_ids_from_heatmap = []
            self._highlighted_sample_ids = []
            self._recreate_figure()

    def _on_sort_change(self, change):
        new_sort = change['new']
        if new_sort != self._current_sort:
            self._current_sort = new_sort
            print(f"NMFHeatmapWidget: Changed sort method to '{new_sort}'.")
            self._recreate_figure()

    def _recreate_figure(self):
        """Internal helper to recreate and update the figure with current settings."""
        try:
            H_matrix, sample_ids_for_current_k, cancer_types = get_prepared_data(self._cfg_path, k_filename=self._current_k_file) 
            
            self._all_original_sample_ids_for_current_k = sample_ids_for_current_k
            
            current_samp_order_indices = get_sample_order(
                self._current_sort, H_matrix, sample_ids_for_current_k, cancer_types, self._cfg_path 
            )
            self._current_heatmap_sample_order_indices = current_samp_order_indices

            new_fig = create_nmf_heatmap_figure(
                cfg_path=self._cfg_path, 
                sort_method=self._current_sort, 
                k_filename=self._current_k_file,
                all_sample_ids=self._all_original_sample_ids_for_current_k, 
                current_heatmap_samp_order_indices=self._current_heatmap_sample_order_indices, 
                selected_sample_ids=self._highlighted_sample_ids 
            )
            
            # FIX: Clear existing traces and then add new ones
            with self.figure.batch_update():
                self.figure.data = [] # Clear all existing traces
                self.figure.add_traces(new_fig.data) # Add all new traces
                self.figure.layout = new_fig.layout # Update layout
            
            # Re-attach selection handler to the main heatmap trace (it's the first trace)
            if len(self.figure.data) > 0:
                main_heatmap_trace = self.figure.data[0] 
                # Ensure only one selection handler is active by setting append=False
                main_heatmap_trace.on_selection(self._on_plotly_selection, append=False) 

        except Exception as e:
            print(f"Error recreating NMF heatmap figure: {e}")
            traceback.print_exc()
            self.figure = go.FigureWidget(create_empty_placeholder_figure(f"Figure Creation Error: {e}"))
            # Ensure the layout shows the control box and the error figure
            if hasattr(self, 'widget_layout') and hasattr(self, 'control_box'):
                self.widget_layout.children = [self.control_box, self.figure]
            else: # Fallback if even control_box/widget_layout failed initialization
                self.widget_layout = widgets.VBox([self.figure])


    def _on_plotly_selection(self, trace, points, state):
        """
        Internal handler for selections made directly on the Plotly heatmap.
        This extracts selected sample IDs and updates the traitlet.
        """
        selected_ids_from_heatmap_view = []
        if points and points.point_inds: 
            selected_x_indices_on_heatmap = np.unique([p['x'] for p in points.points]) 
            
            original_indices_selected = self._current_heatmap_sample_order_indices[selected_x_indices_on_heatmap]
            
            selected_ids_from_heatmap_view = np.array(self._all_original_sample_ids_for_current_k)[original_indices_selected].tolist()
            
        if sorted(selected_ids_from_heatmap_view) != sorted(self.selected_sample_ids_from_heatmap):
            self.selected_sample_ids_from_heatmap = selected_ids_from_heatmap_view
            if selected_ids_from_heatmap_view:
                print(f"NMFHeatmapWidget: Selection made on heatmap: {len(selected_ids_from_heatmap_view)} samples.")
            else:
                print("NMFHeatmapWidget: Heatmap selection cleared.")


    def update_highlight_from_external(self, sample_ids_to_highlight: list[str]):
        """
        Public method to update the heatmap highlighting based on external selection (e.g., UMAP).
        This updates the _highlighted_sample_ids and triggers a figure re-render.
        """
        if sorted(self._highlighted_sample_ids) != sorted(sample_ids_to_highlight):
            print(f"NMFHeatmapWidget: Updating highlight for {len(sample_ids_to_highlight)} samples from external source.")
            self._highlighted_sample_ids = sample_ids_to_highlight
            self._recreate_figure() 


    @observe('selected_sample_ids_from_heatmap')
    def _selected_sample_ids_from_heatmap_changed(self, change):
        pass

    def _repr_mimebundle_(self, **kwargs):
        """
        This method is called by Jupyter to display the widget.
        """
        if hasattr(self, 'widget_layout'):
            return self.widget_layout._repr_mimebundle_(**kwargs)
        elif hasattr(self, 'figure') and self.figure is not None:
            return self.figure._repr_mimebundle_(**kwargs)
        return {}


def create_nmf_widget(cfg_path: str | Path = "config.json") -> NMFHeatmapWidget:
    """Create an NMF heatmap widget."""
    return NMFHeatmapWidget(cfg_path)