"""
NMF Heatmap Widget - anywidget wrapper for interactive NMF visualization.
"""
from __future__ import annotations
import anywidget
import plotly.graph_objects as go
from pathlib import Path
import traceback
import ipywidgets as widgets
import json
# This import is correct, assuming nmf_plotting.py is in the same directory.
from .nmf_plotting import create_nmf_heatmap_figure, create_empty_placeholder_figure

def load_cfg(cfg_path):
    """Load configuration from JSON file."""
    with open(cfg_path, 'r') as f:
        return json.load(f)

class NMFHeatmapWidget(anywidget.AnyWidget):
    """Plotly + anywidget interactive heat-map of NMF sample activities."""
    
    _esm = """
        function render({ model, el }) {
            // Simple container for the Plotly figure
            el.innerHTML = '<div class="nmf-widget-container"></div>';
        }
        
        export default { render };
    """

    def __init__(self, cfg_path: str | Path = "config.json"):
        super().__init__()
    
        # All of your original initialization logic remains exactly the same...
        self._cfg_path = cfg_path
        self._current_sort = "component"
    
        try:
            cfg = load_cfg(cfg_path)
            self.k_files = []
            data_dir = cfg.get("NMF_DATA_DIRECTORY", ".")
            file_pattern = cfg.get("NMF_FILE_PATTERN", "*.csv")
            import glob, re
            from pathlib import Path
            
            for file_path in glob.glob(str(Path(data_dir) / file_pattern)):
                k_match = re.search(r'_[kK](\d+)\.csv$', file_path)
                if k_match:
                    k_value = k_match.group(1)
                    self.k_files.append({
                        "filename": Path(file_path).name, "k_value": k_value,
                        "path": file_path, "display_name": f"K = {k_value}"
                    })
            
            if not self.k_files:
                default_csv = cfg.get("DEFAULT_CSV_FILENAME")
                if default_csv:
                    self._current_k_file = default_csv
                    self.figure = go.FigureWidget(create_nmf_heatmap_figure(cfg_path, self._current_sort))
                    self.sort_dropdown = self._create_sort_controls()
                    self.widget_layout = widgets.VBox([self.sort_dropdown, self.figure])
                else:
                    self.figure = go.FigureWidget(create_empty_placeholder_figure("No data files available"))
                    self.widget_layout = widgets.VBox([self.figure])
            else:
                self.k_files.sort(key=lambda x: int(x["k_value"]))
                self._current_k_file = self.k_files[0]["filename"]
                self.figure = go.FigureWidget(create_nmf_heatmap_figure(
                    cfg_path, self._current_sort, k_value=self.k_files[0]["k_value"]
                ))
                k_options = [(f"K = {k['k_value']}", k["filename"]) for k in self.k_files]
                self.k_dropdown = widgets.Dropdown(
                    options=k_options, value=self._current_k_file,
                    description='NMF Components:', style={'description_width': 'initial'}
                )
                self.k_dropdown.observe(self._on_k_change, names='value')
                self.sort_dropdown = self._create_sort_controls()
                self.control_box = widgets.HBox([self.k_dropdown, self.sort_dropdown])
                self.widget_layout = widgets.VBox([self.control_box, self.figure])
    
        except Exception as e:
            print(f"Error initializing widget: {e}")
            traceback.print_exc()
            self.figure = go.FigureWidget(create_empty_placeholder_figure())
            self.widget_layout = widgets.VBox([self.figure])
            
        print("NMFHeatmapWidget: Initialization complete.")
        # --- CHANGE #1: The explicit display call below has been removed. ---
        # self._display() # This line is now deleted.

    # All of your other methods (_create_k_selector, _on_k_change, etc.)
    # remain exactly the same as you wrote them.
    def _create_k_selector(self):
        if not hasattr(self, 'k_files') or len(self.k_files) <= 1:
            return widgets.HTML("")
        options = [(f"K = {k_file['k_value']}", k_file["filename"]) for k_file in self.k_files]
        dropdown = widgets.Dropdown(
            options=options, value=self._current_k_file,
            description='NMF Components:', style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        dropdown.observe(self._on_k_change, names='value')
        return dropdown
    
    def _on_k_change(self, change):
        new_k_file = change['new']
        if new_k_file != self._current_k_file:
            self._current_k_file = new_k_file
            k_value = next((k_file["k_value"] for k_file in self.k_files if k_file["filename"] == new_k_file), "Unknown")
            print(f"NMFHeatmapWidget: Changed to K={k_value} ({new_k_file})")
            self.figure = self._create_figure(self._cfg_path, self._current_sort, self._current_k_file)
            self.widget_layout.children = [self.control_box, self.figure]

    def _create_sort_controls(self):
        dropdown = widgets.Dropdown(
            options=[
                ('Sort by Component', 'component'), ('Sort Alphabetically', 'alphabetical'),
                ('Sort by Cancer Type', 'cancer_type'), ('Sort by Organ System', 'organ_system')
            ], value=self._current_sort, description='Sort by:',
            style={'description_width': 'initial'}, layout=widgets.Layout(width='200px')
        )
        dropdown.observe(self._on_sort_change, names='value')
        return dropdown
    
    def _on_sort_change(self, change):
        new_sort = change['new']
        if new_sort != self._current_sort:
            self._current_sort = new_sort
            self.figure = self._create_figure(self._cfg_path, self._current_sort, self._current_k_file)
            self.widget_layout.children = [self.control_box, self.figure]

    def _create_figure(self, cfg_path: str | Path, sort_method: str, k_filename: str = None) -> go.FigureWidget:
        try:
            k_value = next((k_file["k_value"] for k_file in self.k_files if k_file["filename"] == k_filename), None)
            print(f"Creating figure with K={k_value}, sort={sort_method}")
            fig = create_nmf_heatmap_figure(cfg_path, sort_method=sort_method, k_filename=k_filename, k_value=k_value)
            return go.FigureWidget(fig)
        except Exception as e:
            print(f"Error creating figure: {e}")
            traceback.print_exc()
            return go.FigureWidget(create_empty_placeholder_figure())
    
    # The _display method has been completely removed, as it's no longer needed.

    # --- CHANGE #2: This special method is now corrected. ---
    def _repr_mimebundle_(self, **kwargs):
        """
        This method is called by Jupyter to display the widget.
        Instead of calling display(), it now returns the widget's layout data,
        allowing Jupyter to handle the rendering correctly and only once.
        """
        if hasattr(self, 'widget_layout'):
            return self.widget_layout._repr_mimebundle_(**kwargs)
        elif hasattr(self, 'figure') and self.figure is not None:
            return self.figure._repr_mimebundle_(**kwargs)
        return {}


def create_nmf_widget(cfg_path: str | Path = "config.json") -> NMFHeatmapWidget:
    """Create an NMF heatmap widget."""
    return NMFHeatmapWidget(cfg_path)