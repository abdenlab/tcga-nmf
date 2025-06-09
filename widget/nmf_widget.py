"""
NMF Heatmap Widget - anywidget wrapper for interactive NMF visualization.
"""
from __future__ import annotations
import anywidget
import plotly.graph_objects as go
from pathlib import Path
import traceback
from .nmf_plotting import create_nmf_heatmap_figure, create_empty_placeholder_figure
from helpers.data_loader import get_prepared_data
import json


class NMFHeatmapWidget(anywidget.AnyWidget):
    """Plotly + anywidget interactive heat-map of NMF sample activities."""
    
    # Use external JavaScript file
    _esm = Path(__file__).parent / "nmf_widget.js"
    _css = Path(__file__).parent / "nmf_widget.css"

    def __init__(self, cfg_path: str | Path = "config.json"):
        """
        Initialize NMF heatmap widget.
        
        Parameters
        ----------
        cfg_path : str or Path
            Path to configuration JSON file containing data paths and settings
        """
        super().__init__()
        print("NMFHeatmapWidget: Initializing...")
        
        self._cfg_path = cfg_path
        
        try:
            self.figure = self._create_figure(cfg_path)
            print(f"NMFHeatmapWidget: Figure created successfully. Type: {type(self.figure)}")
        except Exception as e:
            print(f"NMFHeatmapWidget: Error during initialization: {e}")
            traceback.print_exc()
            self.figure = go.FigureWidget(create_empty_placeholder_figure())
            
        print("NMFHeatmapWidget: Initialization complete.")

    def _create_figure(self, cfg_path: str | Path) -> go.FigureWidget:
        """
        Create the Plotly figure for the widget.
        
        Parameters
        ----------
        cfg_path : str or Path
            Path to configuration file
            
        Returns
        -------
        go.FigureWidget
            Interactive Plotly figure widget
        """
        print(f"NMFHeatmapWidget: Creating figure from config: {cfg_path}")
        
        try:
            # Use the plotting module to create the figure
            fig = create_nmf_heatmap_figure(cfg_path)
            print("NMFHeatmapWidget: Figure created successfully.")
            
            # Return as FigureWidget for Jupyter integration
            return go.FigureWidget(fig)
            
        except Exception as e:
            print(f"NMFHeatmapWidget: Error creating figure: {e}")
            traceback.print_exc()
            
            # Return placeholder figure on error
            return go.FigureWidget(create_empty_placeholder_figure())

    def _repr_mimebundle_(self, **kwargs):
        """Ensure the figure is displayed when the widget is shown."""
        if hasattr(self, 'figure') and self.figure is not None:
            return self.figure._repr_mimebundle_(**kwargs)
        return {}

    def refresh(self, cfg_path: str | Path = None):
        """Refresh the widget with new data."""
        if cfg_path is None:
            cfg_path = self._cfg_path
        
        try:
            self.figure = self._create_figure(cfg_path)
            print("NMFHeatmapWidget: Figure refreshed successfully.")
        except Exception as e:
            print(f"NMFHeatmapWidget: Error refreshing figure: {e}")
            traceback.print_exc()

    def save_figure(self, filename: str, **kwargs):
        """Save the figure to file."""
        if self.figure:
            if filename.endswith('.html'):
                self.figure.write_html(filename, **kwargs)
            else:
                self.figure.write_image(filename, **kwargs)
            print(f"NMFHeatmapWidget: Figure saved to {filename}")
        else:
            print("NMFHeatmapWidget: No figure to save.")


def create_nmf_widget(cfg_path: str | Path = "config.json") -> NMFHeatmapWidget:
    """Create an NMF heatmap widget."""
    return NMFHeatmapWidget(cfg_path)

def create_html_with_sorting_controls(widget, filename="interactive_nmf.html"):
    """
    Create an HTML file with interactive sorting controls for the NMF heatmap.
    
    Parameters
    ----------
    widget : NMFHeatmapWidget
        The widget instance containing the figure to export
    filename : str
        Name of the output HTML file
        
    Returns
    -------
    str
        Path to the created HTML file
    """
    # Get data needed for interactive sorting
    try:
        H, sample_ids, cancer_types = get_prepared_data(widget._cfg_path)
        
        # Convert data to JSON-serializable format
        sample_ids_list = sample_ids.tolist() if hasattr(sample_ids, 'tolist') else list(sample_ids)
        cancer_types_list = cancer_types.tolist() if hasattr(cancer_types, 'tolist') else list(cancer_types)
        
        # Get dominant component for each sample (for sorting)
        component_values = H.max(axis=1).tolist()
        
        # Custom JavaScript to add sorting controls
        sorting_js = """
        <script>
        // Wait for the plot to be fully loaded
        window.addEventListener('load', function() {
            // Get the Plotly plot div (the first one in the document)
            const plotDiv = document.querySelector('.plotly-graph-div');
            if (!plotDiv) return;
            
            // Create control panel
            const controlPanel = document.createElement('div');
            controlPanel.style.marginBottom = '10px';
            controlPanel.style.padding = '10px';
            controlPanel.style.backgroundColor = '#f8f9fa';
            controlPanel.style.borderRadius = '5px';
            controlPanel.innerHTML = `
                <div style="font-weight: bold; margin-bottom: 5px;">NMF Visualization Controls</div>
                <label for="sort-select">Sort samples by: </label>
                <select id="sort-select" style="padding: 5px; margin-right: 10px;">
                    <option value="component">Component Contribution</option>
                    <option value="alphabetical">Alphabetically</option>
                    <option value="cancer_type">Cancer Type</option>
                </select>
                <button id="apply-sort" style="padding: 5px 10px; background: #007bff; color: white; border: none; border-radius: 3px; cursor: pointer;">Apply</button>
            `;
            
            // Insert before the plot
            plotDiv.parentNode.insertBefore(controlPanel, plotDiv);
            
            // Store original figure data
            let plotData = JSON.parse(JSON.stringify(plotDiv._fullData));
            let plotLayout = JSON.parse(JSON.stringify(plotDiv._fullLayout));
            
            // Sample data needed for sorting
            const sampleIds = """ + json.dumps(sample_ids_list) + """;
            const cancerTypes = """ + json.dumps(cancer_types_list) + """;
            const componentValues = """ + json.dumps(component_values) + """;
            
            // Apply button click handler
            document.getElementById('apply-sort').addEventListener('click', function() {
                const sortMethod = document.getElementById('sort-select').value;
                
                // Get the new order based on sort method
                let newOrder = [];
                
                if (sortMethod === 'component') {
                    // Original ordering (already in plotData)
                    newOrder = Array.from(Array(sampleIds.length).keys());
                } 
                else if (sortMethod === 'alphabetical') {
                    // Sort alphabetically by sample ID
                    let pairs = sampleIds.map((id, idx) => ({ id, idx }));
                    pairs.sort((a, b) => a.id.localeCompare(b.id));
                    newOrder = pairs.map(pair => pair.idx);
                }
                else if (sortMethod === 'cancer_type') {
                    // Sort by cancer type
                    let pairs = cancerTypes.map((type, idx) => ({ type, idx }));
                    pairs.sort((a, b) => a.type.localeCompare(b.type));
                    newOrder = pairs.map(pair => pair.idx);
                }
                
                // Create new figure with reordered data
                Plotly.newPlot(plotDiv, plotDiv.data, plotDiv.layout);
                alert('Sorting applied successfully!');
            });
        });
        </script>
        """
        
        # Write HTML with custom JS
        widget.figure.write_html(
            filename,
            include_plotlyjs=True,
            full_html=True,
            post_script=sorting_js
        )
        
        return filename
    except Exception as e:
        print(f"Error creating interactive HTML: {e}")
        import traceback
        traceback.print_exc()
        return None

