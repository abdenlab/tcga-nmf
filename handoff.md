# NMF Visualization Application - Developer Reference

## Project Purpose
Interactive visualization tool for Non-negative Matrix Factorization (NMF) results with bidirectional selection between heatmaps and UMAP plots.

## Core Components

### Data Pipeline
- `data_loader.py`: 
    - `get_prepared_data()` - Primary data loading entry point
    - `_load_from_csv()/_load_from_npy()` - File format-specific loaders (CSV is slower)
    - `load_cfg()` - Configuration parser used throughout
- `data_utils.py`: 
    - `load_all_data()` - Orchestrates all data preparation
    - Returns H-matrix, sample IDs, cancer types, color mappings, and sorted indices

### Visualization Components
- `nmf_plotting.py`:
    - `create_nmf_heatmap_figure()` - Generates main heatmap with 6 subplots
    - `_add_main_heatmap()` - Core heatmap rendering (performance bottleneck)
    - `_add_proportional_bar_chart()` - Adds stacked bar chart of activities
    - `_add_*_strip_with_legend()` - Creates annotation strips (components, cancer types, etc.)
    - `_configure_layout()` - Sets up multi-panel figure layout with legends
- `umap_utils.py`:
    - `perform_umap_reduction()` - UMAP calculation (major performance bottleneck)
    - `create_umap_visualization()` - Creates interactive jscatter UMAP plot with lasso selection

### Interactivity Framework
- `scatter_plot_utils.py`:
    - `create_nmf_scatter_visualization()` - **MAIN ENTRY POINT** - Creates complete visualization
    - `update_selection()` - Key bidirectional function that propagates selections
    - `on_plotly_selection()` - Handles selection events from heatmap
    - `on_umap_selection()` - Handles selection events from UMAP scatter
    - Contains state management for tracking selected samples

### Widget Implementation
- `nmf_widget.py`:
    - `NMFHeatmapWidget` - Custom widget extending anywidget.AnyWidget
    - `create_nmf_widget()` - Factory function
    - `refresh()` - Method to update widget with new data
    - `save_figure()` - Utility to export static visualization
- JavaScript rendering:
    - `nmf_widget.js` - Basic DOM setup and integration with Jupyter
    - `nmf_widget_advanced.js` - Direct Plotly integration with custom events
    - `nmf_widget.css` - Styling for widget container, loading state, etc.

## Key Data Structures

- **H-matrix** - Core NMF component activity matrix:
  - Shape: (n_samples, n_components)
  - Contains non-negative values representing component activity
- **Sample metadata**:
  - Sample IDs
  - Cancer types (TCGA codes like 'BRCA', 'LUAD', etc.)
  - Organ systems (e.g., 'Reproductive System (Female)')
  - Embryonic layers (e.g., 'Ectoderm', 'Mesoderm', 'Endoderm')
- **Color mappings**:
  - Component colors (loaded from nmf_component_color_map.json)
  - Cancer type colors (loaded from cancer_type_color_map.json)
  - Automatically generated colors for organ systems and embryonic layers

## Development Workflow

1. Entry notebook: `HmWidget_Jscatter.ipynb`
2. Typical implementation path:
    - Configure via `config.json` (data paths, component counts, etc.)
    - Load data through `data_loader.py`
    - Create visualizations with `create_nmf_scatter_visualization()`
    - Connect interactive elements via callbacks in Jupyter
3. Testing workflow:
    - Make code changes
    - Use `importlib.reload()` to refresh modules
    - Re-run visualization cells
    - Check browser console for JS errors

## Bidirectional Selection Implementation

The selection mechanism flows through the following steps:

1. **Heatmap to UMAP direction**:
    - User selects points on Plotly heatmap
    - `on_plotly_selection()` captures selection indices
    - `update_selection("heatmap", selected_samples)` is called
    - UMAP plot updates opacity or filters to show only selected samples
    - Selected sample details appear in data table

2. **UMAP to Heatmap direction**:
    - User selects points using jscatter lasso selection
    - `on_umap_selection()` captures selection indices
    - `update_selection("umap", selected_samples)` is called
    - Plotly heatmap highlights selected samples and adjusts view
    - Selected sample details appear in data table

3. **Current implementation limitation**:
    - UMAP selection currently filters rather than using opacity
    - This loses the context of unselected points

## Critical Performance Issues

1. **UMAP Calculation** (`perform_umap_reduction()`):
    - Extremely CPU-intensive with O(n²) scaling behavior
    - Consider pre-computation, caching, or progressive loading
    - Parameters to optimize: n_neighbors, min_dist, spread

2. **Large Matrix Rendering** (`_add_main_heatmap()`):
    - Rendering bottleneck for large datasets
    - Implement downsampling for initial views
    - Consider chunked loading or hierarchical aggregation
    - Potential for WebGL-based rendering via Plotly's Scattergl

3. **Selection Updates** (`update_selection()`):
    - Expensive filtering operations
    - Potential for optimization with indexed lookups
    - Consider caching selection results
    - Profile the specific bottlenecks in selection propagation

4. **Widget Initialization**:
    - Loading all data at startup causes significant delay
    - Implement progressive/lazy loading
    - Split initialization from data loading
    - Add loading indicators with percentage completion

## Known Issues

1. **Opacity Implementation**:
    - Selected/unselected points in UMAP need opacity-based differentiation
    - Current implementation filters rather than applying opacity
    - Modify `update_selection()` to update opacity values instead of filtering

2. **Large Dataset Performance**:
    - Significant slowdown with large matrices (>500 samples)
    - Memory usage spikes during UMAP calculation
    - Plotly rendering becomes sluggish with many data points

3. **Widget Rendering**:
    - Environment-specific rendering issues in different Jupyter environments
    - Fallback to direct Plotly figure rendering is available via `.figure.show()`
    - Check browser console for JavaScript errors

4. **Selection Mechanism**:
    - Performance bottlenecks in bidirectional selection
    - Selection sometimes fails to propagate correctly
    - Edge cases with empty or very large selections need handling

## Recent Development Focus & Next Steps

### Recent Work
- Bidirectional selection between visualizations
- Opacity-based selection visualization
- UI responsiveness improvements
- Event handling for selection propagation

### Priority Improvements
- Implement UMAP opacity-based selection instead of filtering
- Optimize UMAP calculation with progress reporting
- Add caching layer for both data and visualization results
- Implement hierarchical data exploration for large datasets
- Fix widget rendering issues across environments

## Configuration Options

Key settings in `config.json`:
- `DATA_FILENAME_H_MATRIX`: Path to NMF H-matrix data
- `DATA_FILENAME_SAMPLE_IDS`: Path to sample identifiers
- `DATA_FILENAME_CANCER_TYPES`: Path to cancer type annotations
- `JSON_FILENAME_COMPONENT_COLORS`: Component color mapping
- `JSON_FILENAME_CANCER_TYPE_COLORS`: Cancer type color mapping
- `DEFAULT_SORT_METHOD`: Sorting method for samples ("component" or "cancer")
