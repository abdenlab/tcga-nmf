/**
 * NMF Heatmap Widget - Frontend JavaScript for anywidget
 */

function render({ model, el }) {
    console.log('NMF Widget: render() called');
    
    // Clear any existing content
    el.innerHTML = '';
    
    // Create container div
    const container = document.createElement('div');
    container.className = 'nmf-widget-container';
    
    // Create plotly container
    const plotDiv = document.createElement('div');
    plotDiv.className = 'nmf-heatmap';
    plotDiv.style.width = '100%';
    plotDiv.style.height = '600px';
    plotDiv.id = 'nmf-plot-' + Math.random().toString(36).substr(2, 9);
    
    container.appendChild(plotDiv);
    el.appendChild(container);
    
    console.log('NMF Widget: DOM structure created');
    
    // Add loading indicator
    plotDiv.innerHTML = '<div class="nmf-loading">Loading NMF Heatmap...</div>';
    
    // The actual Plotly rendering will be handled by the FigureWidget
    // This just ensures we have the proper DOM structure
}

function initialize({ model, el }) {
    console.log('NMF Widget: initialize() called');
    // Any initialization logic can go here
}

export default { render, initialize };