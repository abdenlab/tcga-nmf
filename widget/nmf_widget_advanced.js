/**
 * Advanced NMF Heatmap Widget - Direct Plotly integration
 */

async function render({ model, el }) {
    console.log('NMF Widget: Advanced render() called');
    
    // Clear any existing content
    el.innerHTML = '';
    
    // Create container
    const container = document.createElement('div');
    container.className = 'nmf-widget-container';
    
    const plotDiv = document.createElement('div');
    plotDiv.className = 'nmf-heatmap';
    plotDiv.style.width = '100%';
    plotDiv.style.height = '600px';
    plotDiv.id = 'nmf-plot-' + Math.random().toString(36).substr(2, 9);
    
    container.appendChild(plotDiv);
    el.appendChild(container);
    
    // Add loading state
    plotDiv.innerHTML = '<div class="nmf-loading">Loading NMF Heatmap...</div>';
    
    // Try to load Plotly if available
    try {
        if (typeof window.Plotly !== 'undefined') {
            console.log('Plotly found, can create custom plots here');
            // You could create custom Plotly plots here if needed
        }
    } catch (error) {
        console.log('Plotly not available:', error);
    }
    
    console.log('NMF Widget: Advanced DOM structure created');
}

export default { render };