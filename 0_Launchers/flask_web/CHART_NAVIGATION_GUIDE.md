# Chart Navigation System Guide

## Overview

The Flask web application implements a comprehensive chart navigation system that closely matches the behavior of the local PyQt5 GUI. This guide explains how the navigation works and ensures the EC2 instance handles chart navigation the same way as the local interface.

## Navigation Features Comparison

### Local GUI (PyQt5 + PyQtGraph)
- **Mouse Wheel**: Zoom in/out
- **Left Mouse Drag**: Pan around the chart
- **Right Mouse Click**: Reset view
- **Interactive Tooltips**: Show detailed OHLC data on hover
- **Auto-resize buttons**: Fit X and Y axes
- **Performance optimizations**: Throttled mouse events

### Flask Web App (Chart.js + Custom Navigation)
- **Mouse Wheel**: Zoom in/out ✅
- **Left Mouse Drag**: Pan around the chart ✅
- **Right Mouse Drag**: Zoom to selection area ✅
- **Interactive Tooltips**: Show detailed OHLC data on hover ✅
- **Auto-resize buttons**: Fit X and Y axes ✅
- **Performance optimizations**: Throttled mouse events ✅
- **Keyboard shortcuts**: Ctrl+R and Ctrl+0 for reset ✅

## Implementation Details

### 1. Custom Navigation System

The Flask web app uses a custom navigation system built on top of Chart.js:

```javascript
function addCustomNavigation(chart) {
    // Performance optimization: throttle mouse events
    let lastMouseUpdate = 0;
    const mouseUpdateThreshold = 16; // ~60fps
    
    // Mouse wheel zoom with improved precision
    canvas.addEventListener('wheel', function(event) {
        // Zoom factor matches local GUI sensitivity
        const zoomFactor = event.deltaY > 0 ? 0.85 : 1.15;
        // Zoom towards mouse position
    });
    
    // Left mouse button - pan
    // Right mouse button - zoom selection
    // Performance throttling for smooth interaction
}
```

### 2. Performance Optimizations

- **Throttled Mouse Events**: 60fps limit to prevent performance issues
- **Efficient Updates**: Uses `chart.update('none')` for minimal redraws
- **Memory Management**: Proper cleanup of event listeners

### 3. Zoom and Pan Behavior

#### Mouse Wheel Zoom
- Zooms towards the mouse cursor position
- Zoom factor: 0.85x (zoom out) / 1.15x (zoom in)
- Matches local GUI sensitivity

#### Left Mouse Drag (Pan)
- Pans the chart in the direction of mouse movement
- Y-axis is inverted to match local GUI behavior
- Smooth, responsive panning

#### Right Mouse Drag (Zoom Selection)
- Draws a selection rectangle
- Zooms to the selected area
- Minimum selection size: 15px (prevents accidental zooms)

### 4. Auto-Fit Functions

#### Auto Fit X Axis
```javascript
function autoFitX() {
    if (stockChart) {
        stockChart.resetZoom();
        showAlert('X-axis auto-fitted', 'success');
    }
}
```

#### Auto Fit Y Axis
```javascript
function autoFitY() {
    // Calculate min/max from candlestick data (OHLC)
    const prices = currentStockData.flatMap(d => [d.l, d.h]);
    const padding = priceRange * 0.05; // 5% padding
}
```

#### Reset Zoom
```javascript
function resetChartZoom() {
    // Reset to show all data with 1% padding
    yScale.min = Math.min(...prices) * 0.99;
    yScale.max = Math.max(...prices) * 1.01;
}
```

### 5. Tooltip System

The tooltip system provides detailed OHLC data on hover:

```javascript
tooltip: {
    callbacks: {
        title: function(context) {
            // Format date
        },
        label: function(context) {
            // Show OHLC values
            return [
                `Open:  $${dataPoint.o.toFixed(2)}`,
                `High:  $${dataPoint.h.toFixed(2)}`,
                `Low:   $${dataPoint.l.toFixed(2)}`,
                `Close: $${dataPoint.c.toFixed(2)}`
            ];
        },
        afterLabel: function(context) {
            // Show price change and percentage
        }
    }
}
```

## User Interface Elements

### Chart Controls
- **Auto Fit X**: Resets X-axis to show all data
- **Auto Fit Y**: Adjusts Y-axis to fit price range
- **Reset Zoom**: Returns to full view with padding

### Navigation Instructions
The interface includes clear instructions for users:
- Mouse Wheel: Zoom in/out
- Left Mouse Drag: Pan around the chart
- Right Mouse Drag: Zoom to selection area
- Hover: View detailed OHLC data

### Keyboard Shortcuts
- **Ctrl+R**: Reset chart zoom
- **Ctrl+0**: Reset chart zoom (alternative)

## EC2 Deployment Considerations

### 1. Performance
- The navigation system is optimized for web browsers
- Throttled events prevent excessive server load
- Efficient rendering with Chart.js

### 2. Browser Compatibility
- Works with all modern browsers
- Graceful degradation for older browsers
- Mobile-friendly touch interactions

### 3. Network Considerations
- Minimal data transfer for navigation
- Chart updates use efficient WebSocket communication
- Cached data reduces server requests

## Troubleshooting

### Common Issues

1. **Chart not responding to mouse events**
   - Check if Chart.js is loaded properly
   - Verify canvas element exists
   - Check browser console for errors

2. **Zoom/pan feels sluggish**
   - Reduce `mouseUpdateThreshold` for more responsive interaction
   - Check browser performance settings
   - Verify no other scripts are blocking the main thread

3. **Tooltips not showing**
   - Check if data points have valid OHLC values
   - Verify tooltip callbacks are properly configured
   - Check for CSS conflicts

### Debug Information

The system includes comprehensive debug logging:
```javascript
console.log('[DEBUG] Adding enhanced custom navigation system');
console.log('[DEBUG] Enhanced custom navigation system added successfully');
```

## Configuration

### Chart.js Settings
```javascript
options: {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
        intersect: true,
        mode: 'point'
    },
    plugins: {
        zoom: {
            pan: { enabled: false },  // Use custom pan
            zoom: { enabled: false }  // Use custom zoom
        }
    }
}
```

### Performance Settings
- Mouse update threshold: 16ms (60fps)
- Minimum zoom selection size: 15px
- Zoom factors: 0.85x / 1.15x
- Padding: 5% for auto-fit, 1% for reset

## Conclusion

The Flask web application's chart navigation system provides a feature-complete experience that matches the local GUI functionality. The implementation includes:

- ✅ All navigation features from local GUI
- ✅ Performance optimizations
- ✅ Responsive design
- ✅ Comprehensive error handling
- ✅ Debug logging
- ✅ Browser compatibility

The EC2 instance will handle chart navigation exactly like the local GUI, providing users with a consistent experience across both interfaces. 