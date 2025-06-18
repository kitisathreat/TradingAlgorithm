# Flask Web Interface for Neural Network Trading System

This Flask-based web interface provides an alternative to the PyQt5 GUI, allowing you to run your trading algorithm interface in any modern web browser.

## Features

- **Web-based Interface**: Access your trading system from any device with a web browser
- **Real-time Stock Data**: Load and analyze stock data with technical indicators
- **Interactive Training**: Submit trading decisions to train the AI model
- **Live Predictions**: Get AI-powered trading predictions
- **Real-time Updates**: WebSocket-based real-time communication
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Quick Start

### Option 1: Windows Batch File (Recommended)
1. Double-click `run_flask_web.bat`
2. Wait for the environment setup to complete
3. Open your browser and go to `http://localhost:5000`

### Option 2: Python Script
1. Open a terminal/command prompt
2. Navigate to this directory
3. Run: `python run_flask_web.py`
4. Open your browser and go to `http://localhost:5000`

## How It Works

The Flask web interface replicates all the key functionality of your PyQt5 GUI:

### Data & Analysis Tab
- **Stock Selection**: Choose from a list of popular stocks
- **Technical Indicators**: View RSI, MACD, Bollinger Bands, and more
- **Interactive Chart**: Real-time stock price chart with Chart.js
- **Trading Decisions**: Submit BUY/SELL/HOLD decisions with reasoning

### Training Tab
- **Training Controls**: Set epochs and start model training
- **Progress Tracking**: Real-time training progress updates
- **Statistics**: View training data distribution and model performance

### Predictions Tab
- **AI Predictions**: Get trading recommendations from your trained model
- **Confidence Scores**: See prediction confidence levels
- **Reasoning**: Understand the AI's decision-making process

### Status Tab
- **System Information**: Check model availability and connection status
- **Feature Overview**: See what capabilities are available

## Technical Details

### Dependencies
- **Flask 2.3.3**: Web framework
- **Flask-SocketIO 5.3.6**: Real-time communication
- **TensorFlow 2.13.0**: Neural network training
- **yfinance 0.2.63**: Stock data fetching
- **Chart.js**: Interactive charts
- **Bootstrap 5**: Responsive UI framework

### Architecture
- **Backend**: Flask with Socket.IO for real-time updates
- **Frontend**: HTML5, CSS3, JavaScript with Bootstrap
- **Charts**: Chart.js for interactive stock charts
- **Communication**: REST API + WebSocket for real-time features

### File Structure
```
flask_web/
├── flask_app.py          # Main Flask application
├── requirements.txt      # Python dependencies
├── run_flask_web.bat    # Windows launcher
├── run_flask_web.py     # Cross-platform launcher
├── templates/
│   └── index.html       # Main web interface
└── README.md           # This file
```

## Advantages Over PyQt5 GUI

1. **Cross-platform**: Works on any device with a web browser
2. **No Installation**: No need to install desktop applications
3. **Remote Access**: Access from anywhere on your network
4. **Mobile Friendly**: Responsive design works on phones and tablets
5. **Easy Updates**: Web interface updates automatically
6. **Familiar Interface**: Uses standard web technologies

## Troubleshooting

### Port Already in Use
If you see "Address already in use" error:
1. Check if another application is using port 5000
2. Stop the other application or modify the port in `flask_app.py`

### Model Trainer Not Available
If the model trainer fails to initialize:
1. Check that all dependencies are installed
2. Verify the orchestrator path is correct
3. Check the console output for specific error messages

### Browser Compatibility
The interface works best with:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Customization

### Adding New Stocks
Edit the `STOCK_SYMBOLS` list in `flask_app.py` to add more stocks.

### Changing the Port
Modify the port number in the `socketio.run()` call at the bottom of `flask_app.py`.

### Styling
The interface uses Bootstrap 5 and custom CSS. Modify the styles in `templates/index.html` to customize the appearance.

## Security Notes

- The web interface is designed for local development
- For production use, add proper authentication and HTTPS
- The current setup allows access from any device on your network

## Support

If you encounter issues:
1. Check the console output for error messages
2. Verify all dependencies are installed correctly
3. Ensure Python 3.9 is being used
4. Check that the orchestrator path is accessible

The Flask web interface provides the same powerful trading algorithm capabilities as your PyQt5 GUI, but with the convenience and accessibility of a web-based interface. 