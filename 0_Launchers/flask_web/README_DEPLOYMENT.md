# Trading Algorithm Web Interface - Deployment Guide

This directory contains the Flask-based web interface for the Neural Network Trading System with production-ready deployment configurations.

## Quick Start

### Local Development
```bash
# Windows/Linux/Mac
python run_flask_web.py

# Or run directly
python flask_app.py
```

### Production Deployment
```bash
# Windows
run_production.bat

# Linux/Mac (use Docker or manual setup)
docker-compose up -d
```

## Deployment Options

### 1. Local Production Server
- Uses Gunicorn for better performance
- Configured for production environment
- Includes proper logging and error handling

### 2. Docker Deployment
```bash
# Build and run with Docker
docker build -t trading-algorithm-web .
docker run -p 5000:5000 trading-algorithm-web

# Or use docker-compose
docker-compose up -d
```

### 3. AWS Elastic Beanstalk
- Use the `.ebextensions/` configuration
- Deploy with `eb deploy`
- Automatic scaling and load balancing

### 4. AWS EC2
- Manual deployment with Nginx and Supervisor setup
- Production-ready configuration
- Use Docker deployment for easier setup

### 5. AWS ECS (Container)
- Use the provided Dockerfile
- Deploy to ECS with load balancer
- Auto-scaling capabilities

## Configuration

### Environment Variables
- `FLASK_ENV`: Set to 'production' for production mode
- `SECRET_KEY`: Secret key for session management
- `LOG_LEVEL`: Logging level (INFO, WARNING, ERROR)
- `CORS_ORIGINS`: Allowed CORS origins
- `MODEL_SAVE_PATH`: Path for saving trained models
- `DATA_CACHE_DIR`: Path for caching stock data

### Production Settings
- Debug mode disabled
- Secure session cookies
- Proper logging configuration
- Health checks enabled
- WebSocket support for real-time updates

## Features

### Enhanced Chart Interactions
- Mouse wheel zoom
- Click and drag to pan
- Auto-fit X and Y axis buttons
- Candlestick color legends

### Comprehensive Stock Selection
- S&P100 stocks sorted by market cap
- Random and optimized picks
- Custom ticker support

### Advanced Technical Indicators
- RSI, MACD, SMA, EMA
- Bollinger Bands with width
- ATR (Average True Range)
- Volume metrics and ratios
- Price change and volatility
- Comprehensive market analysis

## Security Considerations

1. **HTTPS**: Always use HTTPS in production
2. **Secrets**: Use environment variables for sensitive data
3. **CORS**: Configure CORS properly for your domain
4. **Firewall**: Configure security groups appropriately
5. **Updates**: Keep dependencies updated

## Monitoring

- Application logs in `logs/` directory
- Health check endpoint: `/api/get_model_status`
- Performance monitoring with Gunicorn
- Error tracking and alerting

## Troubleshooting

### Common Issues
1. **Port conflicts**: Change port in configuration
2. **Memory issues**: Adjust worker settings in gunicorn.conf.py
3. **WebSocket issues**: Check proxy configuration
4. **Import errors**: Verify Python path and dependencies

### Logs
- Application logs: `logs/flask_app.log`
- Gunicorn logs: `logs/gunicorn.log`
- Nginx logs: `/var/log/nginx/` (if using Nginx)

## Support

For deployment issues:
1. Check the logs for error messages
2. Verify environment variables
3. Test locally before deploying
4. Review AWS documentation for specific services

For application issues:
1. Check the application logs
2. Verify the model trainer is working
3. Test with different stocks
4. Review the technical indicators 