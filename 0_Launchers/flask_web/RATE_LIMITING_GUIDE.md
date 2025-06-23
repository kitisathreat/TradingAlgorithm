# YFinance Rate Limiting Guide for EC2 Deployment

## Overview

This guide explains how to deploy and manage the rate limiting system for yfinance on your EC2 instance to work around rate limiting issues.

## What the Rate Limiting System Does

The rate limiting system provides multiple layers of protection against yfinance rate limiting:

### 1. **Intelligent Rate Limiting**
- Limits requests to 30 per minute and 1000 per hour (configurable)
- Implements exponential backoff with jitter on errors
- Queues requests to prevent overwhelming the API

### 2. **Smart Caching**
- Caches stock data for 15 minutes (configurable)
- Reduces redundant API calls
- Improves response times for repeated requests

### 3. **Fallback Mechanisms**
- Generates realistic synthetic data when yfinance fails
- Multiple retry strategies with different approaches
- Graceful degradation to ensure your app keeps working

### 4. **Monitoring and Management**
- Real-time statistics and monitoring
- API endpoints for management
- Command-line tools for administration

## Files Created

1. **`rate_limiter.py`** - Core rate limiting module
2. **Updated `flask_app.py`** - Integrated rate limiting
3. **Updated `requirements.txt`** - Added necessary dependencies
4. **`RATE_LIMITING_GUIDE.md`** - This guide

## Deployment Steps

### Step 1: Deploy the Updated Code

The rate limiting system is already integrated into your Flask app. When you deploy to EC2:

1. Upload the updated `flask_app.py` and `rate_limiter.py`
2. Update your `requirements.txt`
3. Restart your Flask application

### Step 2: Verify Installation

Check that the rate limiting system is working:

```bash
# SSH into your EC2 instance
ssh -i your-key.pem ec2-user@your-ec2-ip

# Navigate to your Flask app directory
cd /path/to/your/flask/app

# Check if rate limiter is available
python -c "from rate_limiter import get_rate_limiter_stats; print('Rate limiter available')"
```

### Step 3: Test the System

Use the API endpoints to test the rate limiting:

```bash
# Get rate limiter statistics
curl http://your-ec2-ip/api/rate_limiter_stats

# Test data fetching with multiple symbols
curl -X POST http://your-ec2-ip/api/test_rate_limiting \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL", "MSFT", "GOOGL"], "days": 7}'
```

## API Endpoints

### 1. Get Rate Limiter Statistics
```
GET /api/rate_limiter_stats
```
Returns current statistics including:
- Total requests made
- Cache hit rate
- Rate limiting events
- Error counts
- Queue status

### 2. Clear Cache
```
POST /api/clear_yfinance_cache
```
Clears all cached stock data. Useful if you suspect stale data.

### 3. Reset Rate Limits
```
POST /api/reset_rate_limits
```
Resets all rate limiting counters. Use if you're getting blocked.

### 4. Test Rate Limiting
```
POST /api/test_rate_limiting
```
Test data fetching with multiple symbols:
```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "days": 7
}
```

## Configuration Options

You can modify the rate limiting behavior by editing `rate_limiter.py`:

```python
# In the RateLimiter.__init__ method:
def __init__(self, 
             max_requests_per_minute: int = 30,      # Adjust this
             max_requests_per_hour: int = 1000,      # Adjust this
             cache_duration_minutes: int = 15,       # Adjust this
             cache_dir: str = "cache",               # Cache location
             enable_proxy_rotation: bool = False):   # Future feature
```

## Monitoring and Management

### Real-time Monitoring

You can monitor the rate limiting system in real-time by checking the statistics endpoint:

```bash
# Watch statistics in real-time
watch -n 2 'curl -s http://your-ec2-ip/api/rate_limiter_stats | python -m json.tool'
```

### Log Monitoring

The rate limiting system logs important events. Monitor your Flask logs:

```bash
# Follow Flask logs
tail -f /var/log/your-flask-app.log

# Look for rate limiting events
grep -i "rate limit\|backoff\|cache" /var/log/your-flask-app.log
```

### Performance Metrics

Key metrics to monitor:

1. **Success Rate**: Should be > 90%
2. **Cache Hit Rate**: Higher is better (reduces API calls)
3. **Rate Limited Requests**: Should be low (< 5% of total)
4. **Backoff Events**: Should be very low (< 1% of total)

## Troubleshooting

### Problem: Still Getting Rate Limited

**Solutions:**
1. Reduce `max_requests_per_minute` to 20 or 15
2. Increase `cache_duration_minutes` to 30
3. Clear cache and reset rate limits
4. Check if multiple instances are running

### Problem: Synthetic Data Being Used Too Often

**Solutions:**
1. Check your EC2 instance's IP reputation
2. Verify yfinance is working: `python -c "import yfinance as yf; print(yf.Ticker('AAPL').history(period='1d'))"`
3. Consider using a different data source as backup

### Problem: High Error Rates

**Solutions:**
1. Check network connectivity
2. Verify yfinance version: `pip show yfinance`
3. Check for DNS issues
4. Monitor EC2 instance resources

### Problem: Cache Not Working

**Solutions:**
1. Check disk space: `df -h`
2. Verify cache directory permissions
3. Check if cache directory exists: `ls -la cache/`

## Advanced Configuration

### Custom Cache Location

Set a custom cache location for better performance:

```python
# In your Flask app initialization
from rate_limiter import get_rate_limiter

# Use a faster storage location (e.g., /tmp for RAM disk)
rate_limiter = get_rate_limiter()
rate_limiter.cache_dir = Path("/tmp/yfinance_cache")
```

### Multiple Instances

If running multiple Flask instances, consider:

1. Using a shared cache directory
2. Implementing distributed rate limiting
3. Using a Redis cache backend (future enhancement)

### Proxy Rotation (Future Feature)

For severe rate limiting, you can enable proxy rotation:

```python
# This feature is prepared but not fully implemented
rate_limiter = RateLimiter(enable_proxy_rotation=True)
```

## Best Practices

### 1. **Monitor Regularly**
- Check statistics daily
- Set up alerts for high error rates
- Monitor cache hit rates

### 2. **Optimize Cache Usage**
- Use longer cache durations for stable stocks
- Clear cache during market hours for active stocks
- Monitor cache disk usage

### 3. **Handle Failures Gracefully**
- Always have fallback data available
- Implement circuit breakers for repeated failures
- Log all errors for analysis

### 4. **Scale Appropriately**
- Start with conservative rate limits
- Increase gradually based on success rates
- Monitor for signs of rate limiting

## Performance Expectations

With the rate limiting system in place, you should expect:

- **Success Rate**: 95%+ for normal operations
- **Response Time**: < 2 seconds for cached data, < 5 seconds for fresh data
- **Cache Hit Rate**: 60-80% depending on usage patterns
- **Rate Limiting Events**: < 5% of total requests
- **Synthetic Data Usage**: < 10% of total requests

## Support and Maintenance

### Regular Maintenance Tasks

1. **Weekly**: Check rate limiter statistics
2. **Monthly**: Clear old cache files
3. **Quarterly**: Review and adjust rate limits
4. **As Needed**: Monitor for yfinance API changes

### Emergency Procedures

If yfinance completely blocks your EC2 instance:

1. Reset rate limits immediately
2. Clear all cache
3. Reduce rate limits to minimum
4. Consider using a different IP (new EC2 instance)
5. Implement alternative data sources

## Conclusion

The rate limiting system provides robust protection against yfinance rate limiting while maintaining high availability for your trading algorithm. The system automatically handles most rate limiting scenarios and provides fallback mechanisms to ensure your application continues to function.

For questions or issues, check the logs and use the monitoring endpoints to diagnose problems. The system is designed to be self-healing and should handle most rate limiting challenges automatically. 