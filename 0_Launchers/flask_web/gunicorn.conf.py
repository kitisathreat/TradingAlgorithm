"""
Gunicorn configuration for Trading Algorithm Flask Web Interface
Optimized for production deployment on AWS
"""

import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = os.environ.get('GUNICORN_WORKERS', multiprocessing.cpu_count() * 2 + 1)
worker_class = os.environ.get('GUNICORN_WORKER_CLASS', 'eventlet')
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
preload_app = True

# Timeout settings
timeout = 60  # Increased for ML operations
keepalive = 2
graceful_timeout = 60  # Increased for graceful shutdown

# Logging
accesslog = os.environ.get('GUNICORN_ACCESS_LOG', '-')
errorlog = os.environ.get('GUNICORN_ERROR_LOG', '-')
loglevel = os.environ.get('GUNICORN_LOG_LEVEL', 'info')
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'trading-algorithm-web'

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Memory and performance
max_requests_jitter = 50
worker_tmp_dir = '/dev/shm'  # Use RAM for temporary files

# SSL (if using HTTPS)
# keyfile = '/path/to/keyfile'
# certfile = '/path/to/certfile'

# Development settings (override in production)
reload = os.environ.get('GUNICORN_RELOAD', 'false').lower() == 'true'
reload_engine = 'auto'

def when_ready(server):
    """Called just after the server is started"""
    server.log.info("Trading Algorithm Web Interface is ready to serve requests")

def worker_int(worker):
    """Called just after a worker has been initialized"""
    worker.log.info("Worker %s initialized", worker.pid)

def pre_fork(server, worker):
    """Called just before a worker has been forked"""
    server.log.info("Worker %s will be spawned", worker.pid)

def post_fork(server, worker):
    """Called just after a worker has been forked"""
    server.log.info("Worker %s spawned", worker.pid)

def post_worker_init(worker):
    """Called just after a worker has initialized the application"""
    worker.log.info("Worker %s initialized application", worker.pid)

def worker_abort(worker):
    """Called when a worker has been aborted"""
    worker.log.info("Worker %s aborted", worker.pid) 