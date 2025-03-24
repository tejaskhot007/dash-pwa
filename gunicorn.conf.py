# gunicorn.conf.py

# Number of workers (adjust based on Render's free tier resources)
workers = 1  # Free tier has limited CPU, so use 1 worker to avoid overloading
threads = 2  # Use threads to handle concurrent requests

# Bind to the port Render expects
bind = "0.0.0.0:8050"

# Timeout (Render's free tier has a 30-second request timeout, but set Gunicorn's timeout higher to avoid conflicts)
timeout = 60

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"

# Worker class (use sync for simplicity on free tier)
worker_class = "sync"