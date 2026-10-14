FROM python:3.11.9-slim

# Update CACHE_BUST to force a full layer rebuild when Railway serves stale pip layers.
# Format: YYYY-MM-DD. Change this value any time you suspect Railway is caching old packages.
ARG CACHE_BUST=2026-03-26
RUN echo "Cache bust: ${CACHE_BUST}"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY deepsearch/ ./deepsearch/

# Set environment defaults
ENV PYTHONUNBUFFERED=1
ENV WORKER_LOG_FORMAT=json

# Run worker
CMD ["python", "-m", "deepsearch.worker"]
