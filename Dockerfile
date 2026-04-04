# ─── SRE Incident Response Environment ───────────────────────────────────────
# Based on openenv-base for the Meta OpenEnv framework.
# Optimised for HF Spaces CPU Basic tier (2 vCPUs, 8 GB RAM).
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# Install curl for the Docker HEALTHCHECK
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy environment code
COPY . .

# Install server dependencies
RUN pip install --no-cache-dir -r server/requirements.txt

# Expose the default OpenEnv port
EXPOSE 8000

# Docker HEALTHCHECK (every 30 s, 5 s timeout, 3 retries)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start uvicorn bound to all interfaces
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
